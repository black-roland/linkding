import logging
import os
import re
import threading

from django.conf import settings

from bookmarks.models import sanitize_tag_name

logger = logging.getLogger(__name__)

_LABEL_PREFIX = "__label__"
_MIN_WORDS = 3
_MAX_TAG_LENGTH = 64
_KEEP_WORDS = {"why", "how", "what"}

_TITLE_SUFFIXES = (
    r" - youtube$",
    r" / хабр$",
    r" · github$",
    r" \| youtube$",
)
_TITLE_PREFIXES = (r"^github - ",)

_GARBAGE_PHRASES = (
    "смотрите любимые видео, слушайте любимые песни, загружайте собственные ролики и делитесь ими с друзьями, близкими и целым миром.",
    "enjoy the videos and music you love, upload original content, and share it all with friends, family, and the world on youtube.",
    "this media is not supported in your browser",
    "please open telegram to view this post",
    "view in telegram",
)

_ENGLISH_STOPWORDS = """
a about above after again against ain all am an and any are aren aren't as at be
because been before being below between both but by can couldn couldn't d did didn
didn't do does doesn doesn't doing don don't down during each few for from further
had hadn hadn't has hasn hasn't have haven haven't having he he'd he'll her here
hers herself he's him himself his how i i'd if i'll i'm in into is isn isn't it
it'd it'll it's its itself i've just ll m ma me mightn mightn't more most mustn
mustn't my myself needn needn't no nor not now o of off on once only or other our
ours ourselves out over own re s same shan shan't she she'd she'll she's should
shouldn shouldn't should've so some such t than that that'll the their theirs them
themselves then there these they they'd they'll they're they've this those through
to too under until up ve very was wasn wasn't we we'd we'll we're were weren
weren't we've what when where which while who whom why will with won won't wouldn
wouldn't y you you'd you'll your you're yours yourself yourselves you've
"""

_RUSSIAN_STOPWORDS = """
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли
если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без
будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех
никогда можно при наконец два об другой хоть после над больше тот через эти нас
про всего них какая много разве три эту моя впрочем хорошо свою этой перед
иногда лучше чуть том нельзя такой им более всегда конечно всю между
"""

_STOP_WORDS = set(_ENGLISH_STOPWORDS.split()) | set(_RUSSIAN_STOPWORDS.split())
_STOP_WORDS -= _KEEP_WORDS

_model = None
_loaded_path = None
_load_failed_path = None
_missing_logged_path = None
_lock = threading.Lock()


def clear_model_cache():
    global _model, _loaded_path, _load_failed_path, _missing_logged_path
    with _lock:
        _model = None
        _loaded_path = None
        _load_failed_path = None
        _missing_logged_path = None


def prepare_text(title: str | None, description: str | None) -> str:
    title = title or ""
    description = description or ""
    title_is_garbage = _contains_garbage(title)
    description_is_garbage = _contains_garbage(description)
    if title_is_garbage and description_is_garbage:
        return ""

    cleaned_title = "" if title_is_garbage else _clean_text(_clean_title(title))
    cleaned_description = "" if description_is_garbage else _clean_text(description)
    word_count = len(cleaned_title.split()) + len(cleaned_description.split())
    if word_count < _MIN_WORDS:
        return ""
    return f"{cleaned_title} {cleaned_description}".strip()


def predict_tags(title: str | None, description: str | None) -> list[str]:
    path = settings.LD_FASTTEXT_MODEL_PATH
    if not path:
        return []

    text = prepare_text(title, description)
    if not text:
        return []

    model = _get_model(path)
    if model is None:
        return []

    try:
        labels, _probabilities = _predict(
            model,
            text,
            k=max(1, settings.LD_FASTTEXT_MAX_TAGS),
            threshold=settings.LD_FASTTEXT_THRESHOLD,
        )
    except Exception:
        logger.exception("fastText prediction failed")
        return []

    tags = []
    seen = set()
    for label in labels:
        name = sanitize_tag_name(str(label).removeprefix(_LABEL_PREFIX))
        key = name.lower()
        if (
            not name
            or name.startswith("%")
            or len(name) > _MAX_TAG_LENGTH
            or key in seen
        ):
            continue
        seen.add(key)
        tags.append(name)
    return tags


def _contains_garbage(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in _GARBAGE_PHRASES)


def _clean_title(title: str) -> str:
    cleaned = title.strip().lower()
    for suffix in _TITLE_SUFFIXES:
        cleaned = re.sub(suffix, "", cleaned, flags=re.IGNORECASE)
    for prefix in _TITLE_PREFIXES:
        cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = [word for word in text.split() if word not in _STOP_WORDS and len(word) > 1]
    return " ".join(words)


def _get_model(path: str):
    global _model, _loaded_path, _load_failed_path, _missing_logged_path
    if _model is not None and _loaded_path == path:
        return _model
    if _load_failed_path == path:
        return None
    if not os.path.isfile(path):
        if _missing_logged_path != path:
            logger.warning("fastText model not found at %s", path)
            _missing_logged_path = path
        return None

    with _lock:
        if _model is not None and _loaded_path == path:
            return _model
        if _load_failed_path == path:
            return None
        try:
            import fasttext

            logger.info("Loading fastText model from %s", path)
            _model = fasttext.load_model(path)
            _loaded_path = path
            _load_failed_path = None
        except Exception:
            _load_failed_path = path
            logger.exception("Failed to load fastText model from %s", path)
            return None
    return _model


def _predict(model, text: str, k: int, threshold: float):
    # Call the C++ binding directly. fasttext's Python wrapper uses
    # np.array(..., copy=False), which raises on NumPy 2.
    predictions = model.f.predict(text + "\n", k, threshold, "strict")
    if not predictions:
        return [], []
    probabilities, labels = zip(*predictions, strict=True)
    return list(labels), probabilities
