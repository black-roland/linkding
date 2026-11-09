import tempfile
import urllib.parse
from unittest.mock import patch

from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework import status

from bookmarks.models import Bookmark
from bookmarks.services import fasttext_tagging, website_loader
from bookmarks.services.bookmarks import (
    create_bookmark,
    enhance_with_website_metadata,
    update_bookmark,
)
from bookmarks.services.website_loader import WebsiteMetadata
from bookmarks.tests.helpers import BookmarkFactoryMixin, LinkdingApiTestCase


class FakeBinding:
    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def predict(self, text, k, threshold, on_unicode_error):
        self.calls.append((text, k, threshold, on_unicode_error))
        return self.predictions


class FakeModel:
    def __init__(self, predictions):
        self.f = FakeBinding(predictions)


class FastTextTaggingTestCase(TestCase):
    def tearDown(self):
        fasttext_tagging.clear_model_cache()

    def test_prepare_text_matches_training_cleanup(self):
        text = fasttext_tagging.prepare_text(
            "How Python packaging works - YouTube",
            "A guide for publishing libraries. https://example.com/docs",
        )

        self.assertIn("how", text.split())
        self.assertNotIn("youtube", text.split())
        self.assertNotIn("for", text.split())
        self.assertNotIn("https://example.com/docs", text)
        self.assertIn("python", text.split())
        self.assertIn("packaging", text.split())

    def test_prepare_text_drops_garbage_and_short_text(self):
        self.assertEqual(
            fasttext_tagging.prepare_text(
                "Clip - YouTube",
                "Enjoy the videos and music you love, upload original content, and share it all with friends, family, and the world on YouTube.",
            ),
            "",
        )
        self.assertEqual(fasttext_tagging.prepare_text("Hi", ""), "")

    def test_predict_tags_is_disabled_without_model_path(self):
        with override_settings(LD_FASTTEXT_MODEL_PATH=""):
            self.assertEqual(
                fasttext_tagging.predict_tags(
                    "Python packaging guide for developers", "How to publish"
                ),
                [],
            )

    def test_predict_tags_keeps_model_loaded(self):
        fake = FakeModel(
            [
                (0.9, "__label__python"),
                (0.8, "__label__%hidden"),
                (0.7, "__label__" + "x" * 70),
                (0.6, "__label__python"),
            ]
        )
        with (
            tempfile.NamedTemporaryFile() as model_file,
            override_settings(
                LD_FASTTEXT_MODEL_PATH=model_file.name,
                LD_FASTTEXT_THRESHOLD=0.3,
                LD_FASTTEXT_MAX_TAGS=3,
            ),
            patch("fasttext.load_model", return_value=fake) as load_model,
        ):
            tags = fasttext_tagging.predict_tags(
                "Python packaging guide for developers",
                "How to publish a library",
            )
            fasttext_tagging.predict_tags(
                "Python packaging guide for developers",
                "How to publish a library",
            )

        load_model.assert_called_once_with(model_file.name)
        self.assertEqual(tags, ["python"])
        self.assertEqual(fake.f.calls[0][1:], (3, 0.3, "strict"))
        self.assertTrue(fake.f.calls[0][0].endswith("\n"))

    def test_missing_model_returns_no_tags(self):
        with (
            override_settings(LD_FASTTEXT_MODEL_PATH="/tmp/missing-linkding-model.ftz"),
            patch("fasttext.load_model") as load_model,
        ):
            self.assertEqual(
                fasttext_tagging.predict_tags(
                    "Python packaging guide for developers",
                    "How to publish a library",
                ),
                [],
            )
            load_model.assert_not_called()


def _predicted_tags(title, description):
    if title or description:
        return ["python"]
    return []


class FastTextBookmarkServiceTestCase(TestCase, BookmarkFactoryMixin):
    def test_create_adds_predicted_tags(self):
        with patch(
            "bookmarks.services.fasttext_tagging.predict_tags",
            side_effect=_predicted_tags,
        ):
            bookmark = create_bookmark(
                Bookmark(
                    url="https://example.com",
                    title="Python packaging",
                    description="A guide",
                ),
                "manual",
                self.get_or_create_test_user(),
            )

        self.assertCountEqual(bookmark.tag_names, ["manual", "python"])

    def test_update_does_not_reapply_predicted_tags(self):
        bookmark = self.setup_bookmark(title="Python packaging", description="A guide")
        with patch(
            "bookmarks.services.fasttext_tagging.predict_tags",
            side_effect=_predicted_tags,
        ) as predict:
            update_bookmark(bookmark, "manual", self.get_or_create_test_user())

        predict.assert_called_once_with("", "")
        self.assertEqual(bookmark.tag_names, ["manual"])

    def test_create_existing_url_applies_predicted_tags(self):
        user = self.get_or_create_test_user()
        self.setup_bookmark(url="https://example.com", title="Old", description="Old")
        with patch(
            "bookmarks.services.fasttext_tagging.predict_tags",
            side_effect=_predicted_tags,
        ):
            bookmark = create_bookmark(
                Bookmark(
                    url="https://example.com",
                    title="Python packaging",
                    description="A guide",
                ),
                "",
                user,
            )

        self.assertIn("python", bookmark.tag_names)

    def test_enhance_adds_predicted_tags_after_metadata_is_filled(self):
        bookmark = self.setup_bookmark(
            url="https://example.com", title="", description=""
        )
        with (
            patch.object(
                website_loader,
                "load_website_metadata",
                return_value=WebsiteMetadata(
                    "https://example.com",
                    "Python packaging",
                    "A guide",
                    None,
                ),
            ),
            patch(
                "bookmarks.services.fasttext_tagging.predict_tags",
                side_effect=_predicted_tags,
            ),
        ):
            enhance_with_website_metadata(bookmark)

        bookmark.refresh_from_db()
        self.assertEqual(bookmark.title, "Python packaging")
        self.assertIn("python", bookmark.tag_names)


class FastTextBookmarkApiTestCase(LinkdingApiTestCase, BookmarkFactoryMixin):
    def test_check_merges_rules_and_predictions_from_metadata(self):
        self.authenticate()
        profile = self.get_or_create_test_user().profile
        profile.auto_tagging_rules = "example.com rules"
        profile.save()

        with (
            patch.object(
                website_loader,
                "load_website_metadata",
                return_value=WebsiteMetadata(
                    "https://example.com",
                    "Scraped title",
                    "Scraped description",
                    None,
                ),
            ),
            patch(
                "bookmarks.services.fasttext_tagging.predict_tags",
                return_value=["python"],
            ) as predict,
        ):
            url = reverse("linkding:bookmark-check")
            check_url = urllib.parse.quote_plus("https://example.com")
            response = self.get(
                f"{url}?url={check_url}", expected_status_code=status.HTTP_200_OK
            )

        self.assertCountEqual(response.data["auto_tags"], ["rules", "python"])
        predict.assert_called_once_with("Scraped title", "Scraped description")

    def test_check_uses_existing_bookmark_text(self):
        self.authenticate()
        self.setup_bookmark(
            url="https://example.com",
            title="Saved title",
            description="Saved description",
        )
        with (
            patch.object(
                website_loader,
                "load_website_metadata",
                return_value=WebsiteMetadata(
                    "https://example.com",
                    "Scraped title",
                    "Scraped description",
                    None,
                ),
            ),
            patch(
                "bookmarks.services.fasttext_tagging.predict_tags",
                return_value=["python"],
            ) as predict,
        ):
            url = reverse("linkding:bookmark-check")
            check_url = urllib.parse.quote_plus("https://example.com")
            self.get(f"{url}?url={check_url}", expected_status_code=status.HTTP_200_OK)

        predict.assert_called_once_with("Saved title", "Saved description")

    def test_suggest_tags_uses_submitted_text(self):
        self.authenticate()
        profile = self.get_or_create_test_user().profile
        profile.auto_tagging_rules = "example.com rules"
        profile.save()
        with patch(
            "bookmarks.services.fasttext_tagging.predict_tags",
            return_value=["python"],
        ) as predict:
            url = reverse("linkding:bookmark-suggest-tags")
            website_url = urllib.parse.quote_plus("https://example.com")
            response = self.get(
                f"{url}?url={website_url}&title=Edited+title&description=Edited+description",
                expected_status_code=status.HTTP_200_OK,
            )

        self.assertCountEqual(response.data["auto_tags"], ["rules", "python"])
        predict.assert_called_once_with("Edited title", "Edited description")
