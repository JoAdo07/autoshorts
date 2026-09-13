"""Registry invariants for src/content_types.py.

content_types must be importable without torch/cupy/decord so the dashboard
and the pipeline can both depend on it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import content_types as ct  # noqa: E402


GAMING_SEMANTIC_TYPES = {"action", "funny", "clutch", "wtf", "epic_fail", "hype", "skill"}
GAMING_STYLE_KEYS = {
    "gaming", "dramatic", "funny", "minimal", "genz",
    "story_news", "story_roast", "story_creepypasta", "story_dramatic",
}
EXPECTED_TYPE_KEYS = {
    "gaming", "esports", "podcasts", "interviews", "news_commentary",
    "entertainment", "tv_shows", "comedy", "music", "vlogs", "cooking_diy",
    "fitness", "educational", "documentaries", "sports",
}


def test_registry_lists_all_expected_content_types():
    assert {c.key for c in ct.list_content_types()} == EXPECTED_TYPE_KEYS


def test_gaming_keeps_legacy_taxonomy_and_styles():
    gaming = ct.get_content_type("gaming")
    assert set(gaming.semantic_types) == GAMING_SEMANTIC_TYPES
    assert {s.key for s in gaming.styles} == GAMING_STYLE_KEYS
    assert gaming.default_style == "gaming"
    assert gaming.uses_action_heuristics is True


def test_esports_shares_gaming_taxonomy_and_uses_heuristics():
    esports = ct.get_content_type("esports")
    assert set(esports.semantic_types) == GAMING_SEMANTIC_TYPES
    assert esports.uses_action_heuristics is True
    assert esports.family == ct.get_content_type("gaming").family


@pytest.mark.parametrize("key", sorted(EXPECTED_TYPE_KEYS - {"gaming", "esports"}))
def test_non_gaming_types_are_ai_first_with_domain_taxonomy(key):
    c = ct.get_content_type(key)
    assert c.uses_action_heuristics is False
    assert len(c.semantic_types) >= 4
    # Domain taxonomies must not silently reuse the gaming one.
    assert set(c.semantic_types) != GAMING_SEMANTIC_TYPES


@pytest.mark.parametrize("key", sorted(EXPECTED_TYPE_KEYS))
def test_every_type_is_internally_consistent(key):
    c = ct.get_content_type(key)
    style_keys = {s.key for s in c.styles}
    assert c.default_style in style_keys
    assert set(c.category_to_style) == set(c.semantic_types)
    assert set(c.category_to_style.values()) <= style_keys
    assert c.label and c.emoji and c.family and c.description
    assert c.analysis_focus and c.analysis_ignore and c.clip_noun
    for style in c.styles:
        assert style.prompt_guide, f"{style.key} has no prompt guide"
        assert style.example, f"{style.key} has no example caption"
        assert style.voice_key in ct.VOICE_PRESETS, f"{style.key} voice missing"


def test_style_keys_are_globally_unique():
    seen = {}
    for c in ct.list_content_types():
        for s in c.styles:
            assert s.key not in seen or seen[s.key] is c.key or s.key in GAMING_STYLE_KEYS, (
                f"style {s.key} defined by both {seen[s.key]} and {c.key}"
            )
            seen.setdefault(s.key, c.key)
    assert ct.get_style("podcast_quote").key == "podcast_quote"
    assert ct.get_style("does_not_exist") is None


def test_families_group_every_type_once():
    families = ct.list_families()
    assert [f.key for f in families] == [
        "gaming", "talk", "entertainment", "lifestyle", "knowledge", "sports"
    ]
    all_keys = [c.key for f in families for c in f.types]
    assert sorted(all_keys) == sorted(EXPECTED_TYPE_KEYS)


def test_get_video_type_reads_env_and_normalizes(monkeypatch):
    monkeypatch.setenv("VIDEO_TYPE", "  Podcasts ")
    assert ct.get_video_type() == "podcasts"
    monkeypatch.delenv("VIDEO_TYPE")
    assert ct.get_video_type() == "gaming"


def test_get_content_type_falls_back_to_gaming_for_unknown(monkeypatch):
    monkeypatch.setenv("VIDEO_TYPE", "kittens")
    assert ct.get_content_type().key == "gaming"
    assert ct.get_content_type("kittens").key == "gaming"


class TestResolveCaptionStyle:
    def test_explicit_style_wins(self):
        c = ct.get_content_type("podcasts")
        assert ct.resolve_caption_style("podcast_highlight", "quote", c) == "podcast_highlight"

    def test_auto_maps_detected_category_through_type(self):
        c = ct.get_content_type("gaming")
        assert ct.resolve_caption_style("auto", "clutch", c) == "dramatic"
        assert ct.resolve_caption_style("auto", "funny", c) == "funny"

    def test_auto_for_non_gaming_never_returns_gaming_style(self):
        c = ct.get_content_type("podcasts")
        for category in c.semantic_types:
            resolved = ct.resolve_caption_style("auto", category, c)
            assert resolved in {s.key for s in c.styles}

    def test_auto_with_unknown_category_uses_type_default(self):
        c = ct.get_content_type("sports")
        assert ct.resolve_caption_style("auto", "clutch", c) == c.default_style
        assert ct.resolve_caption_style("auto", None, c) == c.default_style
