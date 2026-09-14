"""Pipeline modules must derive prompts, styles and voices from content_types.

These tests are CPU-only: no models, no API calls, no video processing.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock heavy dependencies before importing shorts
for _mod in ("torch", "torchaudio", "cupy", "cupyx", "cupyx.scipy", "cupyx.scipy.ndimage"):
    sys.modules.setdefault(_mod, MagicMock())
_decord = MagicMock()
_decord.cpu = lambda x: f"cpu({x})"
_decord.gpu = lambda x: f"gpu({x})"
sys.modules.setdefault("decord", _decord)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ai_providers as ap  # noqa: E402
import subtitle_generator as sg  # noqa: E402
import tts_generator as tg  # noqa: E402
import story_narrator as sn  # noqa: E402
import shorts  # noqa: E402
from content_types import VOICE_PRESETS, get_content_type, list_content_types  # noqa: E402


# ---------------------------------------------------------------------------
# ai_providers: clip classification prompts
# ---------------------------------------------------------------------------

def test_gaming_clip_prompt_keeps_gameplay_wording_and_taxonomy():
    ct = get_content_type("gaming")
    prompt = ap.build_clip_classification_prompt(ct, source="video")
    assert prompt.startswith("Analyze this gameplay video clip")
    for cat in ct.semantic_types:
        assert cat.upper() in prompt
    assert '"best_category": "action"' in prompt
    assert "CLASSIFICATION GUIDE" in prompt


def test_podcast_clip_prompt_has_no_gaming_language():
    ct = get_content_type("podcasts")
    prompt = ap.build_clip_classification_prompt(ct, source="keyframes")
    assert prompt.startswith("These are keyframes from a podcast video clip")
    lowered = prompt.lower()
    for banned in ("gameplay", "kill", "headshot", "combat"):
        assert banned not in lowered, banned
    assert "QUOTE" in prompt and "REVELATION" in prompt
    assert ct.analysis_focus in prompt
    assert ct.analysis_ignore in prompt
    assert '"best_category": "quote"' in prompt


@pytest.mark.parametrize("ct", list_content_types(), ids=lambda c: c.key)
def test_clip_prompt_json_skeleton_matches_taxonomy(ct):
    prompt = ap.build_clip_classification_prompt(ct, source="video")
    for cat in ct.semantic_types:
        assert f'"{cat}": 0.5' in prompt


# ---------------------------------------------------------------------------
# ai_providers: Gemini deep analysis prompt
# ---------------------------------------------------------------------------

def test_deep_prompt_gaming_keeps_legacy_focus():
    prompt = ap.build_deep_analysis_prompt(get_content_type("gaming"), target_count=3)
    assert "Find 3 distinct, high-quality moments." in prompt
    assert "kills, funny fails, laughs, glitches, hypes" in prompt
    assert "boring travel or looting" in prompt


def test_deep_prompt_sports_uses_type_focus_and_valid_categories():
    ct = get_content_type("sports")
    prompt = ap.build_deep_analysis_prompt(ct, target_count=5)
    assert "Find 5 distinct, high-quality moments." in prompt
    assert ct.analysis_focus in prompt
    assert ct.analysis_ignore in prompt
    assert "looting" not in prompt
    assert "Triple kill" not in prompt
    # Example JSON must only reference categories the model is allowed to return
    import re
    for cat in re.findall(r'"category": "([a-z_]+)"', prompt):
        assert cat in ct.semantic_types


# ---------------------------------------------------------------------------
# ai_providers: caption prompt + tagging prompt
# ---------------------------------------------------------------------------

def test_caption_prompt_gaming_is_unchanged(monkeypatch):
    monkeypatch.setenv("VIDEO_TYPE", "gaming")
    prompt = ap._get_caption_prompt("gaming", 8, 20.0)
    assert "Watch this gameplay video and generate 8 short captions" in prompt
    assert '"HEADSHOT!"' in prompt
    assert "Focus on action moments, close calls, achievements, or funny situations" in prompt


def test_caption_prompt_uses_registry_guide_and_noun(monkeypatch):
    monkeypatch.setenv("VIDEO_TYPE", "podcasts")
    prompt = ap._get_caption_prompt("podcast_quote", 6, 30.0)
    assert "Watch this podcast video" in prompt
    assert "pull-quote" in prompt
    assert "gameplay" not in prompt.lower()
    assert "close calls" not in prompt
    assert get_content_type("podcasts").analysis_focus in prompt


def test_caption_prompt_unknown_style_falls_back_to_type_default(monkeypatch):
    monkeypatch.setenv("VIDEO_TYPE", "cooking_diy")
    ct = get_content_type("cooking_diy")
    prompt = ap._get_caption_prompt("does_not_exist", 6, 30.0)
    assert ct.style(ct.default_style).example in prompt


def test_caption_prompt_story_style_still_uses_story_pacing(monkeypatch):
    monkeypatch.setenv("VIDEO_TYPE", "gaming")
    prompt = ap._get_caption_prompt("story_news", 8, 60.0)
    assert "narrative segments" in prompt
    assert "4-8 seconds" in prompt


def test_tagging_prompt_describes_current_content_type(monkeypatch):
    monkeypatch.setenv("VIDEO_TYPE", "podcasts")
    prompt = ap.get_semantic_tagging_prompt("Clip this part", "podcast_quote")
    assert "Podcasts" in prompt
    assert "gaming/esports" not in prompt


def test_tagging_prompt_gaming_category_context(monkeypatch):
    monkeypatch.setenv("VIDEO_TYPE", "gaming")
    prompt = ap.get_semantic_tagging_prompt("CLUTCH", "funny")
    assert "Gaming" in prompt
    assert get_content_type("gaming").semantic_types["funny"] in prompt


def test_default_category_is_first_semantic_type():
    assert ap.default_category(get_content_type("gaming")) == "action"
    pod = get_content_type("podcasts")
    assert ap.default_category(pod) == next(iter(pod.semantic_types))


def test_clipscore_properties_follow_video_type(monkeypatch):
    monkeypatch.setenv("VIDEO_TYPE", "podcasts")
    cs = ap.ClipScore(clip_path=Path("x.mp4"), original_start=0, original_end=1, detected_category="funny")
    assert cs.caption_style == "podcast_highlight"
    assert cs.voice_preset == VOICE_PRESETS["podcast_host"]
    monkeypatch.setenv("VIDEO_TYPE", "gaming")
    assert cs.caption_style == "funny"


# ---------------------------------------------------------------------------
# subtitle_generator: CAPTION_STYLE=auto must respect VIDEO_TYPE
# ---------------------------------------------------------------------------

def test_auto_caption_style_respects_video_type(monkeypatch):
    monkeypatch.setenv("VIDEO_TYPE", "podcasts")
    assert sg.resolve_clip_caption_style("auto", "funny") == "podcast_highlight"
    assert sg.resolve_clip_caption_style("auto", None) == "podcast_quote"
    monkeypatch.setenv("VIDEO_TYPE", "gaming")
    assert sg.resolve_clip_caption_style("auto", "funny") == "funny"
    assert sg.resolve_clip_caption_style("auto", "clutch") == "dramatic"
    assert sg.resolve_clip_caption_style("auto", None) == "gaming"
    assert sg.resolve_clip_caption_style("genz", "funny") == "genz"


def test_legacy_video_type_default_map_removed():
    assert not hasattr(sg, "VIDEO_TYPE_DEFAULT_CAPTION_STYLE")


def test_missing_caption_style_defaults_to_auto(monkeypatch):
    """An .env without CAPTION_STYLE must not force gaming captions on other types."""
    monkeypatch.delenv("CAPTION_STYLE", raising=False)
    assert sg.get_caption_style() == "auto"
    monkeypatch.setenv("VIDEO_TYPE", "podcasts")
    assert sg.resolve_clip_caption_style(sg.get_caption_style(), "quote") == "podcast_quote"


# ---------------------------------------------------------------------------
# tts_generator: voice presets come from the registry first
# ---------------------------------------------------------------------------

def test_voice_description_for_registry_styles():
    assert tg.generate_voice_description("podcast_quote") == VOICE_PRESETS["podcast_host"]
    assert tg.generate_voice_description("fitness_coach") == VOICE_PRESETS["coach"]
    assert tg.generate_voice_description("music_hype") == VOICE_PRESETS["music_hype"]


def test_voice_description_legacy_paths_unchanged():
    assert tg.generate_voice_description("gaming") == ap.ClipScore.CAPTION_STYLE_VOICE_MAP["gaming"]
    assert tg.generate_voice_description("clutch") == ap.ClipScore.VOICE_PRESET_MAP["clutch"]
    assert tg.generate_voice_description("nonsense") == ap.ClipScore.VOICE_PRESET_MAP["action"]


def test_legacy_style_voice_fallbacks_removed():
    assert not hasattr(tg, "STYLE_VOICE_FALLBACKS")


# ---------------------------------------------------------------------------
# shorts: heuristic bypass follows the registry, not "!= gaming"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "provider,vtype,expected",
    [
        ("gemini", "esports", False),
        ("openai", "esports", False),
        ("gemini", "gaming", False),
        ("gemini", "podcasts", True),
        ("openai", "sports", True),
        ("local", "podcasts", False),
        ("gemini", "not_a_type", False),
    ],
)
def test_bypass_follows_content_type_strategy(provider, vtype, expected):
    assert shorts.should_bypass_heuristic_prefilter(provider, vtype) is expected


def test_shorts_get_video_type_delegates_to_registry(monkeypatch):
    monkeypatch.setenv("VIDEO_TYPE", "  Sports ")
    assert shorts.get_video_type() == "sports"
    monkeypatch.setenv("VIDEO_TYPE", "")
    assert shorts.get_video_type() == "gaming"


# ---------------------------------------------------------------------------
# story_narrator: prompts talk about the content type's clip noun
# ---------------------------------------------------------------------------

def test_story_prompt_uses_clip_noun(monkeypatch):
    monkeypatch.setenv("VIDEO_TYPE", "podcasts")
    prompt = sn.build_story_prompt("story_news", 2, "en", source="video")
    assert "2 podcast clips in sequence" in prompt
    assert "gameplay" not in prompt.lower()
    assert "esports broadcaster" not in prompt
    assert "The clips:" in prompt

    frames_prompt = sn.build_story_prompt("story_news", 2, "en", source="frames")
    assert "2 podcast clips as a sequence of frames" in frames_prompt
    assert "The clips:" not in frames_prompt

    monkeypatch.setenv("VIDEO_TYPE", "gaming")
    prompt = sn.build_story_prompt("story_news", 3, "en", source="video")
    assert "3 gameplay clips in sequence" in prompt
    assert "esports broadcaster" in prompt
