"""Exercise the real dashboard against a scratch .env, without a GPU pipeline."""
from __future__ import annotations

from collections import Counter
from html import unescape
from pathlib import Path
import sys

import pytest
from dotenv import dotenv_values
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from content_types import get_content_type, list_content_types
from dashboard.utils import config

PAGE = ROOT / "src/dashboard/pages/1_Generate.py"


@pytest.fixture
def dashboard_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / ".env")
    (tmp_path / "gameplay").mkdir()
    return tmp_path / ".env"


def open_page(env_path, video_type="gaming", caption_style="auto", page=PAGE):
    env_path.write_text(
        f"VIDEO_TYPE={video_type}\nCAPTION_STYLE={caption_style}\n"
        "ENABLE_TTS=false\nTTS_LANGUAGE=en\nPYCAPS_TEMPLATE=hype\n",
        encoding="utf-8",
    )
    at = AppTest.from_file(str(page), default_timeout=30).run()
    assert not at.exception
    return at


def rendered_text(at):
    return unescape("\n".join(element.value for element in [*at.markdown, *at.caption]))


def select_type(at, key):
    content_type = get_content_type(key)
    at.button_group(key="gen_content_family").set_value(content_type.family).run()
    assert not at.exception
    at.button(key=f"gen_content_type_{key}").click().run()
    assert not at.exception


@pytest.mark.parametrize("content_type", list_content_types(), ids=lambda ct: ct.key)
def test_page_renders_every_content_type(dashboard_env, content_type):
    at = open_page(dashboard_env, content_type.key)
    assert f"Currently: {content_type.emoji} {content_type.label}" in rendered_text(at)
    assert at.button_group(key="gen_content_family").value == content_type.family
    assert at.button(key=f"gen_content_type_{content_type.key}").proto.type == "primary"
    if content_type.uses_action_heuristics:
        assert "Scene detection + audio/visual action scoring" in rendered_text(at)
    else:
        assert "Heuristic pre-filter skipped" in rendered_text(at)
    assert " / ".join(content_type.semantic_types) in rendered_text(at)
    assert content_type.style(content_type.default_style).example in rendered_text(at)


def test_selecting_type_persists_one_schema_key(dashboard_env):
    at = open_page(dashboard_env)
    select_type(at, "podcasts")
    assert not at.exception
    lines = dashboard_env.read_text().splitlines()
    assert [line for line in lines if line.startswith("VIDEO_TYPE=")] == [
        "VIDEO_TYPE=podcasts"
    ]
    assert dotenv_values(dashboard_env)["ENABLE_TTS"] == "false"


def test_selecting_type_persists_missing_auto_caption(dashboard_env):
    dashboard_env.write_text("VIDEO_TYPE=gaming\n", encoding="utf-8")
    at = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert not at.exception
    assert at.radio(key="gen_caption_mode_gaming").value == "auto"
    select_type(at, "podcasts")
    assert at.radio(key="gen_caption_mode_podcasts").value == "auto"
    assert [
        line for line in dashboard_env.read_text().splitlines()
        if line.startswith("CAPTION_STYLE=")
    ] == ["CAPTION_STYLE=auto"]


def test_podcasts_without_caption_style_renders_auto(dashboard_env):
    dashboard_env.write_text("VIDEO_TYPE=podcasts\n", encoding="utf-8")
    at = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert not at.exception
    assert at.radio(key="gen_caption_mode_podcasts").value == "auto"
    assert "He actually said that." in rendered_text(at)


def test_saving_auto_persists_missing_caption_key_once(dashboard_env):
    from dashboard.components.content_picker import _save_caption

    dashboard_env.write_text("VIDEO_TYPE=gaming\n", encoding="utf-8")
    values, extras = config.load_env_values()
    _save_caption("auto", values, extras)
    _save_caption("auto", values, extras)
    assert [
        line for line in dashboard_env.read_text().splitlines()
        if line.startswith("CAPTION_STYLE=")
    ] == ["CAPTION_STYLE=auto"]


def test_changing_type_resets_incompatible_caption(dashboard_env):
    at = open_page(dashboard_env, caption_style="story_news")
    select_type(at, "podcasts")
    saved = dotenv_values(dashboard_env)
    assert saved["VIDEO_TYPE"] == "podcasts"
    assert saved["CAPTION_STYLE"] == "auto"
    assert at.radio(key="gen_caption_mode_podcasts").value == "auto"
    assert "He actually said that." in rendered_text(at)
    assert "HEADSHOT!" not in rendered_text(at)


def test_browsing_families_does_not_change_selection(dashboard_env):
    at = open_page(dashboard_env, "podcasts", "podcast_highlight")
    before = dashboard_env.read_text()
    at.button_group(key="gen_content_family").set_value("lifestyle").run()
    assert not at.exception
    assert "Currently: \U0001f399\ufe0f Podcasts" in rendered_text(at)
    assert dashboard_env.read_text() == before
    assert "Here's the key takeaway." in rendered_text(at)


def test_gaming_groups_select_their_first_style(dashboard_env):
    at = open_page(dashboard_env, caption_style="story_news")
    group = at.radio(key="gen_caption_mode_gaming")
    assert [option.split(" ", 1)[-1] for option in group.options] == [
        "Auto", "Gaming", "GenZ", "Story"
    ]
    for mode, expected in [
        ("genz", "genz"), ("gaming", "gaming"), ("story", "story_news"), ("auto", "auto")
    ]:
        at.radio(key="gen_caption_mode_gaming").set_value(mode).run()
        assert not at.exception
        assert dotenv_values(dashboard_env)["CAPTION_STYLE"] == expected


@pytest.mark.parametrize("content_type", list_content_types(), ids=lambda ct: ct.key)
def test_style_cards_use_registry_and_persist(dashboard_env, content_type):
    at = open_page(dashboard_env, content_type.key)
    mode_key = f"gen_caption_mode_{content_type.key}"
    for style in content_type.styles:
        mode = style.group if content_type.key == "gaming" else "curated"
        at.radio(key=mode_key).set_value(mode).run()
        assert not at.exception
        assert style.blurb in rendered_text(at)
        assert style.example in rendered_text(at)
        at.button(key=f"gen_caption_style_{content_type.key}_{style.key}").click().run()
        assert not at.exception
        assert dotenv_values(dashboard_env)["CAPTION_STYLE"] == style.key
        assert at.button(
            key=f"gen_caption_style_{content_type.key}_{style.key}"
        ).proto.type == "primary"
        previews = [item.value for item in at.markdown if 'class="subtitle-preview"' in item.value]
        assert len(previews) == 1
        assert style.example in unescape(previews[0])


def test_revisiting_type_does_not_restore_stale_style_mode(dashboard_env):
    at = open_page(dashboard_env, caption_style="story_news")
    select_type(at, "podcasts")
    at.radio(key="gen_caption_mode_podcasts").set_value("curated").run()
    select_type(at, "gaming")
    assert dotenv_values(dashboard_env)["CAPTION_STYLE"] == "auto"
    assert at.radio(key="gen_caption_mode_gaming").value == "auto"


def test_saved_settings_replace_stale_picker_state(dashboard_env):
    at = open_page(dashboard_env, caption_style="story_news")
    at.button_group(key="gen_content_family").set_value("lifestyle").run()
    values, extras = config.load_env_values()
    values["VIDEO_TYPE"] = "podcasts"
    extras["CAPTION_STYLE"] = "podcast_highlight"
    config.save_env_values(values, extras)
    at.run()
    assert not at.exception
    assert at.button_group(key="gen_content_family").value == "talk"
    assert at.radio(key="gen_caption_mode_podcasts").value == "curated"
    assert "Here's the key takeaway." in rendered_text(at)
    assert dotenv_values(dashboard_env)["CAPTION_STYLE"] == "podcast_highlight"


def test_invalid_saved_caption_is_reset_on_load(dashboard_env):
    at = open_page(dashboard_env, "podcasts", "story_news")
    assert dotenv_values(dashboard_env)["CAPTION_STYLE"] == "auto"
    assert at.radio(key="gen_caption_mode_podcasts").value == "auto"
    assert "reset to Auto" in rendered_text(at)


def test_family_picker_falls_back_to_horizontal_radio(dashboard_env, monkeypatch):
    import streamlit as st

    monkeypatch.setattr(st, "segmented_control", None)
    at = open_page(dashboard_env, "podcasts")
    assert at.radio(key="gen_content_family").value == "talk"
    at.radio(key="gen_content_family").set_value("knowledge").run()
    at.button(key="gen_content_type_educational").click().run()
    assert not at.exception
    assert dotenv_values(dashboard_env)["VIDEO_TYPE"] == "educational"
    assert "Currently:" in rendered_text(at) and "Educational" in rendered_text(at)


@pytest.mark.parametrize("video_type", ["gaming", "podcasts"])
@pytest.mark.parametrize("segmented", [True, False], ids=["segmented", "radio"])
def test_returning_from_settings_restores_selected_family(
    dashboard_env, monkeypatch, video_type, segmented
):
    if not segmented:
        import streamlit as st

        monkeypatch.setattr(st, "segmented_control", None)
    at = open_page(dashboard_env, video_type, page=ROOT / "src/dashboard/About.py")
    at.switch_page("pages/1_Generate.py").run()
    assert not at.exception
    at.switch_page("pages/3_Settings.py").run()
    assert not at.exception
    assert at.selectbox(key="cfg_VIDEO_TYPE").value == video_type
    at.switch_page("pages/1_Generate.py").run()
    assert not at.exception
    controls = at.button_group if segmented else at.radio
    assert controls(key="gen_content_family").value == get_content_type(video_type).family
    assert at.button(key=f"gen_content_type_{video_type}").proto.type == "primary"
    assert dotenv_values(dashboard_env)["VIDEO_TYPE"] == video_type


def test_quick_settings_language_and_subtitle_remain_persistent(dashboard_env):
    at = open_page(dashboard_env, "podcasts", "podcast_highlight")
    at.number_input(key="quick_scene_limit").set_value(8).run()
    at.toggle(key="quick_enable_tts").set_value(True).run()
    at.selectbox(key="gen_language").set_value("ja").run()
    at.selectbox(key="gen_template").set_value("classic").run()
    at.toggle(key="show_all_logs").set_value(True).run()
    assert not at.exception
    saved = dotenv_values(dashboard_env)
    assert saved["SCENE_LIMIT"] == "8"
    assert saved["ENABLE_TTS"] == "true"
    assert saved["TTS_LANGUAGE"] == "ja"
    assert saved["PYCAPS_TEMPLATE"] == "classic"
    assert saved["CAPTION_STYLE"] == "podcast_highlight"
    assert any("Start Processing" in button.label and not button.disabled for button in at.button)
    assert any("Stop" in button.label and button.disabled for button in at.button)


def test_process_env_extras_override_stale_values_without_mutation():
    from dashboard.utils.process import build_process_env

    values = {"VIDEO_TYPE": "podcasts", "CAPTION_STYLE": "gaming", "SCENE_LIMIT": 4}
    extras = {"CAPTION_STYLE": "podcast_quote", "TTS_LANGUAGE": "ja", "PYCAPS_TEMPLATE": "classic"}
    base = {"PATH": "/usr/bin", "CAPTION_STYLE": "story_news", "TTS_LANGUAGE": "en"}
    result = build_process_env(values, extras, base)
    assert result == {
        "PATH": "/usr/bin", "VIDEO_TYPE": "podcasts", "SCENE_LIMIT": "4",
        "CAPTION_STYLE": "podcast_quote", "TTS_LANGUAGE": "ja", "PYCAPS_TEMPLATE": "classic",
    }
    assert values["CAPTION_STYLE"] == "gaming"
    assert extras["CAPTION_STYLE"] == "podcast_quote"
    assert base["CAPTION_STYLE"] == "story_news"


def test_env_round_trip_never_duplicates_schema_keys(dashboard_env):
    dashboard_env.write_text("VIDEO_TYPE=podcasts\nCAPTION_STYLE=podcast_quote\n")
    values, extras = config.load_env_values()
    extras.update({"VIDEO_TYPE": "gaming", "AI_PROVIDER": "local", "SCENE_LIMIT": "99"})
    config.save_env_values(values, extras)
    first_counts = Counter(
        line.split("=", 1)[0]
        for line in dashboard_env.read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert all(count == 1 for count in first_counts.values())
    for _ in range(2):
        values, extras = config.load_env_values()
        config.save_env_values(values, extras)
    counts = Counter(
        line.split("=", 1)[0]
        for line in dashboard_env.read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert all(count == 1 for count in counts.values())
    assert values["VIDEO_TYPE"] == "podcasts"
    assert "VIDEO_TYPE" not in extras
    assert extras["CAPTION_STYLE"] == "podcast_quote"


@pytest.mark.parametrize("raw", ["false", "0", "no", "off"])
def test_env_round_trip_preserves_disabled_toggles(dashboard_env, raw):
    dashboard_env.write_text(f"ENABLE_TTS={raw}\nENABLE_SUBTITLES={raw}\n")
    values, extras = config.load_env_values()
    config.save_env_values(values, extras)
    saved = dotenv_values(dashboard_env)
    assert saved["ENABLE_TTS"] == "false"
    assert saved["ENABLE_SUBTITLES"] == "false"
