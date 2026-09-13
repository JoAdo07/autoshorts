"""Registry-driven content and caption selection for the Generate page."""
from __future__ import annotations

from html import escape
from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if parent.name == "src":
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import streamlit as st

from content_types import CaptionStyle, ContentType, get_content_type, list_families
from dashboard.utils.config import save_env_values
from dashboard.utils.theme import COLORS


def _save_caption(style_key: str, values: dict, extras: dict) -> None:
    if extras.get("CAPTION_STYLE") != style_key:
        extras["CAPTION_STYLE"] = style_key
        save_env_values(values, extras)


def _select_content_type(type_key: str, values: dict, extras: dict) -> None:
    content_type = get_content_type(type_key)
    values["VIDEO_TYPE"] = content_type.key
    extras.pop("VIDEO_TYPE", None)
    caption = str(extras.setdefault("CAPTION_STYLE", "auto"))
    if caption != "auto" and content_type.style(caption) is None:
        extras["CAPTION_STYLE"] = "auto"
        st.toast(f"Caption style reset to Auto for {content_type.label}")
    save_env_values(values, extras)


def _example_chip(style: CaptionStyle) -> None:
    st.markdown(
        f'<div class="caption-example" style="background:{COLORS["bg_dark"]};'
        f'border-left:3px solid {COLORS["primary"]};border-radius:6px;'
        f'padding:0.65rem 0.8rem;color:{COLORS["text_primary"]};'
        f'font-size:0.95rem;font-weight:600;overflow-wrap:anywhere;">'
        f'{escape(style.example)}</div>',
        unsafe_allow_html=True,
    )


def render_content_type_picker(values: dict, extras: dict) -> ContentType:
    """Choose a family and type, persisting the type as a schema setting."""
    content_type = get_content_type(values.get("VIDEO_TYPE", "gaming"))
    families = {family.key: family for family in list_families()}
    family_key = "gen_content_family"

    # Widget state is removed on page navigation; restore it from the saved type.
    # Browsing a family alone must not change the selected type.
    if family_key not in st.session_state or st.session_state.get("_gen_selected_type") != content_type.key:
        st.session_state[family_key] = content_type.family
        st.session_state["_gen_selected_type"] = content_type.key

    st.markdown("### 🧭 Video Type")
    strategy = "Action heuristics + AI" if content_type.uses_action_heuristics else "AI-first"
    st.caption(f"Currently: {content_type.emoji} {content_type.label} · {strategy}")

    def family_label(key: str) -> str:
        family = families[key]
        return f"{family.emoji} {family.label}"

    segmented_control = getattr(st, "segmented_control", None)
    if callable(segmented_control):
        selected_family = segmented_control(
            "Content family",
            options=list(families),
            key=family_key,
            format_func=family_label,
            required=True,
            width="stretch",
            label_visibility="collapsed",
        )
    else:
        selected_family = st.radio(
            "Content family",
            options=list(families),
            key=family_key,
            format_func=family_label,
            horizontal=True,
            label_visibility="collapsed",
        )

    types = families[selected_family].types
    for start in range(0, len(types), 2):
        row = types[start:start + 2]
        for column, candidate in zip(st.columns(len(row)), row):
            with column, st.container(border=True):
                selected = candidate.key == content_type.key
                st.markdown(f"**{candidate.emoji} {candidate.label}**")
                st.caption(candidate.description)
                if candidate.uses_action_heuristics:
                    badge, color = "⚡ Action heuristics + AI", COLORS["success"]
                else:
                    badge, color = "🧠 AI-first", COLORS["info"]
                st.markdown(
                    f'<span style="color:{color};background:{COLORS["bg_card"]};'
                    f'border:1px solid {COLORS["border_light"]};border-radius:20px;'
                    f'padding:0.2rem 0.5rem;font-size:0.75rem;">{badge}</span>',
                    unsafe_allow_html=True,
                )
                st.button(
                    f"✓ {candidate.label}" if selected else f"Select {candidate.label}",
                    key=f"gen_content_type_{candidate.key}",
                    type="primary" if selected else "secondary",
                    width="stretch",
                    on_click=_select_content_type,
                    args=(candidate.key, values, extras),
                )

    if content_type.uses_action_heuristics:
        st.caption("Scene detection + audio/visual action scoring pre-filter, then AI ranking.")
    else:
        st.caption(
            "Heuristic pre-filter skipped — AI (or Gemini Deep Analysis) picks moments directly. "
            "Tip: enable Gemini Deep Analysis in Settings for best results."
        )
    return content_type


def render_caption_style_picker(content_type: ContentType, values: dict, extras: dict) -> str:
    """Choose a registry style, or Auto for per-moment matching."""
    st.markdown("### 💬 Caption Style")
    caption = str(extras.get("CAPTION_STYLE", "auto"))
    current_style = content_type.style(caption)
    if caption != "auto" and current_style is None:
        _save_caption("auto", values, extras)
        caption = "auto"
        st.caption(f"The saved style is not available for {content_type.label}; reset to Auto.")

    is_gaming = content_type.key == "gaming"
    modes = (
        {"auto": "🎯 Auto", "gaming": "🎮 Gaming", "genz": "🔥 GenZ", "story": "📖 Story"}
        if is_gaming else {"auto": "🎯 Auto", "curated": "🎨 Curated"}
    )
    mode_key = f"gen_caption_mode_{content_type.key}"
    current_mode = "auto" if current_style is None else (
        current_style.group if is_gaming else "curated"
    )
    # Callbacks persist before rerender, so disk remains authoritative on return
    # from another type or the Settings page, rather than an old widget value.
    st.session_state[mode_key] = current_mode

    def save_mode() -> None:
        mode = st.session_state[mode_key]
        if mode == "auto":
            _save_caption("auto", values, extras)
        else:
            styles = [
                style for style in content_type.styles
                if not is_gaming or style.group == mode
            ]
            selected = next((style for style in styles if style.key == caption), styles[0])
            _save_caption(selected.key, values, extras)

    mode = st.radio(
        "Caption style group",
        options=list(modes),
        key=mode_key,
        format_func=modes.__getitem__,
        horizontal=True,
        label_visibility="collapsed",
        on_change=save_mode,
    )

    if mode == "auto":
        categories = " / ".join(content_type.semantic_types)
        st.caption(f"AI detects {categories} and picks the matching style for each moment.")
        default_style = content_type.style(content_type.default_style)
        with st.container(border=True):
            st.markdown(f"**Auto · {content_type.label}**")
            st.caption(f"Example: {default_style.label}. The style adapts to each detected moment.")
            _example_chip(default_style)
    else:
        styles = [
            style for style in content_type.styles
            if not is_gaming or style.group == mode
        ]
        for start in range(0, len(styles), 2):
            row = styles[start:start + 2]
            for column, style in zip(st.columns(len(row)), row):
                with column, st.container(border=True):
                    selected = style.key == caption
                    st.markdown(f"**{style.icon} {style.label}**")
                    st.caption(style.blurb)
                    _example_chip(style)
                    st.button(
                        f"✓ {style.label}" if selected else f"Use {style.label}",
                        key=f"gen_caption_style_{content_type.key}_{style.key}",
                        type="primary" if selected else "secondary",
                        width="stretch",
                        on_click=_save_caption,
                        args=(style.key, values, extras),
                    )
    return caption
