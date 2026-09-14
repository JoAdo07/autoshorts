# Universal Content Types — Design

Date: 2026-09-14 · Branch: `copilot/add-universal-video-support` (PR #3)

## Problem

PR #3 added a `VIDEO_TYPE` switch and per-type caption *styles*, but the
pipeline's *analysis* layer stayed gaming-only:

- `ClipScore.SEMANTIC_TYPES` (action/funny/clutch/wtf/epic_fail/hype/skill)
  drives every Gemini/OpenAI prompt. A podcast is asked to find "kills".
- `CAPTION_STYLE=auto` maps `detected_category` through the gaming
  `CAPTION_STYLE_MAP` before the new `VIDEO_TYPE` default is consulted, so the
  type default is dead code whenever AI analysis runs.
- `esports` bypasses action heuristics that suit it.
- TTS voices for new styles fall back to mismatched gaming presets.
- The style catalogue is duplicated in five places (dashboard, caption
  prompts, subtitle defaults, TTS fallbacks, tagging prefix chain).

## Goals

1. One registry describes every content type: taxonomy, prompts, styles,
   voices, and clip-selection strategy.
2. Gaming output is byte-identical to today.
3. Dashboard scales to 15+ types without a wall of radio buttons.
4. Everything is testable on CPU with mocked GPU libraries.

## Non-goals

- Transcript-driven moment selection (Whisper-first) for talk content.
- New PyCaps visual templates.
- Changing the `gameplay/` queue directory name.

## Architecture

### `src/content_types.py` (new, dependency-free)

```
CaptionStyle(key, icon, label, blurb, example, prompt_guide, voice_key,
             group="curated")   # gaming | genz | story | curated; .is_story
ContentType(key, label, emoji, family, description,
            semantic_types: {category: description},
            analysis_focus: str,            # "what stands out"
            analysis_ignore: str,           # "what to skip"
            clip_noun: str,                 # "gameplay" / "podcast" ...
            category_to_style: {category: style_key},
            styles: [CaptionStyle],
            default_style: str,
            uses_action_heuristics: bool)
```

Helpers: `get_video_type()`, `get_content_type(key=None)`,
`list_content_types()`, `list_families()`, `get_style(style_key)`,
`resolve_caption_style(requested, detected_category, content_type)`,
`get_voice_preset(style_key)`, `VOICE_PRESETS: {voice_key: description}`.

Families: Gaming (gaming, esports) · Talk (podcasts, interviews,
news_commentary) · Entertainment (entertainment, tv_shows, comedy, music) ·
Lifestyle (vlogs, cooking_diy, fitness) · Knowledge (educational,
documentaries) · Sports (sports).

Gaming keeps its seven semantic types, four base styles, `genz`, and the four
`story_*` styles. Esports reuses gaming's taxonomy with its own two styles.
Every other type gets 5–6 domain-specific semantic types.

### Pipeline wiring

| Module | Change |
|---|---|
| `ai_providers.py` | `build_clip_classification_prompt()`, `build_deep_analysis_prompt()`, `describe_tagging_context()`, `default_category()` build every prompt from `get_content_type()`; gaming keeps its hand-tuned classification guide. Deep-analysis moments with unknown categories fall back to the type's first category. `ClipScore.SEMANTIC_TYPES` / `CAPTION_STYLE_MAP` remain as gaming aliases. |
| `subtitle_generator.py` | `resolve_clip_caption_style()` wraps `resolve_caption_style()`; `CAPTION_STYLE` now defaults to `auto` (was `gaming`), matching `.env.example` and `shorts.py`. |
| `tts_generator.py` | `generate_voice_description()` → `get_voice_preset()` first, then legacy gaming maps. New presets: podcast_host, commentator, news_anchor, sports_caster, teacher, documentary_narrator, vlogger, coach, chef, comedian, music_hype, chill_narrator. `STYLE_VOICE_FALLBACKS` removed. |
| `shorts.py` | `should_bypass_heuristic_prefilter()` uses `uses_action_heuristics` (esports keeps heuristics); `_default_category()` replaces hardcoded `"action"`. |
| `story_narrator.py` | `build_story_prompt(style, n, lang, source)` shared by Gemini/OpenAI; uses `clip_noun`; `story_news` role is "esports broadcaster" only for the gaming family. |

### Dashboard

`dashboard/components/content_picker.py` renders:

1. Family tabs → content-type cards (emoji, label, one-line description,
   strategy badge: "Action heuristics + AI" vs "AI-first").
2. Caption style picker driven by `ContentType.styles`: Auto card + style
   cards with a live example-caption chip. Gaming keeps its Auto / Gaming /
   GenZ / Story grouping.
3. Persists `VIDEO_TYPE` (schema) and `CAPTION_STYLE` (extra) via
   `save_env_values` — no duplicate keys.

`1_Generate.py` calls the component; `dashboard/utils/process.build_process_env()`
merges `extras` so the child process sees `CAPTION_STYLE` even if the dashboard
env is stale. The PyCaps preview chip shows the selected style's `example`.

## Testing

- `tests/test_content_types.py` — registry invariants.
- `tests/test_pipeline_content_wiring.py` — prompt builders mention the right
  noun/taxonomy, auto-style resolution respects `VIDEO_TYPE`, voice presets,
  esports keeps heuristics, story prompt noun.
- `tests/test_video_type_support.py` — PR #3's original bypass tests.
- `tests/test_dashboard_generate.py` — `streamlit.testing.v1.AppTest`
  renders Generate for every content type without exceptions.
