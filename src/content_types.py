"""Content-type registry: the single source of truth for universal video support.

Every video type (gaming, podcasts, sports, ...) is described once here:

* the semantic taxonomy the AI analysers classify clips into,
* prompt hints about what stands out / what to ignore,
* the caption styles the dashboard offers and the prompt guide each style uses,
* the TTS voice preset for each style,
* whether action-energy heuristics are a meaningful pre-filter.

This module is dependency-free so both the Streamlit dashboard and the GPU
pipeline can import it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class CaptionStyle:
    key: str
    icon: str
    label: str
    blurb: str
    example: str
    prompt_guide: str
    voice_key: str
    group: str = "curated"

    @property
    def is_story(self) -> bool:
        return self.key.startswith("story_")


@dataclass(frozen=True)
class ContentType:
    key: str
    label: str
    emoji: str
    family: str
    description: str
    clip_noun: str
    semantic_types: Dict[str, str]
    analysis_focus: str
    analysis_ignore: str
    category_to_style: Dict[str, str]
    styles: Sequence[CaptionStyle]
    default_style: str
    uses_action_heuristics: bool = False

    @property
    def strategy_label(self) -> str:
        if self.uses_action_heuristics:
            return "Action heuristics + AI ranking"
        return "AI-first moment discovery"

    def style(self, key: str) -> Optional[CaptionStyle]:
        for s in self.styles:
            if s.key == key:
                return s
        return None

    @property
    def style_keys(self) -> List[str]:
        return [s.key for s in self.styles]


@dataclass(frozen=True)
class Family:
    key: str
    label: str
    emoji: str
    types: Sequence[ContentType] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Voice presets (Qwen3-TTS VoiceDesign descriptions)
# ---------------------------------------------------------------------------

VOICE_PRESETS: Dict[str, str] = {
    "gaming": """gender: Male.
pitch: Mid-range male pitch with sharp upward inflections during exciting moments.
speed: Very fast-paced, rapid-fire delivery matching gaming action.
volume: Loud and projecting, nearly shouting during intense plays.
age: Young adult, 20s to early 30s.
clarity: Highly articulate, every word distinct even at speed.
fluency: Extremely fluent with no hesitations, continuous flow.
accent: American English, neutral.
texture: Bright, energetic vocal quality with slight rasp.
emotion: Intense excitement, hype, constant enthusiasm.
tone: Upbeat, authoritative, commanding attention.
personality: Confident, extroverted, engaging, competitive edge.""",

    "dramatic": """gender: Male.
pitch: Deep, resonant bass with powerful projection.
speed: Slow, deliberate pacing with strategic pauses for impact.
volume: Loud, commanding presence filling the space.
age: Mature adult, 40s to 50s.
clarity: Perfect diction, every word pronounced with gravitas.
fluency: Flawless delivery with cinematic timing.
accent: American English, neutral broadcast quality.
texture: Rich, velvety depth with cinematic warmth.
emotion: Inspiring, epic, grandiose.
tone: Heroic, momentous, like narrating legends.
personality: Authoritative, wise, larger-than-life presence.""",

    "funny": """gender: Male.
pitch: Mid to slightly high male pitch with playful variations.
speed: Moderate pace with deliberate pauses for comedic timing.
volume: Conversational, occasionally louder for punchlines.
age: Young adult, early to mid 20s.
clarity: Clear but relaxed, not overly precise.
fluency: Fluent with intentional hesitations for humor, occasional 'uh', 'like'.
accent: American English, casual GenZ cadence.
texture: Smooth, light vocal quality with natural warmth.
emotion: Amused, ironic, playfully sarcastic.
tone: Laid-back, chill, slightly deadpan with smirk energy.
personality: Witty, self-aware, relatable, gently mocking.""",

    "minimal": """gender: Male.
pitch: Low, steady pitch with minimal variation.
speed: Slow to moderate, unhurried and calm.
volume: Quiet to moderate, intimate and close.
age: Young to middle-aged adult, 25 to 35.
clarity: Clear but understated, effortless articulation.
fluency: Smooth and easy, natural flow.
accent: American English, neutral and unassuming.
texture: Soft, gentle vocal quality without harshness.
emotion: Calm, composed, subtly confident.
tone: Understated, reserved, quiet assurance.
personality: Introverted, thoughtful, self-contained.""",

    "genz": """gender: Male or Female (androgynous lean).
pitch: Mid-range with frequent upward inflections and vocal fry.
speed: Fast with casual slurring, modern speech patterns.
volume: Moderate, conversational social media energy.
age: Late teens to early 20s, Gen Z demographic.
clarity: Casual clarity, some words blend together naturally.
fluency: Very fluent but with filler words, 'literally', 'like', 'bruh'.
accent: American English, internet-influenced speech.
texture: Bright, youthful, slightly nasal quality.
emotion: Ironic detachment mixed with genuine enthusiasm.
tone: Casual, meme-aware, chronically online vibes.
personality: Self-aware, ironic, effortlessly cool, relatable chaos.""",

    "story_news": """gender: Male.
pitch: Mid-range male pitch, energetic and dynamic.
speed: Fast-paced, building excitement with rapid-fire delivery.
volume: Loud, projected, arena-filling energy.
age: Young adult, late 20s to early 30s.
clarity: Sharp, punchy enunciation, esports casting precision.
fluency: Rapid-fire with hype pauses, building momentum.
accent: American English, energetic gaming culture.
texture: Bright, electric vocal quality with infectious enthusiasm.
emotion: Excited, hyped, passionate about the plays.
tone: Enthusiastic, analytical, building tension and release.
personality: Charismatic caster, knowledgeable, gets hyped with the audience.""",

    "story_roast": """gender: Male.
pitch: Mid to high pitch with sarcastic inflections and exaggerated tones.
speed: Variable, speeding up for punchlines, slowing for emphasis.
volume: Moderate to loud, performative and theatrical.
age: Young adult, mid to late 20s.
clarity: Very clear, ensuring every barb lands perfectly.
fluency: Smooth with comedic pauses and timing.
accent: American English, comedy podcast energy.
texture: Bright with playful edge, slight smirk audible.
emotion: Amused mockery, playful cruelty, entertained.
tone: Sarcastic, teasing, roast-comedy style.
personality: Quick-witted, sharp-tongued, charismatic instigator.""",

    "story_creepypasta": """gender: Male.
pitch: Deep, low pitch with minimal variation, ominous undertones.
speed: Very slow, deliberate, each word drawn out for tension.
volume: Quiet to moderate, intimate and unsettling closeness.
age: Middle-aged to older adult, 40s to 50s.
clarity: Crystal clear whisper-level articulation.
fluency: Controlled, methodical pacing building dread.
accent: American English, neutral with slight rasp.
texture: Dark, gravelly with shadowy depth.
emotion: Foreboding, sinister, building unease.
tone: Ominous, creeping horror, inevitable dread.
personality: Mysterious, unsettling, knows something you don't.""",

    "story_dramatic": """gender: Female.
pitch: Rich, resonant mid-range with expressive depth.
speed: Measured, deliberate pacing with dramatic pauses for impact.
volume: Commanding presence, clear projection with emotional range.
age: Mature adult, late 30s to 40s.
clarity: Perfect diction, every word delivered with intention.
fluency: Flawless delivery with cinematic timing and gravitas.
accent: American English, theatrical broadcast quality.
texture: Warm, velvety depth with captivating allure.
emotion: Intense, evocative, drawing listeners into the story.
tone: Epic, momentous, like narrating legends and tragedies.
personality: Wise, commanding, magnetic storyteller presence.""",

    # --- Universal presets -------------------------------------------------
    "podcast_host": """gender: Male.
pitch: Warm mid-range with relaxed, natural variation.
speed: Conversational, unhurried, with thoughtful micro-pauses.
volume: Close-mic intimacy, moderate and steady.
age: Adult, mid 30s.
clarity: Clear, articulate, never over-enunciated.
fluency: Smooth, natural flow like a seasoned host.
accent: American English, neutral and friendly.
texture: Rich, rounded, radio-warm vocal quality.
emotion: Curious, engaged, genuinely interested.
tone: Thoughtful, candid, easy to listen to.
personality: Grounded, insightful, trustworthy conversationalist.""",

    "commentator": """gender: Male.
pitch: Mid-range with confident, opinionated emphasis.
speed: Brisk and punchy, accelerating into key points.
volume: Projected, assertive, leaning into the argument.
age: Adult, late 30s.
clarity: Crisp, every claim lands distinctly.
fluency: Fluent with rhetorical pauses before the payoff.
accent: American English, cable-news pundit cadence.
texture: Firm, slightly edgy vocal quality.
emotion: Convinced, provocative, fired up.
tone: Direct, spirited, debate-ready.
personality: Bold, articulate, unafraid of a hot take.""",

    "news_anchor": """gender: Female.
pitch: Steady mid-range with controlled, authoritative inflection.
speed: Measured, deliberate, broadcast pacing.
volume: Clear and projected, studio presence.
age: Adult, late 30s to 40s.
clarity: Impeccable diction, professional precision.
fluency: Flawless, composed delivery under pressure.
accent: American English, neutral network broadcast.
texture: Polished, smooth, trustworthy vocal quality.
emotion: Serious, composed, quietly urgent.
tone: Authoritative, factual, headline-driven.
personality: Credible, calm, commanding attention.""",

    "sports_caster": """gender: Male.
pitch: Mid-range rising to high excited peaks on big plays.
speed: Fast, breathless play-by-play accelerating with the action.
volume: Loud, stadium-filling, shouting at the climax.
age: Adult, 40s.
clarity: Sharp and punchy even at full volume.
fluency: Rapid-fire with explosive bursts and dramatic pauses.
accent: American English, network sports broadcast.
texture: Powerful, resonant, slightly hoarse from excitement.
emotion: Electrified, awestruck, ecstatic.
tone: Play-by-play intensity building to a roar.
personality: Passionate veteran broadcaster who lives for the moment.""",

    "teacher": """gender: Female.
pitch: Clear mid-range with friendly, encouraging lift.
speed: Moderate and patient, pausing to let ideas land.
volume: Comfortable, classroom-clear.
age: Adult, early 30s.
clarity: Excellent enunciation, every term precise.
fluency: Smooth, well-structured, logically paced.
accent: American English, neutral and approachable.
texture: Warm, bright, inviting vocal quality.
emotion: Enthusiastic about the subject, supportive.
tone: Explanatory, encouraging, confident.
personality: Patient, knowledgeable, genuinely wants you to get it.""",

    "documentary_narrator": """gender: Male.
pitch: Low-to-mid register with gentle, measured variation.
speed: Slow and contemplative, letting images breathe.
volume: Soft to moderate, intimate yet authoritative.
age: Mature adult, 50s to 60s.
clarity: Exquisite diction, unhurried precision.
fluency: Seamless, poetic phrasing with reflective pauses.
accent: British English, refined and gentle.
texture: Velvety, textured warmth with quiet gravitas.
emotion: Wonder, reverence, quiet fascination.
tone: Reflective, awe-inspiring, calmly profound.
personality: Wise observer, curious, deeply respectful of the subject.""",

    "vlogger": """gender: Female.
pitch: Bright mid-to-high range with lively expressive swings.
speed: Upbeat and natural, speeding up when excited.
volume: Conversational, energetic, talking-to-a-friend closeness.
age: Young adult, mid 20s.
clarity: Casual clarity, natural everyday speech.
fluency: Very fluent with spontaneous, unscripted feel.
accent: American English, social-media native.
texture: Warm, sunny, approachable vocal quality.
emotion: Excited, candid, openly emotional.
tone: Friendly, relatable, in-the-moment storytelling.
personality: Charismatic, authentic, your favorite creator friend.""",

    "coach": """gender: Male.
pitch: Low-to-mid range, firm and grounded.
speed: Punchy, clipped commands with driving rhythm.
volume: Loud, motivating, pushing through fatigue.
age: Adult, mid 30s.
clarity: Sharp, direct, no wasted syllables.
fluency: Fluent, rhythmic, rep-by-rep cadence.
accent: American English, gym-floor energy.
texture: Gritty, powerful, athletic vocal quality.
emotion: Fired up, relentless, believing in you.
tone: Commanding, encouraging, no-excuses intensity.
personality: Tough-love motivator, disciplined, inspiring.""",

    "chef": """gender: Female.
pitch: Warm mid-range with gentle, appetizing emphasis.
speed: Steady, instructional, pausing on key steps.
volume: Moderate, kitchen-intimate.
age: Adult, late 30s.
clarity: Clear, precise on technique and ingredients.
fluency: Smooth, confident, practiced demonstration flow.
accent: American English, warm and hospitable.
texture: Rich, comforting, sensory vocal quality.
emotion: Delighted, nurturing, savoring the process.
tone: Instructional yet inviting, mouth-watering descriptions.
personality: Generous host, meticulous, passionate about food and craft.""",

    "comedian": """gender: Male.
pitch: Mid-range with exaggerated swings for setups and payoffs.
speed: Variable, slow-burn setups then rapid punchlines.
volume: Stage-projected, dropping to deadpan for effect.
age: Adult, early 30s.
clarity: Crisp, timing-perfect delivery.
fluency: Masterful pauses, beat-perfect comedic rhythm.
accent: American English, stand-up club cadence.
texture: Expressive, elastic vocal quality.
emotion: Amused, mischievous, riding the crowd's laughter.
tone: Deadpan to explosive, always in on the joke.
personality: Sharp, observational, effortlessly funny performer.""",

    "music_hype": """gender: Male.
pitch: Mid-to-high energetic range with rhythmic bounce.
speed: Fast, beat-synced, hype-man cadence.
volume: Loud, festival-stage projection.
age: Young adult, mid 20s.
clarity: Punchy, rhythmic, ad-lib ready.
fluency: Flowing, musical, riding the groove.
accent: American English, urban hype energy.
texture: Bright, electric, crowd-rousing vocal quality.
emotion: Euphoric, pumped, feeding off the crowd.
tone: Hype-man energy, building to the drop.
personality: Magnetic MC, infectious, keeps the party going.""",

    "chill_narrator": """gender: Female.
pitch: Low, smooth, velvety register.
speed: Slow, languid, late-night radio pacing.
volume: Soft, intimate, close to the mic.
age: Adult, late 20s.
clarity: Effortlessly clear, breathy softness.
fluency: Unhurried, flowing, dreamy phrasing.
accent: American English, neutral and smooth.
texture: Silky, warm, atmospheric vocal quality.
emotion: Relaxed, wistful, quietly moved.
tone: Mood-first, aesthetic, ambient.
personality: Laid-back curator of vibes, effortlessly cool.""",
}


# ---------------------------------------------------------------------------
# Caption styles
# ---------------------------------------------------------------------------

def _style(key, icon, label, blurb, example, prompt_guide, voice_key, group="curated"):
    return CaptionStyle(key, icon, label, blurb, example, prompt_guide, voice_key, group)


GAMING_STYLES = (
    _style("gaming", "🎮", "Gaming", "HEADSHOT! • GG EZ • Punchy gaming captions", "HEADSHOT!",
           """Generate short, punchy captions like gaming content creators use.
Examples: "HEADSHOT!", "clutch play incoming...", "wait for it...", "GG EZ", "POV: you're cracked"
Keep captions 1-5 words. Use ALL CAPS for emphasis on action moments.""", "gaming", group="gaming"),
    _style("dramatic", "🎭", "Dramatic", "The final stand... • Cinematic narration", "The final stand.",
           """Generate dramatic, cinematic captions.
Examples: "The final stand.", "Everything changed.", "No turning back now."
Keep captions short and impactful. Use lowercase for tension, CAPS for climax.""", "dramatic", group="gaming"),
    _style("funny", "😂", "Funny", "skill issue tbh • Meme-style commentary", "skill issue tbh",
           """Generate humorous, meme-style captions.
Examples: "skill issue tbh", "when the plan works (it never does)", "*chuckles* I'm in danger"
Be self-aware and slightly chaotic. Gen-Z humor welcome.""", "funny", group="gaming"),
    _style("minimal", "✨", "Minimal", "nice. • Clean, understated captions", "nice.",
           """Generate minimal, understated captions.
Examples: "nice.", "oh.", "well then."
Keep it subtle. Less is more. Max 3 words per caption.""", "minimal", group="gaming"),
    _style("genz", "🔥", "GenZ", "bruh • no cap • fr fr • Slang-heavy reactions", "bruh 💀",
           """Generate GenZ slang-heavy reactions and commentary.
Examples: "bruh 💀", "no cap this is insane", "he's locked in rn", "finna go crazy", "ate and left no crumbs"
Use modern slang naturally: bruh, no cap, finna, fr fr, ate, slay, lowkey, highkey, bussin, mid, L, W, rizz
Keep captions 1-6 words. Be authentic to GenZ internet culture. Emojis encouraged (💀🔥😭).""", "genz", group="genz"),
    _style("story_news", "📰", "News", "Professional esports broadcaster narration",
           "And we're witnessing championship-level gameplay here.",
           """Generate professional esports broadcaster narrative.
Examples: "And we're witnessing championship-level gameplay here.", "The positioning is absolutely impeccable.", "This could be the defining moment of the match."
Write 2-3 sentences per caption. Professional tone, clear analysis, building excitement. Fewer captions (2-4 max) with longer text.""", "story_news", group="story"),
    _style("story_roast", "🔥", "Roast", "Sarcastic roasting commentary", "Oh no. Oh no no no.",
           """Generate sarcastic, playful roasting commentary.
Examples: "Oh no. Oh no no no.", "Someone's definitely uninstalling after this.", "The audacity. The absolute audacity of this play."
Write 2-3 sentences per caption. Sarcastic but not mean, comedic timing, playful mockery. Fewer captions (2-4 max).""", "story_roast", group="story"),
    _style("story_creepypasta", "👻", "Creepypasta", "Horror/tension narrative style", "Something felt wrong.",
           """Generate horror-style tension narrative.
Examples: "Something felt wrong.", "The game knew what was about to happen.", "And then... it did."
Write 2-3 sentences per caption. Build tension, ominous tone, slow reveals. Fewer captions (2-4 max). Use ellipses for suspense.""", "story_creepypasta", group="story"),
    _style("story_dramatic", "🎭", "Dramatic", "Epic cinematic storytelling", "In the arena of champions, legends are born.",
           """Generate epic cinematic narration.
Examples: "In the arena of champions, legends are born.", "The crowd holds its breath.", "One shot. One chance. Immortality awaits."
Write 2-3 sentences per caption. Epic tone, powerful delivery, movie trailer style. Fewer captions (2-4 max).""", "story_dramatic", group="story"),
)

GAMING_SEMANTIC_TYPES = {
    "action": "Standard combat, shooting, fighting - normal intense gameplay without special moments",
    "funny": "Fails, glitches, ragdoll physics, unexpected humor, comedic timing, awkward moments",
    "clutch": "1vX clutch wins, last-second saves, comebacks from low health, narrowly avoiding death",
    "wtf": "Unexpected chaos, 'wait what?!' moments, random events, physics breaking, confusing situations",
    "epic_fail": "Embarrassing deaths, tragic blunders, game-losing mistakes, throwing the match",
    "hype": "Victory celebrations, 'LET'S GO' energy, peak excitement, winning moments",
    "skill": "Trick shots, flick shots, IQ plays, advanced mechanics, pro-level techniques, impressive aim",
}

GAMING_CATEGORY_TO_STYLE = {
    "action": "gaming",
    "funny": "funny",
    "clutch": "dramatic",
    "wtf": "funny",
    "epic_fail": "funny",
    "hype": "gaming",
    "skill": "dramatic",
}


_TYPES: List[ContentType] = [
    ContentType(
        key="gaming", label="Gaming", emoji="🎮", family="gaming",
        description="Gameplay, streams, and highlight reels",
        clip_noun="gameplay",
        semantic_types=GAMING_SEMANTIC_TYPES,
        analysis_focus="kills, funny fails, laughs, glitches, hypes",
        analysis_ignore="boring travel or looting",
        category_to_style=GAMING_CATEGORY_TO_STYLE,
        styles=GAMING_STYLES,
        default_style="gaming",
        uses_action_heuristics=True,
    ),
    ContentType(
        key="esports", label="Esports / Tournaments", emoji="🏆", family="gaming",
        description="Competitive matches, casts, and tournament broadcasts",
        clip_noun="esports match",
        semantic_types=GAMING_SEMANTIC_TYPES,
        analysis_focus="team fights, clutch rounds, objective steals, caster hype, crowd eruptions",
        analysis_ignore="loading screens, pauses, and downtime between rounds",
        category_to_style={
            "action": "esports_playcast", "funny": "esports_playcast", "clutch": "esports_clutch",
            "wtf": "esports_playcast", "epic_fail": "esports_playcast", "hype": "esports_clutch",
            "skill": "esports_playcast",
        },
        styles=(
            _style("esports_playcast", "🎮", "Playcast", "Caster-style tactical play calls", "TEAMFIGHT NOW!",
                   """Generate esports caster-style captions.
Examples: "TEAMFIGHT NOW!", "Perfect timing on the engage.", "Objective secured."
Fast-paced esports broadcast tone with tactical callouts.""", "story_news"),
            _style("esports_clutch", "🏁", "Clutch", "Clutch round high-tension style", "1v3. No fear.",
                   """Generate esports clutch-moment captions.
Examples: "1v3. No fear.", "Ice-cold decision making.", "CLUTCH FACTOR."
High-tension competitive style, short and intense.""", "dramatic"),
        ),
        default_style="esports_playcast",
        uses_action_heuristics=True,
    ),
    ContentType(
        key="podcasts", label="Podcasts", emoji="🎙️", family="talk",
        description="Long-form conversations and interviews-as-shows",
        clip_noun="podcast",
        semantic_types={
            "quote": "A memorable, quotable one-liner or bold claim that stands alone",
            "insight": "A key takeaway, framework, or piece of advice worth remembering",
            "debate": "Heated disagreement, pushback, or a strong contrarian opinion",
            "funny": "Banter, laughter, a joke that lands, or an awkward-funny beat",
            "story": "A vivid personal anecdote with a clear payoff",
            "revelation": "A surprising admission, confession, or little-known fact",
        },
        analysis_focus="quotable one-liners, strong opinions, surprising admissions, genuine laughter, heated disagreement",
        analysis_ignore="intros, ad reads, small talk, and long monologues with no payoff",
        category_to_style={
            "quote": "podcast_quote", "insight": "podcast_highlight", "debate": "podcast_quote",
            "funny": "podcast_highlight", "story": "podcast_highlight", "revelation": "podcast_quote",
        },
        styles=(
            _style("podcast_quote", "🧠", "Pull Quote", "Quotable pull-quotes and strong takes", "He actually said that.",
                   """Generate crisp pull-quote captions from podcast moments.
Examples: "That's the real problem.", "Clip this part.", "He actually said that."
Keep captions concise and quotable (3-10 words). Prioritize opinionated or insightful lines.""", "podcast_host"),
            _style("podcast_highlight", "🎧", "Highlight", "Conversational key-moment highlights", "Here's the key takeaway.",
                   """Generate conversational podcast highlight captions.
Examples: "This is where it gets interesting.", "Here's the key takeaway.", "Wait, that's huge."
Use natural spoken phrasing. Keep captions 4-12 words with clear context.""", "podcast_host"),
        ),
        default_style="podcast_quote",
    ),
    ContentType(
        key="interviews", label="Interviews", emoji="🗣️", family="talk",
        description="One-on-one interviews, press junkets, Q&As",
        clip_noun="interview",
        semantic_types={
            "quote": "A memorable, quotable answer that stands on its own",
            "hot_take": "A controversial or provocative opinion likely to spark debate",
            "revelation": "A surprising reveal, first-time admission, or exclusive detail",
            "emotional": "A vulnerable, moving, or visibly emotional moment",
            "funny": "A witty answer, joke, or spontaneous laugh",
            "insight": "A thoughtful explanation or lesson learned",
        },
        analysis_focus="quotable answers, hot takes, surprising reveals, emotional honesty, sharp wit",
        analysis_ignore="introductions, pleasantries, and rambling answers without a clear point",
        category_to_style={
            "quote": "interview_quote", "hot_take": "interview_hot_take", "revelation": "interview_hot_take",
            "emotional": "interview_quote", "funny": "interview_quote", "insight": "interview_quote",
        },
        styles=(
            _style("interview_quote", "🗨️", "Quote", "Best quotable lines", "Most honest line yet.",
                   """Generate interview pull-quote captions.
Examples: "That's a bold statement.", "He answered directly.", "Most honest line yet."
Keep captions quotable and context-rich.""", "podcast_host"),
            _style("interview_hot_take", "⚡", "Hot Take", "Controversial/hot-take moments", "This will spark debate.",
                   """Generate hot-take interview captions.
Examples: "This will spark debate.", "Unfiltered opinion.", "People won't agree on this."
Emphasize controversy and reaction-worthy moments.""", "commentator"),
        ),
        default_style="interview_quote",
    ),
    ContentType(
        key="news_commentary", label="News & Commentary", emoji="📰", family="talk",
        description="News segments, punditry, political and current-affairs commentary",
        clip_noun="news segment",
        semantic_types={
            "breaking": "A headline-worthy development, announcement, or urgent update",
            "debate": "Panelists clashing, a sharp rebuttal, or a heated exchange",
            "analysis": "A clear explanation of why something matters or what happens next",
            "reaction": "A strong emotional or shocked reaction to news",
            "quote": "A striking soundbite from a speaker or guest",
        },
        analysis_focus="headline moments, sharp rebuttals, striking soundbites, clear explanations of stakes",
        analysis_ignore="weather, ad breaks, schedule announcements, and repeated recaps",
        category_to_style={
            "breaking": "news_breaking", "debate": "news_debate", "analysis": "news_debate",
            "reaction": "news_breaking", "quote": "news_debate",
        },
        styles=(
            _style("news_breaking", "🚨", "Breaking", "Breaking-news urgency style", "BREAKING UPDATE",
                   """Generate urgent breaking-news style captions.
Examples: "BREAKING UPDATE", "Developing story.", "Major reaction just in."
Keep captions short, factual, and high-impact.""", "news_anchor"),
            _style("news_debate", "🎙️", "Debate", "Debate and counterpoint style", "Strong take incoming.",
                   """Generate commentary/debate style captions.
Examples: "Strong take incoming.", "This point is controversial.", "Counterargument: ..."
Emphasize contrasting viewpoints and discussion beats.""", "commentator"),
        ),
        default_style="news_breaking",
    ),
    ContentType(
        key="entertainment", label="Entertainment", emoji="🎬", family="entertainment",
        description="Movie reactions, anime highlights, pop-culture breakdowns",
        clip_noun="entertainment video",
        semantic_types={
            "reaction": "A big visible reaction - shock, laughter, jaw-drop - to what's on screen",
            "twist": "A plot twist, reveal, or unexpected turn being discussed or shown",
            "funny": "A genuinely funny beat, joke, or comedic reaction",
            "emotional": "A moving, heartfelt, or tear-jerking moment",
            "hype": "Peak excitement - a hyped scene, trailer beat, or fan-favourite moment",
            "recap": "A concise, clear summary of what happened and why it matters",
        },
        analysis_focus="big reactions, twists, hype scenes, emotional beats, sharp recaps",
        analysis_ignore="setup chatter, sponsor reads, and pauses while media loads",
        category_to_style={
            "reaction": "entertainment_reaction", "twist": "entertainment_reaction", "funny": "entertainment_reaction",
            "emotional": "entertainment_recap", "hype": "entertainment_reaction", "recap": "entertainment_recap",
        },
        styles=(
            _style("entertainment_reaction", "😲", "Reaction", "Live reactions to twists and reveals", "I DID NOT SEE THAT COMING.",
                   """Generate entertainment reaction captions.
Examples: "I DID NOT SEE THAT COMING.", "That scene was wild.", "What a twist."
Focus on emotional reactions and surprise moments.""", "vlogger"),
            _style("entertainment_recap", "📝", "Recap", "Quick recap of key scenes", "Here's what happened.",
                   """Generate entertainment recap captions.
Examples: "Quick recap:", "Here's what happened.", "Key moment right here."
Summarize important scenes clearly and briefly.""", "podcast_host"),
        ),
        default_style="entertainment_reaction",
    ),
    ContentType(
        key="tv_shows", label="TV Shows", emoji="📺", family="entertainment",
        description="Episode recaps, scene breakdowns, series highlights",
        clip_noun="TV episode",
        semantic_types={
            "reveal": "A major plot reveal or secret coming to light",
            "cliffhanger": "A tense ending beat or unresolved moment that demands the next episode",
            "funny": "A comedic scene, one-liner, or running gag",
            "emotional": "A heartfelt, tragic, or cathartic character moment",
            "confrontation": "A charged argument, standoff, or dramatic showdown",
            "recap": "A scene that efficiently sums up where the story stands",
        },
        analysis_focus="reveals, cliffhangers, confrontations, emotional payoffs, iconic one-liners",
        analysis_ignore="opening credits, previously-on montages, and establishing shots",
        category_to_style={
            "reveal": "tv_dramatic", "cliffhanger": "tv_dramatic", "funny": "tv_recap",
            "emotional": "tv_dramatic", "confrontation": "tv_dramatic", "recap": "tv_recap",
        },
        styles=(
            _style("tv_recap", "📺", "Recap", "Scene recap and context", "Episode turning point.",
                   """Generate TV show recap captions.
Examples: "Episode turning point.", "Character arc shift.", "Big reveal."
Concise narrative recap with context.""", "podcast_host"),
            _style("tv_dramatic", "🎭", "Dramatic", "Dramatic cliffhanger style", "No one was ready for this.",
                   """Generate dramatic TV-style captions.
Examples: "No one was ready for this.", "Everything changes now.", "The tension is unreal."
Build tension and cliffhanger energy.""", "story_dramatic"),
        ),
        default_style="tv_recap",
    ),
    ContentType(
        key="comedy", label="Comedy", emoji="😂", family="entertainment",
        description="Stand-up sets, sketches, improv, and crowd work",
        clip_noun="comedy set",
        semantic_types={
            "punchline": "A joke landing hard - clear setup and payoff",
            "crowd_reaction": "The audience erupting, groaning, or losing it",
            "callback": "A callback to an earlier bit that pays off",
            "physical": "Physical comedy, expressions, or a visual gag",
            "awkward": "Awkward silence, a bombing joke, or cringe-funny tension",
            "roast": "Crowd work, roasting a heckler, or a savage burn",
        },
        analysis_focus="punchlines that land, crowd eruptions, savage crowd work, perfect callbacks",
        analysis_ignore="stage entrances, mic adjustments, and long setups with no payoff inside the clip",
        category_to_style={
            "punchline": "comedy_punchline", "crowd_reaction": "comedy_reaction", "callback": "comedy_punchline",
            "physical": "comedy_reaction", "awkward": "comedy_reaction", "roast": "comedy_punchline",
        },
        styles=(
            _style("comedy_punchline", "🥁", "Punchline", "Punchline-focused timing", "AND THAT'S THE JOKE.",
                   """Generate comedy punchline captions.
Examples: "AND THAT'S THE JOKE.", "Perfect timing.", "Crowd lost it."
Prioritize setup/payoff rhythm and punchline emphasis.""", "comedian"),
            _style("comedy_reaction", "🤣", "Reaction", "Crowd and reaction-focused captions", "Everyone's dying laughing.",
                   """Generate crowd/reaction comedy captions.
Examples: "He can't keep a straight face.", "Everyone's dying laughing.", "That landed."
Focus on reactions, laughter, and comedic aftermath.""", "funny"),
        ),
        default_style="comedy_punchline",
    ),
    ContentType(
        key="music", label="Music", emoji="🎵", family="entertainment",
        description="Concerts, live sessions, DJ sets, music videos",
        clip_noun="music performance",
        semantic_types={
            "drop": "The beat drop, bass hit, or biggest energy release",
            "chorus": "The hook or chorus - the most singable, memorable part",
            "crowd": "The crowd singing along, jumping, or erupting",
            "solo": "A standout solo, vocal run, or virtuosic moment",
            "vibe": "An atmospheric, aesthetic, mood-heavy passage",
            "transition": "A slick transition, build-up, or tension-and-release",
        },
        analysis_focus="drops, choruses, crowd sing-alongs, standout solos, atmospheric peaks",
        analysis_ignore="soundchecks, tuning, dead air between songs",
        category_to_style={
            "drop": "music_hype", "chorus": "music_hype", "crowd": "music_hype",
            "solo": "music_hype", "vibe": "music_vibe", "transition": "music_vibe",
        },
        styles=(
            _style("music_hype", "🎤", "Hype", "Performance/drop hype captions", "THE DROP IS INSANE",
                   """Generate concert/music hype captions.
Examples: "THE DROP IS INSANE", "Crowd went wild.", "This hook is unreal."
Punchy, rhythmic, high-energy captions.""", "music_hype"),
            _style("music_vibe", "🌙", "Vibe", "Mood and vibe-first minimal captions", "Late-night energy.",
                   """Generate aesthetic music vibe captions.
Examples: "Late-night energy.", "On repeat.", "Pure vibes."
Minimal, mood-first captions (1-5 words).""", "chill_narrator"),
        ),
        default_style="music_hype",
    ),
    ContentType(
        key="vlogs", label="Vlogs", emoji="📷", family="lifestyle",
        description="Daily life, travel, and personal storytelling",
        clip_noun="vlog",
        semantic_types={
            "highlight": "The standout moment of the day - the thing worth showing everyone",
            "story": "A personal story or reflection with a clear arc",
            "funny": "A funny mishap, joke, or spontaneous laugh",
            "emotional": "A vulnerable, heartfelt, or reflective moment",
            "reveal": "A surprise, unboxing, first look, or big announcement",
            "scenic": "A beautiful location, view, or visually striking shot",
        },
        analysis_focus="peak moments, surprises, funny mishaps, heartfelt reflections, stunning visuals",
        analysis_ignore="transit filler, repeated greetings, and unedited camera setup",
        category_to_style={
            "highlight": "vlog_highlight", "story": "vlog_story", "funny": "vlog_highlight",
            "emotional": "vlog_story", "reveal": "vlog_highlight", "scenic": "vlog_highlight",
        },
        styles=(
            _style("vlog_story", "📖", "Story", "Personal storytelling beats", "I didn't expect this.",
                   """Generate personal vlog storytelling captions.
Examples: "I didn't expect this.", "Best part of the trip.", "Core memory unlocked."
Warm, relatable tone. Keep captions 3-10 words.""", "vlogger"),
            _style("vlog_highlight", "✨", "Highlight", "Memorable life/travel highlights", "Peak moment.",
                   """Generate vlog highlight captions.
Examples: "Peak moment.", "Worth every second.", "This made my day."
Focus on memorable moments and emotional payoff.""", "vlogger"),
        ),
        default_style="vlog_story",
    ),
    ContentType(
        key="cooking_diy", label="Cooking & DIY", emoji="🍳", family="lifestyle",
        description="Recipes, builds, crafts, and how-to transformations",
        clip_noun="cooking or DIY video",
        semantic_types={
            "technique": "A key technique or step that makes the result work",
            "reveal": "The final result, plating, or before/after transformation",
            "tip": "A pro tip, shortcut, or common-mistake warning",
            "fail": "Something going wrong and the recovery or reaction",
            "satisfying": "A visually satisfying moment - sizzle, pour, cut, snap-fit",
            "taste_test": "The taste test, first use, or verdict on the result",
        },
        analysis_focus="key techniques, satisfying visuals, transformations, pro tips, taste-test verdicts",
        analysis_ignore="ingredient shopping, long waits, and repeated stirring with no change",
        category_to_style={
            "technique": "cooking_step", "reveal": "cooking_reveal", "tip": "cooking_step",
            "fail": "cooking_reveal", "satisfying": "cooking_reveal", "taste_test": "cooking_reveal",
        },
        styles=(
            _style("cooking_step", "👨‍🍳", "Step", "Step-by-step instructions", "Step 2: fold gently.",
                   """Generate cooking/DIY step captions.
Examples: "Step 2: fold gently.", "Don't overmix here.", "Now plate it."
Instructional and clear, with practical guidance.""", "chef"),
            _style("cooking_reveal", "🍽️", "Reveal", "Transformation/reveal moments", "Final reveal.",
                   """Generate cooking/DIY reveal captions.
Examples: "Final reveal.", "Before vs after.", "This turned out incredible."
Highlight transformation moments and satisfying results.""", "chef"),
        ),
        default_style="cooking_step",
    ),
    ContentType(
        key="fitness", label="Fitness", emoji="💪", family="lifestyle",
        description="Workouts, form breakdowns, PR attempts, transformations",
        clip_noun="workout video",
        semantic_types={
            "form_cue": "A clear form correction or coaching cue being demonstrated",
            "pr": "A max-effort lift, PR attempt, or peak-intensity set",
            "motivation": "A motivational message, mindset talk, or push-through moment",
            "transformation": "A before/after or progress comparison",
            "fail": "A failed rep, funny gym moment, or near-miss",
            "tip": "A practical training or nutrition tip",
        },
        analysis_focus="PR attempts, form cues, peak-intensity effort, transformations, motivating talk",
        analysis_ignore="rest periods, equipment setup, and walking between stations",
        category_to_style={
            "form_cue": "fitness_coach", "pr": "fitness_motivation", "motivation": "fitness_motivation",
            "transformation": "fitness_motivation", "fail": "fitness_coach", "tip": "fitness_coach",
        },
        styles=(
            _style("fitness_coach", "🏋️", "Coach", "Form cues and coaching prompts", "Keep your core tight.",
                   """Generate coaching-style fitness captions.
Examples: "Keep your core tight.", "Drive through the heels.", "Control the tempo."
Actionable form cues in concise language.""", "coach"),
            _style("fitness_motivation", "💥", "Motivation", "Motivation and intensity boosts", "One more rep.",
                   """Generate motivational fitness captions.
Examples: "One more rep.", "No excuses today.", "Progress over perfection."
Short, motivating captions with high intent.""", "coach"),
        ),
        default_style="fitness_coach",
    ),
    ContentType(
        key="educational", label="Educational", emoji="📚", family="knowledge",
        description="Lectures, explainers, tutorials, and courses",
        clip_noun="lesson",
        semantic_types={
            "explanation": "A clear explanation of a core concept",
            "aha": "The 'aha' moment - an insight that makes everything click",
            "demo": "A worked example, demonstration, or live walkthrough",
            "misconception": "Busting a myth or correcting a common mistake",
            "summary": "A crisp summary or key-takeaway recap",
            "question": "A compelling question or hook that sets up the lesson",
        },
        analysis_focus="clear explanations, aha insights, myth-busting, worked examples, crisp summaries",
        analysis_ignore="housekeeping announcements, slide transitions, and technical setup",
        category_to_style={
            "explanation": "educational_explainer", "aha": "educational_aha", "demo": "educational_explainer",
            "misconception": "educational_aha", "summary": "educational_explainer", "question": "educational_aha",
        },
        styles=(
            _style("educational_explainer", "🧩", "Explainer", "Step-by-step explainer captions", "Here's why this works.",
                   """Generate clear educational explainer captions.
Examples: "Here's why this works.", "Step 1: isolate the variable.", "This is the core concept."
Be concise, structured, and easy to follow. Prioritize clarity over hype.""", "teacher"),
            _style("educational_aha", "💡", "Aha Moment", "Aha-moment and key insight captions", "Now it clicks.",
                   """Generate 'aha moment' educational captions.
Examples: "Now it clicks.", "This changes everything.", "That's the trick."
Highlight key insights and breakthroughs in simple language.""", "teacher"),
        ),
        default_style="educational_explainer",
    ),
    ContentType(
        key="documentaries", label="Documentaries", emoji="🎥", family="knowledge",
        description="Nature, history, science, and investigative films",
        clip_noun="documentary",
        semantic_types={
            "insight": "A key fact or insight that reframes the subject",
            "spectacle": "A visually stunning shot or natural spectacle",
            "revelation": "A surprising discovery or twist in the investigation",
            "emotional": "A moving human or animal moment",
            "tension": "A suspenseful sequence - the hunt, the countdown, the risk",
            "archival": "Striking archival footage or a historical moment",
        },
        analysis_focus="stunning visuals, surprising discoveries, suspenseful sequences, reframing insights",
        analysis_ignore="title cards, credits, and slow establishing shots without narration",
        category_to_style={
            "insight": "documentary_insight", "spectacle": "documentary_wonder", "revelation": "documentary_insight",
            "emotional": "documentary_wonder", "tension": "documentary_wonder", "archival": "documentary_insight",
        },
        styles=(
            _style("documentary_insight", "🔍", "Insight", "Informative insight narration", "This explains the pattern.",
                   """Generate documentary insight captions.
Examples: "Key insight:", "This explains the pattern.", "The data tells a story."
Informative, thoughtful, and precise.""", "documentary_narrator"),
            _style("documentary_wonder", "🌍", "Wonder", "Awe-focused spectacle captions", "Hard to believe this is real.",
                   """Generate documentary wonder captions.
Examples: "Nature at full scale.", "Hard to believe this is real.", "A rare moment captured."
Emphasize awe, discovery, and visual spectacle.""", "documentary_narrator"),
        ),
        default_style="documentary_insight",
    ),
    ContentType(
        key="sports", label="Sports", emoji="⚽", family="sports",
        description="Matches, races, fights, and highlight packages",
        clip_noun="sports broadcast",
        semantic_types={
            "score": "A goal, basket, touchdown, knockout, or race finish",
            "clutch": "A last-second, high-pressure play that decides the outcome",
            "skill": "A highlight-reel move - dribble, dunk, trick, perfect technique",
            "celebration": "Players or fans celebrating, the stadium erupting",
            "controversy": "A contentious call, foul, fight, or heated exchange",
            "fail": "A blooper, miss, or embarrassing mistake",
        },
        analysis_focus="scores, clutch plays, highlight-reel skill, celebrations, controversial calls",
        analysis_ignore="timeouts, replays of routine plays, commercials, and warm-ups",
        category_to_style={
            "score": "sports_playbyplay", "clutch": "sports_playbyplay", "skill": "sports_playbyplay",
            "celebration": "sports_hype", "controversy": "sports_playbyplay", "fail": "sports_hype",
        },
        styles=(
            _style("sports_playbyplay", "📣", "Play-by-Play", "Play-by-play commentator energy", "HE'S THROUGH!",
                   """Generate energetic sports play-by-play captions.
Examples: "HE'S THROUGH!", "LAST-SECOND FINISH!", "WHAT A COMEBACK!"
Short, high-energy, broadcast-like. Use CAPS for peak moments.""", "sports_caster"),
            _style("sports_hype", "🔥", "Hype", "Crowd hype and clutch moment punches", "ABSOLUTE SCENES!",
                   """Generate crowd-hype sports captions.
Examples: "THE STADIUM ERUPTS!", "ABSOLUTE SCENES!", "GOOSEBUMPS."
Very punchy captions (1-6 words) focused on emotion and momentum.""", "sports_caster"),
        ),
        default_style="sports_playbyplay",
    ),
]

_TYPE_INDEX: Dict[str, ContentType] = {c.key: c for c in _TYPES}
_STYLE_INDEX: Dict[str, CaptionStyle] = {}
for _c in _TYPES:
    for _s in _c.styles:
        _STYLE_INDEX.setdefault(_s.key, _s)

_FAMILIES = (
    ("gaming", "Gaming", "🎮"),
    ("talk", "Talk & Commentary", "🎙️"),
    ("entertainment", "Entertainment", "🎬"),
    ("lifestyle", "Lifestyle", "✨"),
    ("knowledge", "Knowledge", "📚"),
    ("sports", "Sports", "⚽"),
)

DEFAULT_TYPE = "gaming"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_video_type() -> str:
    """Normalized VIDEO_TYPE from the environment (defaults to gaming)."""
    return os.getenv("VIDEO_TYPE", DEFAULT_TYPE).strip().lower() or DEFAULT_TYPE


def get_content_type(key: Optional[str] = None) -> ContentType:
    """Look up a content type; unknown keys fall back to gaming."""
    k = (key if key is not None else get_video_type()).strip().lower()
    return _TYPE_INDEX.get(k, _TYPE_INDEX[DEFAULT_TYPE])


def list_content_types() -> List[ContentType]:
    return list(_TYPES)


def list_families() -> List[Family]:
    return [
        Family(key, label, emoji, tuple(c for c in _TYPES if c.family == key))
        for key, label, emoji in _FAMILIES
    ]


def get_style(style_key: Optional[str]) -> Optional[CaptionStyle]:
    if not style_key:
        return None
    return _STYLE_INDEX.get(style_key.strip().lower())


def resolve_caption_style(
    requested: Optional[str],
    detected_category: Optional[str],
    content_type: Optional[ContentType] = None,
) -> str:
    """Turn a CAPTION_STYLE setting into a concrete style key.

    Explicit styles are returned as-is. ``auto`` maps the AI-detected category
    through the content type's own category→style table and falls back to the
    type's default style, so a podcast never resolves to a gaming style.
    """
    ct = content_type or get_content_type()
    style = (requested or "auto").strip().lower()
    if style != "auto":
        return style
    if detected_category:
        mapped = ct.category_to_style.get(detected_category.strip().lower())
        if mapped:
            return mapped
    return ct.default_style


def get_voice_preset(style_key: Optional[str]) -> Optional[str]:
    """Voice description for a caption style, or None if the style is unknown."""
    style = get_style(style_key)
    if style is None:
        return None
    return VOICE_PRESETS.get(style.voice_key)
