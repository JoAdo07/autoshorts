"""Fresh-process GPU regression checks; never imported by the mocked suite."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def command(*args, timeout=60):
    result = subprocess.run(args, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result


def probe(path):
    return json.loads(command("ffprobe", "-v", "error", "-show_streams", "-show_format",
                              "-of", "json", str(path)).stdout)


def stream(path, kind):
    return next(s for s in probe(path)["streams"] if s["codec_type"] == kind)


def video(directory, name="source.mp4"):
    return directory / name


def params(directory, name="source.mp4", duration=0.8):
    from shorts import RenderParams
    return RenderParams(video(directory, name), 0.0, duration, 160, 240,
                        0, 0, 160, 90, 160, 240, False)


def check_fixtures(directory):
    command("ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=24000:duration=2",
            "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video(directory)))
    command("ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=white:size=160x90:rate=10:duration=2",
            "-c:v", "h264_nvenc", str(video(directory, "static.mp4")))
    command("ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=red:size=160x90:rate=10", "-frames:v", "1",
            "-c:v", "h264_nvenc", str(video(directory, "single.mp4")))
    command("ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=160x288:rate=10:duration=1",
            "-c:v", "h264_nvenc", str(video(directory, "portrait.mp4")))
    # Cut exactly at a decode-batch boundary (128 frames), with enough quiet
    # frames for the scene filter to recognize both sides.
    command("ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=black:size=160x90:rate=30:duration=4.2666667",
            "-f", "lavfi", "-i", "color=white:size=160x90:rate=30:duration=1",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
            "-c:v", "h264_nvenc", str(video(directory, "cut.mp4")))
    import soundfile as sf
    sample_rate = 24000
    t = np.arange(sample_rate * 2) / sample_rate
    burst = np.sin(2 * np.pi * 440 * t) * np.where(t < 1, 0.01, 0.8)
    sf.write(directory / "burst.wav", burst, sample_rate)
    sf.write(directory / "short.wav", np.zeros(100), sample_rate)
    sf.write(directory / "silence.wav", np.zeros(sample_rate), sample_rate)
    sf.write(directory / "voice.wav", 0.5 * np.sin(2 * np.pi * 800 * t), sample_rate)
    long_t = np.arange(sample_rate * 6) / sample_rate
    sf.write(directory / "long_voice.wav", 0.5 * np.sin(2 * np.pi * 800 * long_t), sample_rate)


def check_hardware(directory):
    import torch
    from decord import VideoReader, gpu, bridge
    assert torch.cuda.is_available(), "CUDA is required when --run-gpu is requested"
    bridge.set_bridge("torch")
    reader = VideoReader(str(video(directory)), ctx=gpu(0))
    frames = reader.get_batch([0, 1])
    assert frames.is_cuda and frames.dtype == torch.uint8
    assert tuple(frames.shape) == (2, 90, 160, 3)


def check_decoder_env(directory):
    import os
    import dotenv
    os.environ.pop("DECORD_EOF_RETRY_MAX", None)

    def load_decoder_config():
        os.environ["DECORD_EOF_RETRY_MAX"] = "1048576"

    dotenv.load_dotenv = load_decoder_config
    import shorts
    reader = shorts.VideoReader(str(video(directory)), ctx=shorts.gpu(0))
    assert reader.get_batch([0, 1]).is_cuda


def check_blur_constant(directory):
    import torch
    from shorts import gaussian_blur_torch
    for shape in [(8, 12, 3), (2, 8, 12, 3)]:
        image = torch.full(shape, 100.0, device="cuda")
        blurred = gaussian_blur_torch(image, 3.0)
        assert blurred.is_cuda and blurred.shape == image.shape
        torch.testing.assert_close(blurred, image, rtol=1e-5, atol=1e-4)
    image = torch.full((12, 12, 3), 100, device="cuda", dtype=torch.uint8)
    blurred = gaussian_blur_torch(image, 2.0)
    assert blurred.dtype == torch.uint8
    assert int(blurred.min()) >= 99


def check_blur_impulse(directory):
    import torch
    from shorts import blur_gpu
    image = torch.zeros((31, 31, 3), device="cuda")
    image[15, 15, 0] = 1
    result = blur_gpu(image, sigma=2)
    assert result.is_cuda
    assert 0 < result[15, 15, 0] < 1
    torch.testing.assert_close(result[:, :, 0], result[:, :, 0].flip(0))
    assert float(result[:, :, 0].sum()) == pytest_approx(1, 0.0001)
    assert float(result[:, :, 1:].abs().sum()) == 0


def pytest_approx(value, tolerance):
    import pytest
    return pytest.approx(value, abs=tolerance)


def check_hsv_colors(directory):
    import torch
    from shorts import rgb_to_hsv_torch
    rgb = torch.tensor([[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0],
                        [255, 255, 255]], device="cuda").float().view(5, 3, 1, 1)
    got = rgb_to_hsv_torch(rgb).squeeze(-1).squeeze(-1)
    expected = torch.tensor([[0, 255, 255], [60, 255, 255], [120, 255, 255],
                             [0, 0, 0], [0, 0, 255]], device="cuda").float()
    torch.testing.assert_close(got, expected, atol=0.1, rtol=0)
    torch.testing.assert_close(rgb_to_hsv_torch(rgb / 255), rgb_to_hsv_torch(rgb))


def check_video_static(directory):
    from shorts import analyze_video_content
    result = analyze_video_content(video(directory, "static.mp4"))
    assert len(result.scenes) == 1
    assert result.scenes[0][0].get_seconds() == 0
    assert result.scenes[0][1].get_seconds() == pytest_approx(2.0, 0.1)
    assert len(result.video_times) == len(result.video_scores)
    assert np.isfinite(result.video_scores).all()


def check_video_single_frame(directory):
    from shorts import analyze_video_content
    result = analyze_video_content(video(directory, "single.mp4"))
    assert len(result.video_times) == len(result.video_scores) == 1
    assert np.isfinite(result.video_scores).all()


def check_video_batch_boundary(directory):
    from shorts import analyze_video_content
    result = analyze_video_content(video(directory, "cut.mp4"))
    assert len(result.scenes) == 2
    assert result.scenes[0][1].get_seconds() == pytest_approx(128 / 30, 0.05)
    assert result.scenes[1][0].get_seconds() == pytest_approx(128 / 30, 0.05)
    assert len(result.video_scores) == len(result.video_times)


def check_analysis_decode_failure(directory):
    import pytest
    import shorts
    original_reader = shorts.VideoReader

    class FailingReader:
        def __init__(self, *args, **kwargs):
            self.reader = original_reader(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.reader, name)

        def __len__(self):
            return len(self.reader)

        def __getitem__(self, index):
            return self.reader[index]

        def get_batch(self, indices):
            if 128 in indices:
                raise OSError("Corrupt decode batch")
            return self.reader.get_batch(indices)

    shorts.VideoReader = FailingReader
    with pytest.raises(RuntimeError, match="128"):
        shorts.analyze_video_content(video(directory, "cut.mp4"))


def check_audio_burst(directory):
    from shorts import compute_audio_action_profile
    times, scores = compute_audio_action_profile(directory / "burst.wav")
    assert len(times) == len(scores) > 0
    assert np.isfinite(scores).all()
    assert scores[(times > 1.2) & (times < 1.8)].mean() > scores[times < 0.8].mean()


def check_audio_short(directory):
    from shorts import compute_audio_action_profile
    times, scores = compute_audio_action_profile(directory / "short.wav")
    assert len(times) == len(scores) > 0
    assert np.isfinite(scores).all()


def check_audio_silence(directory):
    from shorts import compute_audio_action_profile
    times, scores = compute_audio_action_profile(directory / "silence.wav")
    assert len(times) == len(scores) > 0
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores, 0, atol=1e-5)


def check_render_audio(directory):
    from shorts import render_video_gpu
    output = directory / "render-audio.mp4"
    render_video_gpu(params(directory), output)
    s = stream(output, "video")
    assert (s["width"], s["height"]) == (160, 240)
    assert float(s["duration"]) == pytest_approx(0.8, 0.1)
    assert stream(output, "audio")["codec_name"] == "aac"
    command("ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-")


def check_render_silent(directory):
    from shorts import render_video_gpu
    output = directory / "render-silent.mp4"
    render_video_gpu(params(directory, "static.mp4"), output)
    assert float(stream(output, "video")["duration"]) == pytest_approx(0.8, 0.1)
    assert not any(s["codec_type"] == "audio" for s in probe(output)["streams"])


def check_render_portrait(directory):
    from shorts import render_video_gpu
    output = directory / "render-portrait.mp4"
    p = replace(params(directory, "portrait.mp4"), crop_w=160, crop_h=288,
                output_width=160, output_height=288, is_vertical_bg=True)
    render_video_gpu(p, output)
    s = stream(output, "video")
    assert (s["width"], s["height"]) == (160, 288)
    assert float(s["duration"]) == pytest_approx(0.8, 0.1)


def check_render_offset_crop(directory):
    import cv2
    from shorts import VideoReader, gpu, render_video_gpu
    output = directory / "render-offset.mp4"
    p = replace(params(directory), start_time=0.5, crop_x=20, crop_y=10,
                crop_w=120, crop_h=60)
    render_video_gpu(p, output)
    reader = VideoReader(str(video(directory)), ctx=gpu(0))
    frame = reader[5].cpu().numpy()
    expected = cv2.resize(frame[10:70, 20:140], (160, 80)).astype(float)
    data = command("ffmpeg", "-v", "error", "-i", str(output), "-frames:v", "1",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-").stdout
    actual = np.frombuffer(data, dtype=np.uint8).reshape(240, 160, 3)[80:160].astype(float)
    assert np.mean(np.abs(actual - expected)) < 8


def check_render_failure(directory):
    import pytest
    from shorts import render_video_gpu
    output = directory / "invalid-render.mp4"
    # NVENC cannot encode an odd-sized yuv420p output. Never report a successful
    # render after FFmpeg exits or the rawvideo pipe breaks.
    with pytest.raises((RuntimeError, ValueError)):
        render_video_gpu(replace(params(directory), output_width=159), output)
    assert not output.exists()


def check_render_encoder_failure(directory):
    import pytest
    from shorts import render_video_gpu
    output = directory / "invalid-container.invalid"
    with pytest.raises(RuntimeError):
        render_video_gpu(params(directory), output)
    assert not output.exists()
    assert not output.with_suffix(".aac").exists()


def check_render_isolated_success(directory):
    from shorts import render_video_gpu_isolated
    output = directory / "isolated.mp4"
    render_video_gpu_isolated(params(directory), output)
    assert float(stream(output, "video")["duration"]) == pytest_approx(0.8, 0.1)
    assert not output.with_suffix(".aac").exists()


def check_render_isolated_failure(directory):
    import pytest
    from shorts import render_video_gpu_isolated
    with pytest.raises(RuntimeError):
        render_video_gpu_isolated(replace(params(directory), source_path=directory / "missing.mp4"),
                                  directory / "missing-render.mp4")


def crash_render(params, output_path, **kwargs):
    import os
    import signal
    output_path.write_bytes(b"partial render")
    output_path.with_suffix(".aac").write_bytes(b"partial audio")
    os.kill(os.getpid(), signal.SIGKILL)


def check_render_isolated_crash(directory):
    import pytest
    import shorts
    shorts.render_video_gpu = crash_render
    for name, previous in [("killed.mp4", None), ("existing.mp4", b"previous good clip")]:
        output = directory / name
        if previous is not None:
            output.write_bytes(previous)
        with pytest.raises(RuntimeError):
            shorts.render_video_gpu_isolated(params(directory), output)
        if previous is None:
            assert not output.exists()
        else:
            assert output.read_bytes() == previous
        assert not output.with_suffix(".aac").exists()


def audio_rms(path, start=0.1, end=0.5):
    audio = command("ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
                    "-f", "f32le", "-ac", "1", "-ar", "24000", "-").stdout
    samples = np.frombuffer(audio, dtype="<f4")
    return float(np.sqrt(np.mean(samples[int(start * 24000):int(end * 24000)] ** 2)))


def check_mix_silent_gain(directory):
    from tts_generator import mix_audio_with_video
    output = directory / "mix-gain.mp4"
    assert mix_audio_with_video(video(directory, "static.mp4"), directory / "voice.wav",
                               output, voiceover_volume=0.2)
    assert audio_rms(output) == pytest_approx(0.5 / np.sqrt(2) * 0.2, 0.01)


def check_mix_original_gain(directory):
    from tts_generator import mix_audio_with_video
    output = directory / "mix-original-gain.mp4"
    assert mix_audio_with_video(video(directory), directory / "voice.wav", output,
                               game_audio_volume=0, voiceover_volume=0.2)
    assert audio_rms(output) == pytest_approx(0.5 / np.sqrt(2) * 0.2, 0.01)


def subtitle_gain(directory, name):
    import os
    from types import SimpleNamespace
    import ai_providers
    import tts_generator
    from subtitle_generator import generate_subtitles
    os.environ.update(ENABLE_SUBTITLES="true", ENABLE_TTS="true", SUBTITLE_MODE="ai_captions",
                      VIDEO_TYPE="educational", CAPTION_STYLE="educational_explainer",
                      TTS_GAME_AUDIO_VOLUME="0", TTS_VOICEOVER_VOLUME="0.2",
                      ENABLE_CAPTION_EMOJIS="false")
    ai_providers.generate_ai_captions = lambda *args, **kwargs: ai_providers.CaptionResult(
        captions=[ai_providers.Caption(0.1, 1.8, "Clear helpful captions")], provider="fixture", success=True)
    t = np.arange(24000 * 6) / 24000
    voice = (0.5 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)
    model = SimpleNamespace(generate_voice_design=lambda **kwargs: ([voice], 24000))
    tts = SimpleNamespace(_model=model, _ensure_initialized=lambda: None,
                          config=tts_generator.TTSConfig())
    tts_generator.QwenTTS.get_instance = lambda *args, **kwargs: tts
    output = directory / f"subtitle-gain-{name}"
    meta = asdict(params(directory, name))
    result = generate_subtitles(video(directory, name), output, "explanation", render_meta=meta)
    assert result and result.is_file()
    assert audio_rms(result) == pytest_approx(0.5 / np.sqrt(2) * 0.2, 0.01)
    assert audio_rms(result, 2.5, 3.0) == pytest_approx(0.5 / np.sqrt(2) * 0.2, 0.01)
    assert float(stream(result, "video")["duration"]) >= 6


def check_subtitle_gain_source(directory):
    subtitle_gain(directory, "source.mp4")


def check_subtitle_gain_silent(directory):
    subtitle_gain(directory, "static.mp4")


def check_mix_long_voice(directory):
    from tts_generator import mix_audio_with_video
    output = directory / "mix-long.mp4"
    assert mix_audio_with_video(video(directory), directory / "long_voice.wav", output)
    assert float(stream(output, "video")["duration"]) >= 6.0
    assert float(stream(output, "audio")["duration"]) >= 5.9


def check_mix_source_eof(directory):
    from tts_generator import RenderMeta, mix_audio_with_video
    meta = RenderMeta(video(directory), 0.0, 2.0, 160, 90, 0, 0, 160, 90, 160, 90, False)
    output = directory / "mix-eof.mp4"
    assert mix_audio_with_video(video(directory), directory / "long_voice.wav", output,
                               render_meta=meta)
    assert float(stream(output, "video")["duration"]) >= 6.0
    assert float(stream(output, "audio")["duration"]) >= 5.9


def check_subtitle_paths(directory):
    from subtitle_generator import SubtitleConfig, _apply_ffmpeg_subtitles
    srt = directory / "captions [test], it's.srt"
    srt.write_text("1\n00:00:00,100 --> 00:00:01,800\nGPU subtitle test\n", encoding="utf-8")
    output = directory / "subtitles.mp4"
    assert _apply_ffmpeg_subtitles(video(directory), srt, output, SubtitleConfig(font_size=14))
    assert float(stream(output, "video")["duration"]) == pytest_approx(2, 0.1)
    assert stream(output, "audio")["codec_name"] == "aac"
    assert frames_differ(video(directory), output)


def frames_differ(first, second):
    def pixels(path):
        return command("ffmpeg", "-v", "error", "-ss", "0.5", "-i", str(path),
                       "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-").stdout
    a = np.frombuffer(pixels(first), dtype=np.uint8).astype(float)
    b = np.frombuffer(pixels(second), dtype=np.uint8).astype(float)
    return np.mean(np.abs(a - b)) > 2


def check_pycaps(directory):
    from subtitle_generator import _run_pycaps_worker, SubtitleConfig
    srt = directory / "pycaps.srt"
    srt.write_text("1\n00:00:00,100 --> 00:00:01,800\nClear helpful captions\n", encoding="utf-8")
    output = directory / "pycaps.mp4"
    _run_pycaps_worker(str(video(directory)), str(srt), str(output), SubtitleConfig(font_size=14))
    assert output.is_file()
    assert float(stream(output, "video")["duration"]) == pytest_approx(2, 0.1)
    assert stream(output, "audio")["codec_name"] == "aac"
    assert frames_differ(video(directory), output)


def check_candidate_extraction(directory):
    import ai_providers
    real_run = ai_providers.subprocess.run

    def gpu_only(args, **kwargs):
        assert "libx264" not in args, "Candidate extraction unexpectedly fell back to CPU"
        return real_run(args, **kwargs)

    ai_providers.subprocess.run = gpu_only
    output = directory / "candidate.mp4"
    try:
        assert ai_providers.extract_candidate_clip(video(directory), 0.5, 0.8, output)
    finally:
        ai_providers.subprocess.run = real_run
    assert stream(output, "video")["width"] == 640
    assert float(stream(output, "video")["duration"]) == pytest_approx(0.8, 0.15)
    assert stream(output, "audio")["codec_name"] == "aac"


def run_process(directory, deep=False):
    import os
    import shorts
    if deep:
        import ai_providers

        class FixtureAnalyzer:
            def is_available(self):
                return True

            def analyze_full_video(self, *args, **kwargs):
                return [(0.2, 1.7, "explanation", 0.95)]

        def unexpected_heuristics(*args, **kwargs):
            raise AssertionError("Successful deep analysis must bypass local heuristics")

        ai_providers.GeminiAnalyzer = FixtureAnalyzer
        shorts.analyze_video_content = unexpected_heuristics
        os.environ.update(AI_PROVIDER="gemini", AI_ANALYSIS_ENABLED="true",
                          GEMINI_DEEP_ANALYSIS="true", VIDEO_TYPE="educational")
    else:
        os.environ.update(AI_ANALYSIS_ENABLED="false", VIDEO_TYPE="gaming")
    output_dir = directory / ("deep-output" if deep else "local-output")
    output_dir.mkdir()
    config = shorts.ProcessingConfig(min_short_length=1, max_short_length=1, scene_limit=1)
    shorts.process_video(video(directory), config, output_dir)
    outputs = list(output_dir.glob("*.mp4"))
    assert len(outputs) == 1
    assert float(stream(outputs[0], "video")["duration"]) == pytest_approx(1, 0.1)
    assert stream(outputs[0], "audio")["codec_name"] == "aac"
    command("ffmpeg", "-v", "error", "-i", str(outputs[0]), "-f", "null", "-")


def check_process_local(directory):
    run_process(directory)


def check_process_deep(directory):
    run_process(directory, deep=True)


def check_real_footage(directory):
    import os
    from shorts import (VideoReader, gpu, analyze_video_content, compute_audio_action_profile,
                        RenderParams, render_video_gpu)
    source = Path(os.environ["AUTOSHORTS_GPU_SAMPLE"])
    metadata = probe(source)
    duration = float(metadata["format"]["duration"])
    assert 3 <= duration <= 30, "Supply a pre-extracted 3-30 second excerpt, not the full recording"
    reader = VideoReader(str(source), ctx=gpu(0))
    frames = reader.get_batch([0, len(reader) // 2])
    assert frames.is_cuda
    _, height, width, _ = frames.shape
    del reader, frames
    analysis = analyze_video_content(source)
    assert analysis.scenes
    assert analysis.scenes[0][0].get_seconds() == 0
    assert analysis.scenes[-1][1].get_seconds() == pytest_approx(duration, 0.2)
    assert len(analysis.video_times) == len(analysis.video_scores) > 0
    assert np.isfinite(analysis.video_scores).all()
    has_audio = any(s["codec_type"] == "audio" for s in metadata["streams"])
    if has_audio:
        times, scores = compute_audio_action_profile(source)
        assert len(times) == len(scores) > 0 and np.isfinite(scores).all()
    output = directory / "footage-render.mp4"
    p = RenderParams(source, 1.0, 2.0, 360, 640, 0, 0, width, height, 360, 640, False)
    render_video_gpu(p, output)
    result = stream(output, "video")
    assert (result["width"], result["height"]) == (360, 640)
    assert float(result["duration"]) == pytest_approx(2, 0.1)
    assert any(s["codec_type"] == "audio" for s in probe(output)["streams"]) == has_audio
    command("ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-")


def check_qwen_tts(directory):
    from tts_generator import QwenTTS, TTSConfig, get_audio_duration, generate_voice_description
    import torch
    import os
    os.environ["VIDEO_TYPE"] = "educational"
    tts = QwenTTS.get_instance(TTSConfig(enabled=True, device="cuda"))
    try:
        output = directory / "qwen.wav"
        assert tts.generate("A clear explanation makes all the difference.", output,
                            generate_voice_description("educational_explainer"))
        assert 0.5 < get_audio_duration(output) < 20
        assert tts._model.model.device.type == "cuda"
    finally:
        QwenTTS.clear_instance()
        torch.cuda.empty_cache()


def check_whisper(directory):
    from subtitle_generator import transcribe_audio
    import os
    os.environ["WHISPER_MODEL"] = "tiny.en"
    import whisper
    original_load = whisper.load_model

    def load_on_gpu(*args, **kwargs):
        model = original_load(*args, **kwargs)
        assert next(model.parameters()).is_cuda
        return model

    whisper.load_model = load_on_gpu
    if not (directory / "qwen.wav").exists():
        check_qwen_tts(directory)
    path = transcribe_audio(directory / "qwen.wav", directory / "whisper.srt")
    text = path.read_text().lower()
    assert "-->" in text
    assert "explanation" in text or "difference" in text


def check_model_roundtrip(directory):
    import re
    from tts_generator import get_audio_duration, mix_audio_with_video
    from subtitle_generator import _run_pycaps_worker, SubtitleConfig
    if not (directory / "qwen.wav").exists():
        check_qwen_tts(directory)
    if not (directory / "whisper.srt").exists():
        check_whisper(directory)
    duration = get_audio_duration(directory / "qwen.wav")
    mixed = directory / "speech-mixed.mp4"
    assert mix_audio_with_video(video(directory, "static.mp4"), directory / "qwen.wav", mixed,
                               voiceover_volume=0.8)
    assert float(stream(mixed, "video")["duration"]) >= duration - 0.1
    assert float(stream(mixed, "audio")["duration"]) >= duration - 0.1
    output = directory / "speech-captioned.mp4"
    _run_pycaps_worker(str(mixed), str(directory / "whisper.srt"), str(output), SubtitleConfig())
    assert float(stream(output, "video")["duration"]) >= duration - 0.1
    assert float(stream(output, "audio")["duration"]) >= duration - 0.1
    assert audio_rms(output) > 0.001
    endings = re.findall(r"--> (\d+):(\d+):(\d+),(\d+)", (directory / "whisper.srt").read_text())
    assert endings
    for h, m, s, ms in endings:
        assert int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000 <= duration + 0.1


def check_production_models(directory):
    import os
    import ai_providers
    from tts_generator import QwenTTS
    from subtitle_generator import generate_subtitles
    os.environ.update(ENABLE_SUBTITLES="true", ENABLE_TTS="true", SUBTITLE_MODE="ai_captions",
                      VIDEO_TYPE="educational", CAPTION_STYLE="educational_explainer",
                      TTS_GAME_AUDIO_VOLUME="0", TTS_VOICEOVER_VOLUME="0.2",
                      ENABLE_CAPTION_EMOJIS="false")
    ai_providers.generate_ai_captions = lambda *args, **kwargs: ai_providers.CaptionResult(
        captions=[ai_providers.Caption(0.1, 1.8, "A clear explanation makes all the difference.")],
        provider="fixture", success=True)
    tts = QwenTTS.get_instance()
    try:
        tts._ensure_initialized()
        assert tts._model.model.device.type == "cuda"
        original_generate = tts._model.generate_voice_design
        recorded = []

        def generate(**kwargs):
            wavs, rate = original_generate(**kwargs)
            recorded.append((wavs[0], rate))
            return wavs, rate

        tts._model.generate_voice_design = generate
        output = directory / "production-speech.mp4"
        result = generate_subtitles(video(directory), output, "explanation",
                                    render_meta=asdict(params(directory)))
        assert result and result.is_file() and recorded
        samples, rate = recorded[0]
        expected = float(np.sqrt(np.mean(samples[int(0.2 * rate):int(0.7 * rate)] ** 2))) * 0.2
        assert audio_rms(result, 0.3, 0.8) == pytest_approx(expected, 0.003)
        speech_end = 0.1 + len(samples) / rate
        assert float(stream(result, "video")["duration"]) >= speech_end - 0.1
        assert float(stream(result, "audio")["duration"]) >= speech_end - 0.1
    finally:
        QwenTTS.clear_instance()


if __name__ == "__main__":
    globals()[f"check_{sys.argv[1]}"](Path(sys.argv[2]))
