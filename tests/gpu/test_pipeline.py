"""Real GPU checks run outside the lightweight suite's global library mocks."""
import os
import signal
from pathlib import Path
import subprocess
import sys

import pytest


RUNNER = Path(__file__).with_name("checks.py")
pytestmark = pytest.mark.gpu


def run_check(name, directory, timeout=180, extra_env=None):
    env = {
        **os.environ,
        "AI_PROVIDER": "local",
        "GEMINI_API_KEY": "",
        "OPENAI_API_KEY": "",
        "DEBUG_SKIP_ANALYSIS": "false",
        "DEBUG_SKIP_RENDER": "false",
        "ENABLE_SUBTITLES": "false",
        "ENABLE_TTS": "false",
        "OMP_NUM_THREADS": "2",
        "DECORD_EOF_RETRY_MAX": "1048576",
        **(extra_env or {}),
    }
    with subprocess.Popen(
        [sys.executable, str(RUNNER), name, str(directory)],
        cwd=directory, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        finally:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()
    assert process.returncode == 0, stdout + stderr


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    directory = tmp_path_factory.mktemp("gpu-media")
    run_check("fixtures", directory)
    return directory


@pytest.mark.parametrize("name", [
    "hardware", "decoder_env", "blur_constant", "blur_impulse", "hsv_colors",
    "video_static", "video_single_frame", "video_batch_boundary", "analysis_decode_failure",
    "audio_burst", "audio_short", "audio_silence",
    "render_audio", "render_silent", "render_portrait", "render_offset_crop", "render_failure",
    "render_isolated_failure", "render_isolated_crash", "mix_silent_gain", "mix_long_voice",
    "mix_source_eof", "subtitle_paths", "mix_original_gain",
    "subtitle_gain_source", "subtitle_gain_silent",
    "render_encoder_failure", "render_isolated_success",
    "candidate_extraction", "process_local", "process_deep", "pycaps",
])
def test_real_gpu_pipeline(name, media):
    run_check(name, media)


@pytest.mark.models
@pytest.mark.parametrize("name", ["qwen_tts", "whisper", "model_roundtrip", "production_models"])
def test_real_model_inference(name, media):
    run_check(name, media, timeout=1200)


def test_real_footage(request, tmp_path):
    samples = request.config.getoption("--gpu-sample")
    if not samples:
        pytest.skip("requires a bounded --gpu-sample excerpt")
    for index, sample in enumerate(samples):
        directory = tmp_path / str(index)
        directory.mkdir()
        run_check("real_footage", directory,
                  extra_env={"AUTOSHORTS_GPU_SAMPLE": str(Path(sample).resolve())})
