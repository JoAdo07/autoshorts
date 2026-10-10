import sys
from unittest.mock import MagicMock
from pathlib import Path

# --- Mock GPU libraries BEFORE importing shorts ---
# We must mock decord, cupy, torchaudio, torch so that shorts.py can be imported
# even if these libraries are missing or if we are on a CPU-only node.

# Mock torch
torch_mock = MagicMock()
torch_mock.cuda.is_available.return_value = False
torch_mock.device.return_value = "cpu"
torch_mock.tensor = lambda x, **kwargs: MagicMock()
# Mock basic tensor ops used in shorts
torch_mock.abs = MagicMock()
torch_mock.mean = MagicMock()
torch_mock.sqrt = MagicMock()
torch_mock.cat = MagicMock()
torch_mock.from_numpy = lambda x: x
torch_mock.from_dlpack = MagicMock()
torch_mock.to_dlpack = MagicMock()
torch_mock.nn.functional.interpolate = MagicMock()
sys.modules["torch"] = torch_mock

# Mock torchaudio
torchaudio_mock = MagicMock()
sys.modules["torchaudio"] = torchaudio_mock

# Mock decord
decord_mock = MagicMock()
decord_mock.bridge.set_bridge = MagicMock()
decord_mock.cpu = lambda x: f"cpu({x})"
decord_mock.gpu = lambda x: f"gpu({x})"
sys.modules["decord"] = decord_mock

# Mock cupy
cupy_mock = MagicMock()
cupy_mock.asarray = MagicMock(side_effect=lambda x: x)
cupy_mock.asnumpy = MagicMock(side_effect=lambda x: x)
cupy_mock.from_dlpack = MagicMock()
cupy_mock.to_dlpack = MagicMock()
sys.modules["cupy"] = cupy_mock

# Mock cupyx
cupyx_mock = MagicMock()
sys.modules["cupyx"] = cupyx_mock
sys.modules["cupyx.scipy"] = MagicMock()
sys.modules["cupyx.scipy.ndimage"] = MagicMock()


# Ensure the src directory is on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Import shorts AFTER mocking
import shorts  # noqa: E402
from shorts import (  # noqa: E402
    combine_scenes,
    select_background_resolution,
    ProcessingConfig,
    _SecondsTime,
)


# Helper to create scene tuples
def make_scene(start: float, end: float):
    return (_SecondsTime(start), _SecondsTime(end))


def test_select_background_resolution():
    assert select_background_resolution(800) == (720, 1280)
    assert select_background_resolution(1500) == (1440, 2560)
    assert select_background_resolution(2100) == (2160, 3840)


def test_combine_scenes_merges_short_scenes():
    config = ProcessingConfig(min_short_length=5, max_short_length=10, max_combined_scene_length=15)
    scenes = [
        make_scene(0, 5),
        make_scene(5, 7),
        make_scene(7, 9),
        make_scene(9, 11),
        make_scene(11, 13),
        make_scene(13, 18),
    ]
    combined = combine_scenes(scenes, config)
    assert len(combined) == 1
    start, end = combined[0]
    assert start.get_seconds() == 5
    assert end.get_seconds() == 13

# Blur, sequential decoding and deep-analysis rendering are exercised with real
# tensors and media in tests/gpu, rather than assertions on MagicMock results.
