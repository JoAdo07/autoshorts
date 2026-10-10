import pytest


def pytest_addoption(parser):
    parser.addoption("--run-gpu", action="store_true", help="Run real CUDA/NVDEC/NVENC integration tests")
    parser.addoption("--run-models", action="store_true", help="Also run real Whisper and Qwen3-TTS inference")
    parser.addoption("--gpu-sample", action="append", default=[],
                     help="Validate a pre-extracted video excerpt (maximum 30 seconds); repeat for multiple files")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: real GPU integration test, executed in a clean subprocess")
    config.addinivalue_line("markers", "models: downloads and runs real local model weights")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "models" in item.keywords and not config.getoption("--run-models"):
            item.add_marker(pytest.mark.skip(reason="requires --run-models"))
        elif "gpu" in item.keywords and not config.getoption("--run-gpu"):
            item.add_marker(pytest.mark.skip(reason="requires --run-gpu"))
