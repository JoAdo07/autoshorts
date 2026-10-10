import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group containment")
@pytest.mark.parametrize("mode", ["timeout", "failure"])
def test_gpu_check_reaps_descendants(mode, tmp_path):
    path = Path(__file__).parent / "gpu" / "test_pipeline.py"
    spec = importlib.util.spec_from_file_location("gpu_harness", path)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    harness.RUNNER = path.with_name("process_tree.py")
    try:
        with pytest.raises((subprocess.TimeoutExpired, AssertionError)):
            harness.run_check(mode, tmp_path, timeout=1)
        pid = int((tmp_path / "child.pid").read_text())
        status = Path(f"/proc/{pid}/stat")
        def active():
            try:
                return status.read_text().split()[2] != "Z"
            except (FileNotFoundError, ProcessLookupError):
                return False

        deadline = time.monotonic() + 1
        while active() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not active()
    finally:
        pid_path = tmp_path / "child.pid"
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
