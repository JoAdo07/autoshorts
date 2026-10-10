"""Deliberately orphan a child to exercise test-harness containment."""
from pathlib import Path
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
(Path(sys.argv[2]) / "child.pid").write_text(str(child.pid))
if sys.argv[1] == "timeout":
    time.sleep(60)
sys.exit(2)
