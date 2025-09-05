import json
import os
import subprocess
import sys
from typing import Any, Dict

# Path to Python interpreter with Open3D installed. Set via environment variable.
# GEO3D_PY = os.environ.get("GEO3D_PY")
GEO3D_PY = r"D:\ProgramData\anaconda3\envs\geo3d\python.exe"

# Path to the worker script (same directory as this file by default)
O3D_WORKER = os.path.join(os.path.dirname(__file__), "o3d_worker.py")


def run_o3d_worker(*args: str) -> Dict[str, Any]:
    """Run the Open3D worker in the external environment.

    If the ``GEO3D_PY`` environment variable is set, that interpreter is used.
    Otherwise ``sys.executable`` is used as a fallback.
    Returns the parsed JSON output from the worker, or a dict with ``raw_stdout``
    if the output is not JSON.
    """

    interpreter = GEO3D_PY or sys.executable
    cmd = [interpreter, O3D_WORKER, *map(str, args)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError("o3d_worker failed:\n" + (res.stderr or res.stdout))
    try:
        return json.loads(res.stdout.strip())
    except Exception:
        return {"ok": True, "raw_stdout": res.stdout}
