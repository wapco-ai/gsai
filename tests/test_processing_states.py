import os
import sys
import types
from threading import Thread

# Ensure repository root is on path for module imports
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Ensure required environment variable is set before importing app
os.environ.setdefault("METASHAPE_EXECUTABLE", "dummy_exec")

# Stub heavy dependencies to keep tests light
sys.modules.setdefault("image_classifier", types.ModuleType("image_classifier"))

from app import app, update_process_state, PROCESS_STATE_LOCK


def test_concurrent_process_state_updates():
    """Concurrent updates to PROCESSING_STATES should be thread-safe."""
    with app.app_context():
        process_id = "test"

        # Reset any previous state
        with PROCESS_STATE_LOCK:
            app.config["PROCESSING_STATES"].clear()

        def worker(idx: int) -> None:
            with app.app_context():
                update_process_state(process_id, {f"key{idx}": idx})

        threads = [Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all updates are present and consistent
        with PROCESS_STATE_LOCK:
            state = app.config["PROCESSING_STATES"][process_id]
            assert len(state) == 50
            for i in range(50):
                assert state[f"key{i}"] == i
