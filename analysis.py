from __future__ import annotations

"""Registry and utility functions for optional point cloud analyses.

This module allows registering analysis functions and exposing them to the
application so that users can select which analyses to run. Each analysis
is a callable that accepts a point cloud path and optional keyword
arguments. Results are typically written to JSON files in the same
output directory.
"""

from typing import Callable, Dict, List, Any
import os
import json

_ANALYSES: Dict[str, Dict[str, Any]] = {}


def register_analysis(name: str, description: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a new analysis.

    Parameters
    ----------
    name:
        Unique identifier for the analysis.
    description:
        Human readable description presented to the user.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _ANALYSES[name] = {"func": func, "description": description}
        return func

    return decorator


def list_available() -> List[Dict[str, str]]:
    """Return metadata for all registered analyses.

    Returns
    -------
    list of dict
        Each dictionary contains ``name`` and ``description`` keys.
    """

    return [
        {"name": key, "description": value["description"]}
        for key, value in _ANALYSES.items()
    ]


def run_analysis(name: str, *args: Any, **kwargs: Any) -> Any:
    """Execute a registered analysis by *name*.

    Raises
    ------
    ValueError
        If *name* does not correspond to a registered analysis.
    """

    entry = _ANALYSES.get(name)
    if not entry:
        raise ValueError(f"Unknown analysis: {name}")
    return entry["func"](*args, **kwargs)


@register_analysis("sign_bbox", "پردازش جعبه‌های محدودکننده تابلوها")
def compute_sign_bounding_box(ply_path: str, output_dir: str, label_value: int = 43) -> List[Dict[str, float]]:
    """Generate a bounding box around points with the given *label_value*.

    The point cloud must contain ``label`` field. Results are written to a
    ``sign_bboxes.json`` file in ``output_dir`` and also returned.
    """
    try:
        from plyfile import PlyData
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependencies may be missing
        raise RuntimeError(f"Required dependency missing for analysis: {exc}") from exc

    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]
    if "label" not in vertex.data.dtype.names:
        raise RuntimeError("Point cloud lacks 'label' field needed for analysis")

    mask = vertex["label"] == label_value
    if mask.sum() == 0:
        results: List[Dict[str, float]] = []
    else:
        pts = np.vstack((vertex["x"], vertex["y"], vertex["z"])).T[mask]
        min_xyz = pts.min(axis=0)
        max_xyz = pts.max(axis=0)
        size = max_xyz - min_xyz
        results = [
            {
                "label": int(label_value),
                "min": min_xyz.tolist(),
                "max": max_xyz.tolist(),
                "size": size.tolist(),
            }
        ]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "sign_bboxes.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    return results
