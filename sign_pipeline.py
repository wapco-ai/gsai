"""Simple pipeline for detecting urban signs and mapping results to point clouds.

This module provides a lightweight example implementation that scans a directory of
input images, marks each as containing an urban sign, and stores the results in a
JSON file. If a point cloud (PLY) exists in the output directory the module also
adds ``class`` and ``label`` fields using :func:`metashape_script.add_class_field_to_ply`.

Each execution should use a dedicated ``output_dir``.  When multiple runs target
the same directory a file lock is used to coordinate access, but sharing
directories between unrelated runs is unsupported.

The intent is to demonstrate how a dedicated sign processing stage can be
integrated into the existing project without requiring heavy dependencies.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Optional

from filelock import FileLock

from metashape_script import add_class_field_to_ply
from settings import ENABLE_PLY_VALIDATION
import analysis

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def detect_signs(image_dir: str) -> List[Dict[str, str]]:
    """Detect urban signs within *image_dir*.

    The current implementation simply treats every image as containing an
    ``urban_sign`` label. In a real-world scenario this function could load a
    YOLO or SegFormer model to perform actual detection.
    """
    detections: List[Dict[str, str]] = []
    for fname in os.listdir(image_dir):
        if os.path.splitext(fname)[1].lower() in ALLOWED_EXTENSIONS:
            detections.append({"image": fname, "label": "urban_sign"})
    return detections


def _summarise(detections: List[Dict[str, str]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for det in detections:
        label = det.get("label", "unknown")
        summary[label] = summary.get(label, 0) + 1
    return summary


def process_sign_pipeline(
    image_dir: str,
    output_dir: str,
    analyses: Optional[List[str]] = None,
    add_class_field: bool = True,
    validate: bool = ENABLE_PLY_VALIDATION,
) -> Dict[str, int]:
    """Run the sign detection pipeline.

    Args:
        image_dir: Directory containing the images that were processed by
            Metashape.
        output_dir: Directory containing the point cloud outputs. A JSON file
            named ``signs.json`` will be written here describing the detected
            signs. If *add_class_field* is True and a PLY file is present, a new
            file with added ``class`` and ``label`` fields will be generated
            alongside it.
        validate: When ``True`` run extra integrity checks on PLY files. Disable
            for faster processing once files are trusted.

    Returns:
        A dictionary summarising the counts of detected sign classes.
    """
    detections = detect_signs(image_dir)
    summary = _summarise(detections)

    if summary:
        json_path = os.path.join(output_dir, "signs.json")
        lock_path = json_path + ".lock"
        ply_path: Optional[str] = None
        with FileLock(lock_path):
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"detections": detections, "summary": summary},
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )

            # Locate a PLY point cloud and optionally add classification fields
            for root, _, files in os.walk(output_dir):
                # Prefer an already classified file if present
                for file in files:
                    if file.lower().endswith("_with_class.ply"):
                        ply_path = os.path.join(root, file)
                        break
                if ply_path:
                    break
                for file in files:
                    if file.lower().endswith(".ply"):
                        ply_path = os.path.join(root, file)
                        if add_class_field:
                            try:
                                add_class_field_to_ply(
                                    ply_path, validate=validate
                                )
                            except Exception:
                                pass
                        break
                if ply_path:
                    break

        if analyses and ply_path:
            for name in analyses:
                try:
                    analysis.run_analysis(name, ply_path, output_dir)
                except Exception:
                    pass

    return summary


def available_analyses() -> List[Dict[str, str]]:
    """Expose registered analysis options for user selection."""
    return analysis.list_available()
