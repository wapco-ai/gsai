"""Simple pipeline for detecting urban signs and mapping results to point clouds.

This module provides a lightweight example implementation that scans a directory of
input images, marks each as containing an urban sign, and stores the results in a
JSON file. If a point cloud (PLY) exists in the output directory the module also
adds ``class`` and ``label`` fields using :func:`metashape_script.add_class_field_to_ply`.

The intent is to demonstrate how a dedicated sign processing stage can be
integrated into the existing project without requiring heavy dependencies.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Optional

from metashape_script import add_class_field_to_ply
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

    Returns:
        A dictionary summarising the counts of detected sign classes.
    """
    detections = detect_signs(image_dir)
    summary = _summarise(detections)

    if summary:
        json_path = os.path.join(output_dir, "signs.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"detections": detections, "summary": summary},
                fh,
                ensure_ascii=False,
                indent=2,
            )

        # Locate first PLY point cloud and optionally add classification fields
        ply_path = None
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith(".ply") and not file.lower().endswith("_with_class.ply"):
                    ply_path = os.path.join(root, file)
                    if add_class_field:
                        try:
                            add_class_field_to_ply(ply_path)
                            with_class = os.path.splitext(ply_path)[0] + "_with_class.ply"
                            if os.path.exists(with_class):
                                ply_path = with_class
                        except Exception:
                            pass
                    break
            else:
                continue
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
