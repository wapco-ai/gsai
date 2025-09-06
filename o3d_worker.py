import json
import sys
import open3d as o3d
import numpy as np
from settings import PLY_ROUND_COORDS, PLY_ROUND_DECIMALS


def downsample(in_path: str, out_path: str, ratio: float):
    pcd = o3d.io.read_point_cloud(in_path)
    sampled = pcd.random_down_sample(ratio)
    points = np.asarray(sampled.points)
    if PLY_ROUND_COORDS:
        points = np.round(points, PLY_ROUND_DECIMALS)
    sampled.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(out_path, sampled, write_ascii=True)
    return {
        "ok": True,
        "points_before": len(pcd.points),
        "points_after": len(sampled.points),
    }


def count_points(ply_path: str):
    pcd = o3d.io.read_point_cloud(ply_path)
    return {"ok": True, "points": len(pcd.points)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "usage"}))
        sys.exit(1)

    cmd = sys.argv[1]
    try:
        if cmd == "downsample" and len(sys.argv) >= 5:
            in_ply = sys.argv[2]
            out_ply = sys.argv[3]
            ratio = float(sys.argv[4])
            result = downsample(in_ply, out_ply, ratio)
        elif cmd == "count" and len(sys.argv) >= 3:
            result = count_points(sys.argv[2])
        else:
            result = {"ok": False, "error": "invalid args"}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result))
    sys.exit(0 if result.get("ok") else 1)
