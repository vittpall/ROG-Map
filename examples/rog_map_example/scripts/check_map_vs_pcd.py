#!/usr/bin/env python3
"""Check ROG-Map's occupancy against the ground-truth PCD the simulator renders.

Every occupied voxel should coincide with a real point in the scene. A low match
rate means the pose feeding the map is wrong (drift, frame mismatch); missing
structure usually means raycasting/ray_range is not reaching it.

  rosrun rog_map_example check_map_vs_pcd.py [scene.pcd]
"""

import sys
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

DEFAULT_PCD = "/home/vittorio/rog_ws/src/ROG-Map/examples/occlusion_boundary/pcd/wall_scene.pcd"
TOPIC = "/rm_node/rog_map/occ"


def load_pcd(path):
    """Read xyz from an ascii or binary PCD. MARSIM ships both."""
    with open(path, "rb") as f:
        fields, count = [], 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError("no DATA section in %s" % path)
            tok = line.decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "FIELDS":
                fields = tok[1:]
            elif tok[0] == "POINTS":
                count = int(tok[1])
            elif tok[0] == "DATA":
                fmt = tok[1]
                break
        n = len(fields)
        if fmt == "binary":
            a = np.frombuffer(f.read(count * n * 4), dtype=np.float32).reshape(-1, n)
        elif fmt == "ascii":
            a = np.loadtxt(f, dtype=np.float32).reshape(-1, n)
        else:
            raise ValueError("unsupported PCD format: %s" % fmt)
        idx = [fields.index(c) for c in ("x", "y", "z")]
        a = a[:, idx]
        return a[np.isfinite(a).all(axis=1)]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PCD
    rospy.init_node("check_map_vs_pcd", anonymous=True)
    res = rospy.get_param("/rm_node/rog_map/resolution", 0.1)

    truth = load_pcd(path)
    print("truth  : %d pts  x %.1f..%.1f  y %.1f..%.1f  z %.1f..%.1f"
          % (len(truth), truth[:, 0].min(), truth[:, 0].max(), truth[:, 1].min(),
             truth[:, 1].max(), truth[:, 2].min(), truth[:, 2].max()))

    msg = rospy.wait_for_message(TOPIC, PointCloud2, timeout=20.0)
    occ = np.array([p[:3] for p in pc2.read_points(msg, skip_nans=True)])
    if not len(occ):
        print("map    : EMPTY — check ray_range and that /cloud_registered is publishing")
        return
    print("map    : %d voxels  frame=%s  x %.1f..%.1f  y %.1f..%.1f  z %.1f..%.1f"
          % (len(occ), msg.header.frame_id, occ[:, 0].min(), occ[:, 0].max(),
             occ[:, 1].min(), occ[:, 1].max(), occ[:, 2].min(), occ[:, 2].max()))

    # An occupied voxel is corroborated if any truth point lands in it or a neighbour.
    keys = set(map(tuple, np.floor(truth / res).astype(np.int32)))
    v = np.floor(occ / res).astype(np.int32)
    hit = sum(any((x + i, y + j, z + k) in keys
                  for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1))
              for x, y, z in v)
    print("match  : %d/%d occupied voxels backed by ground truth (%.1f%%)"
          % (hit, len(v), 100.0 * hit / len(v)))
    if hit / len(v) < 0.9:
        print("         LOW — pose feeding the map is likely drifting or in the wrong frame")


if __name__ == "__main__":
    main()
