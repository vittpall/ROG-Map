#!/usr/bin/env python3
"""Generate a minimal test scene for validating occlusion-boundary detection.

Geometry: one ground plane + one vertical wall. Nothing else.

WHY A GROUND PLANE IS INCLUDED
------------------------------
MARSIM only emits a point where a ray actually hits something, and ROG-Map only
marks free space along rays that produced a return. With a wall alone, the only
carved free space is the narrow cone pointing at the wall; everything else stays
UNKNOWN because it was never swept. Frontier would then trace the edge of that
cone -- an artefact of "no returns", not an occlusion -- and the test would prove
nothing.

The ground gives returns in every direction, carving a proper free disc around
the drone. The wall then cuts a clean, unambiguous shadow wedge out of that disc.

EXPECTED RESULT (this is the actual assertion of the test)
----------------------------------------------------------
  occlusion_frontier (red)  : hugs the wall's silhouette -- its two vertical
                              side edges and top edge -- and lines the shadow
                              wedge BEHIND the wall (y > wall_y).
  open_frontier     (grey)  : the sensing-horizon shell out at ~15 m, far from
                              any geometry.

Red appearing IN FRONT of the wall (y < wall_y), or no red at all along the wall
edges, means the detector is wrong.
"""

import argparse
import numpy as np


def make_grid(u_rng, v_rng, step):
    u = np.arange(u_rng[0], u_rng[1] + 1e-9, step)
    v = np.arange(v_rng[0], v_rng[1] + 1e-9, step)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return uu.ravel(), vv.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="wall_scene.pcd")
    # Ground: sampled a little finer than the 0.1 m map resolution so rays cannot
    # slip between points and punch spurious holes in the floor.
    ap.add_argument("--ground-half", type=float, default=18.0)
    ap.add_argument("--ground-step", type=float, default=0.07)
    # Wall: finite, so it has genuine left/right/top silhouette edges. An infinite
    # wall would only ever produce a single flat shadow with no interesting edges.
    ap.add_argument("--wall-y", type=float, default=5.0)
    ap.add_argument("--wall-x-half", type=float, default=3.0)
    ap.add_argument("--wall-z", type=float, default=3.0)
    ap.add_argument("--wall-step", type=float, default=0.04)
    ap.add_argument("--wall-thick", type=float, default=0.2)
    args = ap.parse_args()

    pts = []

    gx, gy = make_grid((-args.ground_half, args.ground_half),
                       (-args.ground_half, args.ground_half), args.ground_step)
    pts.append(np.stack([gx, gy, np.zeros_like(gx)], axis=1))

    # Wall given real thickness: a zero-thickness sheet can be pierced by rays
    # arriving at a grazing angle, which would leak light into the shadow.
    wx, wz = make_grid((-args.wall_x_half, args.wall_x_half),
                       (0.0, args.wall_z), args.wall_step)
    n_layers = max(1, int(round(args.wall_thick / args.wall_step)))
    for i in range(n_layers):
        y = args.wall_y + i * args.wall_step
        pts.append(np.stack([wx, np.full_like(wx, y), wz], axis=1))

    cloud = np.concatenate(pts, axis=0).astype(np.float32)

    hdr = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {len(cloud)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(cloud)}\nDATA binary\n"
    )
    with open(args.out, "wb") as f:
        f.write(hdr.encode())
        f.write(cloud.tobytes())

    print(f"wrote {args.out}: {len(cloud)} points")
    print(f"  ground {2*args.ground_half:.0f}x{2*args.ground_half:.0f} m at z=0")
    print(f"  wall   y={args.wall_y} m, x in [{-args.wall_x_half},{args.wall_x_half}], "
          f"z in [0,{args.wall_z}], thickness {args.wall_thick} m")
    print(f"  bounds min {cloud.min(0).round(2)}  max {cloud.max(0).round(2)}")


if __name__ == "__main__":
    main()
