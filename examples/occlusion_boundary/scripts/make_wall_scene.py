#!/usr/bin/env python3
"""Generate a test scene for occlusion-boundary detection: one vertical wall.

Expected result: occlusion_frontier hugs the wall's silhouette (its two vertical
side edges and top edge) and lines the shadow wedge behind it (y > wall_y).

--ground adds a floor. Off by default, because a bare wall is the cleanest
occlusion test: every detected boundary has one obvious cause.

Turn it ON for FAST-LIO. A lidar-inertial estimator has to register successive
scans against geometry, and a single wall in a 36 m world means any view not
pointing at it returns ZERO points -- FAST-LIO then logs "No point, skip this
scan", dead-reckons on IMU alone and diverges within seconds. The floor is what
guarantees returns in every direction.

--ground also plants a ring of peripheral pillars, because floor+wall alone is
DEGENERATE for a point-to-plane estimator: the ground normal (z) constrains
z/roll/pitch and the wall normal (y) constrains y, but NOTHING constrains x or
yaw -- a scan slides freely along both planes with zero residual change, so x
and yaw drift while flying and every scan stamps the wall into the map at a
shifted pose ("random walls"). The pillars add normals in x and break the yaw
symmetry, sitting at the map edge so the wall-test corridor stays clean.
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
    ap.add_argument("--wall-y", type=float, default=10.0)
    ap.add_argument("--wall-x-half", type=float, default=7.0)
    ap.add_argument("--wall-z", type=float, default=5.0)
    ap.add_argument("--wall-step", type=float, default=0.04)
    ap.add_argument("--wall-thick", type=float, default=0.2)
    # Sampled finer than the 0.1 m map resolution so rays cannot slip between
    # points and punch spurious holes in the floor.
    ap.add_argument("--ground", action="store_true",
                    help="add a ground plane (required for FAST-LIO)")
    ap.add_argument("--ground-half", type=float, default=18.0)
    ap.add_argument("--ground-step", type=float, default=0.07)
    args = ap.parse_args()

    # Real thickness: a zero-thickness sheet can be pierced by rays arriving at a
    # grazing angle, which leaks light into the shadow.
    wx, wz = make_grid((-args.wall_x_half, args.wall_x_half),
                       (0.0, args.wall_z), args.wall_step)
    n_layers = max(1, int(round(args.wall_thick / args.wall_step)))
    pts = [np.stack([wx, np.full_like(wx, args.wall_y + i * args.wall_step), wz],
                    axis=1)
           for i in range(n_layers)]

    if args.ground:
        gx, gy = make_grid((-args.ground_half, args.ground_half),
                           (-args.ground_half, args.ground_half), args.ground_step)
        pts.append(np.stack([gx, gy, np.zeros_like(gx)], axis=1))

        # Peripheral pillars (see docstring): 0.6 m square columns, clear of the
        # drone start (0,3), the wall span, and the shadow region behind it.
        for cx, cy in ((-13.0, -8.0), (13.0, -8.0), (-13.0, 6.0),
                       (13.0, 6.0), (0.0, -13.0)):
            for face in range(4):
                u, w = make_grid((-0.3, 0.3), (0.0, 2.5), 0.05)
                if face % 2 == 0:      # faces with normal +/-x
                    x = np.full_like(u, cx + (0.3 if face else -0.3))
                    pts.append(np.stack([x, cy + u, w], axis=1))
                else:                  # faces with normal +/-y
                    y = np.full_like(u, cy + (0.3 if face == 1 else -0.3))
                    pts.append(np.stack([cx + u, y, w], axis=1))

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
    print(f"  wall y={args.wall_y} m, x in [{-args.wall_x_half},{args.wall_x_half}], "
          f"z in [0,{args.wall_z}], thickness {args.wall_thick} m")
    if args.ground:
        print(f"  ground {2*args.ground_half:.0f}x{2*args.ground_half:.0f} m at z=0, "
              "5 pillars at the periphery (x/yaw observability for LIO)")
    else:
        print("  no ground: FAST-LIO will starve on views away from the wall")
    print(f"  bounds min {cloud.min(0).round(2)}  max {cloud.max(0).round(2)}")


if __name__ == "__main__":
    main()
