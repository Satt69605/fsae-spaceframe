"""
run_analysis.py: run any or all load cases on a spaceframe JSON.

    python run_analysis.py frames/frame_at_node.json
    python run_analysis.py frames/frame_at_node.json --case torsion --plot twist.png
    python run_analysis.py frames/frame_at_node.json --case bending --driver 75 --engine 65 --g 3
"""
import argparse

import numpy as np

from fesolver.chassis import load_frame, torsion_test
from fesolver.loadcases import bending_test, bulkhead_test
from fesolver.postprocess import member_forces, print_top_compression, buckling_check, print_buckling, print_energy


def plot_deformed(data, r, path, scale, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx, xyz, u = r["index"], r["coords"], r["u"]
    moved = xyz + scale * u.reshape(-1, 6)[:, :3]

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(projection="3d")
    for m in data["members"]:
        a, b = idx[m["n1"]], idx[m["n2"]]
        ax.plot(*xyz[[a, b]].T, color="lightgrey", lw=1)
        ax.plot(*moved[[a, b]].T, color="tab:red", lw=1.5)
    for group, colour in (("front_pickups", "gold"), ("rear_pickups", "black")):
        p = xyz[[idx[n] for n in data[group]]]
        ax.scatter(*p.T, color=colour, s=40, depthshade=False)
    ax.set_box_aspect(np.ptp(xyz, axis=0))
    ax.set_xlabel("x rearward (mm)"); ax.set_ylabel("y left (mm)"); ax.set_zlabel("z up (mm)")
    ax.set_title(f"{title}  |  grey = original, red = deformed x{scale:g}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  plot          {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame", help="JSON written by spaceframe.py")
    ap.add_argument("--case", default="all", choices=["all", "torsion", "bending", "bulkhead"])
    ap.add_argument("--force", type=float, default=500.0, help="torsion: N per front pickup")
    ap.add_argument("--driver", type=float, default=80.0, help="bending: driver mass, kg")
    ap.add_argument("--engine", type=float, default=70.0, help="bending: powertrain mass, kg")
    ap.add_argument("--g", type=float, default=1.0, help="bending: g factor (3 = 3 g bump)")
    ap.add_argument("--bulkhead-load", type=float, default=10000.0, help="bulkhead: total N")
    ap.add_argument("--forces", action="store_true", help="list the most compressed members")
    ap.add_argument("--buckling", action="store_true", help="Euler/Johnson buckling check per member")
    ap.add_argument("--energy", action="store_true", help="strain energy share per tube group")
    ap.add_argument("--k-factor", type=float, default=1.0,
                    help="effective length factor: 1.0 pinned (conservative), ~0.7 welded")
    ap.add_argument("--plot", help="save a deformed-shape PNG (one case at a time)")
    ap.add_argument("--scale", type=float, default=200.0)
    args = ap.parse_args()

    data = load_frame(args.frame)
    print(args.frame)
    cases = ["torsion", "bending", "bulkhead"] if args.case == "all" else [args.case]

    for case in cases:
        print(f"\n[{case}]")
        if case == "torsion":
            r = torsion_test(data, force=args.force)
            title = f"torsion {r['stiffness_Nm_per_deg']:.0f} Nm/deg"
        elif case == "bending":
            r = bending_test(data, driver_kg=args.driver, engine_kg=args.engine, g_factor=args.g)
            title = f"bending, max sag {r['max_sag_mm']:.3f} mm"
        else:
            r = bulkhead_test(data, load_N=args.bulkhead_load)
            title = f"bulkhead, face {r['face_deflection_mm']:.3f} mm"

        if args.forces:
            print_top_compression(member_forces(data, r))
        if args.buckling:
            print_buckling(buckling_check(data, r, k=args.k_factor))
        if args.energy:
            print_energy(data, r)
        if args.plot and len(cases) == 1:
            plot_deformed(data, r, args.plot, args.scale, title)


if __name__ == "__main__":
    main()
