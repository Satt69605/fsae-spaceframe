"""
run_sweep.py: varies one tube group at a time and records what it buys.

    python run_sweep.py                          # quick sweep, 12 tubes per group
    python run_sweep.py --max-tubes 0            # every legal tube (~1900 runs)
    python run_sweep.py --cases torsion,bending,bulkhead
    python run_sweep.py --groups MAIN_HOOP,HARNESS_BAR --top 20

Writes a CSV (default results/sweep.csv) with one row per candidate frame.
"""
import argparse
import time

from fesolver.optimise import FREE_MASS_G
from fesolver import sweep as sweep_mod
from fesolver.sweep import one_at_a_time, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sweep.csv")
    ap.add_argument("--max-tubes", type=int, default=12,
                    help="tubes sampled per group; 0 = every legal tube")
    ap.add_argument("--cases", default="torsion",
                    help="comma separated: torsion,bending,bulkhead")
    ap.add_argument("--groups", default="", help="comma separated; default all")
    ap.add_argument("--top", type=int, default=12, help="rows in the ranking printed")
    ap.add_argument("--min-wall", type=float, default=0.0,
                    help="manufacturability guard: ignore tubes thinner than this (mm)")
    args = ap.parse_args()

    sweep_mod.MIN_WALL = args.min_wall
    cases = tuple(c.strip() for c in args.cases.split(",") if c.strip())
    groups = [g.strip() for g in args.groups.split(",") if g.strip()] or None

    t0 = time.time()
    rows, path = write_csv(one_at_a_time(groups=groups, cases=cases,
                                         max_tubes=args.max_tubes or None), args.out)
    base = rows[0]
    print(f"\nbaseline      {base['stiffness_Nm_per_deg']:.1f} Nm/deg at "
          f"{base['mass_kg']:.2f} kg ({base['stiffness_per_kg']:.1f} Nm/deg/kg)")
    print(f"candidates    {len(rows)-1} in {time.time()-t0:.1f} s -> {path}")

    bad = [r for r in rows if r["checks_failed"]]
    print(f"rules         {len(bad)} candidate(s) failed a rules check")

    # Rank by value per gram only where the mass change is big enough for the ratio to mean anything.
    # A +8 Nm/deg change that costs 1 g gives a huge ratio that says more about the denominator than about the design.
    gain = [r for r in rows[1:]
            if r.get("d_mass_g", 0) > FREE_MASS_G and r.get("d_stiffness", 0) > 0]
    gain.sort(key=lambda r: -r["Nm_per_deg_per_g"])
    print(f"\nbest stiffness per gram added (top {args.top}):")
    print(f"  {'group':16s} {'tube':11s} {'+mass g':>8s} {'+Nm/deg':>9s} {'Nm/deg/g':>9s}")
    for r in gain[:args.top]:
        tube = f"{r['od']:g}x{r['wall']:g}"
        print(f"  {r['group']:16s} {tube:11s} {r['d_mass_g']:8.0f} "
              f"{r['d_stiffness']:9.1f} {r['Nm_per_deg_per_g']:9.4f}")

    free = [r for r in rows[1:]
            if r.get("d_mass_g", 0) <= FREE_MASS_G and r.get("d_stiffness", 0) > 0]
    free.sort(key=lambda r: -r["d_stiffness"])
    if free:
        print(f"\nstiffness for no meaningful extra mass (<= {FREE_MASS_G:.0f} g, stiffer):")
        for r in free[:5]:
            tube = f"{r['od']:g}x{r['wall']:g}"
            print(f"  {r['group']:16s} {tube:11s} {r['d_mass_g']:8.0f} g "
                  f"{r['d_stiffness']:+9.1f} Nm/deg")


if __name__ == "__main__":
    main()
