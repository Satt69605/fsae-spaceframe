"""
run_optimise.py: combine tube changes and produce the stiffness/mass Pareto front.

    python run_optimise.py                      # dominance pass + 2 kg budget
    python run_optimise.py --budget 0           # dominance pass only (fast-ish)
    python run_optimise.py --budget 4000 --max-tubes 24 --steps 20

Writes results/optimise_steps.csv, results/pareto.csv, results/pareto.png.
Takes a couple of minutes as it's thousands of full FE solves.
"""
import argparse
import csv
import os
import time

from fesolver import sweep as sweep_mod
from fesolver.sweep import default_spec
from fesolver.optimise import (Evaluator, dominance_pass, greedy_upgrade,
                               pareto_front, spec_diff)


def write_rows(rows, path, fields):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_pareto(history, front, base, final, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter([r["mass_kg"] for r in history], [r["stiffness_Nm_per_deg"] for r in history],
               s=8, color="lightgrey", label=f"designs evaluated ({len(history)})")
    ax.plot([r["mass_kg"] for r in front], [r["stiffness_Nm_per_deg"] for r in front],
            "-o", color="tab:blue", ms=4, label=f"Pareto front ({len(front)})")
    ax.scatter([base["mass_kg"]], [base["stiffness_Nm_per_deg"]], s=90, color="black",
               zorder=5,
               label=f"baseline ({base['stiffness_Nm_per_deg']:.0f} Nm/deg, "
                     f"{base['mass_kg']:.2f} kg)")
    ax.scatter([final["mass_kg"]], [final["stiffness_Nm_per_deg"]], s=90, marker="*",
               color="tab:red", zorder=5,
               label=f"greedy result ({final['stiffness_Nm_per_deg']:.0f} Nm/deg, "
                     f"{final['mass_kg']:.2f} kg)")
    ax.set_xlabel("frame mass (kg)")
    ax.set_ylabel("torsional stiffness (Nm/deg)")
    ax.set_title("Stiffness vs mass: every point is a rules-legal frame")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  plot          {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=2000.0, help="mass budget in grams")
    ap.add_argument("--max-tubes", type=int, default=16, help="tubes sampled per group")
    ap.add_argument("--steps", type=int, default=20, help="max greedy steps")
    ap.add_argument("--min-wall", type=float, default=0.0,
                    help="manufacturability guard: ignore tubes thinner than this (mm)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    sweep_mod.MIN_WALL = args.min_wall
    t0 = time.time()
    ev = Evaluator()
    base = ev(default_spec(), "baseline")

    print("\n[dominance pass: stiffer, not heavier]")
    spec, after_free = dominance_pass(ev, max_tubes=args.max_tubes)
    print(f"  after pass    {after_free['stiffness_Nm_per_deg']:8.1f} Nm/deg  "
          f"{after_free['mass_kg']:6.2f} kg  "
          f"({after_free['stiffness_Nm_per_deg']/base['stiffness_Nm_per_deg']-1:+.1%} stiffness, "
          f"{after_free['mass_kg']-base['mass_kg']:+.2f} kg)")

    steps = []
    final = after_free
    if args.budget > 0:
        print(f"\n[greedy: spend {args.budget:.0f} g, best value per gram first]")
        spec, final, steps = greedy_upgrade(ev, spec=spec, budget_g=args.budget,
                                            max_tubes=args.max_tubes, step_limit=args.steps)

    front = pareto_front(ev.history)
    write_rows(steps, f"{args.outdir}/optimise_steps.csv",
               ["group", "od", "wall", "d_mass_g", "d_stiffness", "mass_kg",
                "stiffness_Nm_per_deg", "value_Nm_per_deg_per_g"])
    write_rows(front, f"{args.outdir}/pareto.csv",
               ["mass_kg", "stiffness_Nm_per_deg", "stiffness_per_kg",
                "torsion_buckling_util", "torsion_stress_util", "label"])
    plot_pareto(ev.history, front, base, final, f"{args.outdir}/pareto.png")

    print(f"\n[result]")
    print(f"  baseline      {base['stiffness_Nm_per_deg']:8.1f} Nm/deg  {base['mass_kg']:6.2f} kg  "
          f"{base['stiffness_per_kg']:5.1f} Nm/deg/kg")
    print(f"  final         {final['stiffness_Nm_per_deg']:8.1f} Nm/deg  {final['mass_kg']:6.2f} kg  "
          f"{final['stiffness_per_kg']:5.1f} Nm/deg/kg")
    print(f"  change        {final['stiffness_Nm_per_deg']/base['stiffness_Nm_per_deg']-1:+.1%} stiffness, "
          f"{final['mass_kg']-base['mass_kg']:+.2f} kg, "
          f"{final['stiffness_per_kg']/base['stiffness_per_kg']-1:+.1%} per kg")
    print(f"  worst buckling utilisation {final['torsion_buckling_util']:.3f}, "
          f"stress {final['torsion_stress_util']:.3f}")
    print("\n  tube changes from baseline:")
    for g, (old, new) in spec_diff(default_spec(), spec).items():
        print(f"    {g:16s} {old[0]:g}x{old[1]:g}  ->  {new[0]:g}x{new[1]:g}")
    print(f"\n  {len(ev.history)} designs evaluated in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
