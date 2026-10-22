"""
run_case_studies.py: conducts four short studies that test what the project claims.

    python case_studies/run_case_studies.py

Prints every number quoted in case_studies/CASE_STUDIES.md, so the document
can be regenerated and checked rather than trusted blindly.

  1. Load path: what triangulating the front axle station is worth
  2. Geometry: the main hoop diagonal the tube-size optimiser cannot find
  3. Material: 4130 steel vs 6061-T6 aluminium in the same geometry
  4. Real data: where these numbers sit against published FS chassis tests
"""
import copy
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fesolver.chassis import load_frame, torsion_test          # noqa: E402
from fesolver.loadcases import bending_test, bulkhead_test     # noqa: E402
from fesolver.postprocess import (buckling_check, energy_by_group,  # noqa: E402
                                  strain_energy)

FRAME = os.path.join(ROOT, "frames", "frame_at_node.json")


def summary(data):
    """Stiffness, mass, and worst utilisations over all three load cases."""
    t = torsion_test(data, verbose=False)
    b = bending_test(data, verbose=False)
    h = bulkhead_test(data, verbose=False)
    worst = {}
    for name, r in (("torsion", t), ("bending", b), ("bulkhead", h)):
        rows = buckling_check(data, r)
        worst[name] = (rows[0]["utilisation"], max(x["yield_utilisation"] for x in rows))
    mass = sum(m["mass"] for m in data["members"])
    return {"k": t["stiffness_Nm_per_deg"], "mass": mass, "k_per_kg": t["stiffness_Nm_per_deg"] / mass,
            "sag": b["max_sag_mm"], "bulkhead": h["face_deflection_mm"], "worst": worst, "torsion": t}


def add_member(data, n1, n2, like_group):
    """Add a tube between two existing nodes, sized like an existing group."""
    ref = next(m for m in data["members"] if m["group"] == like_group)
    a, b = data["nodes"][n1], data["nodes"][n2]
    L = math.dist(a, b)
    new = dict(ref)
    new.update({"n1": n1, "n2": n2, "group": "MH_DIAGONAL", "length": L,
                "mass": data["material"]["rho"] * ref["area"] * L})
    data["members"].append(new)
    return new


def with_material(data, E, nu, rho, Sy, Sy_welded):
    """Same geometry, different material; mass rescales with density."""
    d = copy.deepcopy(data)
    ratio = rho / d["material"]["rho"]
    d["material"] = {"E": E, "G": E / (2 * (1 + nu)), "nu": nu, "rho": rho,
                     "Sy": Sy, "Sy_welded": Sy_welded}
    for m in d["members"]:
        m["mass"] *= ratio
    return d


def line(label, s, base=None):
    extra = ""
    if base:
        extra = (f"  ({s['k']/base['k']-1:+.1%} stiffness, "
                 f"{(s['mass']-base['mass'])*1000:+.0f} g)")
    print(f"  {label:34s} {s['k']:7.1f} Nm/deg  {s['mass']:6.2f} kg  "
          f"{s['k_per_kg']:5.1f} Nm/deg/kg{extra}")


def without_group(data, group):
    """same frame with one tube group deleted."""
    d = copy.deepcopy(data)
    d["members"] = [m for m in d["members"] if m["group"] != group]
    return d


def study_1():
    print("\n=== 1. Load path: triangulating the front axle station ===")
    untri = without_group(load_frame(FRAME), "PICKUP_SUPPORT")   # delete the 5 bracing tubes
    before = summary(untri)
    after = summary(load_frame(FRAME))
    line("untriangulated axle station", before)
    line("triangulated (PICKUP_SUPPORT)", after, before)
    for name, data in (("untriangulated", untri), ("triangulated", load_frame(FRAME))):
        rows, _ = strain_energy(data, torsion_test(data, verbose=False))
        g = {x["group"]: x["fraction"] for x in energy_by_group(rows)}
        fbs = g.get("FBS_UPPER", 0) + g.get("FBS_LOWER", 0)
        print(f"  {name:34s} bulkhead-support energy share {fbs:6.1%}")
    added = after["mass"] - before["mass"]
    gain = after["k"] - before["k"]
    print(f"  value of the change                {gain/ (added*1000):.3f} Nm/deg per gram")


def study_2():
    print("\n=== 2. Geometry: a diagonal across the main hoop ===")
    base = load_frame(FRAME)
    s0 = summary(base)
    line("baseline", s0)

    u = s0["torsion"]["u"].reshape(-1, 6)
    i = s0["torsion"]["index"]["MH_TOP_R"]
    print(f"  baseline sway at MH_TOP_R          {u[i,1]:+.2f} mm lateral")

    for label, n1, n2, note in (
            ("seat-back diagonal MH_BOT_L-MH_HARN_R", "MH_BOT_L", "MH_HARN_R",
             "below the harness bar, behind the seat back"),
            ("head-zone diagonal MH_HARN_L-MH_TOP_R", "MH_HARN_L", "MH_TOP_R",
             "above the harness bar, behind the driver's head")):
        d = copy.deepcopy(base)
        m = add_member(d, n1, n2, "SIS_UPPER")                 # a 25.4x1.6 class-B tube
        s = summary(d)
        uu = s["torsion"]["u"].reshape(-1, 6)
        line(label, s, s0)
        print(f"    {note}; tube {m['od']}x{m['wall']}, {m['length']:.0f} mm, "
              f"{m['mass']*1000:.0f} g; sway now {uu[s['torsion']['index']['MH_TOP_R'],1]:+.2f} mm; "
              f"{(s['k']-s0['k'])/(m['mass']*1000):.2f} Nm/deg per gram")
    print("  compare: the whole tube-size optimisation, 2 kg budget, "
          "+786.6 Nm/deg for 1790 g = 0.44 Nm/deg per gram")
    print("  NOTE: the added member is outside the generator, so the T3 checks are not re-run")


def study_3():
    print("\n=== 3. Material: 4130 steel vs 6061-T6 aluminium, same geometry ===")
    steel = load_frame(FRAME)
    alu = with_material(steel, E=68900.0, nu=0.33, rho=2.70e-6, Sy=241.0, Sy_welded=103.0)
    ss, sa = summary(steel), summary(alu)
    line("steel (E 200 GPa, welded Sy 180)", ss)
    line("6061-T6 (E 68.9 GPa, welded Sy 103)", sa, ss)
    print(f"  specific stiffness E/rho           steel {200000/7.85e-6/1e9:.1f}  "
          f"aluminium {68900/2.70e-6/1e9:.1f}  (x1e9, N.mm/kg)")
    for case in ("torsion", "bending", "bulkhead"):
        (bs, ys), (ba, ya) = ss["worst"][case], sa["worst"][case]
        print(f"  {case:9s} worst buckling {bs:.3f} -> {ba:.3f}   "
              f"worst stress {ys:.3f} -> {ya:.3f}")
    print(f"  bulkhead load to first yield       steel {10/ss['worst']['bulkhead'][1]:.1f} kN  "
          f"aluminium {10/sa['worst']['bulkhead'][1]:.1f} kN  (linear scaling of the 10 kN case)")


def study_4():
    print("\n=== 4. Against published Formula Student chassis data ===")
    s = summary(load_frame(FRAME))
    lo, hi = s["k"] * 0.8, s["k"] * 0.9
    print(f"  this frame, predicted              {s['k']:.0f} Nm/deg")
    print(f"  as-built, if FE over-predicts 10-20%: {lo:.0f} to {hi:.0f} Nm/deg")
    for label, k in (("optimised, 2 kg budget", 2498.8), ("optimised, 4 kg budget", 3141.0)):
        print(f"  {label:34s} {k:.0f} Nm/deg -> as-built {k*0.8:.0f} to {k*0.9:.0f}")


if __name__ == "__main__":
    study_1()
    study_2()
    study_3()
    study_4()
