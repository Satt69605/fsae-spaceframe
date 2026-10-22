"""
postprocess.py: what each tube is actually carrying once u is known.

The solve gives displacements. Member forces come from feeding those displacements back into each element:

    f_local = k_local @ T @ u_element

f_local[0] is the axial force at end 1 in local x. By sign convention here,
+ve axial = tension, -ve = compression. Compression is the one that
matters later, because thin tubes buckle long before the steel yields.
"""

import numpy as np

from .elements import frame3d_k, frame3d_k_local, frame3d_T
from .assemble import node_dofs
from .chassis import DOF


def member_forces(data, result):
    """Returns a list of dicts, one per member, with its axial force in N."""
    index, coords, u = result["index"], result["coords"], result["u"]
    E, G = data["material"]["E"], data["material"]["G"]

    out = []
    for m in data["members"]:
        a, b = index[m["n1"]], index[m["n2"]]
        T, L = frame3d_T(coords[a], coords[b])
        k_local = frame3d_k_local(E, G, m["area"], m["I"], m["I"], m["J"], L)
        ue = u[node_dofs(a, DOF) + node_dofs(b, DOF)]
        f_local = k_local @ (T @ ue)

        axial = -f_local[0]          # + tension, - compression

        c = m["od"] / 2.0
        M_end1 = np.hypot(f_local[4], f_local[5])
        M_end2 = np.hypot(f_local[10], f_local[11])
        M_max = max(M_end1, M_end2)

        out.append({
            "n1": m["n1"], "n2": m["n2"], "group": m["group"],
            "length": L, "area": m["area"], "I": m["I"], "od": m["od"],
            "axial_N": axial,
            "axial_stress_MPa": axial / m["area"],
            "moment_Nmm": M_max,
            "bending_stress_MPa": M_max * c / m["I"],
            "combined_stress_MPa": abs(axial / m["area"]) + M_max * c / m["I"],
        })
    return out


def top_compression(forces, n=5):
    return sorted(forces, key=lambda f: f["axial_N"])[:n]


def print_top_compression(forces, n=5):
    print(f"  most compressed members (top {n}):")
    for f in top_compression(forces, n):
        print(f"    {f['group']:16s} {f['n1']:>11s}-{f['n2']:<11s} "
              f"{f['axial_N']:9.1f} N  ({f['axial_stress_MPa']:6.1f} MPa)")


# ----------------------------------------------------------------------------
# Buckling
# ----------------------------------------------------------------------------
# Euler's formula gives the load at which that happens for a slender column. For short, 
# stocky members Euler is optimistic and nonsense in the limit, so the standard fix is the 
# Johnson parabola, which blends smoothly into yielding. Both are used here, chosen by slenderness:
#
#   slenderness  lam = k*L / r,  r = sqrt(I/A)   (r = radius of gyration)
#   transition   lam_c = sqrt(2 pi^2 E / Sy)
#   Euler   (lam >= lam_c):  sigma_cr = pi^2 E / lam^2
#   Johnson (lam <  lam_c):  sigma_cr = Sy - (Sy^2 / (4 pi^2 E)) * lam^2
#
# k is the effective length factor. k = 1.0 (pinned ends) is CONSERVATIVE for a
# welded spaceframe, where joints give partial rotational restraint (k ~ 0.7).

import math


def critical_load(E, Sy, area, I, L, k=1.0):
    """Returns (P_cr in N, which formula was used, slenderness)."""
    r = math.sqrt(I / area)
    lam = k * L / r
    lam_c = math.sqrt(2 * math.pi**2 * E / Sy)
    if lam >= lam_c:
        sigma_cr = math.pi**2 * E / lam**2
        mode = "euler"
    else:
        sigma_cr = Sy - (Sy**2 / (4 * math.pi**2 * E)) * lam**2
        mode = "johnson"
    return sigma_cr * area, mode, lam


def buckling_check(data, result, k=1.0, use_welded_strength=True):
    """
    Per-member buckling and yield utilisation for the load case in `result`.

    utilisation = |compressive force| / P_cr    -> 1.0 means on the limit
    margin      = P_cr / |compressive force|    -> how many times the load it takes

    Yield strength defaults to the welded value, because a spaceframe member is
    weakest in its heat-affected zone next to the joint.
    """
    E = data["material"]["E"]
    Sy = data["material"]["Sy_welded"] if use_welded_strength else data["material"]["Sy"]

    rows = []
    for f in member_forces(data, result):
        P_cr, mode, lam = critical_load(E, Sy, f["area"], f["I"], f["length"], k)
        comp = max(-f["axial_N"], 0.0)                     # compression only
        rows.append({**f,
                     "P_cr_N": P_cr,
                     "mode": mode,
                     "slenderness": lam,
                     "compression_N": comp,
                     "utilisation": comp / P_cr,
                     "margin": (P_cr / comp) if comp > 0 else float("inf"),
                     "yield_utilisation": f["combined_stress_MPa"] / Sy,
                     "interaction": comp / P_cr + f["bending_stress_MPa"] / Sy})
    return sorted(rows, key=lambda r: -r["utilisation"])


def print_buckling(rows, n=5):
    worst = rows[0]
    worst_stress = max(rows, key=lambda r: r["yield_utilisation"])
    worst_int = max(rows, key=lambda r: r["interaction"])
    print(f"  buckling      worst utilisation {worst['utilisation']:.3f} "
          f"(margin {worst['margin']:.1f}x) on {worst['group']} "
          f"{worst['n1']}-{worst['n2']}")
    print(f"  stress        worst {worst_stress['yield_utilisation']:.3f} of welded yield "
          f"(axial+bending) on {worst_stress['group']} "
          f"{worst_stress['n1']}-{worst_stress['n2']}")
    print(f"  interaction   worst {worst_int['interaction']:.3f} "
          f"(buckling + bending, 1.0 = limit)")
    print(f"  top {n} compressed members:")
    print(f"    {'group':16s} {'member':24s} {'comp N':>9s} {'P_cr N':>10s} "
          f"{'util':>6s} {'lam':>6s} mode")
    for r in rows[:n]:
        print(f"    {r['group']:16s} {r['n1']+'-'+r['n2']:24s} {r['compression_N']:9.0f} "
              f"{r['P_cr_N']:10.0f} {r['utilisation']:6.3f} {r['slenderness']:6.1f} {r['mode']}")
    if all(r["utilisation"] < 1 for r in rows):
        print("  verdict       no member buckles under this load case")
    else:
        bad = [r for r in rows if r["utilisation"] >= 1]
        print(f"  verdict       {len(bad)} member(s) OVER the buckling limit")


# ----------------------------------------------------------------------------
# Strain energy
# ----------------------------------------------------------------------------
#energy stored in a member as it deforms:
#
#     U_member = 0.5 * u_e^T k_e u_e
#
# Summed over all members it equals the work done by the loads, 0.5 * F.u, which
# is an exact identity for a linear solve and therefore a free correctness check.

def strain_energy(data, result):
    """per-member strain energy in N*mm. Returns (rows, total)."""
    index, coords, u = result["index"], result["coords"], result["u"]
    E, G = data["material"]["E"], data["material"]["G"]

    rows, total = [], 0.0
    for m in data["members"]:
        a, b = index[m["n1"]], index[m["n2"]]
        k = frame3d_k(E, G, m["area"], m["I"], m["I"], m["J"], coords[a], coords[b])
        ue = u[node_dofs(a, DOF) + node_dofs(b, DOF)]
        U = 0.5 * ue @ k @ ue
        total += U
        rows.append({"n1": m["n1"], "n2": m["n2"], "group": m["group"],
                     "mass": m.get("mass"), "energy": U})
    for r in rows:
        r["fraction"] = r["energy"] / total
    return sorted(rows, key=lambda r: -r["energy"]), total


def energy_by_group(rows):
    groups = {}
    for r in rows:
        g = groups.setdefault(r["group"], {"group": r["group"], "energy": 0.0,
                                           "mass": 0.0, "members": 0})
        g["energy"] += r["energy"]
        g["mass"] += r["mass"] or 0.0
        g["members"] += 1
    total = sum(g["energy"] for g in groups.values())
    for g in groups.values():
        g["fraction"] = g["energy"] / total
        g["energy_per_kg"] = g["energy"] / g["mass"] if g["mass"] else float("nan")
    return sorted(groups.values(), key=lambda g: -g["energy"])


def check_energy_balance(result, rows_total):
    """Sum of member energies must equal 0.5*F.u. Relative error, should be ~1e-12."""
    work = 0.5 * result["F"] @ result["u"]
    return abs(rows_total - work) / abs(work)


def print_energy(data, result, n_groups=8, n_members=5):
    rows, total = strain_energy(data, result)
    groups = energy_by_group(rows)
    print(f"  strain energy {total:9.1f} N.mm total")
    if "F" in result:
        print(f"  check         energy balance vs 0.5*F.u: "
              f"{check_energy_balance(result, total):.1e}")
    print(f"  energy by group (top {n_groups}):")
    print(f"    {'group':18s} {'share':>7s} {'mass kg':>8s} {'share/kg':>9s}")
    for g in groups[:n_groups]:
        print(f"    {g['group']:18s} {100*g['fraction']:6.1f}% {g['mass']:8.2f} "
              f"{100*g['fraction']/g['mass']:9.1f}")
    print(f"  single members holding the most energy (top {n_members}):")
    for r in rows[:n_members]:
        print(f"    {r['group']:18s} {r['n1']+'-'+r['n2']:24s} {100*r['fraction']:5.1f}%")
