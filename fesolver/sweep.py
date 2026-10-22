"""
sweep.py: generate multiple frames, evaluate each, collect the numbers.

The generator (`spaceframe.py`) is imported as a module here, not run as a
script, so a candidate frame is built in memory and handed straight to the
solver. No JSON files on disk, no subprocesses: one candidate costs about 50 ms.

Scope, deliberately kept to the standard formulation:
  - what varies : tube size (OD x wall) per tube group, from the purchasable
                  catalogue. Node positions are fixed as topology and geometry
                  don't change.
  - what's "best"  : the stiffness/mass trade-off, reported as a Pareto front.
                   No weighted objective to argue about.
  - constraints  : FS rules legality (guaranteed by the generator, re-checked
                   here) and buckling utilisation below 1.
"""

import csv
import os
import sys

# spaceframe.py lives in the repo root, one level above this package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spaceframe as sf  # noqa: E402

from .chassis import torsion_test  # noqa: E402
from .loadcases import bending_test, bulkhead_test  # noqa: E402
from .postprocess import buckling_check  # noqa: E402


def default_spec():
    """The generator's baseline tube choice, as a plain dict we can edit."""
    return dict(sf.ChassisParams().spec)


# Manufacturability guard. The rules set no minimum wall for non-structural groups, so an unconstrained search 
# will happily pick 0.9 mm walls that are awkward to weld to a 2 mm hoop without burning through. 
# Set this (or pass --min-wall) before running a sweep intended to produce a buildable design.
MIN_WALL = 0.0


def legal_tubes(group, min_wall=None):
    """Every catalogue tube this group is allowed to use, lightest first."""
    floor = MIN_WALL if min_wall is None else min_wall
    key = sf.MEMBER_GROUPS[group]
    if key == "N":
        tubes = list(sf.CATALOGUE)
    else:
        rc = sf.RULE_CLASSES[key]
        tubes = [t for t in sf.CATALOGUE if t.meets(rc)]
    return [t for t in tubes if t.t >= floor - 1e-9]


def build_frame(spec=None, **params):
    """Build one candidate frame in memory. Returns (data, failed_checks)."""
    s = default_spec()
    if spec:
        s.update(spec)
    p = sf.ChassisParams(spec=s, **params)
    frame = sf.build(p)
    failed = [c.rule for c in sf.check(frame) if not c.ok]
    return frame.to_dict(), failed


def actual_tube(data, group):
    """What the group ended up as: the generator substitutes illegal requests."""
    for m in data["members"]:
        if m["group"] == group:
            return m["od"], m["wall"]
    return None, None


def evaluate(data, cases=("torsion",)):
    out = {"mass_kg": sum(m["mass"] for m in data["members"]),
           "members": len(data["members"])}

    if "torsion" in cases:
        r = torsion_test(data, verbose=False)
        b = buckling_check(data, r)
        out["stiffness_Nm_per_deg"] = r["stiffness_Nm_per_deg"]
        out["torsion_buckling_util"] = b[0]["utilisation"]
        out["torsion_stress_util"] = max(x["yield_utilisation"] for x in b)
    if "bending" in cases:
        r = bending_test(data, verbose=False)
        out["bending_sag_mm"] = r["max_sag_mm"]
        out["bending_stiffness_N_per_mm"] = r["bending_stiffness_N_per_mm"]
    if "bulkhead" in cases:
        r = bulkhead_test(data, verbose=False)
        b = buckling_check(data, r)
        out["bulkhead_defl_mm"] = r["face_deflection_mm"]
        out["bulkhead_buckling_util"] = b[0]["utilisation"]

    if "stiffness_Nm_per_deg" in out:
        out["stiffness_per_kg"] = out["stiffness_Nm_per_deg"] / out["mass_kg"]
    return out


def baseline(cases=("torsion",)):
    data, failed = build_frame()
    row = {"group": "BASELINE", "od": None, "wall": None, "checks_failed": ";".join(failed)}
    row.update(evaluate(data, cases))
    return row


def _subsample(items, limit):
    """evenly spaced subset, always keeping the lightest and heaviest."""
    if not limit or len(items) <= limit:
        return items
    step = (len(items) - 1) / (limit - 1)
    return [items[round(i * step)] for i in range(limit)]


def one_at_a_time(groups=None, cases=("torsion",), progress=True, max_tubes=None):
    """Change one group's tube at a time, leave everything else at baseline."""
    base = baseline(cases)
    yield base

    groups = groups or [g for g in sf.MEMBER_GROUPS if g in default_spec()]
    for group in groups:
        base_od, base_wall = default_spec()[group]
        for tube in _subsample(legal_tubes(group), max_tubes):
            if abs(tube.od - base_od) < 1e-9 and abs(tube.t - base_wall) < 1e-9:
                continue                               # that's the baseline
            data, failed = build_frame(spec={group: (tube.od, tube.t)})
            got_od, got_wall = actual_tube(data, group)
            if got_od is None:
                break            # group has no members in this pickup mode
            if abs(got_od - tube.od) > 1e-9 or abs(got_wall - tube.t) > 1e-9:
                continue         # generator substituted the request; not this tube

            row = {"group": group, "od": tube.od, "wall": tube.t,
                   "checks_failed": ";".join(failed)}
            row.update(evaluate(data, cases))
            row["d_mass_g"] = (row["mass_kg"] - base["mass_kg"]) * 1000.0
            if "stiffness_Nm_per_deg" in row:
                row["d_stiffness"] = (row["stiffness_Nm_per_deg"]
                                      - base["stiffness_Nm_per_deg"])
                row["Nm_per_deg_per_g"] = (row["d_stiffness"] / row["d_mass_g"]
                                           if abs(row["d_mass_g"]) > 1e-9 else float("nan"))
            yield row
        if progress:
            print(f"  swept {group}", flush=True)


def write_csv(rows, path):
    rows = list(rows)
    fields, seen = [], set()
    for r in rows:                      # union of keys, first-seen order
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return rows, path
