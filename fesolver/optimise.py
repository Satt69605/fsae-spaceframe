"""
optimise.py: combine tube changes, then report the stiffness/mass trade-off.

The one at a time sweep shows what each group buys on its own. It cannot
tell what two changes buy together, because groups share load paths: two
upgrades can overlap (less than the sum) or complement each other (more).
Everything here changes several groups at once and re-solves, so interactions
are included by construction.

Three pieces:

1. dominance_pass  -- apply every change that is stiffer and not heavier,
                       then repeat until nothing is left. Pure profit; no weightage.
2. greedy_upgrade   -- spend a mass budget one step at a time, each step taking
                       the best Nm/deg per gram available at that point: the ranking is recomputed after every
                       change, on the real combined frame.
3. pareto_front     -- from everything evaluated,the designs that are not
                       beaten on both mass and stiffness at once are kept.

Note: Greedy is not guaranteed to find the global optimum. It's transparent,
reproducible and easy to defend, which for 19 groups of catalogue tubes is the
right trade. A GA would be more machinery for a search this small.

"""

import copy

from .sweep import build_frame, default_spec, evaluate, legal_tubes, _subsample

# A change worth less mass than this counts as "free": the per-gram ratio is meaningless when divided by a near-zero denominator.
FREE_MASS_G = 10.0


def _spec_key(spec):
    return tuple(sorted((g, round(od, 3), round(t, 3)) for g, (od, t) in spec.items()))


class Evaluator:

    def __init__(self, cases=("torsion",)):
        self.cases = cases
        self.cache = {}
        self.history = []          # every distinct design evaluated

    def __call__(self, spec, label=""):
        key = _spec_key(spec)
        if key in self.cache:
            return self.cache[key]
        data, failed = build_frame(spec=spec)
        res = evaluate(data, self.cases)
        res["checks_failed"] = ";".join(failed)
        res["spec"] = copy.deepcopy(spec)
        res["label"] = label
        self.cache[key] = res
        self.history.append(res)
        return res


def candidate_changes(spec, groups, max_tubes):
    for group in groups:
        cur_od, cur_t = spec[group]
        for tube in _subsample(legal_tubes(group), max_tubes):
            if abs(tube.od - cur_od) < 1e-9 and abs(tube.t - cur_t) < 1e-9:
                continue
            trial = dict(spec)
            trial[group] = (tube.od, tube.t)
            yield group, tube, trial


def dominance_pass(ev, spec=None, groups=None, max_tubes=None, verbose=True):
    spec = dict(spec or default_spec())
    groups = groups or list(spec)
    base = ev(spec, "start")
    if verbose:
        print(f"  start         {base['stiffness_Nm_per_deg']:8.1f} Nm/deg  "
              f"{base['mass_kg']:6.2f} kg")

    while True:
        best = None
        for group, tube, trial in candidate_changes(spec, groups, max_tubes):
            r = ev(trial)
            if r["checks_failed"]:
                continue
            heavier = (r["mass_kg"] - base["mass_kg"]) * 1000.0
            gain = r["stiffness_Nm_per_deg"] - base["stiffness_Nm_per_deg"]
            if heavier <= 0 and gain > 1e-6 and (best is None or gain > best[0]):
                best = (gain, group, tube, trial, r)
        if best is None:
            break
        gain, group, tube, spec, base = best[0], best[1], best[2], best[3], best[4]
        if verbose:
            print(f"  keep          {group:16s} -> {tube}  "
                  f"{gain:+7.1f} Nm/deg  {(base['mass_kg']):6.2f} kg")
    return spec, base


def greedy_upgrade(ev, spec=None, budget_g=2000.0, groups=None, max_tubes=16,
                   step_limit=40, verbose=True):
    """
    Spend a mass budget, best value per gram first, re-ranking after each step.

    Returns (spec, result, steps) where steps is one row per accepted change.
    """
    spec = dict(spec or default_spec())
    groups = groups or list(spec)
    cur = ev(spec, "greedy start")
    start_mass = cur["mass_kg"]
    steps = []

    for _ in range(step_limit):
        best = None
        for group, tube, trial in candidate_changes(spec, groups, max_tubes):
            r = ev(trial)
            if r["checks_failed"]:
                continue
            d_mass = (r["mass_kg"] - cur["mass_kg"]) * 1000.0
            d_k = r["stiffness_Nm_per_deg"] - cur["stiffness_Nm_per_deg"]
            if d_k <= 0:
                continue
            spent = (r["mass_kg"] - start_mass) * 1000.0
            if spent > budget_g:
                continue
            # free or near-free changes are ranked by gain, not by a ratio with a vanishing denominator
            value = d_k / max(d_mass, FREE_MASS_G)
            if best is None or value > best[0]:
                best = (value, group, tube, trial, r, d_mass, d_k)
        if best is None:
            break
        value, group, tube, spec, cur, d_mass, d_k = best
        steps.append({"group": group, "od": tube.od, "wall": tube.t,
                      "d_mass_g": d_mass, "d_stiffness": d_k,
                      "mass_kg": cur["mass_kg"],
                      "stiffness_Nm_per_deg": cur["stiffness_Nm_per_deg"],
                      "value_Nm_per_deg_per_g": value})
        if verbose:
            print(f"  step {len(steps):2d}       {group:16s} -> {str(tube):9s} "
                  f"{d_mass:+7.0f} g {d_k:+7.1f} Nm/deg  -> "
                  f"{cur['stiffness_Nm_per_deg']:7.1f} @ {cur['mass_kg']:.2f} kg")
    return spec, cur, steps


def pareto_front(results):
    """
    Non-dominated designs: nothing else is lighter and stiffer.

    Returned sorted by mass. Each point is a design that could actually be built.
    """
    pts = [r for r in results if not r.get("checks_failed")]
    front = []
    for r in pts:
        dominated = any(
            o["mass_kg"] <= r["mass_kg"] + 1e-12
            and o["stiffness_Nm_per_deg"] >= r["stiffness_Nm_per_deg"] - 1e-12
            and (o["mass_kg"] < r["mass_kg"] - 1e-12
                 or o["stiffness_Nm_per_deg"] > r["stiffness_Nm_per_deg"] + 1e-12)
            for o in pts)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r["mass_kg"])


def spec_diff(a, b):
    """Groups whose tube differs between two specs."""
    return {g: (a[g], b[g]) for g in a if a[g] != b[g]}
