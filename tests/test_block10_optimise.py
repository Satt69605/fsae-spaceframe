"""Block 10: combining changes, and the Pareto front."""
import os
from _util import rel, run_all
from fesolver.sweep import default_spec
from fesolver.optimise import (Evaluator, dominance_pass, greedy_upgrade,
                               pareto_front, spec_diff)

SMALL = ["MAIN_HOOP", "HARNESS_BAR", "SIS_UPPER"]     # keep the tests quick

def test_pareto_maths():
    """Hand-built points: only the non-dominated ones survive, sorted by mass."""
    pts = [{"mass_kg": 30, "stiffness_Nm_per_deg": 1000},   # on the front
           {"mass_kg": 32, "stiffness_Nm_per_deg": 1500},   # on the front
           {"mass_kg": 34, "stiffness_Nm_per_deg": 1200},   # dominated by the 32 kg point
           {"mass_kg": 31, "stiffness_Nm_per_deg": 900},    # dominated by the 30 kg point
           {"mass_kg": 36, "stiffness_Nm_per_deg": 1800}]   # on the front
    front = pareto_front(pts)
    assert [p["mass_kg"] for p in front] == [30, 32, 36]

def test_pareto_ignores_illegal_designs():
    pts = [{"mass_kg": 30, "stiffness_Nm_per_deg": 1000, "checks_failed": ""},
           {"mass_kg": 25, "stiffness_Nm_per_deg": 9999, "checks_failed": "T3.2.1"}]
    assert [p["mass_kg"] for p in pareto_front(pts)] == [30]

def test_dominance_pass_never_adds_mass_and_always_adds_stiffness():
    ev = Evaluator()
    base = ev(default_spec())
    spec, res = dominance_pass(ev, groups=SMALL, max_tubes=6, verbose=False)
    assert res["mass_kg"] <= base["mass_kg"] + 1e-9
    assert res["stiffness_Nm_per_deg"] >= base["stiffness_Nm_per_deg"]
    assert res["checks_failed"] == ""

def test_greedy_improves_and_respects_the_budget():
    ev = Evaluator()
    base = ev(default_spec())
    budget = 800.0
    spec, res, steps = greedy_upgrade(ev, budget_g=budget, groups=SMALL, max_tubes=6,
                                      step_limit=4, verbose=False)
    assert res["stiffness_Nm_per_deg"] > base["stiffness_Nm_per_deg"]
    assert (res["mass_kg"] - base["mass_kg"]) * 1000.0 <= budget + 1e-6
    assert all(s["d_stiffness"] > 0 for s in steps)          # every step is an improvement
    assert [s["stiffness_Nm_per_deg"] for s in steps] == sorted(
        s["stiffness_Nm_per_deg"] for s in steps)            # monotonic

def test_only_the_named_groups_move():
    ev = Evaluator()
    spec, _, _ = greedy_upgrade(ev, budget_g=500.0, groups=SMALL, max_tubes=6,
                                step_limit=2, verbose=False)
    assert set(spec_diff(default_spec(), spec)) <= set(SMALL)

def test_evaluator_caches_repeats():
    ev = Evaluator()
    ev(default_spec())
    ev(default_spec())
    assert len(ev.history) == 1

def test_combined_change_is_not_the_sum_of_separate_changes():
    """Groups share load paths, so changes interact. This documents that they do."""
    ev = Evaluator()
    base = ev(default_spec())
    a = dict(default_spec()); a["MAIN_HOOP"] = (30.0, 2.0)
    b = dict(default_spec()); b["HARNESS_BAR"] = (30.0, 2.0)
    both = dict(a); both["HARNESS_BAR"] = (30.0, 2.0)
    da = ev(a)["stiffness_Nm_per_deg"] - base["stiffness_Nm_per_deg"]
    db = ev(b)["stiffness_Nm_per_deg"] - base["stiffness_Nm_per_deg"]
    dboth = ev(both)["stiffness_Nm_per_deg"] - base["stiffness_Nm_per_deg"]
    assert da > 0 and db > 0 and dboth > 0
    assert abs(dboth - (da + db)) > 1e-6        # not additive

if __name__ == "__main__":
    run_all(globals())
