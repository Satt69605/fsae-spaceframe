"""Block 9: the sweep driver. Basically asks if it build the frames it claims to build?"""
import json, os
import numpy as np
from _util import rel, run_all
import spaceframe as sf
from fesolver.sweep import (default_spec, legal_tubes, build_frame, actual_tube,
                            evaluate, baseline, one_at_a_time)
from fesolver.chassis import load_frame

HERE = os.path.dirname(os.path.abspath(__file__))
STORED = os.path.join(HERE, "..", "frames", "frame_at_node.json")

def test_in_memory_frame_equals_the_stored_json():
    """The frame the sweep builds must be the frame we validated the solver on."""
    data, failed = build_frame()
    stored = load_frame(STORED)
    assert not failed
    assert data["nodes"] == stored["nodes"]
    assert data["members"] == stored["members"]

def test_baseline_matches_the_pinned_result():
    b = baseline()
    assert rel(b["stiffness_Nm_per_deg"], 1712.2021) < 1e-5
    assert rel(b["mass_kg"], 34.18) < 1e-3

def test_legal_tubes_all_meet_their_rule_class():
    for group, key in sf.MEMBER_GROUPS.items():
        if key == "N":
            continue
        rc = sf.RULE_CLASSES[key]
        for t in legal_tubes(group):
            assert t.t >= rc.min_wall * 0.999 and t.area >= rc.min_area * 0.999
            assert t.I >= rc.min_I * 0.999

def test_changing_one_group_changes_only_that_group():
    group, tube = "SIS_UPPER", sf.Tube(30.0, 2.0)
    data, _ = build_frame(spec={group: (tube.od, tube.t)})
    stored = load_frame(STORED)
    for new, old in zip(data["members"], stored["members"]):
        assert new["group"] == old["group"]
        if new["group"] == group:
            assert new["od"] == tube.od and new["wall"] == tube.t
        else:
            assert new["od"] == old["od"] and new["wall"] == old["wall"]

def test_a_stiffer_tube_gives_a_stiffer_frame():
    """Same group, strictly larger I and area -> stiffness must not fall."""
    base = baseline()["stiffness_Nm_per_deg"]
    data, _ = build_frame(spec={"SIS_UPPER": (30.0, 2.0)})     # bigger I than 25.4x1.6
    assert evaluate(data)["stiffness_Nm_per_deg"] > base

def test_every_swept_candidate_is_rules_legal():
    rows = list(one_at_a_time(groups=["MAIN_HOOP", "SIS_LOWER"], max_tubes=6, progress=False))
    assert len(rows) > 5
    assert all(r["checks_failed"] == "" for r in rows)
    assert all(r["mass_kg"] > 0 and r["stiffness_Nm_per_deg"] > 0 for r in rows)

def test_requested_tube_is_the_tube_actually_built():
    for od, wall in [(30.0, 2.0), (31.75, 1.6), (25.4, 1.6)]:
        data, _ = build_frame(spec={"REAR_IMPACT": (od, wall)})
        assert actual_tube(data, "REAR_IMPACT") == (od, wall)

if __name__ == "__main__":
    run_all(globals())
