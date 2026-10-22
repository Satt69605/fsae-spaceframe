"""Block 5: the full chassis, end to end.

Two regression pins on the real frame. If either number changes, the solver's
behaviour changed -> find out why before trusting anything downstream.

Validation of the solver itself is elsewhere: closed-form beam solutions in
Blocks 1-4 and the PyNite cross-check in tools/crosscheck_pynite.py.
"""
import copy
import os
from _util import run_all
from fesolver.chassis import load_frame, torsion_test

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME = os.path.join(HERE, "..", "frames", "frame_at_node.json")


def without_group(data, group):
    """The same frame with one tube group deleted."""
    d = copy.deepcopy(data)
    d["members"] = [m for m in d["members"] if m["group"] != group]
    return d


def test_frame_regression():
    r = torsion_test(load_frame(FRAME), verbose=False)
    assert abs(r["stiffness_Nm_per_deg"] - 1712.2021) < 0.001, r["stiffness_Nm_per_deg"]
    assert r["force_balance_N"] < 1e-6
    assert r["solve_residual"] < 1e-9


def test_frame_without_axle_triangulation_regression():
    """Deleting the 5 PICKUP_SUPPORT tubes exercises a different member set and
    a different load path (case study 1). Pinned so that path cannot drift."""
    r = torsion_test(without_group(load_frame(FRAME), "PICKUP_SUPPORT"), verbose=False)
    assert abs(r["stiffness_Nm_per_deg"] - 953.9238) < 0.001, r["stiffness_Nm_per_deg"]
    assert r["solve_residual"] < 1e-9


def test_triangulating_the_axle_station_helps():
    full = torsion_test(load_frame(FRAME), verbose=False)["stiffness_Nm_per_deg"]
    bare = torsion_test(without_group(load_frame(FRAME), "PICKUP_SUPPORT"),
                        verbose=False)["stiffness_Nm_per_deg"]
    assert full > 1.5 * bare


if __name__ == "__main__":
    run_all(globals())
