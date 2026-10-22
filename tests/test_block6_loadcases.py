"""Blocks 6-7: bending and bulkhead load cases + member forces.

there's no textbook formula for a whole chassis, so these are properties the
answer must have: symmetry, linearity, reciprocity, equilibrium. A bug in
supports, loads or assembly breaks at least one of them.
"""
import os
import numpy as np
from _util import rel, run_all
from fesolver.chassis import load_frame, build_model, DOF
from fesolver.assemble import node_dofs, solve
from fesolver.loadcases import bending_test, bulkhead_test
from fesolver.postprocess import member_forces

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME = os.path.join(HERE, "..", "frames", "frame_at_node.json")
data = load_frame(FRAME)
Z = 2

def zdisp(r, name):
    return r["u"][node_dofs(r["index"][name], DOF)[Z]]

def test_mirrored_frame_gives_the_same_answer():
    """Mirror the whole frame in y and the vertical answer must be identical.

    The frame itself is not left-right symmetric as the floor X-braces are
    single diagonals, so this mirror test is the right check. It catches handedness/sign bugs in the 3D rotation."""
    import copy
    mirrored = copy.deepcopy(data)
    mirrored["nodes"] = {n: [x, -y, z] for n, (x, y, z) in data["nodes"].items()}
    a = bending_test(data, verbose=False)
    b = bending_test(mirrored, verbose=False)
    for name in a["names"]:
        assert abs(zdisp(a, name) - zdisp(b, name)) < 1e-9 * abs(a["max_sag_mm"])

def test_bending_is_linear():
    """Double the g factor, double the sag. Linear FE must do this exactly."""
    a = bending_test(data, g_factor=1.0, verbose=False)
    b = bending_test(data, g_factor=3.0, verbose=False)
    assert rel(b["max_sag_mm"], 3 * a["max_sag_mm"]) < 1e-10
    assert rel(b["bending_stiffness_N_per_mm"], a["bending_stiffness_N_per_mm"]) < 1e-10

def test_supports_carry_the_whole_load():
    for r in (bending_test(data, verbose=False), bulkhead_test(data, verbose=False)):
        assert r["force_balance_N"] < 1e-6

def test_reciprocity():
    """Maxwell-Betti: deflection at A from a load at B equals deflection at B
    from the same load at A. True for any correct symmetric stiffness matrix."""
    names, index, coords, K = build_model(data)
    fixed = [d for n in data["rear_pickups"] for d in node_dofs(index[n], DOF)[:3]]
    A = node_dofs(index["FH_TOP_L"], DOF)[Z]
    B = node_dofs(index["MHB_FOOT_R"], DOF)[0]
    FA = np.zeros(K.shape[0]); FA[A] = 1000.0
    FB = np.zeros(K.shape[0]); FB[B] = 1000.0
    uA, _ = solve(K, FA, fixed)
    uB, _ = solve(K, FB, fixed)
    assert rel(uA[B], uB[A]) < 1e-8

def test_member_axial_sign_and_magnitude():
    """One bar pulled along its own axis: axial force must equal the load, positive."""
    P, L = 1234.0, 500.0
    mini = {
        "units": {"length": "mm", "modulus": "MPa"},
        "material": {"E": 200e3, "G": 76923.08},
        "nodes": {"A": [0, 0, 0], "B": [L / np.sqrt(3)] * 3},
        "members": [{"n1": "A", "n2": "B", "group": "T", "area": 100.0,
                     "I": 8000.0, "J": 16000.0, "od": 25.4}],
    }
    names, index, coords, K = build_model(mini)
    d = (coords[1] - coords[0]) / L
    F = np.zeros(12); F[6:9] = P * d                      # pull node B along the bar
    u, _ = solve(K, F, range(6))
    r = {"index": index, "coords": coords, "u": u}
    f = member_forces(mini, r)[0]
    assert rel(f["axial_N"], P) < 1e-8                    # positive = tension
    assert rel(u[6:9] @ d, P * L / (200e3 * 100.0)) < 1e-9

if __name__ == "__main__":
    run_all(globals())
