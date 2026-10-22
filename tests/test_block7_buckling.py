"""Block 7: Euler / Johnson buckling check."""
import math, os
import numpy as np
from _util import rel, run_all
from fesolver.chassis import load_frame, torsion_test
from fesolver.loadcases import bending_test
from fesolver.postprocess import critical_load, buckling_check

E, Sy = 200e3, 180.0
A, I = 119.63, 8508.82                       # 25.4 x 1.6 tube
LAM_C = math.sqrt(2 * math.pi**2 * E / Sy)

HERE = os.path.dirname(os.path.abspath(__file__))
data = load_frame(os.path.join(HERE, "..", "frames", "frame_at_node.json"))

def test_slender_member_matches_euler():
    L = 3000.0                               # long enough to be in the Euler regime
    P, mode, lam = critical_load(E, Sy, A, I, L)
    assert mode == "euler" and lam > LAM_C
    assert rel(P, math.pi**2 * E * I / L**2) < 1e-12

def test_short_member_uses_johnson_and_stays_below_yield():
    P, mode, lam = critical_load(E, Sy, A, I, 100.0)
    assert mode == "johnson" and lam < LAM_C
    assert P < Sy * A                        # never stronger than squashing the tube

def test_two_formulas_agree_at_the_transition():
    r = math.sqrt(I / A)
    L = LAM_C * r                            # exactly at the transition slenderness
    P_hi, _, _ = critical_load(E, Sy, A, I, L * 1.000001)
    P_lo, _, _ = critical_load(E, Sy, A, I, L * 0.999999)
    assert rel(P_hi, P_lo) < 1e-5            # the curves meet, no jump

def test_longer_is_weaker():
    Ps = [critical_load(E, Sy, A, I, L)[0] for L in (200, 400, 800, 1600, 3200)]
    assert all(b < a for a, b in zip(Ps, Ps[1:]))

def test_effective_length_factor_helps():
    """k = 0.7 (welded joints) must give a higher capacity than k = 1.0 (pinned)."""
    assert critical_load(E, Sy, A, I, 2000.0, k=0.7)[0] > critical_load(E, Sy, A, I, 2000.0)[0]

def test_utilisation_scales_with_load():
    a = buckling_check(data, bending_test(data, g_factor=1.0, verbose=False))
    b = buckling_check(data, bending_test(data, g_factor=3.0, verbose=False))
    assert rel(b[0]["utilisation"], 3 * a[0]["utilisation"]) < 1e-9

def test_frame_survives_the_load_cases():
    for rows in (buckling_check(data, torsion_test(data, verbose=False)),
                 buckling_check(data, bending_test(data, g_factor=3.0, verbose=False))):
        assert rows == sorted(rows, key=lambda r: -r["utilisation"])
        assert rows[0]["utilisation"] < 1.0
        assert all(r["compression_N"] >= 0 for r in rows)

if __name__ == "__main__":
    run_all(globals())
