"""Block 1: one 2D beam. Cantilever with tip load and tip moment."""
import numpy as np
from _util import rel, run_all
from fesolver.elements import beam2d_k

E, L = 200e3, 1000.0                          # MPa, mm
D, t = 25.4, 1.65
I = np.pi / 64 * (D**4 - (D - 2*t)**4)        # mm^4

def cantilever(F):
    K = beam2d_k(E, I, L)                     # [v1, th1, v2, th2]
    free = [2, 3]                             # node 1 clamped
    return np.linalg.solve(K[np.ix_(free, free)], F)

def test_tip_load():
    P = 100.0
    assert rel(cantilever([P, 0])[0], P * L**3 / (3 * E * I)) < 1e-10

def test_tip_moment():
    M = 5e4
    assert rel(cantilever([0, M])[1], M * L / (E * I)) < 1e-10

def test_symmetric():
    K = beam2d_k(E, I, L)
    assert np.allclose(K, K.T)

if __name__ == "__main__":
    run_all(globals())
