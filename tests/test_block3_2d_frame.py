"""Block 3: 2D frame elements (axial + bending) at arbitrary angles."""
import numpy as np
from _util import rel, run_all
from fesolver.elements import frame2d_k
from fesolver.assemble import assemble, solve

E, A, I = 200e3, 119.63, 8508.82
DOF = 3                                      # u, v, theta

def test_inclined_cantilever():
    """Cantilever at 35 deg. Load split into along-beam and across-beam parts."""
    L, ang = 800.0, np.radians(35)
    ex = np.array([np.cos(ang), np.sin(ang)]); ey = np.array([-np.sin(ang), np.cos(ang)])
    K = assemble(2, DOF, [(0, 1, frame2d_k(E, A, I, (0, 0), tuple(L * ex)))])
    Fg = np.array([30.0, -70.0])             # global load at tip
    F = np.zeros(6); F[3:5] = Fg
    u, _ = solve(K, F, [0, 1, 2])
    d = u[3:5]
    assert rel(d @ ex, (Fg @ ex) * L / (E * A)) < 1e-9           # axial stretch
    assert rel(d @ ey, (Fg @ ey) * L**3 / (3 * E * I)) < 1e-9    # bending deflection

def test_L_frame():
    """Vertical column H clamped at base, horizontal arm L, downward tip load P."""
    H, L, P = 600.0, 400.0, 100.0
    K = assemble(3, DOF, [
        (0, 1, frame2d_k(E, A, I, (0, 0), (0, H))),
        (1, 2, frame2d_k(E, A, I, (0, H), (L, H))),
    ])
    F = np.zeros(9); F[2*DOF + 1] = -P
    u, _ = solve(K, F, [0, 1, 2])
    hand = P*L**3/(3*E*I) + P*L**2*H/(E*I) + P*H/(E*A)   # arm bending + column rotation + column squash
    assert rel(-u[2*DOF + 1], hand) < 1e-10

if __name__ == "__main__":
    run_all(globals())
