"""Block 4: 3D frame element (12x12). Cantilever pointing in a random 3D direction."""
import numpy as np
from _util import rel, run_all
from fesolver.elements import frame3d_k, frame3d_T
from fesolver.assemble import assemble, solve

E, G = 200e3, 76923.08
A, I, J = 119.63, 8508.82, 17017.63
L = 700.0
DOF = 6

def cantilever(direction, load6):
    p2 = L * np.asarray(direction, float) / np.linalg.norm(direction)
    K = assemble(2, DOF, [(0, 1, frame3d_k(E, G, A, I, I, J, (0, 0, 0), p2))])
    F = np.zeros(12); F[6:] = load6
    u, _ = solve(K, F, range(6))
    return u[6:], p2 / L

def test_axial_bending_torsion_any_direction():
    rng = np.random.default_rng(1)
    for _ in range(20):
        d = rng.normal(size=3)
        ex = d / np.linalg.norm(d)
        perp = np.cross(ex, rng.normal(size=3)); perp /= np.linalg.norm(perp)

        u, _ = cantilever(d, np.r_[50 * ex, 0, 0, 0])                       # axial
        assert rel(u[:3] @ ex, 50 * L / (E * A)) < 1e-9

        u, _ = cantilever(d, np.r_[80 * perp, 0, 0, 0])                     # transverse
        assert rel(u[:3] @ perp, 80 * L**3 / (3 * E * I)) < 1e-9

        u, _ = cantilever(d, np.r_[0, 0, 0, 3e4 * ex])                      # torque
        assert rel(u[3:] @ ex, 3e4 * L / (G * J)) < 1e-9

        u, _ = cantilever(d, np.r_[0, 0, 0, 3e4 * perp])                    # bending moment
        assert rel(u[3:] @ perp, 3e4 * L / (E * I)) < 1e-9

def test_vertical_beam_branch():
    """The 'nearly vertical' branch in frame3d_T must also be right."""
    u, _ = cantilever((0, 0, 1), np.r_[100, 0, 0, 0, 0, 0])
    assert rel(u[0], 100 * L**3 / (3 * E * I)) < 1e-9

def test_six_rigid_body_modes():
    """An unsupported beam can move/rotate freely in 6 ways -> exactly 6 zero eigenvalues."""
    k = frame3d_k(E, G, A, I, I, J, (0, 0, 0), (300, 200, -100))
    assert np.allclose(k, k.T)
    eig = np.linalg.eigvalsh(k)
    assert np.sum(np.abs(eig) < 1e-8 * eig.max()) == 6
    assert np.all(eig > -1e-8 * eig.max())

if __name__ == "__main__":
    run_all(globals())
