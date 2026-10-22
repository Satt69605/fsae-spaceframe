"""Block 2: join two 2D beams at a shared middle node (assembly)."""
import numpy as np
from _util import rel, run_all
from fesolver.elements import beam2d_k
from fesolver.assemble import assemble, solve

E, I, Ltot = 200e3, 8508.82, 1000.0
L = Ltot / 2                                  
P = 100.0
DOF = 2                                       
V = 0

def two_elements():
    k = beam2d_k(E, I, L)
    return assemble(3, DOF, [(0, 1, k), (1, 2, k)])    # nodes 0 - 1 - 2

def test_cantilever_tip_and_midpoint():
    K = two_elements()
    F = np.zeros(6); F[2*DOF + V] = P                  # tip load at node 2
    u, R = solve(K, F, fixed_dofs=[0, 1])              # clamp node 0
    assert rel(u[4], P * Ltot**3 / (3*E*I)) < 1e-10              # tip
    assert rel(u[2], 5 * P * Ltot**3 / (48*E*I)) < 1e-10         # midpoint
    assert abs(R[0] + P) < 1e-8                                  # support pushes back with -P

def test_simply_supported_centre_load():
    K = two_elements()
    F = np.zeros(6); F[1*DOF + V] = -P                 # load at the shared middle node
    u, _ = solve(K, F, fixed_dofs=[0, 4])              # v=0 at both ends, rotations free
    assert rel(u[2], -P * Ltot**3 / (48*E*I)) < 1e-10

def test_shared_node_adds_stiffness():
    K, k = two_elements(), beam2d_k(E, I, L)
    # middle node's diagonal = end-2 part of element A + end-1 part of element B
    assert np.allclose(K[2:4, 2:4], k[2:4, 2:4] + k[0:2, 0:2])

if __name__ == "__main__":
    run_all(globals())
