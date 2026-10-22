"""
elements.py: the stiffness matrix of one beam in three cases.

Each function takes material + section + geometry and returns a matrix k such that, for that single
beam on its own, f = k @ u   (end forces = k times end displacements).

Units: anything consistent. The chassis JSON uses N, mm, MPa.

Block 1: beam2d_k      4x4    [v1, th1, v2, th2]               bending only
Block 3: frame2d_k     6x6    [u1, v1, th1, u2, v2, th2]       + axial, any angle
Block 4: frame3d_k     12x12  [u v w thx thy thz] x 2 nodes   + torsion, 3D
"""

import numpy as np


# ----------------------------------------------------------------------------
# Block 1 - 2D beam, bending only, lying along the x axis
# ----------------------------------------------------------------------------
def beam2d_k(E, I, L):
    """4x4 Euler-Bernoulli beam. DOF order: [v1, theta1, v2, theta2]."""
    return (E * I / L**3) * np.array([
        [ 12,     6*L,    -12,     6*L   ],
        [ 6*L,    4*L**2, -6*L,    2*L**2],
        [-12,    -6*L,     12,    -6*L   ],
        [ 6*L,    2*L**2, -6*L,    4*L**2],
    ], dtype=float)


# ----------------------------------------------------------------------------
# Block 3 - 2D frame element: axial + bending, at any angle in the x-y plane
# ----------------------------------------------------------------------------
def frame2d_k_local(E, A, I, L):
    """6x6 in the beam's own axes (local x runs along the beam).
    DOF order: [u1, v1, th1, u2, v2, th2]  (u = along beam, v = sideways)."""
    k = np.zeros((6, 6))
    a = E * A / L                      # axial spring, like a bar
    k[np.ix_([0, 3], [0, 3])] = a * np.array([[1, -1], [-1, 1]])
    k[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = beam2d_k(E, I, L)   # Block 1, reused
    return k


def frame2d_T(x1, y1, x2, y2):
    """6x6 rotation: turns global displacements into LOCAL ones (u_local = T @ u_global)."""
    L = np.hypot(x2 - x1, y2 - y1)
    c, s = (x2 - x1) / L, (y2 - y1) / L
    R = np.array([[ c, s, 0],
                  [-s, c, 0],
                  [ 0, 0, 1]])        # rotation about z is unchanged by rotating the beam
    T = np.zeros((6, 6))
    T[:3, :3] = R
    T[3:, 3:] = R
    return T, L


def frame2d_k(E, A, I, p1, p2):
    """6x6 in global axes for a beam from point p1=(x,y) to p2=(x,y)."""
    T, L = frame2d_T(*p1, *p2)
    return T.T @ frame2d_k_local(E, A, I, L) @ T


# ----------------------------------------------------------------------------
# Block 4 - 3D frame element: axial + torsion + bending in two planes
# ----------------------------------------------------------------------------
def frame3d_k_local(E, G, A, Iy, Iz, J, L):
    """12x12 in the beam's OWN axes.
    DOF order per node: [u, v, w, thx, thy, thz]; node 1 is 0-5, node 2 is 6-11.
      u / thx : along the beam  -> axial (EA) and torsion (GJ)
      v / thz : bending in the local x-y plane (uses Iz)
      w / thy : bending in the local x-z plane (uses Iy)
    """
    k = np.zeros((12, 12))

    # axial (bar) -- same as Block 3
    k[np.ix_([0, 6], [0, 6])] = (E * A / L) * np.array([[1, -1], [-1, 1]])

    # torsion -- mathematically identical to a bar, with GJ instead of EA
    k[np.ix_([3, 9], [3, 9])] = (G * J / L) * np.array([[1, -1], [-1, 1]])

    # bending in x-y: exactly the Block 1 matrix on [v1, thz1, v2, thz2]
    k[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] = beam2d_k(E, Iz, L)

    # bending in x-z: the Block 1 matrix again, but on [w1, thy1, w2, thy2].
    # The 6L terms flip sign because a positive thy (right-hand rule about y)
    # tilts the beam DOWN in z, i.e. slope dw/dx = -thy.
    S = np.diag([1, -1, 1, -1])
    k[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] = S @ beam2d_k(E, Iy, L) @ S

    return k


def frame3d_T(p1, p2):
    """12x12 rotation from global to local axes for a beam from p1 to p2.

    local x = along the beam. local y and z just need to be perpendicular. For round tubes Iy == Iz.
    """
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    L = np.linalg.norm(d)
    ex = d / L

    ref = np.array([0.0, 0.0, 1.0])            # global up
    if abs(ex @ ref) > 0.99:                   # beam nearly vertical -> use global y
        ref = np.array([0.0, 1.0, 0.0])
    ey = np.cross(ref, ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)

    R = np.vstack([ex, ey, ez])                # rows = local axes written in global coords
    T = np.zeros((12, 12))
    for b in range(4):                         # node1 transl, node1 rot, node2 transl, node2 rot
        T[3*b:3*b+3, 3*b:3*b+3] = R
    return T, L


def frame3d_k(E, G, A, Iy, Iz, J, p1, p2):
    """12x12 in global axes for a beam from p1=(x,y,z) to p2=(x,y,z)."""
    T, L = frame3d_T(p1, p2)
    return T.T @ frame3d_k_local(E, G, A, Iy, Iz, J, L) @ T
