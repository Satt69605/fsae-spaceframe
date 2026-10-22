"""
assemble.py: glue element matrices into one global K, fix supports, solve.

This file is the same for 2D beams, 2D frames and 3D frames. The only thing
that changes is `dof_per_node` (2, 3 or 6) and which element matrices are added.
"""

import numpy as np


def node_dofs(node, dof_per_node):
    """Global DOF numbers owned by a node. Node i owns i*n ... i*n + n-1."""
    start = node * dof_per_node
    return list(range(start, start + dof_per_node))


def assemble(n_nodes, dof_per_node, elements):
    """
    elements: iterable of (node_a, node_b, k_element_in_global_axes)

    Each element touches only its own two nodes, so its matrix is added into
    the rows/columns of those nodes. Where two beams share a node, their
    stiffnesses add.
    """
    ndof = n_nodes * dof_per_node
    K = np.zeros((ndof, ndof))
    for a, b, k in elements:
        dofs = node_dofs(a, dof_per_node) + node_dofs(b, dof_per_node)
        K[np.ix_(dofs, dofs)] += k
    return K


def solve(K, F, fixed_dofs):
    """
    Solve K u = F with some displacements forced to zero.

    Returns (u, R): full displacement vector and reaction forces.
    Reactions must balance the applied loads, which give a sanity check.
    """
    ndof = K.shape[0]
    fixed = np.array(sorted(set(fixed_dofs)), dtype=int)
    free = np.setdiff1d(np.arange(ndof), fixed)

    Kff = K[np.ix_(free, free)]                      # the "delete rows and columns" step
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(Kff, F[free])          # raises Linalgerror if singular

    R = K @ u - F                                    # nonzero only at the fixed DOF
    return u, R
