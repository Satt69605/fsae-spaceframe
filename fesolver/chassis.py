"""
chassis.py: Block 5. Turn spaceframe.py's JSON into a torsional stiffness.

Steps (numbers match the explanation in README):
  1. read JSON
  2. number the nodes (from the MEMBER list, so unconnected reference points are skipped)
  3. build a 12x12 for every member and add it into the global K
  4. fix all 3 translations at the rear pickups (rotations free)
  5. push front-left pickups up, front-right down
  6. solve K u = F
  7. twist angle -> torque / angle = Nm/deg
"""

import json
import math

import numpy as np

from .elements import frame3d_k
from .assemble import assemble, node_dofs, solve

DOF = 6  # per node in 3D: u, v, w, thx, thy, thz


def load_frame(path):
    with open(path) as f:
        data = json.load(f)
    if data["units"]["length"] != "mm" or data["units"]["modulus"] != "MPa":
        raise ValueError(f"Expected mm / MPa, got {data['units']}")
    return data


def build_model(data):
    """Steps 2-3. Returns (names, index, coords, K)."""
    # step 2: only nodes that some member actually uses (README trap 1)
    names = []
    for m in data["members"]:
        for n in (m["n1"], m["n2"]):
            if n not in names:
                names.append(n)
    index = {n: i for i, n in enumerate(names)}
    coords = np.array([data["nodes"][n] for n in names], dtype=float)

    E, G = data["material"]["E"], data["material"]["G"]

    # step 3: one 12x12 per member, rotated to global axes, added into K
    elements = []
    for m in data["members"]:
        a, b = index[m["n1"]], index[m["n2"]]
        k = frame3d_k(E, G, m["area"], m["I"], m["I"], m["J"], coords[a], coords[b])
        elements.append((a, b, k))
    K = assemble(len(names), DOF, elements)
    return names, index, coords, K


def torsion_test(data, force=500.0, verbose=True):
    """Steps 4-7. Returns a dict of results. Units in: N, mm. Result: Nm/deg."""
    names, index, coords, K = build_model(data)
    front = data["front_pickups"]            # e.g. FA_UPR_L, FA_UPR_R, FA_LWR_L, FA_LWR_R
    rear = data["rear_pickups"]
    left = [n for n in front if n.endswith("_L")]
    right = [n for n in front if n.endswith("_R")]

    # step 4: boundary conditions: all 3 translations at each rear pickup
    fixed = []
    for n in rear:
        fixed += node_dofs(index[n], DOF)[:3]

    # step 5: loads: +z on the left front pickups, -z on the right
    F = np.zeros(K.shape[0])
    Z = 2  # z translation is the 3rd DOF of a node
    for n in left:
        F[node_dofs(index[n], DOF)[Z]] += force
    for n in right:
        F[node_dofs(index[n], DOF)[Z]] -= force

    # step 6: solve
    u, R = solve(K, F, fixed)

    # sanity checks: the solve really satisfied K u = F, and supports balance loads
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    residual = np.linalg.norm(K[np.ix_(free, free)] @ u[free] - F[free]) / np.linalg.norm(F)
    # per axis: sum of support reactions + sum of applied loads must be zero
    force_balance = max(abs(R[ax::DOF].sum() + F[ax::DOF].sum()) for ax in range(3))
    if not np.all(np.isfinite(u)) or residual > 1e-6:
        raise RuntimeError(f"Solve failed (residual {residual:.2e}). Is the frame a mechanism?")

    # step 7: torque and twist
    def z_disp(n):
        return u[node_dofs(index[n], DOF)[Z]]

    def y_pos(n):
        return coords[index[n], 1]

    torque_Nmm = sum(force * y_pos(n) for n in left) + sum(-force * y_pos(n) for n in right)
    spacing = np.mean([y_pos(n) for n in left]) - np.mean([y_pos(n) for n in right])
    dz = np.mean([z_disp(n) for n in left]) - np.mean([z_disp(n) for n in right])
    twist_deg = math.degrees(math.atan(dz / spacing))

    torque_Nm = torque_Nmm / 1000.0
    stiffness = torque_Nm / twist_deg
    mass = sum(m["mass"] for m in data["members"])

    result = {
        "nodes": len(names),
        "members": len(data["members"]),
        "dof_total": K.shape[0],
        "dof_free": len(free),
        "torque_Nm": torque_Nm,
        "lateral_spacing_mm": spacing,
        "twist_deg": twist_deg,
        "stiffness_Nm_per_deg": stiffness,
        "mass_kg": mass,
        "specific_stiffness_Nm_per_deg_per_kg": stiffness / mass,
        "solve_residual": residual,
        "force_balance_N": force_balance,
        "u": u,
        "coords": coords,
        "names": names,
        "index": index,
        "F": F,
        "K": K,
    }
    if verbose:
        print(f"  nodes {len(names)}, members {len(data['members'])}, "
              f"K is {K.shape[0]}x{K.shape[0]}, {len(free)} free DOF")
        print(f"  torque        {torque_Nm:9.1f} Nm  (lateral spacing {spacing:.1f} mm)")
        print(f"  twist         {twist_deg:9.4f} deg")
        print(f"  STIFFNESS     {stiffness:9.1f} Nm/deg")
        print(f"  mass          {mass:9.2f} kg  -> {stiffness/mass:.1f} Nm/deg/kg")
        print(f"  checks        residual {residual:.1e}, support force balance {force_balance:.1e} N")
    return result
