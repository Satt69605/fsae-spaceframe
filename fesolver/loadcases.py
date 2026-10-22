"""
loadcases.py: load cases 2 and 3. Same K, same solve, different loads and supports.

Everything here reuses chassis.build_model(). The only thing that changes
between load cases is:
    - which DOF are held (supports)
    - where the forces go (loads)
    - what you measure afterwards

Load case 1 (torsion): in chassis.torsion_test().
Load case 2 (bending): driver + powertrain mass, car supported at its wheels.
Load case 3 (bulkhead): rearward push on the impact-attenuator face.

Units in: N, mm. g = 9810 mm/s^2.
"""

import numpy as np

from .assemble import node_dofs, solve
from .chassis import DOF, build_model

X, Y, Z = 0, 1, 2


def mass_to_N(kg):
    return kg * 9.81

DRIVER_NODES = ["FH_BOT_L", "FH_BOT_R", "MH_BOT_L", "MH_BOT_R"]   # cockpit floor rails
ENGINE_NODES = ["MHB_FOOT_L", "MHB_FOOT_R", "REAR_AXLE_L", "REAR_AXLE_R"]  # rear bay
BULKHEAD_NODES = ["FBH_BOT_L", "FBH_BOT_R", "FBH_TOP_L", "FBH_TOP_R"]     # IA mounting face

# Bending supports
BENDING_SUPPORT_NODES = ["FA_LWR_L", "FA_LWR_R", "REAR_AXLE_L", "REAR_AXLE_R"]


def _check_nodes(index, names, what):
    missing = [n for n in names if n not in index]
    if missing:
        raise KeyError(f"{what}: node(s) not in this frame: {missing}")


def _spread(F, index, nodes, total_N, axis):
    for n in nodes:
        F[node_dofs(index[n], DOF)[axis]] += total_N / len(nodes)


def _vertical_report(u, index, names, coords):
    dz = u.reshape(-1, DOF)[:, Z]
    i = int(np.argmin(dz))
    return dz[i], names[i]


# ----------------------------------------------------------------------------
# Load case 2: vertical bending
# ----------------------------------------------------------------------------
def bending_test(data, driver_kg=80.0, engine_kg=70.0, g_factor=1.0, verbose=True):
    """
    Driver and powertrain weight pressing down. Supports: vertical at all four pickups, 
    with minimal lateral restraint
    """
    names, index, coords, K = build_model(data)
    _check_nodes(index, DRIVER_NODES + ENGINE_NODES + BENDING_SUPPORT_NODES, "bending")

    # supports: z at all four corners; x+y at the front-left; y at the rear-left
    fl, rl = BENDING_SUPPORT_NODES[0], BENDING_SUPPORT_NODES[2]
    fixed = [node_dofs(index[n], DOF)[Z] for n in BENDING_SUPPORT_NODES]
    fixed += [node_dofs(index[fl], DOF)[X], node_dofs(index[fl], DOF)[Y]]
    fixed += [node_dofs(index[rl], DOF)[Y]]

    # loads
    F = np.zeros(K.shape[0])
    driver_N = mass_to_N(driver_kg) * g_factor
    engine_N = mass_to_N(engine_kg) * g_factor
    _spread(F, index, DRIVER_NODES, -driver_N, Z)
    _spread(F, index, ENGINE_NODES, -engine_N, Z)

    u, R = solve(K, F, fixed)
    sag, sag_node = _vertical_report(u, index, names, coords)
    total_N = driver_N + engine_N

    result = {
        "case": "bending",
        "driver_N": driver_N,
        "engine_N": engine_N,
        "total_load_N": total_N,
        "max_sag_mm": sag,
        "max_sag_node": sag_node,
        "bending_stiffness_N_per_mm": total_N / abs(sag),
        "force_balance_N": max(abs(R[ax::DOF].sum() + F[ax::DOF].sum()) for ax in range(3)),
        "u": u, "coords": coords, "names": names, "index": index, "K": K, "F": F,
    }
    if verbose:
        print(f"  loads         driver {driver_N:.0f} N + powertrain {engine_N:.0f} N "
              f"= {total_N:.0f} N at {g_factor:g} g")
        print(f"  max sag       {sag:9.4f} mm at {sag_node}")
        print(f"  stiffness     {result['bending_stiffness_N_per_mm']:9.1f} N/mm")
        print(f"  check         support force balance {result['force_balance_N']:.1e} N")
    return result


# ----------------------------------------------------------------------------
# Load case 3: front bulkhead
# ----------------------------------------------------------------------------
def bulkhead_test(data, load_N=10000.0, verbose=True):
    """
    Rearward push spread over the impact-attenuator mounting face; frame held at
    the rear pickups (same convention as the torsion case).
    """
    names, index, coords, K = build_model(data)
    _check_nodes(index, BULKHEAD_NODES, "bulkhead")

    fixed = []
    for n in data["rear_pickups"]:
        fixed += node_dofs(index[n], DOF)[:3]

    F = np.zeros(K.shape[0])
    _spread(F, index, BULKHEAD_NODES, +load_N, X)   # +x is rearward

    u, R = solve(K, F, fixed)
    ux = u.reshape(-1, DOF)[:, X]
    face = np.mean([ux[index[n]] for n in BULKHEAD_NODES])

    result = {
        "case": "bulkhead",
        "load_N": load_N,
        "face_deflection_mm": face,
        "stiffness_N_per_mm": load_N / abs(face),
        "per_node_mm": {n: ux[index[n]] for n in BULKHEAD_NODES},
        "force_balance_N": max(abs(R[ax::DOF].sum() + F[ax::DOF].sum()) for ax in range(3)),
        "u": u, "coords": coords, "names": names, "index": index, "K": K, "F": F,
    }
    if verbose:
        print(f"  load          {load_N:.0f} N rearward over {len(BULKHEAD_NODES)} bulkhead nodes")
        print(f"  face moves    {face:9.4f} mm rearward")
        print(f"  stiffness     {result['stiffness_N_per_mm']:9.1f} N/mm")
        print(f"  check         support force balance {result['force_balance_N']:.1e} N")
    return result
