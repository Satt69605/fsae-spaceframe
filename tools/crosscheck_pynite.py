"""
Optional: rebuild the same model in PyNite (an established open-source 3D frame
solver) and compare. Agreement means the solver is right.

    pip install PyNiteFEA
    python tools/crosscheck_pynite.py frames/frame_at_node.json
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Pynite import FEModel3D
from fesolver.chassis import load_frame, torsion_test

path = sys.argv[1]
d = json.load(open(path))
m = FEModel3D()
for mem in d["members"]:
    for n in (mem["n1"], mem["n2"]):
        if n not in m.nodes:
            m.add_node(n, *d["nodes"][n])
m.add_material("steel", d["material"]["E"], d["material"]["G"], d["material"]["nu"], d["material"]["rho"])
for i, mem in enumerate(d["members"]):
    m.add_section(f"s{i}", mem["area"], mem["I"], mem["I"], mem["J"])
    m.add_member(f"M{i}", mem["n1"], mem["n2"], "steel", f"s{i}")
for n in d["rear_pickups"]:
    m.def_support(n, True, True, True, False, False, False)
L = [n for n in d["front_pickups"] if n.endswith("_L")]
R = [n for n in d["front_pickups"] if n.endswith("_R")]
for n in L: m.add_node_load(n, "FZ", 500.0)
for n in R: m.add_node_load(n, "FZ", -500.0)
m.analyze_linear(check_statics=False, log=False)

dz = sum(m.nodes[n].DZ["Combo 1"] for n in L) / len(L) - sum(m.nodes[n].DZ["Combo 1"] for n in R) / len(R)
sp = sum(d["nodes"][n][1] for n in L) / len(L) - sum(d["nodes"][n][1] for n in R) / len(R)
T = sum(500.0 * d["nodes"][n][1] for n in L) - sum(500.0 * d["nodes"][n][1] for n in R)
k_pynite = (T / 1000) / math.degrees(math.atan(dz / sp))
k_ours = torsion_test(load_frame(path), verbose=False)["stiffness_Nm_per_deg"]
print(f"solver {k_ours:.4f} Nm/deg")
print(f"PyNite {k_pynite:.4f} Nm/deg")
print(f"relative difference {abs(k_ours - k_pynite) / k_pynite:.1e}")
