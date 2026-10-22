"""Block 8: strain energy breakdown."""
import os
import numpy as np
from _util import rel, run_all
from fesolver.chassis import load_frame, torsion_test
from fesolver.loadcases import bending_test, bulkhead_test
from fesolver.postprocess import strain_energy, energy_by_group, check_energy_balance

HERE = os.path.dirname(os.path.abspath(__file__))
data = load_frame(os.path.join(HERE, "..", "frames", "frame_at_node.json"))

def cases():
    return [torsion_test(data, verbose=False),
            bending_test(data, verbose=False),
            bulkhead_test(data, verbose=False)]

def test_energy_equals_work_done_by_the_loads():
    """Exact identity for a linear solve: sum of member energies = 0.5 F.u."""
    for r in cases():
        rows, total = strain_energy(data, r)
        assert check_energy_balance(r, total) < 1e-10

def test_every_member_stores_non_negative_energy():
    for r in cases():
        rows, _ = strain_energy(data, r)
        assert all(x["energy"] >= -1e-12 for x in rows)

def test_shares_sum_to_one():
    for r in cases():
        rows, _ = strain_energy(data, r)
        assert rel(sum(x["fraction"] for x in rows), 1.0) < 1e-12
        assert rel(sum(g["fraction"] for g in energy_by_group(rows)), 1.0) < 1e-12

def test_shares_do_not_depend_on_load_size():
    a, _ = strain_energy(data, torsion_test(data, force=500.0, verbose=False))
    b, tb = strain_energy(data, torsion_test(data, force=1000.0, verbose=False))
    ga, gb = energy_by_group(a), energy_by_group(b)
    for x, y in zip(ga, gb):
        assert x["group"] == y["group"]
        assert rel(x["fraction"], y["fraction"]) < 1e-9

def test_energy_scales_with_load_squared():
    _, u1 = strain_energy(data, torsion_test(data, force=500.0, verbose=False))
    _, u2 = strain_energy(data, torsion_test(data, force=1000.0, verbose=False))
    assert rel(u2, 4 * u1) < 1e-9

if __name__ == "__main__":
    run_all(globals())
