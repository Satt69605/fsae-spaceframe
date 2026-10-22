import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def rel(fe, hand):
    return abs(fe - hand) / abs(hand)

def run_all(module_globals):
    """Lets each test file run with plain `python tests/test_x.py` (no pytest needed)."""
    tests = [(n, f) for n, f in module_globals.items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"  PASS  {name}")
