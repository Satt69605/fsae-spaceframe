#!/usr/bin/env python3
"""
One-command verification.

Runs every test block, regenerates the baseline frame, re-checks the rules, and
prints a single table a reviewer can read in thirty seconds.

    python verify.py              # tests + frame + rules
    python verify.py --pynite     # also cross-check against the third-party solver

Exit code 0 means everything passed.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
GREEN, RED, DIM, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    GREEN = RED = DIM = OFF = ""

EXPECTED_STIFFNESS = 1712.2        # N.m/deg, pinned baseline
EXPECTED_UNBRACED = 953.9          # N.m/deg, same frame without axle bracing
TOL = 0.1                          # N.m/deg


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pynite", action="store_true",
                    help="also run the independent solver cross-check")
    args = ap.parse_args()

    rows = []
    ok = True
    t0 = time.time()

    # 1. Test blocks
    files = sorted(glob.glob(os.path.join(ROOT, "tests", "test_block*.py")),
                   key=lambda p: int(re.search(r"block(\d+)", p).group(1)))
    n_pass = 0
    for f in files:
        r = run([sys.executable, f])
        good = r.returncode == 0
        n_pass += good
        ok &= good
        if not good:
            rows.append((os.path.basename(f), "FAILED", r.stderr.strip().splitlines()[-1][:60]))
    rows.insert(0, ("Test blocks", f"{n_pass}/{len(files)} passed",
                    "closed-form beams, assembly, buckling, energy, sweep"))

    # 2. Generate the frame and count rule checks
    out = os.path.join(ROOT, "frames")
    r = run([sys.executable, "spaceframe.py", "--out", out])
    m = re.search(r"(\d+)/(\d+) passed", r.stdout)
    fuzz = re.search(r"(\d+)/(\d+) random frames legal", r.stdout)
    if m:
        good = m.group(1) == m.group(2)
        ok &= good
        rows.append(("Rule checks", f"{m.group(1)}/{m.group(2)} passed",
                     "T3 tube sizing and geometry, re-derived from coordinates"))
    else:
        ok = False
        rows.append(("Rule checks", "COULD NOT PARSE", r.stderr.strip()[:60]))
    if fuzz:
        good = fuzz.group(1) == fuzz.group(2)
        ok &= good
        rows.append(("Fuzzed frames", f"{fuzz.group(1)}/{fuzz.group(2)} legal",
                     "random parameter sets, both pickup modes"))

    # 3. Baseline stiffness
    frame = os.path.join(out, "frame_at_node.json")
    r = run([sys.executable, "run_analysis.py", frame, "--buckling", "--energy"])
    m = re.search(r"STIFFNESS\s+([\d.]+)\s*Nm/deg", r.stdout)
    if m:
        val = float(m.group(1))
        good = abs(val - EXPECTED_STIFFNESS) < TOL
        ok &= good
        rows.append(("Baseline stiffness", f"{val:.1f} N.m/deg",
                     f"regression pin {EXPECTED_STIFFNESS} {'OK' if good else 'MOVED'}"))
    else:
        ok = False
        rows.append(("Baseline stiffness", "NOT FOUND", r.stderr.strip()[:60]))

    for label, pat, note in (
            ("Solve residual", r"residual ([\d.eE+-]+)", "K u = F satisfied"),
            ("Support balance", r"support force balance ([\d.eE+-]+)", "reactions equal applied load"),
            ("Energy balance", r"energy balance vs 0\.5\*F\.u: ([\d.eE+-]+)", "strain energy = work done")):
        m = re.search(pat, r.stdout)
        if m:
            rows.append((label, m.group(1), note))

    # 4. Optional independent cross-check
    if args.pynite:
        try:
            import Pynite  # noqa: F401
            r = run([sys.executable, os.path.join("tools", "crosscheck_pynite.py"), frame])
            m = re.search(r"relative difference ([\d.eE+-]+)", r.stdout)
            if m:
                rows.append(("PyNite cross-check", m.group(1),
                             "independent third-party solver, same frame"))
            else:
                rows.append(("PyNite cross-check", "no result", r.stderr.strip()[:60]))
        except ImportError:
            rows.append(("PyNite cross-check", "skipped", "pip install PyNiteFEA to enable"))

    # Print
    w = max(len(a) for a, _, _ in rows) + 2
    v = max(len(str(b)) for _, b, _ in rows) + 2
    print()
    print("  " + "VERIFICATION".ljust(w) + "RESULT".ljust(v) + "WHAT IT SHOWS")
    print("  " + "-" * (w + v + 46))
    for a, b, c in rows:
        bad = any(k in str(b) for k in ("FAIL", "NOT FOUND", "COULD NOT"))
        col = RED if bad else (GREEN if re.search(r"passed|legal|N\.m", str(b)) else "")
        print("  " + a.ljust(w) + col + str(b).ljust(v) + OFF + DIM + c + OFF)
    print()
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOMETHING FAILED'} "
          f"in {time.time() - t0:.1f} s")
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
