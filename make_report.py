"""
make_report.py: produces the sensitivity table as a PDF

    python make_report.py                     # full sweep, all legal tubes
    python make_report.py --min-wall 1.2        # only weldable walls
    python make_report.py --max-tubes 16        # faster, coarser

Writes results/sensitivity_table.pdf. One row per tube group, sorted by what a
gram spent there buys.
"""
import argparse
import time

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle)

from fesolver import sweep as sweep_mod
from fesolver.chassis import torsion_test
from fesolver.postprocess import energy_by_group, strain_energy
from fesolver.optimise import FREE_MASS_G
from fesolver.sweep import build_frame, default_spec, one_at_a_time


def collect(max_tubes, min_wall):
    sweep_mod.MIN_WALL = min_wall

    data, _ = build_frame()
    base_r = torsion_test(data, verbose=False)
    rows_e, _ = strain_energy(data, base_r)
    energy = {g["group"]: g["fraction"] for g in energy_by_group(rows_e)}
    mass_by_group = {}
    for m in data["members"]:
        mass_by_group[m["group"]] = mass_by_group.get(m["group"], 0.0) + m["mass"]

    rows = list(one_at_a_time(max_tubes=max_tubes or None, progress=False))
    base = rows[0]

    per_group = {}
    for r in rows[1:]:
        g = per_group.setdefault(r["group"], {"group": r["group"], "free": None, "value": None})
        if r["d_stiffness"] <= 0:
            continue
        if r["d_mass_g"] <= FREE_MASS_G:
            if g["free"] is None or r["d_stiffness"] > g["free"]["d_stiffness"]:
                g["free"] = r
        else:
            if g["value"] is None or r["Nm_per_deg_per_g"] > g["value"]["Nm_per_deg_per_g"]:
                g["value"] = r

    out = []
    for group, g in per_group.items():
        spec = default_spec()[group]
        out.append({
            "group": group,
            "current": f"{spec[0]:g}x{spec[1]:g}",
            "mass": mass_by_group.get(group, 0.0),
            "energy": energy.get(group, 0.0),
            "free": g["free"],
            "value": g["value"],
        })
    out.sort(key=lambda r: -(r["value"]["Nm_per_deg_per_g"] if r["value"] else 0))
    return base, out


def build_pdf(base, groups, path, min_wall):
    doc = SimpleDocTemplate(path, pagesize=landscape(A4),
                            leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm,
                            title="FS chassis torsional stiffness sensitivity")
    ss = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=ss["Normal"], fontSize=7.6, leading=9.6)
    story = []

    story.append(Paragraph(
        "Formula Student spaceframe: torsional stiffness sensitivity by tube group",
        ParagraphStyle("t", parent=ss["Title"], fontSize=15, spaceAfter=2)))
    story.append(Paragraph(
        f"Baseline {base['stiffness_Nm_per_deg']:.0f} Nm/deg at {base['mass_kg']:.2f} kg "
        f"({base['stiffness_per_kg']:.1f} Nm/deg/kg). Load case: 408.6 Nm applied at the "
        f"front axle pickups, all three translations fixed at the four rear pickups. "
        f"Every row is a rules-legal frame from the T3 catalogue"
        + (f", walls &ge; {min_wall:g} mm." if min_wall else "."), small))
    story.append(Spacer(1, 5))

    head = ["Tube group", "Current", "Mass\n(kg)", "Strain\nenergy",
            "Free upgrade\n(no extra mass)", "&Delta;Nm/deg",
            "Best value\nupgrade", "+g", "&Delta;Nm/deg", "Nm/deg\nper g", "Rank"]
    body = [[Paragraph(f"<b>{h.replace(chr(10), '<br/>')}</b>", small) for h in head]]

    rank = 0
    for r in groups:
        v, f = r["value"], r["free"]
        rank += 1
        body.append([
            Paragraph(r["group"], small),
            r["current"],
            f"{r['mass']:.2f}",
            f"{100*r['energy']:.1f}%",
            f"{f['od']:g}x{f['wall']:g}" if f else "-",
            f"{f['d_stiffness']:+.1f}" if f else "-",
            f"{v['od']:g}x{v['wall']:g}" if v else "-",
            f"{v['d_mass_g']:.0f}" if v else "-",
            f"{v['d_stiffness']:+.1f}" if v else "-",
            f"{v['Nm_per_deg_per_g']:.3f}" if v else "-",
            str(rank) if v else "-",
        ])

    t = Table(body, repeatRows=1, hAlign="LEFT",
              colWidths=[33*mm, 17*mm, 14*mm, 15*mm, 30*mm, 19*mm, 26*mm, 13*mm, 19*mm, 18*mm, 12*mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("TEXTCOLOR", (4, 1), (5, -1), colors.HexColor("#1a6b2f")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>How to read this.</b> <i>Strain energy</i> is the share of the frame's elastic energy "
        "held by that group under torsion: a group holding 5% cannot buy more than 5%, however much "
        "steel goes into it. <i>Free upgrade</i> is a different catalogue tube of the same or lower "
        "mass that is stiffer, legal because Table 4 sets minimum wall, area and second moment but "
        "not a diameter, and I grows with OD<super>4</super> while mass grows with area. "
        "<i>Best value upgrade</i> is the largest Nm/deg per gram added. Changes are measured one "
        "group at a time against the baseline, so they are not additive: groups share load paths, "
        "and combining them is what run_optimise.py does.", small))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Method.</b> 3D Euler-Bernoulli beam elements, 6 DOF per node, written from scratch in "
        "NumPy. Validated against closed-form cantilever solutions (~1e-10) and cross-checked "
        "against PyNite, a third-party solver (~1e-12). Buckling by Euler above the transition "
        "slenderness and the Johnson parabola below it, against welded yield strength. "
        "Geometry is rules-informed and verified against section T3; cockpit templates, Percy "
        "clearance, roll hoop bend geometry and SES approval are out of scope, so this is not a "
        "scrutineering result.", small))
    doc.build(story)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sensitivity_table.pdf")
    ap.add_argument("--max-tubes", type=int, default=0, help="0 = every legal tube")
    ap.add_argument("--min-wall", type=float, default=0.0)
    args = ap.parse_args()

    t0 = time.time()
    base, groups = collect(args.max_tubes, args.min_wall)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    build_pdf(base, groups, args.out, args.min_wall)
    print(f"wrote {args.out} ({len(groups)} groups, {time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
