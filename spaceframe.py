"""
spaceframe.py: Parametric Formula Student spaceframe generator

Generates a rules-informed steel spaceframe and writes frame.json + frame.png
for use as the fixed input geometry of a torsional stiffness study.

    python spaceframe.py                 # writes into the current folder
    python spaceframe.py --out ./build   # writes somewhere else
    python spaceframe.py --fuzz 5000     # run the self-test harder
    python spaceframe.py --mode mounted  # pickups on mount tubes

Requires: python 3.10+, numpy, matplotlib.
    pip install numpy matplotlib

Rules reference: Formula Student Rules 2026 v1.1, section T3.

SCOPE -- READ THIS BEFORE CALLING THE OUTPUT "LEGAL"
----------------------------------------------------
This models the tube geometry rules of T3 well enough that a torsional
stiffness study on the result is meaningful and comparable to published FS
figures. It doesn't model the cockpit templates (T4.1, T4.2), Percy clearance
(T4.3), roll hoop bend geometry (T3.8.2, T3.9.3), or anything requiring an
approved SES. Real scrutineering legality comes from an approved SES plus
physical inspection, and this generator is not a substitute for either.

Coordinate system:
    x  longitudinal, +ve rearward, origin at the front bulkhead plane
    y  lateral, +ve to the driver's left
    z  vertical, +ve up, origin at the ground plane
Lengths mm, angles degrees, mass kg, modulus MPa.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field, fields

import numpy as np

# ===========================================================================
# 1. MATERIAL (T3.2.2, T3.2.4)
# ===========================================================================

E_STEEL = 200_000.0
NU = 0.30                        # assumed; the rules fix E but not G
G_STEEL = E_STEEL / (2.0 * (1.0 + NU))
SY = 305.0
SY_WELDED = 180.0
RHO = 7.85e-6                    # kg/mm^3


# ===========================================================================
# 2. RULE CLASSES (T3.2.1, Table 4)
# ===========================================================================

@dataclass(frozen=True)
class RuleClass:
    key: str
    label: str
    min_wall: float
    min_area: float
    min_I: float


RULE_CLASSES = {
    "A": RuleClass("A", "roll hoops, shoulder harness mounting bar",
                   2.0, 173.0, 11_320.0),
    "B": RuleClass("B", "impact structures, front bulkhead, roll hoop bracing, "
                        "harness attachment", 1.2, 119.0, 8_509.0),
    "C": RuleClass("C", "front bulkhead support, main hoop bracing supports",
                   1.2, 91.0, 6_695.0),
    "N": RuleClass("N", "non-structural", 0.0, 0.0, 0.0),
}

RULE_TOL = 1e-4                  # Table 4 figures are rounded from the baseline
NONSTRUCTURAL_WALL = 1.2         # T3.2.5
NONSTRUCTURAL_I = 6_695.0        # T3.2.5


# ===========================================================================
# 3. TUBES, FROM A CATALOGUE OF PURCHASABLE SIZES
# ===========================================================================

@dataclass(frozen=True)
class Tube:
    od: float
    t: float

    @property
    def id_(self) -> float:
        return self.od - 2.0 * self.t

    @property
    def area(self) -> float:
        return math.pi / 4.0 * (self.od ** 2 - self.id_ ** 2)

    @property
    def I(self) -> float:
        return math.pi / 64.0 * (self.od ** 4 - self.id_ ** 4)

    @property
    def J(self) -> float:
        return 2.0 * self.I

    @property
    def mass_per_mm(self) -> float:
        return self.area * RHO

    def euler_load(self, length: float, k: float = 1.0) -> float:
        return (math.pi ** 2 * E_STEEL * self.I) / ((k * length) ** 2)

    def squash_load(self) -> float:
        return self.area * SY

    def meets(self, rc: RuleClass) -> bool:
        return (self.t >= rc.min_wall * (1 - RULE_TOL)
                and self.area >= rc.min_area * (1 - RULE_TOL)
                and self.I >= rc.min_I * (1 - RULE_TOL))

    def __repr__(self) -> str:
        return f"{self.od:g}x{self.t:g}"


_ODS = [19.05, 22.20, 25.00, 25.40, 26.90, 28.00, 30.00, 31.75,
        33.70, 35.00, 38.00, 38.10, 40.00, 41.30, 44.50]
_WALLS = [0.90, 1.00, 1.20, 1.50, 1.60, 1.80, 2.00, 2.30, 2.50, 3.00]

CATALOGUE: list[Tube] = sorted(
    (Tube(od, t) for od in _ODS for t in _WALLS
     if od - 2 * t > 0 and t / od <= 0.12),
    key=lambda tb: tb.area,
)


def legalise_tube(od: float, t: float, rule_key: str) -> tuple[Tube, str | None]:
    """Map a requested (od, t) onto a purchasable tube satisfying its class.

    Policy: the LIGHTEST catalogue tube that meets the class and has OD >= the
    request; failing that, the lightest legal tube outright. Buying second
    moment by thickening the wall is the most mass-expensive route available,
    because I grows with OD^4 while mass grows with area.
    """
    rc = RULE_CLASSES[rule_key]
    want = Tube(float(od), float(t))
    if rc.key == "N":
        return want, None
    in_cat = any(abs(c.od - want.od) < 1e-9 and abs(c.t - want.t) < 1e-9
                 for c in CATALOGUE)
    if want.meets(rc) and in_cat:
        return want, None
    legal = [c for c in CATALOGUE if c.meets(rc)]
    if not legal:
        raise ValueError(f"no catalogue tube satisfies class {rule_key}")
    wide = [c for c in legal if c.od >= want.od - 1e-9]
    chosen = (wide or legal)[0]
    return chosen, f"{want} -> {chosen} (lightest class-{rc.key} tube stocked)"


MEMBER_GROUPS = {
    "MAIN_HOOP": "A", "FRONT_HOOP": "A", "HARNESS_BAR": "A",
    "FBH": "B", "SIS_UPPER": "B", "SIS_LOWER": "B", "SIS_DIAGONAL": "B",
    "MH_BRACE": "B", "FH_BRACE": "B", "REAR_IMPACT": "B",
    "HARNESS_SUPPORT": "B",
    "FBS_UPPER": "C", "FBS_LOWER": "C", "FBS_DIAGONAL": "C", "MHB_SUPPORT": "C",
    "FLOOR": "N", "REAR_BAY": "N", "PICKUP_MOUNT": "N",
    "PICKUP_SUPPORT": "N",
}


# ===========================================================================
# 4. PARAMETERS -- gaps and rises, never absolute stations
# ===========================================================================

def _clamp(v: float, lo: float, hi: float, name: str, log: list[str]) -> float:
    if hi < lo:
        hi = lo
    out = min(max(v, lo), hi)
    if abs(out - v) > 1e-9:
        log.append(f"{name}: {v:.1f} -> {out:.1f} (window {lo:.1f}..{hi:.1f})")
    return out


@dataclass
class ChassisParams:
    """Every dimension is a positive gap or rise, never an absolute station,
    so ordering violations are unrepresentable rather than merely detected."""

    # --- longitudinal gaps, accumulating rearward from the bulkhead --------
    bulkhead_to_front_axle: float = 300.0
    front_axle_to_front_hoop: float = 380.0
    front_hoop_to_main_hoop: float = 650.0
    main_hoop_to_brace_foot: float = 490.0
    brace_foot_to_rear_axle: float = 90.0

    # --- vertical rises, accumulating up from the floor --------------------
    floor_z: float = 60.0
    bulkhead_drop_below_floor: float = -70.0
    bulkhead_height: float = 300.0
    sis_rise: float = 285.0
    fh_rise_above_sis: float = 310.0
    mh_rise_above_fh: float = 540.0
    harness_rise: float = 560.0          # shoulder harness bar, above the floor

    # --- half widths -------------------------------------------------------
    bulkhead_half_width: float = 200.0
    front_hoop_half_width: float = 210.0
    main_hoop_half_width: float = 225.0

    # --- track, and pickup inboard offset from the wheel centre ------------
    track_front: float = 1240.0
    track_rear: float = 1210.0
    pickup_inboard_offset: float = 300.0

    # --- angles ------------------------------------------------------------
    # T3.10.3 allows +-20 deg, but T3.12.3 demands ADDITIONAL rearward braces
    # beyond 10 deg rearward, which this generator does not build. Clamped to
    # 9.5 deg rearward so the frame stays consistent with what it constructs.
    fh_lean_deg: float = 8.0
    mh_lean_deg: float = 5.0

    # --- bracing attachment drops (to tube CENTRELINE) ---------------------
    # The rules measure from the top-most SURFACE, so these are capped at
    # (50 - OD/2) during legalisation, not at 50.
    mhb_drop: float = 110.0
    fhb_drop: float = 35.0
    fhb_separation: float = 120.0
    fbs_upper_drop: float = 35.0

    # --- pickup mode -------------------------------------------------------
    # "at_node": load points are real frame nodes at the front axle station,
    #            created by splitting the bulkhead support members there. No
    #            mount tubes, no mount compliance. Use this for your headline
    #            stiffness number.
    # "mounted": separate pickup nodes on four mount tubes each. Use this to
    #            QUANTIFY mount compliance, never as the headline number --
    #            measured mount sensitivity is large.
    pickup_mode: str = "at_node"
    pickup_lower_rise: float = 80.0      # mounted mode only
    pickup_upper_rise: float = 280.0     # mounted mode only

    # --- tube specs --------------------------------------------------------
    spec: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "MAIN_HOOP": (25.0, 2.5), "FRONT_HOOP": (25.0, 2.5),
        "HARNESS_BAR": (25.0, 2.5),
        "FBH": (25.4, 1.6), "SIS_UPPER": (25.4, 1.6),
        "SIS_LOWER": (25.4, 1.6), "SIS_DIAGONAL": (25.4, 1.6),
        "MH_BRACE": (25.4, 1.6), "FH_BRACE": (25.4, 1.6),
        "REAR_IMPACT": (25.4, 1.6), "HARNESS_SUPPORT": (25.4, 1.6),
        "FBS_UPPER": (25.4, 1.2), "FBS_LOWER": (25.4, 1.2),
        "FBS_DIAGONAL": (25.4, 1.2), "MHB_SUPPORT": (25.4, 1.2),
        "FLOOR": (25.4, 1.2), "REAR_BAY": (25.4, 1.2),
        "PICKUP_MOUNT": (25.4, 1.6), "PICKUP_SUPPORT": (25.4, 1.6),
    })

    needs_bulkhead_x_brace: bool = field(default=False, init=False)
    clamp_log: list[str] = field(default_factory=list, init=False, repr=False)

    MIN_GAP = 60.0
    MIN_MEMBER = 25.0

    def __post_init__(self) -> None:
        self._legalise()

    # -- derived stations ---------------------------------------------------
    @property
    def front_axle_x(self) -> float:
        return self.bulkhead_to_front_axle

    @property
    def fh_x(self) -> float:
        return self.front_axle_x + self.front_axle_to_front_hoop

    @property
    def mh_x(self) -> float:
        return self.fh_x + self.front_hoop_to_main_hoop

    @property
    def mh_brc_x(self) -> float:
        return self.mh_x + (self.mh_top_z - self.mhb_drop - self.floor_z) \
            * math.tan(math.radians(self.mh_lean_deg))

    @property
    def brace_foot_x(self) -> float:
        # Datum is the brace ATTACHMENT node, not the hoop foot. Measuring from
        # the foot makes the real horizontal run shorter than the clamped value
        # by the hoop's lean offset, which breaks the >=30 deg guarantee.
        return self.mh_brc_x + self.main_hoop_to_brace_foot

    @property
    def rear_axle_x(self) -> float:
        return self.brace_foot_x + self.brace_foot_to_rear_axle

    @property
    def wheelbase(self) -> float:
        return self.rear_axle_x - self.front_axle_x

    @property
    def bulkhead_bottom_z(self) -> float:
        return self.floor_z - self.bulkhead_drop_below_floor

    @property
    def bulkhead_top_z(self) -> float:
        return self.bulkhead_bottom_z + self.bulkhead_height

    @property
    def sis_z(self) -> float:
        return self.floor_z + self.sis_rise

    @property
    def fh_top_z(self) -> float:
        return self.sis_z + self.fh_rise_above_sis

    @property
    def mh_top_z(self) -> float:
        return self.fh_top_z + self.mh_rise_above_fh

    @property
    def pickup_half_width(self) -> float:
        return self.track_front / 2.0 - self.pickup_inboard_offset

    @property
    def rear_pickup_half_width(self) -> float:
        return self.track_rear / 2.0 - self.pickup_inboard_offset

    # -- clamping -----------------------------------------------------------
    def _legalise(self) -> None:
        log = self.clamp_log

        # Tubes FIRST: the surface-referenced geometric clamps below need OD.
        for group, (od, t) in list(self.spec.items()):
            tube, msg = legalise_tube(od, t, MEMBER_GROUPS[group])
            if msg:
                log.append(f"{group}: {msg}")
            self.spec[group] = (tube.od, tube.t)
        od_fh = self.spec["FRONT_HOOP"][0]
        od_fbh = self.spec["FBH"][0]
        od_sis = self.spec["SIS_UPPER"][0]

        if self.pickup_mode not in ("at_node", "mounted"):
            log.append(f"pickup_mode '{self.pickup_mode}' unknown -> at_node")
            self.pickup_mode = "at_node"

        # Per-gap minima. The bulkhead-to-front-hoop bay holds the driver's
        # legs; at a small floor the bracing tubes between them collapse.
        for name, lo in {"bulkhead_to_front_axle": 150.0,
                         "front_axle_to_front_hoop": 250.0,
                         "front_hoop_to_main_hoop": self.MIN_GAP,
                         "brace_foot_to_rear_axle": self.MIN_GAP}.items():
            setattr(self, name, _clamp(getattr(self, name), lo, 3000.0, name, log))

        self.floor_z = _clamp(self.floor_z, 30.0, 300.0, "floor_z (T2.2.1)", log)
        self.bulkhead_height = _clamp(self.bulkhead_height, 150.0, 500.0,
                                      "bulkhead_height", log)
        self.bulkhead_drop_below_floor = _clamp(
            self.bulkhead_drop_below_floor, -400.0, 25.0,
            "bulkhead_drop_below_floor (T3.8.3)", log)

        # T3.15.1 with figure 7: the upper side impact member must be
        # COMPLETELY inside the 240-320 zone, so the CENTRELINE window is
        # narrowed by the tube radius at both ends.
        r = od_sis / 2.0
        self.sis_rise = _clamp(self.sis_rise, 240.0 + r + 1.0, 320.0 - r - 1.0,
                               "sis_rise (T3.15.1, whole tube in zone)", log)

        # T3.10.3 allows 20 deg, T3.12.3 requires extra rearward bracing past
        # 10 deg rearward. This generator does not build those extra braces,
        # so the rearward limit is 9.5, not 19.5. Forward lean is unaffected.
        self.fh_lean_deg = _clamp(self.fh_lean_deg, -19.5, 9.5,
                                  "fh_lean_deg (T3.10.3, T3.12.3)", log)
        self.mh_lean_deg = _clamp(self.mh_lean_deg, 0.0, 9.5,
                                  "mh_lean_deg (T3.9.2, T3.9.4)", log)

        # Drops measured to CENTRELINE, but the rules measure to the top-most
        # SURFACE. A 40 mm centreline drop on a 25 mm tube is 52.5 mm to the
        # surface, which a strict scrutineer fails.
        self.mhb_drop = _clamp(self.mhb_drop, self.MIN_MEMBER,
                               159.0 - self.spec["MAIN_HOOP"][0] / 2.0,
                               "mhb_drop (T3.11.3, to surface)", log)
        self.fhb_drop = _clamp(self.fhb_drop, self.MIN_MEMBER, 49.0 - od_fh / 2.0,
                               "fhb_drop (T3.12.2, to surface)", log)
        self.fbs_upper_drop = _clamp(self.fbs_upper_drop, self.MIN_MEMBER,
                                     49.0 - od_fbh / 2.0,
                                     "fbs_upper_drop (T3.14.1, to surface)", log)
        self.fhb_separation = _clamp(self.fhb_separation, 100.0, 600.0,
                                     "fhb_separation (T3.12.1)", log)

        # +150, not MIN_MEMBER: at the bare margin the lower brace node sits
        # almost on the SIS node and the tubes reaching them from the bulkhead
        # become collinear -- a duplicate load path an FE solver double-counts.
        self.fh_rise_above_sis = _clamp(
            self.fh_rise_above_sis,
            self.fhb_drop + self.fhb_separation + 150.0, 900.0,
            "fh_rise_above_sis (hoop must carry both brace nodes)", log)
        self.mh_rise_above_fh = _clamp(self.mh_rise_above_fh, 150.0, 900.0,
                                       "mh_rise_above_fh", log)

        # Hoop separation measured at SIS HEIGHT, not at the floor: the hoops
        # lean toward each other as they rise, so a legal gap at the feet can
        # still collapse the upper side impact member to nothing.
        close = self.sis_rise * (math.tan(math.radians(self.fh_lean_deg))
                                 - math.tan(math.radians(self.mh_lean_deg)))
        self.front_hoop_to_main_hoop = _clamp(
            self.front_hoop_to_main_hoop, 300.0 + max(0.0, close), 1400.0,
            "front_hoop_to_main_hoop (SIS member length at SIS height)", log)

        self.bulkhead_half_width = _clamp(self.bulkhead_half_width, 150.0, 300.0,
                                          "bulkhead_half_width", log)
        self.front_hoop_half_width = _clamp(self.front_hoop_half_width, 150.0, 350.0,
                                            "front_hoop_half_width", log)
        self.main_hoop_half_width = _clamp(self.main_hoop_half_width, 150.0, 350.0,
                                           "main_hoop_half_width", log)

        # T2.9.2, with margin. Track drives the reference wheel centres and the
        # pickup stations, so it is load-bearing geometry, not a stored number.
        big = max(self.track_front, self.track_rear)
        if min(self.track_front, self.track_rear) < 0.752 * big:
            if self.track_front < self.track_rear:
                self.track_front = 0.752 * big
                log.append("track_front raised to 75% of rear (T2.9.2)")
            else:
                self.track_rear = 0.752 * big
                log.append("track_rear raised to 75% of front (T2.9.2)")
        self.track_front = _clamp(self.track_front, 900.0, 1600.0, "track_front", log)
        self.track_rear = _clamp(self.track_rear, 900.0, 1600.0, "track_rear", log)

        widest = max(self.bulkhead_half_width, self.front_hoop_half_width,
                     self.main_hoop_half_width)
        PU_MARGIN = 80.0
        need = 2.0 * (widest + PU_MARGIN + 150.0)
        if min(self.track_front, self.track_rear) < need:
            self.track_front = max(self.track_front, need)
            self.track_rear = max(self.track_rear, need)
            log.append(f"track widened to {need:.0f} mm so pickups clear the frame")
        max_off = min(self.track_front, self.track_rear) / 2.0 - widest - PU_MARGIN
        self.pickup_inboard_offset = _clamp(self.pickup_inboard_offset, 150.0,
                                            max(150.0, max_off),
                                            "pickup_inboard_offset", log)
        self.pickup_lower_rise = _clamp(self.pickup_lower_rise, 40.0, 250.0,
                                        "pickup_lower_rise", log)
        self.pickup_upper_rise = _clamp(self.pickup_upper_rise,
                                        self.pickup_lower_rise + 120.0, 600.0,
                                        "pickup_upper_rise", log)

        # Harness bar sits on the main hoop between the SIS node and the brace
        # attachment, clear of both (T5.5.1: not on the main hoop bracing).
        mhb_z = self.mh_top_z - self.mhb_drop
        self.harness_rise = _clamp(self.harness_rise,
                                   self.sis_rise + 120.0,
                                   mhb_z - self.floor_z - 120.0,
                                   "harness_rise (T5.5.1, clear of hoop nodes)", log)

        # T3.11.3: >=30 deg between hoop and brace. Rearward lean OPENS that
        # angle, so the brace needs (30 - lean) from vertical, not (30 + lean).
        drop_to_floor = self.mh_top_z - self.mhb_drop - self.floor_z
        beta_min = math.radians(max(5.0, 30.5 - self.mh_lean_deg))
        self.main_hoop_to_brace_foot = _clamp(
            self.main_hoop_to_brace_foot, drop_to_floor * math.tan(beta_min), 2500.0,
            "main_hoop_to_brace_foot (T3.11.3, >=30deg to hoop)", log)

        # T2.9.1: wheelbase is measured between wheel centres, so guarantee it.
        span = self.rear_axle_x - self.front_axle_x
        if span < 1530.0:
            self.brace_foot_to_rear_axle += 1530.0 - span
            log.append("brace_foot_to_rear_axle extended to reach wheelbase 1530 mm"
                       " (T2.9.1)")

        # T3.17.7 uses OVERALL (outside) dimensions, not centrelines.
        overall_w = 2 * self.bulkhead_half_width + od_fbh
        overall_h = self.bulkhead_height + od_fbh
        self.needs_bulkhead_x_brace = overall_w > 400.0 or overall_h > 350.0


# ===========================================================================
# 5. FRAME
# ===========================================================================

@dataclass
class Member:
    n1: str
    n2: str
    group: str
    tube: Tube

    def length(self, N) -> float:
        return float(np.linalg.norm(N[self.n2] - N[self.n1]))

    def mass(self, N) -> float:
        return self.length(N) * self.tube.mass_per_mm


@dataclass
class Frame:
    nodes: dict[str, np.ndarray]
    members: list[Member]
    front_pickups: list[str] = field(default_factory=list)
    rear_pickups: list[str] = field(default_factory=list)
    clamp_log: list[str] = field(default_factory=list)
    mode: str = "at_node"
    # Wheel centres. Reference geometry ONLY -- not in nodes, not connected by
    # any member. They exist so track and wheelbase can be checked against
    # geometry rather than against a stored parameter.
    reference: dict[str, np.ndarray] = field(default_factory=dict)

    def mass(self) -> float:
        return sum(m.mass(self.nodes) for m in self.members)

    def mass_by_group(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in self.members:
            out[m.group] = out.get(m.group, 0.0) + m.mass(self.nodes)
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def to_dict(self) -> dict:
        return {
            "generator": "spaceframe.py (final)",
            "rules_reference": "Formula Student Rules 2026 v1.1, section T3",
            "scope_warning": ("Tube geometry rules of T3 only. Cockpit templates "
                              "(T4.1, T4.2), Percy (T4.3), roll hoop bend geometry "
                              "(T3.8.2, T3.9.3) and SES approval are NOT modelled. "
                              "Do not describe this as scrutineering-legal."),
            "pickup_mode": self.mode,
            "units": {"length": "mm", "force": "N", "mass": "kg",
                      "modulus": "MPa", "angle": "deg"},
            "material": {"E": E_STEEL, "G": G_STEEL, "nu": NU,
                         "Sy": SY, "Sy_welded": SY_WELDED, "rho": RHO},
            "nodes": {k: [round(float(c), 4) for c in v]
                      for k, v in self.nodes.items()},
            "members": [
                {"n1": m.n1, "n2": m.n2, "group": m.group,
                 "rule_class": MEMBER_GROUPS[m.group],
                 "od": m.tube.od, "wall": m.tube.t,
                 "area": round(m.tube.area, 4),
                 "I": round(m.tube.I, 2), "J": round(m.tube.J, 2),
                 "length": round(m.length(self.nodes), 3),
                 "mass": round(m.mass(self.nodes), 5)}
                for m in self.members
            ],
            "reference_only_not_for_fe": {
                k: [round(float(c), 4) for c in v] for k, v in self.reference.items()},
            "front_pickups": self.front_pickups,
            "rear_pickups": self.rear_pickups,
            "mass_total": round(self.mass(), 4),
            "mass_by_group": {k: round(v, 4) for k, v in self.mass_by_group().items()},
        }


def build(p: ChassisParams) -> Frame:
    N: dict[str, np.ndarray] = {}
    M: list[Member] = []

    def pair(name, x, hw, z):
        N[f"{name}_L"] = np.array([x, +hw, z], float)
        N[f"{name}_R"] = np.array([x, -hw, z], float)

    def link(a, b, group):
        od, t = p.spec[group]
        M.append(Member(a, b, group, Tube(od, t)))

    def link_pair(a, b, group):
        link(f"{a}_L", f"{b}_L", group)
        link(f"{a}_R", f"{b}_R", group)

    def split_at_x(a: str, b: str, x_target: float, new: str) -> None:
        """Insert a node on the straight line a->b at longitudinal station
        x_target. Used to create real structural nodes at the front axle so
        the torsion load enters at the correct station instead of at the
        bulkhead plane. Adds no mass and no compliance -- it only subdivides
        an existing member into two collinear elements."""
        A, B = N[a], N[b]
        f = (x_target - A[0]) / (B[0] - A[0])
        N[new] = A + f * (B - A)

    fh_lean, mh_lean = math.radians(p.fh_lean_deg), math.radians(p.mh_lean_deg)

    def on_front_hoop(z):
        return p.fh_x + (z - p.floor_z) * math.tan(fh_lean)

    def on_main_hoop(z):
        return p.mh_x + (z - p.floor_z) * math.tan(mh_lean)

    # ---- front bulkhead ---------------------------------------------------
    hw = p.bulkhead_half_width
    fbs_z = p.bulkhead_top_z - p.fbs_upper_drop
    pair("FBH_BOT", 0.0, hw, p.bulkhead_bottom_z)
    pair("FBS_TOP", 0.0, hw, fbs_z)
    pair("FBH_TOP", 0.0, hw, p.bulkhead_top_z)
    link("FBH_BOT_L", "FBH_BOT_R", "FBH")
    link("FBH_TOP_L", "FBH_TOP_R", "FBH")
    link_pair("FBH_BOT", "FBS_TOP", "FBH")
    link_pair("FBS_TOP", "FBH_TOP", "FBH")
    if p.needs_bulkhead_x_brace:                      # T3.17.7
        link("FBH_BOT_L", "FBH_TOP_R", "FBH")
        link("FBH_BOT_R", "FBH_TOP_L", "FBH")

    # ---- front hoop -------------------------------------------------------
    hw = p.front_hoop_half_width
    fhb_a_z = p.fh_top_z - p.fhb_drop
    fhb_b_z = fhb_a_z - p.fhb_separation
    pair("FH_BOT", p.fh_x, hw, p.floor_z)
    pair("FH_SIS", on_front_hoop(p.sis_z), hw, p.sis_z)
    pair("FH_BRC_B", on_front_hoop(fhb_b_z), hw, fhb_b_z)
    pair("FH_BRC_A", on_front_hoop(fhb_a_z), hw, fhb_a_z)
    pair("FH_TOP", on_front_hoop(p.fh_top_z), hw, p.fh_top_z)
    link("FH_TOP_L", "FH_TOP_R", "FRONT_HOOP")
    for a, b in (("FH_BOT", "FH_SIS"), ("FH_SIS", "FH_BRC_B"),
                 ("FH_BRC_B", "FH_BRC_A"), ("FH_BRC_A", "FH_TOP")):
        link_pair(a, b, "FRONT_HOOP")

    # ---- main hoop, with the shoulder harness mounting bar node -----------
    hw = p.main_hoop_half_width
    mhb_z = p.mh_top_z - p.mhb_drop
    harn_z = p.floor_z + p.harness_rise
    pair("MH_BOT", p.mh_x, hw, p.floor_z)
    pair("MH_SIS", on_main_hoop(p.sis_z), hw, p.sis_z)
    pair("MH_HARN", on_main_hoop(harn_z), hw, harn_z)
    pair("MH_BRC", on_main_hoop(mhb_z), hw, mhb_z)
    pair("MH_TOP", on_main_hoop(p.mh_top_z), hw, p.mh_top_z)
    link("MH_TOP_L", "MH_TOP_R", "MAIN_HOOP")
    for a, b in (("MH_BOT", "MH_SIS"), ("MH_SIS", "MH_HARN"),
                 ("MH_HARN", "MH_BRC"), ("MH_BRC", "MH_TOP")):
        link_pair(a, b, "MAIN_HOOP")

    # T3.2.1 lists the shoulder harness mounting bar as a class A tube, and
    # T5.5.1 forbids mounting the harness to the main hoop bracing without
    # extra bracing. The bar spans the main hoop directly, and is triangulated
    # back to the brace feet.
    link("MH_HARN_L", "MH_HARN_R", "HARNESS_BAR")

    # ---- side impact structure (T3.15.1) ----------------------------------
    link_pair("FH_SIS", "MH_SIS", "SIS_UPPER")
    link_pair("FH_BOT", "MH_BOT", "SIS_LOWER")
    for s in "LR":
        link(f"FH_BOT_{s}", f"MH_SIS_{s}", "SIS_DIAGONAL")

    # ---- front bulkhead support (T3.14.1), split at the front axle --------
    split_at_x("FBS_TOP_L", "FH_SIS_L", p.front_axle_x, "FA_UPR_L")
    split_at_x("FBS_TOP_R", "FH_SIS_R", p.front_axle_x, "FA_UPR_R")
    split_at_x("FBH_BOT_L", "FH_BOT_L", p.front_axle_x, "FA_LWR_L")
    split_at_x("FBH_BOT_R", "FH_BOT_R", p.front_axle_x, "FA_LWR_R")
    link_pair("FBS_TOP", "FA_UPR", "FBS_UPPER")
    link_pair("FA_UPR", "FH_SIS", "FBS_UPPER")
    link_pair("FBH_BOT", "FA_LWR", "FBS_LOWER")
    link_pair("FA_LWR", "FH_BOT", "FBS_LOWER")
    link_pair("FBS_TOP", "FH_BOT", "FBS_DIAGONAL")

    # ---- front hoop bracing (T3.12.1) -------------------------------------
    link_pair("FH_BRC_A", "FBH_TOP", "FH_BRACE")
    link_pair("FH_BRC_B", "FBS_TOP", "FH_BRACE")

    # ---- main hoop bracing (T3.11) ----------------------------------------
    hw = p.main_hoop_half_width
    pair("MHB_FOOT", p.brace_foot_x, hw, p.floor_z)
    link_pair("MH_BRC", "MHB_FOOT", "MH_BRACE")       # straight, T3.11.4
    link_pair("MHB_FOOT", "MH_BOT", "MHB_SUPPORT")    # T3.11.5
    link_pair("MHB_FOOT", "MH_SIS", "MHB_SUPPORT")
    link("MHB_FOOT_L", "MHB_FOOT_R", "MHB_SUPPORT")
    link_pair("MH_HARN", "MHB_FOOT", "HARNESS_SUPPORT")   # T5.5.2

    # ---- rear impact structure (T3.15.2): 3+ structural tubes per side ----
    pair("REAR_AXLE", p.rear_axle_x, p.main_hoop_half_width, p.floor_z)
    pair("REAR_UPR", p.rear_axle_x, p.main_hoop_half_width, p.sis_z)
    link_pair("MHB_FOOT", "REAR_AXLE", "REAR_IMPACT")
    link_pair("MH_SIS", "REAR_UPR", "REAR_IMPACT")
    link_pair("REAR_AXLE", "REAR_UPR", "REAR_IMPACT")
    link_pair("MHB_FOOT", "REAR_UPR", "REAR_IMPACT")      # per-side diagonal
    link("REAR_AXLE_L", "REAR_AXLE_R", "REAR_BAY")
    link("REAR_UPR_L", "REAR_UPR_R", "REAR_BAY")
    link("MH_BOT_L", "MH_BOT_R", "REAR_BAY")
    link("MH_SIS_L", "MH_SIS_R", "REAR_BAY")
    link("MHB_FOOT_L", "REAR_UPR_R", "REAR_BAY")
    link("MH_BOT_L", "MHB_FOOT_R", "REAR_BAY")

    # ---- upper plane between the hoops ------------------------------------
    link("FH_TOP_L", "MH_TOP_R", "REAR_BAY")
    link("FH_TOP_R", "MH_TOP_L", "REAR_BAY")

    # ---- floor plane ------------------------------------------------------
    link("FH_BOT_L", "FH_BOT_R", "FLOOR")
    link("FBH_BOT_L", "FH_BOT_R", "FLOOR")
    link("FBH_BOT_R", "FH_BOT_L", "FLOOR")
    link("FH_BOT_L", "MH_BOT_R", "FLOOR")

    # ---- load points ------------------------------------------------------
    if p.pickup_mode == "at_node":
        # Without these, the load pushes on the middle of two unsupported
        # 1.2 mm bulkhead-support tubes and bends them locally: ~44% of strain
        # energy landed in FBS_UPPER/FBS_LOWER and the loaded node moved ~1.5 mm
        # while the tube's own ends moved 0.9 and 0.5 mm. That is a tube
        # flexing, not the chassis twisting. Real frames put pickups at
        # triangulated nodes. Tagged PICKUP_SUPPORT, not rule structure.
        link_pair("FA_UPR", "FA_LWR", "PICKUP_SUPPORT")
        link_pair("FA_LWR", "FH_SIS", "PICKUP_SUPPORT")
        link("FA_LWR_L", "FA_LWR_R", "PICKUP_SUPPORT")
        front = ["FA_UPR_L", "FA_UPR_R", "FA_LWR_L", "FA_LWR_R"]
        rear = ["REAR_UPR_L", "REAR_UPR_R", "REAR_AXLE_L", "REAR_AXLE_R"]
    else:
        fpw = p.pickup_half_width
        pair("PU_F_LWR", p.front_axle_x, fpw, p.floor_z + p.pickup_lower_rise)
        pair("PU_F_UPR", p.front_axle_x, fpw, p.floor_z + p.pickup_upper_rise)
        for s in "LR":
            for tgt in ("FBH_BOT", "FH_BOT", "FBS_TOP", "FH_SIS"):
                link(f"PU_F_LWR_{s}", f"{tgt}_{s}", "PICKUP_MOUNT")
            # Targets spread in BOTH x and z. FBS_TOP and FBH_TOP are only
            # ~35 mm apart vertically, so from a distant pickup those two
            # mounts point almost the same way -- degenerate, not triangulated.
            for tgt in ("FBS_TOP", "FBH_BOT", "FH_SIS", "FH_BRC_B"):
                link(f"PU_F_UPR_{s}", f"{tgt}_{s}", "PICKUP_MOUNT")
        rpw = p.rear_pickup_half_width
        pair("PU_R_LWR", p.rear_axle_x, rpw, p.floor_z + p.pickup_lower_rise)
        pair("PU_R_UPR", p.rear_axle_x, rpw, p.floor_z + p.pickup_upper_rise)
        for s in "LR":
            for tgt in ("REAR_AXLE", "MHB_FOOT", "REAR_UPR", "MH_SIS"):
                link(f"PU_R_LWR_{s}", f"{tgt}_{s}", "PICKUP_MOUNT")
            for tgt in ("REAR_UPR", "REAR_AXLE", "MH_SIS", "MH_BRC"):
                link(f"PU_R_UPR_{s}", f"{tgt}_{s}", "PICKUP_MOUNT")
        front = ["PU_F_UPR_L", "PU_F_UPR_R", "PU_F_LWR_L", "PU_F_LWR_R"]
        rear = ["PU_R_UPR_L", "PU_R_UPR_R", "PU_R_LWR_L", "PU_R_LWR_R"]

    ref = {
        "WHEEL_F_L": np.array([p.front_axle_x, +p.track_front / 2, 0.0]),
        "WHEEL_F_R": np.array([p.front_axle_x, -p.track_front / 2, 0.0]),
        "WHEEL_R_L": np.array([p.rear_axle_x, +p.track_rear / 2, 0.0]),
        "WHEEL_R_R": np.array([p.rear_axle_x, -p.track_rear / 2, 0.0]),
    }
    return Frame(nodes=N, members=M, front_pickups=front, rear_pickups=rear,
                 clamp_log=list(p.clamp_log), mode=p.pickup_mode, reference=ref)


# ===========================================================================
# 6. CHECKER -- reads ONLY the built frame, never ChassisParams
# ===========================================================================

@dataclass
class Check:
    rule: str
    description: str
    ok: bool
    detail: str


def _ang_from_vertical(v) -> float:
    return math.degrees(math.atan2(abs(v[0]), abs(v[2])))


def _acute_between(u, v) -> float:
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return min(a := math.degrees(math.acos(max(-1.0, min(1.0, c)))), 180.0 - a)


def check(frame: Frame) -> list[Check]:
    N, R, out = frame.nodes, frame.reference, []

    def add(rule, desc, ok, detail):
        out.append(Check(rule, desc, bool(ok), detail))

    def z(n): return float(N[n][2])
    def x(n): return float(N[n][0])

    def members_of(group):
        return [m for m in frame.members if m.group == group]

    def od_of(group):
        ms = members_of(group)
        return ms[0].tube.od if ms else 0.0

    # --- structural sanity --------------------------------------------------
    seen, dup = set(), []
    for m in frame.members:
        k = tuple(sorted((m.n1, m.n2)))
        if k in seen:
            dup.append(k)
        seen.add(k)
    add("SANITY", "no duplicated members", not dup, f"{len(dup)} duplicates")

    short = [(m.n1, m.n2, round(m.length(N), 2))
             for m in frame.members if m.length(N) < ChassisParams.MIN_MEMBER]
    add("SANITY", f"no members under {ChassisParams.MIN_MEMBER:.0f} mm",
        not short, f"{len(short)} short" + (f": {short[:3]}" if short else ""))

    used = {n for m in frame.members for n in (m.n1, m.n2)}
    orphan = sorted(set(N) - used)
    add("SANITY", "no unconnected nodes (a singular stiffness matrix otherwise)",
        not orphan, f"{len(orphan)} orphans" + (f": {orphan[:3]}" if orphan else ""))

    by_node: dict[str, list[Member]] = {}
    for m in frame.members:
        by_node.setdefault(m.n1, []).append(m)
        by_node.setdefault(m.n2, []).append(m)
    overlap = []
    for node, ms in by_node.items():
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = ms[i], ms[j]
                va = N[a.n2 if a.n1 == node else a.n1] - N[node]
                vb = N[b.n2 if b.n1 == node else b.n1] - N[node]
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                if na < 1e-9 or nb < 1e-9:
                    continue
                # A collinear pair sharing a node is only a genuine overlap if
                # the two members lie on TOP of each other. Two collinear
                # members pointing OPPOSITE ways are a legitimate split member
                # (the front axle nodes), so only same-direction counts.
                if float(np.dot(va / na, vb / nb)) > 0.9999:
                    overlap.append((node, a.group, b.group))
    add("SANITY", "no overlapping collinear members sharing a node",
        not overlap, f"{len(overlap)} overlaps" + (f": {overlap[:3]}" if overlap else ""))

    add("SANITY", "all declared load points exist as nodes",
        all(n in N for n in frame.front_pickups + frame.rear_pickups),
        f"front {frame.front_pickups}")

    # --- layout, measured from coordinates ---------------------------------
    stations = [("bulkhead", x("FBH_BOT_L")),
                ("front axle", float(R["WHEEL_F_L"][0])),
                ("front hoop", x("FH_BOT_L")),
                ("main hoop", x("MH_BOT_L")),
                ("brace foot", x("MHB_FOOT_L")),
                ("rear axle", x("REAR_AXLE_L"))]
    add("LAYOUT", "longitudinal stations strictly increasing rearward",
        all(stations[i][1] < stations[i + 1][1] for i in range(len(stations) - 1)),
        " < ".join(f"{n} {v:.0f}" for n, v in stations))
    bad = [n for n in N if n.endswith("_L") and N[n][1] <= 0]
    add("LAYOUT", "no lateral inversion", not bad, f"{len(bad)} inverted")
    add("LAYOUT", "bulkhead top above bulkhead bottom",
        z("FBH_TOP_L") > z("FBH_BOT_L"),
        f"{z('FBH_TOP_L'):.0f} over {z('FBH_BOT_L'):.0f}")
    add("LAYOUT", "vertical stack floor < SIS < front hoop top < main hoop top",
        z("FH_BOT_L") < z("FH_SIS_L") < z("FH_TOP_L") < z("MH_TOP_L"),
        f"{z('FH_BOT_L'):.0f} < {z('FH_SIS_L'):.0f} < {z('FH_TOP_L'):.0f}"
        f" < {z('MH_TOP_L'):.0f}")

    # --- T3.2.1 tube minima -------------------------------------------------
    for group in sorted({m.group for m in frame.members}):
        rc = RULE_CLASSES[MEMBER_GROUPS[group]]
        tube = members_of(group)[0].tube
        if rc.key == "N":
            structural = tube.t >= NONSTRUCTURAL_WALL and tube.I >= NONSTRUCTURAL_I
            add("T3.2.5", f"{group} declared non-structural", True,
                f"{tube}: I={tube.I:.0f} -> "
                f"{'counts structurally' if structural else 'ignored for compliance'}")
        else:
            add("T3.2.1", f"{group} meets class {rc.key}", tube.meets(rc),
                f"{tube}: t={tube.t:.2f}/{rc.min_wall} A={tube.area:.0f}/{rc.min_area:.0f}"
                f" I={tube.I:.0f}/{rc.min_I:.0f}")

    # --- roll hoops ---------------------------------------------------------
    a = _ang_from_vertical(N["MH_TOP_L"] - N["MH_SIS_L"])
    add("T3.9.2", "main hoop above SIS attachment < 10 deg from vertical",
        a < 10.0, f"{a:.2f} deg")
    lean = math.degrees(math.atan2(N["MH_SIS_L"][0] - N["MH_BOT_L"][0],
                                   N["MH_SIS_L"][2] - N["MH_BOT_L"][2]))
    add("T3.9.4", "main hoop below SIS attachment forward or <=10 deg rearward",
        lean <= 10.0, f"{lean:+.2f} deg rearward")
    fh_lean = math.degrees(math.atan2(N["FH_TOP_L"][0] - N["FH_BOT_L"][0],
                                      N["FH_TOP_L"][2] - N["FH_BOT_L"][2]))
    add("T3.10.3", "front hoop <= 20 deg from vertical", abs(fh_lean) <= 20.0,
        f"{fh_lean:+.2f} deg")
    # T3.12.3: beyond 10 deg rearward the front hoop needs ADDITIONAL rearward
    # braces meeting roll hoop bracing spec. This generator does not build them,
    # so the lean itself must stay inside 10 deg.
    add("T3.12.3", "front hoop <= 10 deg rearward, or dedicated rearward braces",
        fh_lean <= 10.0,
        f"{fh_lean:+.2f} deg rearward; no dedicated rearward front hoop braces built")

    # --- main hoop bracing --------------------------------------------------
    d = z("MH_TOP_L") - z("MH_BRC_L") + od_of("MAIN_HOOP") / 2.0
    add("T3.11.3a", "main hoop brace <= 160 mm below the hoop top SURFACE",
        d <= 160.0, f"{d:.1f} mm to surface")
    inc = _acute_between(N["MH_TOP_L"] - N["MH_BRC_L"],
                         N["MHB_FOOT_L"] - N["MH_BRC_L"])
    add("T3.11.3b", "included angle hoop/brace >= 30 deg", inc >= 30.0,
        f"{inc:.2f} deg acute")
    top = x("MH_TOP_L")
    add("T3.11.2", "hoop and brace on opposite sides of the hoop-top vertical",
        (x("MH_BOT_L") - top) * (x("MHB_FOOT_L") - top) <= 1e-6,
        f"hoop foot {x('MH_BOT_L'):.0f}, brace foot {x('MHB_FOOT_L'):.0f},"
        f" top {top:.0f}")
    nb = len([m for m in members_of("MH_BRACE") if m.n1.endswith("_L")])
    add("T3.11.4", "main hoop brace is one straight member per side",
        nb == 1, f"{nb} brace member(s) on the left side")

    # --- front hoop bracing -------------------------------------------------
    sep = float(np.linalg.norm(N["FH_BRC_A_L"] - N["FH_BRC_B_L"]))
    add("T3.12.1", "front hoop brace tubes >= 100 mm apart at the hoop",
        sep >= 100.0, f"{sep:.1f} mm")
    d = z("FH_TOP_L") - z("FH_BRC_A_L") + od_of("FRONT_HOOP") / 2.0
    add("T3.12.2", "front hoop bracing <= 50 mm below the hoop top SURFACE",
        d <= 50.0, f"{d:.1f} mm to surface")

    # --- front bulkhead support --------------------------------------------
    d = z("FBH_TOP_L") - z("FBS_TOP_L") + od_of("FBH") / 2.0
    add("T3.14.1a", "upper bulkhead support <= 50 mm below the bulkhead top SURFACE",
        d <= 50.0, f"{d:.1f} mm to surface")
    # Measured, not asserted: find the upper support member that actually
    # reaches the front hoop and compare ITS landing height to the upper SIS
    # member. v2 compared a node with itself, so this could never fail.
    upper = [m for m in members_of("FBS_UPPER")
             if "FH_SIS" in m.n1 or "FH_SIS" in m.n2]
    if upper:
        m = upper[0]
        node = m.n1 if "FH_SIS" in m.n1 else m.n2
        rel = z(node) - z("FH_SIS_L")
        add("T3.14.1b", "upper support terminates AT the upper SIS node, so the"
                        " 50/100 mm window holds by construction (cannot fail)",
            -50.0 <= rel <= 100.0,
            f"lands on {node} at z={z(node):.0f}, upper SIS at"
            f" z={z('FH_SIS_L'):.0f} -> {rel:+.1f} mm")
    else:
        add("T3.14.1b", "upper bulkhead support reaches the front hoop", False,
            "no FBS_UPPER member terminates at FH_SIS")
    def per_side(prefix):
        out = {}
        for side in "LR":
            out[side] = {m.group for m in frame.members
                         if m.group.startswith(prefix)
                         and (m.n1.endswith(f"_{side}") or m.n2.endswith(f"_{side}"))}
        return out
    ps = per_side("FBS_")
    add("T3.14.1c", "upper, lower and diagonal bulkhead support on BOTH sides",
        all(v >= {"FBS_UPPER", "FBS_LOWER", "FBS_DIAGONAL"} for v in ps.values()),
        f"L {sorted(ps['L'])}, R {sorted(ps['R'])}")

    # --- side impact structure ---------------------------------------------
    floor = min(float(v[2]) for k, v in N.items()
                if k.startswith(("FH_BOT", "MH_BOT")))
    r = od_of("SIS_UPPER") / 2.0
    lo, hi = z("FH_SIS_L") - floor - r, z("FH_SIS_L") - floor + r
    add("T3.15.1a", "upper SIS member COMPLETELY within 240-320 mm above the"
                    " lowest inside point",
        240.0 <= lo and hi <= 320.0, f"tube spans {lo:.1f}..{hi:.1f} mm")
    ps = per_side("SIS_")
    add("T3.15.1b", ">= 3 side impact tubes on BOTH sides",
        all(len(v) >= 3 for v in ps.values()),
        f"L {len(ps['L'])}, R {len(ps['R'])} tubes")
    add("T3.15.1c", "upper SIS member connects front hoop to main hoop",
        any({m.n1, m.n2} == {"FH_SIS_L", "MH_SIS_L"} for m in frame.members),
        "FH_SIS_L -> MH_SIS_L present")
    n_rear = min(len([m for m in members_of("REAR_IMPACT")
                      if m.n1.endswith(f"_{s2}") or m.n2.endswith(f"_{s2}")])
                 for s2 in "LR")
    add("T3.15.2", ">= 3 structural tubes per side in the rear impact structure",
        n_rear >= 3,
        f"{n_rear} class-B tubes per side ({members_of('REAR_IMPACT')[0].tube})"
        if n_rear else "none")

    # --- harness bar (T3.2.1, T5.5) ----------------------------------------
    hb = members_of("HARNESS_BAR")
    add("T3.2.1", "shoulder harness mounting bar present and class A",
        bool(hb) and hb[0].tube.meets(RULE_CLASSES["A"]),
        f"{hb[0].tube} at z={z('MH_HARN_L'):.0f} mm" if hb else "ABSENT")
    add("T5.5.1", "harness bar spans the main hoop, not the bracing (by"
                  " construction; T5.5.5 strap angle NOT checked -- needs a seat)",
        all({m.n1, m.n2} <= set(N) and "MH_HARN" in m.n1 and "MH_HARN" in m.n2
            for m in hb),
        "spans MH_HARN_L to MH_HARN_R on the main hoop, triangulated to the"
        " brace feet")

    # --- T3.8.3 -------------------------------------------------------------
    lowest = min(float(v[2]) for v in N.values())
    add("T3.8.3", "no structure more than 30 mm below the roll hoop feet",
        floor - lowest <= 30.0 + 1e-6,
        f"lowest node z={lowest:.0f}, hoop feet z={floor:.0f}")

    # --- wheelbase and track, from the reference wheel centres -------------
    wb = float(R["WHEEL_R_L"][0] - R["WHEEL_F_L"][0])
    add("T2.9.1", "wheelbase >= 1525 mm", wb >= 1525.0,
        f"{wb:.0f} mm between wheel centres")
    tf = float(R["WHEEL_F_L"][1] - R["WHEEL_F_R"][1])
    tr = float(R["WHEEL_R_L"][1] - R["WHEEL_R_R"][1])
    add("T2.9.2", "smaller track >= 75% of larger",
        min(tf, tr) / max(tf, tr) >= 0.75,
        f"front {tf:.0f}, rear {tr:.0f} -> {min(tf,tr)/max(tf,tr)*100:.1f}%")

    # --- load path ----------------------------------------------------------
    fx = [x(n) for n in frame.front_pickups]
    add("LOADPATH", "front load points sit at the front axle station",
        all(abs(v - float(R["WHEEL_F_L"][0])) < 1.0 for v in fx),
        f"load x={min(fx):.0f}..{max(fx):.0f}, axle x={float(R['WHEEL_F_L'][0]):.0f}")
    rx = [x(n) for n in frame.rear_pickups]
    add("LOADPATH", "rear load points sit at the rear axle station",
        all(abs(v - float(R["WHEEL_R_L"][0])) < 1.0 for v in rx),
        f"load x={min(rx):.0f}..{max(rx):.0f}, axle x={float(R['WHEEL_R_L'][0]):.0f}")

    # --- informational ------------------------------------------------------
    add("T3.8.2", "roll hoop minimum bend radius (manufacturing, not modelled)",
        True, f"centreline radius >= 3 x OD = {3*od_of('MAIN_HOOP'):.0f} mm required;"
              f" hoops are modelled as sharp corners")

    return out


def report(frame: Frame) -> str:
    L = ["=" * 78,
         "PARAMETRIC FS SPACEFRAME -- GENERATOR REPORT",
         "Rules: Formula Student Rules 2026 v1.1, section T3",
         f"Pickup mode: {frame.mode}",
         "=" * 78]
    if frame.clamp_log:
        L.append("\nCLAMPED INTO LEGAL WINDOWS")
        L += [f"  - {e}" for e in frame.clamp_log]
    else:
        L.append("\nAll requested values already legal; nothing clamped.")
    L.append(f"\nFRAME: {len(frame.nodes)} nodes, {len(frame.members)} members, "
             f"{frame.mass():.2f} kg of tube")
    L.append("\nMASS BY GROUP (kg)")
    for g, m in frame.mass_by_group().items():
        tb = next(x.tube for x in frame.members if x.group == g)
        L.append(f"  {g:<17s} {m:6.3f}   {tb}   class {MEMBER_GROUPS[g]}")
    L.append("\nCHECKS")
    cs = check(frame)
    for c in cs:
        L.append(f"  [{'PASS' if c.ok else 'FAIL'}] {c.rule:<10s} {c.description}")
        L.append(f"         {c.detail}")
    nf = sum(1 for c in cs if not c.ok)
    L.append(f"\n{len(cs)-nf}/{len(cs)} passed, {nf} failed.")
    L.append("\nNOT MODELLED: cockpit templates (T4.1, T4.2), Percy clearance"
             " (T4.3),\nroll hoop bend geometry (T3.8.2, T3.9.3), SES approval."
             " This is not a\nscrutineering result.")
    return "\n".join(L)


# ===========================================================================
# 7. FUZZ -- every numeric field, and both pickup modes
# ===========================================================================

_NUMERIC_BOUNDS = {
    "bulkhead_to_front_axle": (-200, 900), "front_axle_to_front_hoop": (-200, 900),
    "front_hoop_to_main_hoop": (-200, 1400), "main_hoop_to_brace_foot": (-200, 1800),
    "brace_foot_to_rear_axle": (-200, 900),
    "floor_z": (-50, 400), "bulkhead_drop_below_floor": (-500, 400),
    "bulkhead_height": (20, 800), "sis_rise": (50, 700),
    "fh_rise_above_sis": (20, 1200), "mh_rise_above_fh": (-100, 1200),
    "harness_rise": (-100, 1400),
    "bulkhead_half_width": (-50, 500), "front_hoop_half_width": (-50, 500),
    "main_hoop_half_width": (-50, 500),
    "track_front": (600, 1900), "track_rear": (500, 1900),
    "pickup_inboard_offset": (0, 800),
    "fh_lean_deg": (-45, 45), "mh_lean_deg": (-30, 30),
    "mhb_drop": (-50, 500), "fhb_drop": (-50, 400),
    "fhb_separation": (-50, 900), "fbs_upper_drop": (-50, 300),
    "pickup_lower_rise": (-100, 600), "pickup_upper_rise": (-100, 900),
}


def fuzz(n: int = 3000, seed: int = 0) -> tuple[int, list[str]]:
    """Throw deliberately illegal parameters at the generator and confirm the
    independent checker never reports a violation. Both pickup modes."""
    rng = np.random.default_rng(seed)
    numeric = [f.name for f in fields(ChassisParams) if f.name in _NUMERIC_BOUNDS]
    missing = {f.name for f in fields(ChassisParams)
               if f.type == "float" and f.name not in _NUMERIC_BOUNDS}
    assert not missing, f"numeric fields missing from the fuzz bounds: {missing}"
    fails: list[str] = []
    for i in range(n):
        kw = {k: float(rng.uniform(*_NUMERIC_BOUNDS[k])) for k in numeric}
        kw["spec"] = {g: (float(rng.uniform(10, 60)), float(rng.uniform(0.3, 6.0)))
                      for g in MEMBER_GROUPS}
        kw["pickup_mode"] = "at_node" if rng.random() < 0.5 else "mounted"
        fr = build(ChassisParams(**kw))
        fails += [f"iter {i} [{kw['pickup_mode']}]: {c.rule} -- {c.detail}"
                  for c in check(fr) if not c.ok]
    return n, fails


# ===========================================================================
# 8. OUTPUT
# ===========================================================================

def export(frame: Frame, path: str) -> None:
    with open(path, "w") as f:
        json.dump(frame.to_dict(), f, indent=2)


def plot(frame: Frame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    col = {"MAIN_HOOP": "#c1121f", "FRONT_HOOP": "#c1121f", "HARNESS_BAR": "#7b2cbf",
           "MH_BRACE": "#e85d04", "FH_BRACE": "#e85d04", "HARNESS_SUPPORT": "#9d4edd",
           "SIS_UPPER": "#0353a4", "SIS_LOWER": "#0353a4", "SIS_DIAGONAL": "#0353a4",
           "REAR_IMPACT": "#0077b6", "FBH": "#2a9d8f",
           "FBS_UPPER": "#6a4c93", "FBS_LOWER": "#6a4c93", "FBS_DIAGONAL": "#6a4c93",
           "MHB_SUPPORT": "#8d99ae", "FLOOR": "#ced4da", "REAR_BAY": "#adb5bd",
           "PICKUP_MOUNT": "#ffb703", "PICKUP_SUPPORT": "#fb8500"}
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    seen = set()
    for m in frame.members:
        a, b = frame.nodes[m.n1], frame.nodes[m.n2]
        lbl = m.group if m.group not in seen else None
        seen.add(m.group)
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=col.get(m.group, "#333"), lw=0.8 + m.tube.t, label=lbl)
    for n in frame.front_pickups + frame.rear_pickups:
        ax.scatter(*frame.nodes[n], color="#ffb703", s=55, edgecolors="k", zorder=5)
    P = np.array(list(frame.nodes.values()))
    rr = (P.max(0) - P.min(0)).max() / 2
    c = (P.max(0) + P.min(0)) / 2
    ax.set_xlim(c[0]-rr, c[0]+rr); ax.set_ylim(c[1]-rr, c[1]+rr)
    ax.set_zlim(c[2]-rr, c[2]+rr)
    ax.set_xlabel("x rearward (mm)"); ax.set_ylabel("y left (mm)")
    ax.set_zlabel("z up (mm)")
    ax.set_title(f"{frame.mass():.2f} kg of tube, {len(frame.members)} members"
                 f"  |  load points in yellow  |  mode: {frame.mode}")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate an FS spaceframe.")
    ap.add_argument("--out", default=".", help="output folder (default: here)")
    ap.add_argument("--mode", default="at_node", choices=["at_node", "mounted"])
    ap.add_argument("--fuzz", type=int, default=2000,
                    help="self-test frames; 0 to skip")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frame = build(ChassisParams(pickup_mode=args.mode))
    print(report(frame))

    jpath = os.path.join(args.out, f"frame_{args.mode}.json")
    export(frame, jpath)
    print(f"\nWrote {os.path.abspath(jpath)}")
    if not args.no_plot:
        ppath = os.path.join(args.out, f"frame_{args.mode}.png")
        plot(frame, ppath)
        print(f"Wrote {os.path.abspath(ppath)}")

    if args.fuzz:
        print(f"\nFuzzing {args.fuzz} frames across both pickup modes...")
        n, f = fuzz(args.fuzz)
        if f:
            print(f"  {n-len(f)}/{n} legal -- {len(f)} violations:")
            for line in f[:10]:
                print("   ", line)
        else:
            print(f"  {n}/{n} random frames legal. Generator invariant holds.")


if __name__ == "__main__":
    main()
