# Case studies

What the solver and optimiser are actually good for.

Every number below is printed by `python case_studies/run_case_studies.py`, so this document can be regenerated and checked rather than trusted blindly.

**Baseline for all studies:** `frames/frame_at_node.json` — 36 nodes, 81 members, 34.18 kg, 1712.2 Nm/deg. Torsion load 408.6 Nm at the front axle pickups, all three translations fixed at the four rear pickups.

---

## 1. A load-path failure, and its fix

**Question:** Does it matter *where* the suspension load enters the frame if the tubes are all the same size?

**Setup:** The front suspension load enters at the front axle station, on nodes that sit part-way along two 25.4 × 1.2 mm bulkhead-support tubes. The current frame braces those nodes with five tubes (the `PICKUP_SUPPORT` group). The study deletes those five tubes and re-solves, so the load is pushed into the mid-span of the unsupported bulkhead-support tubes, which is how an earlier version of the generator built the frame.

**Result:**

| | Stiffness | Mass | Per kg |
|---|---|---|---|
| Untriangulated axle station | 953.9 Nm/deg | 32.37 kg | 29.5 Nm/deg/kg |
| Triangulated | 1712.2 Nm/deg | 34.18 kg | 50.1 Nm/deg/kg |
| Change | **+79.5%** | +1.8 kg | **+70%** |

Share of strain energy in the two bulkhead-support groups: **43.6%** untriangulated, **8.0%** triangulated.

**What went wrong:** Nearly half of all the elastic energy in the chassis sat in two thin tubes bending locally, where the load was pushed into their mid-span. The number being measured was a tube flexing, not a chassis twisting. Therefore, the FE model was technically correct as it solved the structure it was given, but the structure was badly designed at the load input.

**Conclusion:** Load has to enter a spaceframe at a braced node. The strain energy breakdown shows this: no single displacement looked wrong, but the energy distribution was obviously lopsided. Of every change in this project, this one is worth the most, and no amount of tube resizing would have found it.

---

## 2. What the optimiser cannot see: a main hoop diagonal

**Question:** The tube-size optimiser only changes diameters and walls. Is it missing anything bigger?

**Setup:** Under torsion, the largest displacement anywhere in the baseline frame is 2.80 mm of sideways sway at the top of the main hoop. The hoop is an unbraced rectangle, so it racks into a parallelogram without any tube having to stretch as it's the cheapest way for the frame to deform. Two diagonals were added across the main hoop, each a 25.4 × 1.6 mm class-B tube, and the frame is re-solved. The harness bar divides the hoop at shoulder height, so one diagonal crosses the region below it and the other the region above it.

**Result:**

| | Stiffness | Added | Hoop sway | Nm/deg per g |
|---|---|---|---|---|
| Baseline | 1712.2 Nm/deg | — | 2.80 mm | — |
| Seat-back diagonal (below the harness bar) | 1828.5 (+6.8%) | 676 g | 2.37 mm | 0.17 |
| Head-zone diagonal (above the harness bar) | 2335.5 (**+36.4%**) | 687 g | 1.27 mm | **0.91** |

For comparison, the entire tube-size optimisation with a 2 kg budget gains +786.6 Nm/deg for 1790 g: **0.44 Nm/deg per gram**.

**Conclusion:** One well-placed tube above the harness bar is worth about twice as much per gram as the whole tube-size optimisation and delivers 80% of its gain for 38% of its mass. Therefore, topology essentially beats sizing here.

However, the head-zone diagonal crosses the zone behind the driver's head and shoulders, where the headrest, harness routing, and driver egress live. The seat-back diagonal fits behind the seat back and is worth far less. The engineering question is not just "whether a diagonal is good," but "where can one go such that the driver can live with it". That trade is a packaging decision the solver can inform but not make.

**Caveat:** The diagonal is added outside the generator, so the T3 rule checks were not re-run on these two frames.

---

## 3. Material choice: steel vs aluminium, and a near-failure

**Question:** Because Aluminium is a third the weight of steel, would an aluminium frame
of the same geometry perform better?

**Setup:** Same geometry and tube sizes, material swapped using Aluminum Association design-manual values:

| | E | Density | Welded yield |
|---|---|---|---|
| Steel | 200 GPa | 7850 kg/m³ | 180 MPa |
| 6061-T6 aluminium | 68.9 GPa | 2700 kg/m³ | 103 MPa (parent 241 MPa) |

**Result**

| | Stiffness | Mass | Per kg |
|---|---|---|---|
| Steel | 1712.2 Nm/deg | 34.18 kg | 50.1 Nm/deg/kg |
| 6061-T6 aluminium | 588.6 Nm/deg | 11.76 kg | 50.1 Nm/deg/kg |

| Worst utilisation (1.0 = limit) | Steel | Aluminium |
|---|---|---|
| Torsion, buckling | 0.108 | 0.211 |
| Torsion, stress | 0.219 | 0.384 |
| Bulkhead, buckling | 0.206 | 0.401 |
| Bulkhead, stress | 0.485 | **0.848** |

Bulkhead load to first yield: steel 20.6 kN, aluminium 11.8 kN.

**What this shows:**

1. **Stiffness per kg is identical-** Steel and aluminium have almost exactly the
   same specific stiffness (E divided by density). Swapping material in the same
   geometry saves 22 kg and loses 66% of the stiffness, ending up with the same stiffness per kilogram and no gain.
2. **The aluminium frame comes close to failure-** In a frame made of one material,
   member forces don't depend on E, so stresses are unchanged; however, welded
   aluminium is only 57% as strong, so bulkhead utilisation rises from 0.49 to
   0.85. The weld (not the tube) is the weak point.
3. **Aluminium only performs with different geometry-** Aluminium can perform better, but only if the tube sizes change with it: I grows with OD⁴ while mass grows only with area, so a large-diameter aluminium tube can be both stiffer and lighter than the steel tube it replaces. Therefore, this would entail redrawing the tube schedule and re-checking every joint instead of simply building the same frame out of different metal.

**Conclusion:** Material choice changes the design, not just the numbers. The FS
rules also require alternative materials to demonstrate equivalence through the
SES, which this project does not model.

**Caveat on carbon fibre:** Even though carbon fibre is prevalent in most professional motorsport events like F1, it was deliberately not modelled here. A composite chassis is a monocoque: a shell of layered, direction-dependent laminate, bonded rather than welded; it's
not a frame of isotropic tubes, and it needs a shell finite-element model with
laminate properties. Feeding carbon-fibre numbers into this beam solver would
produce a figure that looks precise but means nothing. For scale, a carbon-fibre
sandwich monocoque tub has been cited at around 30,000 Nm/deg for the tub alone
(Featherston et al.).

---

## 4. Against published Formula Student data

**Question:** Are these numbers realistic, and does the optimisation matter for a real car?

**Published figures:**

- **4ZE Racing —** their FE model predicted up to 1561.4 Nm/deg. They then built
  the chassis and tested it physically: the measured stiffness was about 10%
  off the ANSYS prediction.
- **University of Michigan (Velie) —** the team set a design target of 2100
  Nm/deg; their FE model of the resulting frame gave 1900 Nm/deg. Because built
  chassis typically measure 10-20% softer than their models, the report argues a
  design should aim for at least ~1900 Nm/deg so the finished car still has
  enough stiffness.
- **Cardiff (Featherston et al.) —** a steel spaceframe was loaded on a torsion
  rig and its movement tracked with 3D motion capture, giving an average of 2930
  Nm/deg. This is a physical measurement, not a simulation. The same paper notes
  a typical Formula SAE target of around 1500 Nm/deg.

**This project:**

| | Predicted | As built, if FE over-predicts by 10–20% |
|---|---|---|
| Baseline | 1712 Nm/deg | 1370–1541 Nm/deg |
| Optimised, 2 kg budget | 2499 Nm/deg | 1999–2249 Nm/deg |
| Optimised, 4 kg budget | 3141 Nm/deg | 2513–2827 Nm/deg |

**What this shows-** The baseline sits inside the published range, which is a plausibility check on the whole pipeline. But once the typical FE-to-test shortfall is allowed for, the baseline as built would likely fall **below** the ~1900 Nm/deg minimum the Michigan work recommends. The 2 kg optimised frame clears it in every case. That is the practical argument for the optimiser: it turns a frame that is probably too soft once built into one with margin, for 1.8 kg, without touching the geometry.

**Caveat-** These are different cars with different geometry, load points and support conditions. The comparison is a range check, not validation. The solver's validation is the closed-form and PyNite cross-checks in the README; nothing here substitutes for testing a physical frame.

---

## References

- 4ZE Racing, "Design and optimization of FSAE chassis using FEA", IOP Conf. Ser.:
  Mater. Sci. Eng. 402 012184 —
  <https://iopscience.iop.org/article/10.1088/1757-899X/402/1/012184/pdf>
- H. D. Velie, "Chassis Torsional Rigidity Analysis for a Formula SAE Racecar",
  University of Michigan —
  <https://deptapps.engin.umich.edu/open/rise/getreport?pid=104&fv=2&file=Chassis+Torsional+Rigidity+Analysis+for+a+Formula+SAE+Racecar.pdf>
- C. A. Featherston et al., "Measuring the Torsional Stiffness of a Space Frame
  Chassis Using 3D Motion Capture Techniques", 2006 —
  <https://www.researchgate.net/publication/229825867>
- W. B. Riley, A. R. George, "Design, Analysis and Testing of a Formula SAE Car
  Chassis", SAE 2002-01-3300 (widely cited; not used for numbers here)
- Aluminum Association, *Aluminum Design Manual*: 6061-T6 minimum yield 35 ksi
  (241 MPa), welded 15 ksi (103 MPa)
