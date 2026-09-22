# Sliced loft: mesh → section outlines → loft (research + plan)

Status (2026-09-22, later the same day): **implemented as v0.4.0** —
`stl2prism/section_fit.py` (plane cut + curve fit), `stl2prism/sliced_loft.py`
(the loft), `method='loft'` through pipeline / CLI / API / web UI, the
Fusion loft script, `tests/test_sliced_loft.py`. The Phase 0 findings and
what changed against the plan are in "What Phase 0 found" below; the
design decisions further down still hold except where that section says
otherwise.

## What Phase 0 found (and what the implementation does)

- **Ring curves.** Neither the prototype's chord-length interpolating
  splines (minutes) nor `makeSplineApprox` rings (13 s per cornered run,
  and `ThruSections` failed outright on the controller housing) work:
  ThruSections must unify every ring's knot vector. Uniform interpolating
  splines build in 0.7 s but carry one pole per sample, and the 11k-pole
  surface could not be tessellated in minutes. The implementation fits
  every ring of a run by least squares onto **one shared uniform periodic
  cubic B-spline basis** (`fit_ring_poles`: K poles, doubling from 16
  until the samples sit within 0.05 mm, cap 320 or N/2) and builds the
  OCC curves from those poles directly. Surface = K x M poles, no knot
  unification, and a sphere is 16 x 60.
- **Smooth loft.** `BRepOffsetAPI_ThruSections(solid, ruled=False)` with
  cadquery's defaults; `SetSmoothing` / `SetMaxDegree` make it fail.
  Through exact circles it is exact; through the fitted rings 0.7 s for 60
  sections.
- **Dome tips flare.** A slice within a hair of a dome tip (a 0.2 mm ring)
  makes the smooth surface flare by centimetres, and a smooth loft to an
  apex *vertex* flares too. End slices under 50 % of their neighbour's
  area are dropped and the tip is a short **ruled cone to the apex**
  (0.2 mm cap error at 0.2 mm spacing instead of 0.8).
- **Steep stretches.** A smooth B-spline through sections whose area
  changes by more than 15 % per step overshoots by 0.3–0.6 mm and makes
  BRepMesh produce 500k triangles. Such pairs are lofted **ruled**; the
  rest of the run smooth; the pieces are glued (`_stretches`). Sphere:
  17 faces, 0.09 mm max both ways, 0.03 % volume, 13k triangles.
- **`cq.Shape.Volume()` is 1–2 % wrong on B-spline faces** (a lofted
  60x40x6 box measured 14596). `pipeline.accurate_volume` (adaptive
  `BRepGProp.VolumeProperties`, eps 1e-5) now feeds `validate` and every
  loft check. The plan's "smooth loft volume 5 % off" was this artefact.
- **Fallbacks.** A smooth stretch that is invalid or off the trapezoid
  integral of its section areas by > 1 % is rebuilt ruled; a run off by
  > 5 % even ruled is refused (its rings do not correspond); every piece
  (run, hole cut, cone, bridge) is fail-safe and listed in
  `info['skipped']`. Bridges are ruled two-ring lofts, because
  `extrudeLinear` refuses B-spline wires as "not planar".
- **Step levels** are clustered with a gap of max(0.05 mm, interval/2): a
  parting rim exported at three heights 0.01 mm apart is one step.
- **Thinning** keeps ≤ 60 sections per run, crowding where the area
  changes by > 2 % between neighbours.
- **The controller housing halves** (thin shells with dozens of openings)
  loft, but slowly and mostly ruled: topology changes every slice near
  the openings, so runs are short and hole cuts of B-spline solids
  dominate the time. That is the "features that cross the axis" limit
  below, not a bug; a smooth outer skin is the intended use.

Section tracing was checked against a Fusion cut of the RC-N2 controller
(`samples/fixed-rc-n2-360.stl`, file in 0.1 mm units → `--scale 0.1`;
Fusion's plane = normal to file Z at bbox-centre Z − 26 mm):
`tools/trace_section.py … --axis z --offset -26 --scale 0.1` gives the same
outline (mirrored), and at tol 0.08 fits it as 114 lines, 120 arcs, 1
circle and 29 splines within 0.103 mm, in 3 s. On organic sections the
prismatic engine's frame snapping and tangent junction solving made the
fit *worse* (twice as many primitives off by more than tol), so
`fit_loop` leaves them off unless `clean=True`.

## What one does by hand in Fusion

1. Design workspace → **Mesh** tab → Create → **Create Mesh Section
   Sketch**: pick the mesh body and a section plane (offset / angle
   handles). Fusion puts the plane–mesh intersection into a new sketch as
   a "Mesh Section" polyline.
2. Repeat at a chosen spacing to get a stack of outlines (like CT / MRI
   slices).
3. In each sketch, Sketch → **Fit Curves to Mesh Section** turns the
   polyline into real sketch curves (lines, arcs, splines).
4. **Loft** through the profiles → one smooth solid instead of thousands
   of facets.

Sources: [Create a mesh section sketch](https://help.autodesk.com/cloudhelp/ENU/Fusion-Mesh/files/MESH-CREATE-MESH-SECTION-SKETCH.htm),
[Create mesh section sketch (Fusion blog, 2019)](https://www.autodesk.com/products/fusion-360/blog/april-8-2019-product-update-whats-new/create-mesh-section-sketch/),
[Fit curves to mesh sections](https://help.autodesk.com/view/fusion360/ENU/?guid=SKT-SKETCH-CREATE-FIT-CURVE-MESH-SECTION),
[Loft Feature API sample](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/LoftFeatureSample_Sample.htm),
[LoftFeatureInput.loftSections](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/LoftFeatureInput_loftSections.htm).

## Can stl2prism do the same? Yes, with limits

The whole workflow can be automated outside Fusion:

| Fusion step | Our equivalent |
|---|---|
| Mesh Section Sketch | `trimesh.section_multiplane` (one call, all heights, fast) |
| Fit Curves to Mesh Section | resample each loop to N points, fit a closed B-spline |
| Loft | OpenCascade `BRepOffsetAPI_ThruSections` (`cq.Solid.makeLoft`) |
| Fusion timeline | generated Fusion script: offset planes + fitted-spline sketches + Loft |

So the option gives two things: a STEP with very few faces (one
B-spline face per run of slices, plus caps), and a Fusion script that
repeats the manual workflow as a parametric timeline.

**Where it is good:** organic and turned shapes whose outline changes
smoothly along one axis: controller shells, handles, lens caps, hulls,
scanned housings. These are exactly the parts the prismatic and
face-group engines do worst on.

**Where it is bad (by design, not fixable by finer slices):**

- Features that cross the slice axis: a hole drilled sideways becomes a
  smeared trough, because no slice sees it as a circle. The prismatic
  engine already handles those; loft is not a replacement for it.
- Sharp corners *around* an outline: a single closed spline rounds them.
  Splitting each outline at corners and lofting edge-by-edge only works
  when every slice in the run has the same number of corners. Not in v1.
- Step changes *along* the axis (a shoulder): a smooth loft overshoots
  across a jump. Handled by breaking the loft into runs at every flat
  face perpendicular to the axis (see below), so each run is smooth
  inside and the step is a real planar face.
- True cylinders and planes become B-spline faces. Fusion can still sketch
  on and measure them, but they are not "cylinder r = 5.000" objects.

**Slice spacing.** 0.2 mm is accurate enough for hobby printing. At 0.2 mm the slicing
is trivial (a 33 mm part is 165 slices) but a *smooth* loft through 165
sections is not (see prototype findings). Plan: slice at 0.2 mm for
detection and accuracy, loft through a thinned subset (every k-th slice,
capped at ~60 sections per run), and let the deviation check confirm the
result is within 0.2 mm anyway.

## Prototype findings (tools/sliced_loft_proto.py, tools/sliced_loft_profile.py)

Run: `.venv/bin/python tools/sliced_loft_profile.py sphere|bottle|<mesh> [axis] [interval]`.

What works:

- Slicing, loop extraction (outer + holes), resampling to equal arc
  length, CCW orientation, start-point alignment between slices (roll to
  the minimum-distance offset), and matching loops slice to slice by
  overlap all behave. The servo bracket at 0.83 mm gave 43 sections, 4
  step levels, 7 runs, correct topology.
- Step levels: cluster the mesh's faces with |n·axis| > 0.999 by height;
  any level holding > 2% of the bounding cross-section area is a step.
  Slice at level ± 1e-3 mm and snap both sections onto the level, so the
  run above and the run below touch exactly (no sliver, no gap).

What failed, and what to try first in the implementation session:

1. **The smooth loft is the bottleneck.** `cq.Solid.makeLoft(wires,
   ruled=False)` through periodic *interpolating* splines with up to 240
   points each did not finish a 40-slice sphere in 600 s, and the servo
   bracket took 610 s. Fixes to measure, in order:
   - fit each outline with `cq.Edge.makeSplineApprox(pts, tol=0.02)`
     (far fewer poles than interpolating 240 points), then loft;
   - drive `BRepOffsetAPI_ThruSections` directly with
     `SetSmoothing(True)`, `SetMaxDegree(3)`, `CheckCompatibility(True)`;
   - `ruled=True` as a fallback: fast, one face per slice pair, exact
     within d²/(8R). With thinned sections (2 mm apart) that is ~0.05 mm
     on a 10 mm radius, so still inside the 0.2 mm target.
   - alternative build that skips ThruSections: least-squares
     `GeomAPI_PointsToBSplineSurface` on the M×N grid of resampled
     points, `BRepBuilderAPI_MakeFace`, planar caps, sew, `MakeSolid`.
     One face per run, cost O(M·N). Seam is C0 (acceptable).
2. **One loft raised `BRep_API: command not done`** on a 2-section run
   (a short 2-outer run on the bracket). Guard: a run needs ≥ 2 sections
   with distinct heights; extrude a 1-section run instead of lofting it.
3. **The final fuse produced a null / invalid shape** (9 solids, 0.4 % of
   the volume). Causes to check: negative-volume lofts (orientation),
   and fuzzy `fuse(tol=1e-3)` on touching B-spline solids. Use
   `fuse(glue=True)` for solids that share a planar cap, check each
   solid's volume sign before fusing, and fall back to writing the runs
   as separate solids if the fuse is still invalid.

## Design of the option

Decisions already made (do not re-open):

- Name: **Sliced loft**; pipeline mode string `'loft'`.
- Default slice spacing **0.2 mm**; axis `auto` = longest extent, or
  `x`/`y`/`z`.
- Smooth loft by default (fewest faces); `ruled` as an option.
- Run breaks at flat-face levels and at topology changes (loop count or
  hole count differs). Gaps between runs that are *not* at a level are
  bridged by extruding the last outline of the earlier run to the next
  run's first plane.
- A loft result is **always written** when it builds (the user chose the
  method); the gate rows are shown as PASS / FAIL for information. The
  faceted fallback only applies when the loft cannot be built at all.
- Holes along the axis: loft the hole loops as their own solids and cut.
- Section thinning for both the OCC loft and the Fusion script: at most
  ~60 sections per run, always keeping the first, the last and the
  snapped step sections.

### Plumbing (mirror the existing options exactly)

- `stl2prism/sliced_loft.py` (new): `step_levels`, `slice_heights`,
  `sections`, `resample`, `align`, `match`, `build_runs`,
  `loft_body(mesh, axis, interval, ruled=False, verbose=True) ->
  (TopoDS_Shape, build_info)`; `build_info` holds the thinned section
  points per run (for the Fusion script) and counts (sections, runs,
  step levels, holes).
- `pipeline.run(..., method='auto', slice_mm=0.2, slice_axis='auto')`;
  `_convert_body` takes the same; when `method == 'loft'` it calls
  `loft_body`, validates with the existing `validate`, logs
  `_log_check`, and returns `(shape, 'loft', metrics)` with
  `metrics['build'] = {'mode': 'loft', ...}`. Also thread through
  `_shell_task` / `_convert_all_pooled` (the worker payload).
- CLI: `--method {auto,loft}`, `--slice-mm 0.2`, `--slice-axis
  {auto,x,y,z}`, `--loft-ruled`.
- `backend/main.py ConvertParams`: `method: Literal['auto','loft'] =
  'auto'`, `slice_mm: float = Field(0.2, gt=0, le=50)`, `slice_axis:
  Literal['auto','x','y','z'] = 'auto'`, `loft_ruled: bool = False`.
  `backend/worker.py`: pass them to `run`.
- `stl2prism/fusion_export.py`: `emit_fusion_loft_script(bodies_info)`:
  reuse `_plane` and `_pick`; per section a sketch on an offset plane
  with `sketchFittedSplines.add(ObjectCollection of
  sk.modelToSketchSpace(Point3D))` and `isClosed = True` (verify the
  property name in Fusion); per run `lofts.createInput(newBody if first
  else join)`, `loftSections.add(profile)` for each section, `isSolid =
  True`; holes as a second loft with `cut`. Written by `_write_script`
  as `<out>_fusion.py` when the body's build mode is `'loft'`, so the
  existing `/fusion-script` endpoint and UI link serve it unchanged.
- Frontend: `store.js` defaults `method: 'auto', slice_mm: 0.2,
  slice_axis: 'auto', loft_ruled: false`; `ParamsPanel.vue` gets a
  "Method" select above the acceptance gate (Prismatic / face-group
  (auto) vs Sliced loft) and, only when loft is chosen, the spacing,
  axis and ruled fields with an explain block; `ReportPanel.vue` gets a
  `'loft'` verdict ("Sliced loft solid: N sections along <axis> at
  0.2 mm, M runs; one smooth face per run"), keeps the gate table for
  mode `'loft'`, and the Fusion script hint says it rebuilds section
  sketches + Loft.
- Tests: `tests/test_sliced_loft.py` with synthetic parts from
  `tests/synth.py` style helpers: `icosphere`, a revolved bottle, an
  ellipsoid, `plate_holes` (axis-parallel holes), `stepped_shaft` (step
  levels). Assert: mode `'loft'`, one solid, `BRepCheck` valid, face
  count ≤ runs·2 + holes + a small slack, `dev_p95 ≤ 0.1`, volume error
  ≤ 1 %, STEP re-reads as one solid (use `_reimport` from
  `tests/test_pipeline.py`). Plus a smoke test that the Fusion loft
  script is emitted and mentions `loftFeatures`.
- README: a "Sliced loft" section under "What it does" and the new CLI
  flags; version bump to 0.4.0.

### Order of work

1. Phase 0 (measure): the four loft-speed fixes above on the sphere and
   bottle from `tools/sliced_loft_profile.py`. Pick the fastest that keeps
   `dev_p95` under 0.1 mm at 2 mm thinned sections. Fix the fuse.
2. Module + pipeline + CLI, with the tests.
3. Fusion script.
4. Web UI.
5. README, version, then the samples: `Sony E-Mount Body Cap.stl`,
   `joystick_claw_1.stl`, `rc2-clean-controller-v2.stl` (largest body
   only) for numbers in the README.
