# stl2prism

## Why

I do a lot of hobby 3D-printing projects, designing parts for my own use.
Again and again I needed to model *around* a mesh part — a bracket from
Thingiverse, a scanned housing, a controller shell — and Fusion 360 gave me
no easy way to do it: on the free hobby tier, **Mesh → Solid** turns the
mesh into a solid made of hundreds or thousands of facet triangles, which is
impossible to measure, sketch on, or model against. So I built this: it
turns a mesh into a solid with real planes, cylinders and holes that I can
actually design against once it is imported into Fusion 360 (see the
[comparison](#compared-with-fusion-360s-mesh-to-solid) below).

## What it does

Convert STL / OBJ (also PLY, OFF, 3MF, GLB) meshes into **prismatic STEP
solids** — clean BREP with true planes, cylinders and cones you can sketch
on, dimension against, and constrain — replicating the core of Fusion 360's
paid "Prismatic" mesh conversion, plus something Fusion does not give you: an
**editable CadQuery script** of the sketches and extrudes it recognised.
Guaranteed output: when a mesh isn't an extrusion, the **face-group engine**
groups it into surface regions and gives each a real plane, cylinder, cone,
sphere or torus face (Fusion's face-group approach, in the open; a
rolling-ball fillet around a boss or a hole mouth is one torus face); regions nothing fits
keep their exact facets; when even that fails, the tool falls back to a
faceted (valid, manifold, coplanar-merged, tolerance-reduced) STEP solid
instead of failing.

## Screenshots

The web app: drop a mesh, inspect it, set the acceptance gate, convert, and
read the fidelity report before downloading the STEP (and, for extrusions,
the CadQuery / Fusion 360 scripts).

| | |
|---|---|
| ![Mesh loaded, acceptance gate and options](docs/img/ui-loaded.png) *Mesh loaded; units, acceptance gate, face-group engine toggle.* | ![Prismatic result](docs/img/ui-servo-prismatic.png) *Servo bracket → prismatic solid: 1,644 triangles → 23 faces (15 planes, 8 cylinders), max deviation 0.060 mm, every gate passed, STEP + scripts offered.* |
| ![Face-group result: frame](docs/img/ui-frame-facegroup.png) *Frame (countersinks, multi-direction material) → face-group solid: 4,200 triangles → 101 faces (v0.3.3), 9 cylinders + 7 cones, max deviation 0.040 mm.* | ![Face-group result: joystick claw](docs/img/ui-claw-facegroup.png) *Joystick claw → face-group solid: 1,902 triangles → 250 faces incl. 43 cylinders and 20 spheres.* |

### Compared with Fusion 360's Mesh to Solid

The same `servo_bracket_1.stl`, opened in Fusion 360 three ways. Left:
Fusion's own *Mesh → Solid* (the free tier's faceted conversion — every
triangle becomes a face). Middle: the STL mesh as loaded. Right: the STEP
from stl2prism — 23 faces, planes and cylinders, holes that are real holes.

![Fusion Mesh to Solid (left), the STL mesh (middle) and the stl2prism STEP (right)](docs/img/side-by-side.png)

| | |
|---|---|
| ![Fusion Mesh to Solid, zoomed](docs/img/fusion-mesh-solid.png) *Fusion Mesh → Solid, zoomed in: one face per triangle, so the "solid" carries all 1,644 facets and cannot be sketched on, filleted or measured like a modelled part.* | ![stl2prism STEP, zoomed](docs/img/Stl-prism-zoom.png) *stl2prism output, zoomed in: flat faces are single planes, the blend is one cylinder, the edges are where the design has them.* |

## Install

```bash
pip install .            # core
pip install .[scan]      # + pymeshlab, for 3D-scan repair (Poisson; Linux x86_64)
```

## Usage

```bash
stl2prism part.stl                 # -> part.step  (+ part.py CadQuery script)
stl2prism part.obj --units cm      # file is in cm (Fusion's OBJ default); scale to mm
stl2prism part.stl out.step --tol 0.05 --accept-max 0.3 --accept-vol-pct 3
stl2prism scan.stl --reduce-tol 0.1     # faceted output: simplify curved regions within 0.1 mm
stl2prism scan.stl --force-prismatic    # attempt prismatic on scan input
stl2prism part.stl --no-face-groups     # skip the face-group engine (prismatic -> faceted only)
```

Or from Python:

```python
from stl2prism import run
result = run("part.stl", "part.step")
print(result["mode"], result["metrics"], result["script"])   # 'prismatic' | 'facegroup' | 'faceted' | 'mixed'
```

Exit code 0 on success. The log reports which route produced the output
(`prismatic`, `facegroup` or `faceted`) and the measured fidelity (surface
deviation both ways, bore deviation, volume error) of prismatic and
face-group results.

Outputs next to the STEP: for prismatic results `<out>.py` (CadQuery) and
`<out>_fusion.py` (Fusion 360 sketches + extrudes — verified: a fully
parametric timeline you can edit); for face-group results
`<out>_fusion_bfill.py` — an **experimental Fusion 360 Boundary Fill
script**. It recreates every fitted plane, cylinder, cone, sphere and torus
slightly oversized as a temporary body, runs Boundary Fill, and keeps the
cells that lie inside the original mesh (the mesh travels inside the
script). Fusion's own kernel then computes the exact edges between the
faces, so the solid comes out without the polyline edges of the STEP.

To run either script in Fusion: put the `.py` in an empty folder of its
own, then Utilities → Add-Ins → Scripts and Add-Ins → **+** → choose that
folder → Run. Progress appears in the Text Commands panel (View → Show
Text Commands).

Boundary Fill status: verified on parts the engine fits cleanly (planes,
cylinders, cones, spheres, fillets, holes — exact face counts). Torus blends
(fillets around curved edges) are one torus tool each; an apex cone (a
pencil tip) is one solid cone tool; a tapered / variable-radius fillet kept
as bands in the STEP becomes a single approximate torus or cone tool per
chain (tangent band chains defeat the cell computation). The printed
outlook is an OCC dry run of the actual script — the tools are rebuilt,
the cells computed and the enclosed volume measured — so OK / LIKELY TO
FAIL reflects the arrangement itself, not a guess. Still fails on bodies
whose gently curved plates segment into near-parallel plane strips (the
frame — see Roadmap). Bodies with more than 200 regions get no script;
`EXPAND` at the top of the script sets the oversize (1 mm by default).

## Web app

A browser UI for the same pipeline: drag a mesh in, inspect it in 3D, set
the acceptance tolerances, convert, and download the STEP (and, for
prismatic results, the CadQuery and Fusion 360 scripts) with a fidelity
report (route, surface deviation vs. your limits, volume error, face counts
and surface types, regions kept as facets).

### Run locally (dev)

```bash
python -m venv .venv && .venv/bin/pip install -e . fastapi 'uvicorn[standard]' python-multipart
.venv/bin/uvicorn backend.main:app --port 8000     # API
cd frontend && npm install && npm run dev           # UI on :5173, proxies /api
```

### Run as a container

```bash
docker compose up --build        # then open http://localhost:8321
./deploy.sh                      # same, but stamps the image with the git commit,
                                 # shown top-right in the UI and at /api/version
```

To run it on a home NAS (any x86_64 box with Docker — the NAS vendor, Synology,
QNAP; clone on the NAS, build natively, optional Tailscale HTTPS front), follow
[docs/DEPLOY-NAS.md](docs/DEPLOY-NAS.md).

Environment knobs: `STL2PRISM_DATA` (job storage dir, default `/data` in
the container), `STL2PRISM_JOB_TTL` (seconds before old jobs are purged,
default 86400), `STL2PRISM_MAX_UPLOAD` (bytes, default 200 MB).

## How it works

The pipeline implements the classical reverse-engineering architecture
(segmentation -> primitive fitting -> constraint solving -> rebuild), using
the *extrusion-cylinder* decomposition strategy rather than free surface
stitching — which sidesteps the brittle face-intersection/topology problem
that makes general mesh-to-BREP hard — extended with lofts, cross-axis
features and local patching so that one axis need not explain everything.

1. **Prep** (`mesh_prep`) — load, weld, repair. Vertices closer than 1e-6 of
   the bounding box are welded (exporters leave micron cracks that split a
   part), duplicate/degenerate faces dropped, OBJ per-corner normals/UVs
   merged away. Units (`--units mm|cm|in|ft|m`) scale the mesh on load; the UI
   suggests a unit from the bounding box. Connected shells are split into
   bodies; a shell *inside* another is an internal cavity and is attached to
   its body as a void, so a hollow part becomes one hollow solid. Scan-like
   input (dense, low dihedral, no exactly-coplanar facets) is rebuilt via the
   pymeshlab repair ladder and decimated with topology preservation.
2. **Axis discovery** (`extrusion`) — face normals are clustered on the
   Gaussian sphere; each candidate axis is offered raw *and* snapped to
   global XYZ (when within 5°) and *scored* by the volume fraction of the
   part that has a constant (or linearly varying) cross-section along it,
   with the perpendicular-face area as a tie-breaker. Snapping is a
   hypothesis, not a decision: a part tilted 3° keeps its true axis.
3. **Slab decomposition** — planar faces perpendicular to the axis vote for
   height levels; each slab is cross-sectioned at several heights *and* just
   inside its ends. Where the section starts or stops changing (a chamfer,
   countersink or boss top) or its topology changes, a level is inserted by
   bisection and snapped to a mesh-vertex height. Constancy compares section
   *shapes* (IoU + boundary distance), not just areas.
4. **Profile fitting** (`profile_fit`) — each polygon ring is segmented into
   **lines and circular arcs**: recursive split (at real corners first),
   Taubin circle fits with deviation measured against the whole polyline
   (chord interiors included, so a big circle through the ends of a straight
   wall cannot pass), a cyclic merge pass, boundary refinement between
   neighbours, arc radii/centres re-fitted on the mesh vertices (which lie
   exactly on the CAD surface — section vertices sit on chords), and
   junction solving: line/line at their intersection, tangent line/arc
   fillets solved exactly, lines squared to the dominant frame.
5. **Constraint snapping** (GlobFit-lite, `rebuild._global_snap`) — radii and
   centres are clustered *globally across all slabs* and snapped to cluster
   means. This turns facet-noise families like r = 5.242..5.257 into a
   single design radius.
6. **Rebuild** (`rebuild`) — constant slabs are extruded; slabs whose section
   varies linearly (drafts, chamfers, countersinks, tapered ribs) are
   **lofted with analytic faces** — planes between matched lines, cones /
   cylinders between matched arcs and circles. Equal consecutive slabs are
   merged, Booleans use a fuzzy tolerance, and a finishing pass drops
   micro-edges, unifies same-domain faces and checks validity.
7. **Cross-axis features** (`features`) — curved facet regions are split
   into coaxial primitives and classified: **cylinders** (bores not parallel
   to the main axis, radius/axis refined by least squares on the vertices,
   blind ends kept blind) and **cones** (countersinks, chamfered hole
   mouths) are subtracted as analytic features.
8. **Validate + gate** (`pipeline`) — deterministic, *symmetric* deviation
   (mesh → solid on area-uniform samples plus every vertex; solid → mesh),
   bore deviation, volume error. Passing → prismatic. Failing → the next
   rungs, each held to the same gate:
   * **Face-group engine** (`facegroups`) — the mesh's coplanar components
     are seeds for a greedy, fit-driven region growing (never across a
     dihedral > 25°): a region grows while a plane / cylinder / cone / sphere
     (simplest first) explains its **vertices** within a few microns and its
     facet interiors within the fit tolerance — the second test is what stops
     a wide plane and its first fillet strip from being fitted by an exact,
     absurdly large cylinder. A rolling-ball blend around a curved edge (a
     boss-base fillet, a filleted hole mouth) comes out of growth as a chain
     of short cylinder bands whose axes are all tangents of one circle; the
     engine then fits a **torus** to each such chain (seeded from the bands'
     own axes) and absorbs every neighbour it explains. Directions, coaxial
     axes, coplanar offsets and equal radii are then snapped within
     measurement uncertainty (reverted
     if a snap moves a surface off its vertices), and every region becomes
     one trimmed face on its fitted surface: boundary polylines projected
     onto the surface (a shared vertex table keeps both sides of every edge
     identical for sewing), `MakeFace` + `ShapeFix_Face`, seams placed
     through boundary vertices. Regions no primitive fits, or whose face
     will not build, keep their exact facets. Sew → largest solid → heal →
     check → gate. Mode `facegroup`; no sketch/extrude script for it.
   * **Hybrid patch** (`hybrid`): the prismatic solid with its deviating
     regions boxed and replaced by the exact faceted geometry,
     `(P − B) ∪ (F ∩ B)`, re-checked. Keeps the script.
   * **Faceted** route: coplanar triangles merged into single planar faces
     *before* sewing, curved regions decimated within `--reduce-tol`, sewn
     into a manifold solid.
9. **Export** — one STEP (AP214) with **named bodies and faces coloured by
   surface type**, plus a **CadQuery script** (`<out>.py`) that rebuilds the
   recognised sketches, extrudes, lofts and feature cuts with named
   parameters (radii `R_n`, heights `H_n`) — edit a value, re-run, get a new
   STEP.

### Research basis

The architecture follows the standard two-phase scan-to-BREP paradigm
(segmentation + fitting) established by Schnabel et al.'s Efficient RANSAC
(2007) and surveyed in recent literature; the extrusion-cylinder
decomposition is the classical analogue of Point2Cyl (CVPR 2022) / PrismCAD;
global constraint snapping follows GlobFit (Li et al.); validation-gated
output with honest fallback and local patching is our own addition.

## Measured results (v0.3)

Default gates (fit tol 0.08 mm, p95 ≤ 0.25, max ≤ 0.26, bore ≤ 0.10 mm,
volume ≤ 2 %). "faces" = ADVANCED_FACE count in the STEP.

| part | triangles | mode | faces | max dev | vol err |
|---|---|---|---|---|---|
| servo_bracket_1 | 1,644 | prismatic | 23 (was 60) | 0.060 mm | 0.08 % |
| servo_bracket_2 | 1,712 | prismatic | 19 (was 28) | 0.044 mm | 0.06 % |
| top_arm_1 | 3,896 | prismatic (chamfered bosses as cones) | 44 (was 1,245 faceted) | 0.050 mm | 0.00 % |
| top_arm_2 | 3,816 | prismatic | 40 (was 1,213 faceted) | 0.050 mm | 0.05 % |
| joystick_claw_1 | 2,134 | face-group (27 cyl, 5 spheres, 8 cones, 5 tori) | 153 (was 495 in v0.3.3, 671 patched) | 0.019 mm | 0.07 % |
| joystick_claw_2 | 1,902 | face-group (17 cyl, 12 spheres, 13 cones, 5 tori) | 208 (was 226 in v0.3.2, 341 faceted) | 0.047 mm | 0.04 % |
| frame | 4,200 | face-group (9 cyl, 7 cones) | 101 (was 160 in v0.3.2, 516 faceted) | 0.040 mm | 0.03 % |
| Mesh_90p (scan) | 2.17 M | faceted (scan route) | 39,575 | — | 0.00 % |

The remaining planar faces on the face-group parts are blends (tapered
corner fillets, free-form) that v1 keeps as exact facets.

v0.3.5 (the dry run made honest and fast): the OCC dry run behind the
Boundary Fill outlook now builds exactly the tools the script builds — the
script's own arc growth is executed from its text (the emulation had grown
arcs by at most 30° where Fusion grows by up to 180° and closes near-full
arcs into circles), a cone tool's two ends share one angular window, the
SKIP list is honoured, unbuildable tools are skipped and counted, and cells
are classified by the same procedure as the runtime (probe point, interior
point, face probe for slivers, closest-volume fallback). It is ~50× faster
(one classifier per cell behind a bounding-box test: claw_1's conversion
270 s → 36 s) and judges every body on its own — a region none of whose
probe points lies in a cell, or a body the script had to leave out, is a
FAIL; a kernel error or a body past the time budget is "not checked"
rather than a guess; the script's own header carries the same verdict as
the UI. Also: merged tapered-fillet cones are snapped to their tangent
planes; pieces of one torus/sphere/cone fit (pinch repair) become one tool
instead of two coincident ones; blend chains are walked in chain order; the
segmenter's acceptance test and seeders are shared with the blend merge;
region ids in the script's log carry the engine ids they were made from
(`12<3,4,5>`). claw_1's dry run: 101.2 % enclosed, OK.

v0.3.4 (pinched boundaries, blend tools, real Boundary Fill outlook): a
region whose boundary pinches (an annulus whose hole touches the rim at one
vertex) is repaired by peeling the faces at the pinch instead of falling to
triangles — joystick_claw_1's biggest plane was one such region, 290
triangles for what is now 3 planar faces (claw_1: 495 → 153). Regions that
still get no analytic face are emitted as one planar face per coplanar
facet group instead of raw triangles. For the Boundary Fill script only,
chains of tangent blend bands (tapered / variable-radius fillets) are
merged into single approximate torus or cone tools — a G1 chain of tangent
bands defeats the kernel's cell computation, and a cutting tool only has to
stay within the fit tolerance; the STEP keeps the exact bands. The Boundary
Fill outlook is now an OCC dry run of the actual script (build the tools,
compute the cells, measure the enclosed volume) instead of a heuristic:
claw_1 gets its first OK (cells enclose 100.3 % of the mesh).

v0.3.3 (apex and tapered cones): a pointed cone is one conical face closed
at the tip by a degenerate edge (a pencil part: 1 plane + 1 cylinder +
1 cone), and a tapered pin is a few cones instead of dozens of short
cylinder bands — the cone seeding now centres the axis with a per-slab
circle fit (a sector of a real cone used to be rejected before the
least-squares fit ever ran), and a post-growth cone-chain merge joins
near-coaxial bands, cone sectors and ring-like "sphere" bands (any two
vertex rings lie exactly on some sphere) into single cones. The frame
dropped 160 → 101 faces, the claws to 495 / 208. The Boundary Fill script
emits an apex-reaching cone as one solid cone tool.

v0.3.2 (torus fits): a rolling-ball blend around a curved edge is one torus
face instead of a chain of short cylinder bands (boss-base fillet: 82 → 9
faces; filleted hole mouth: 8 faces; the joystick claws lost 39 and 24
faces). The band chains are found after region growing and the torus is
seeded from the bands' own axes (every band axis is a tangent of the blend's
centre circle), so parts without such chains are untouched. The Boundary
Fill script emits one `createTorus` tool per blend.

v0.3.1 (profile-fit fidelity): the prismatic profile fitter no longer lets
one circle absorb an exactly straight wall next to a gentle arc (a flat
mesh face used to come out as an r = 50–200 mm cylinder, a different one
per slab), walls and arcs are unified across slabs (no more micron-wide
seam faces on flat faces), and a perpendicular face under 0.25 mm from its
neighbour is a level of its own when it is not edge-connected to it (a
0.15 mm ledge used to be merged away). servo_bracket_1 went 60 → 23 faces
with the same deviation.

Synthetic CAD parts (see `tests/test_matrix.py`, `tests/test_facegroups.py`):
plates with holes/fillets, obround slots, hex pockets, stepped shafts, cross
and blind holes, hollow parts, drafted blocks, chamfers, countersinks, small
interior steps, tilted and rotated parts, tiny (3 mm) and huge (1.5 m) parts
convert to the exact face count with sub-0.1 mm deviation on the prismatic
route; a sphere boss (6 planes + 1 sphere), a top-edge-filleted box (6
planes + 4 cylinders), a plate with a spherical dimple, an icosphere (one
spherical face), a boss with a filleted base (7 planes + 1 cylinder + 1
torus) and a plate with a filleted hole mouth (6 + 1 + 1 torus) come out
exact through the face-group engine.

## Limitations (v0.3)

* The **CadQuery / Fusion sketch script** only exists for prismatic results
  (the extrusion engine is the only route that yields sketch + extrude
  structure); face-group results get the Boundary Fill script instead, which
  works only on cleanly fitted parts (see Usage).
* The face-group engine fits **planes, cylinders, cones (apex cones and
  tapered pins included), spheres and tori** (constant-radius rolling-ball
  blends around curved edges); variable-radius corner fillets and free-form
  blends keep their exact facets or stay as short cylinder bands, and a
  torus is only found where the blend spans at least three such bands
  (about 10° of the ring). A gently curved taper is approximated by a few
  cones, as many as the fit tolerance needs. It targets CAD exports
  (coplanar facet pairs); scans stay on the faceted route. Face edges are
  the projected mesh polylines (chords), not surface–surface intersection
  curves yet.
* Scan input defaults to faceted (`--force-prismatic` overrides); the scan
  repair ladder needs pymeshlab (Linux x86_64).
* The CadQuery script reproduces the recognised extrusion structure; patched
  regions are not in the script.

## Roadmap

In rough priority order:

- **Curved-plate consolidation for Boundary Fill.** A body whose gently
  curved plates segment into many near-parallel plane strips (the frame)
  never closes its cells: the strips' tools cannot reach each other at
  grazing angles. Needs curved fits over gently-bent plane chains, tools
  first. (Torus fits landed in v0.3.2; apex cones and tapered pins in
  v0.3.3; pinch repair, blend-chain tools and the OCC dry-run outlook in
  v0.3.4 — variable-radius fillets stay exact bands in the STEP by design.)
- **Clean intersection edges on face-group solids.** Today each analytic face
  in the STEP is bounded by the mesh's own polyline edges. Done via the
  Fusion 360 Boundary Fill script for cleanly fitted parts (experimental,
  see Usage); still to do natively in the STEP.
- **Named features in the generated script** — `hole()`, counterbores and
  `fillet()` calls instead of raw cylinder cuts.
- **Region growing for 3D scans** — real faces on scanned parts instead of a
  faceted solid.
- **Faster conversions** — warm worker process, bodies converted in parallel.

## License

**PolyForm Noncommercial 1.0.0** — see [LICENSE](LICENSE). In short: you may
use, copy, change and share this software for personal, hobby, educational,
research and other noncommercial purposes. Any commercial use needs the
author's permission — get in touch.

Third-party components keep their own licences (CadQuery Apache-2.0,
OpenCascade LGPL-2.1 with exception, trimesh MIT, shapely/networkx BSD).
The optional `scan` extra pulls in **pymeshlab, which is GPL-3.0**; it is not
part of this package's licence and is installed only if you ask for it.
