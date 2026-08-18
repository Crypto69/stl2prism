# stl2prism

Convert STL / OBJ (also PLY, OFF, 3MF, GLB) meshes into **prismatic STEP
solids** — clean BREP with true planes, cylinders and cones you can sketch
on, dimension against, and constrain — replicating the core of Fusion 360's
paid "Prismatic" mesh conversion, plus something Fusion does not give you: an
**editable CadQuery script** of the sketches and extrudes it recognised.
Guaranteed output: when part of a mesh isn't prismatic, that region is
patched with exact faceted geometry; when none of it is, the tool falls back
to a faceted (valid, manifold, coplanar-merged, tolerance-reduced) STEP solid
instead of failing.

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
```

Or from Python:

```python
from stl2prism import run
result = run("part.stl", "part.step")
print(result["mode"], result["metrics"], result["script"])   # 'prismatic' | 'faceted' | 'mixed'
```

Exit code 0 on success. The log reports which mode produced the output and
the measured fidelity (surface deviation both ways, volume error) of
prismatic results.

## Web app

A browser UI for the same pipeline: drag a mesh in, inspect it in 3D, set
the acceptance tolerances, convert, and download the STEP (and the CadQuery
script) with a fidelity report (mode, surface deviation vs. your limits,
volume error, face counts and surface types, patched regions).

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

For deployment on the x86_64 NAS (clone on the NAS, build
natively, optional tailscale HTTPS front), follow the runbook in
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
   bore deviation, volume error. Passing → prismatic. Failing locally →
   **hybrid patch** (`hybrid`): the deviating regions are boxed and replaced
   by the exact faceted geometry, `(P − B) ∪ (F ∩ B)`, and re-checked.
   Failing everywhere → faceted route: coplanar triangles merged into single
   planar faces *before* sewing, curved regions decimated within
   `--reduce-tol`, sewn into a manifold solid.
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
output with honest fallback and local patching is our own addition. See
[DEEP-REVIEW.md](DEEP-REVIEW.md) for the full review, comparison with Fusion
360 / commercial reverse-engineering tools and the roadmap; the working
to-do list is [FIX-PLAN.md](FIX-PLAN.md).

## Measured results (v0.2)

Default gates (fit tol 0.08 mm, p95 ≤ 0.25, max ≤ 0.26, bore ≤ 0.10 mm,
volume ≤ 2 %). "faces" = ADVANCED_FACE count in the STEP.

| part | triangles | mode | faces | max dev | vol err |
|---|---|---|---|---|---|
| servo_bracket_1 | 1,644 | prismatic | 60 (was 76) | 0.066 mm | 0.03 % |
| servo_bracket_2 | 1,712 | prismatic | 28 (was 35) | 0.035 mm | 0.07 % |
| top_arm_1 | 3,896 | prismatic (chamfered bosses as cones) | 45 (was 1,245 faceted) | 0.071 mm | 0.18 % |
| top_arm_2 | 3,816 | prismatic | 41 (was 1,213 faceted) | 0.074 mm | 0.18 % |
| joystick_claw_1 | 2,134 | prismatic, fillet region patched | 671 (was 817 faceted) | 0.050 mm | 0.09 % |
| joystick_claw_2 | 1,902 | faceted (reduced) | 341 (was 811) | — | 0.05 % |
| frame | 4,200 | faceted (reduced) — complex blends | 516 (was 646) | — | 0.01 % |
| Mesh_90p (scan) | — | faceted | — | — | — |

Synthetic CAD parts (see `tests/test_matrix.py`): plates with holes/fillets,
obround slots, hex pockets, stepped shafts, cross and blind holes, hollow
parts, drafted blocks, chamfers, countersinks, small interior steps, tilted
and rotated parts, tiny (3 mm) and huge (1.5 m) parts all convert to the
exact face count with sub-0.1 mm deviation.

## Limitations (v0.2)

* **Single primary axis per body** for solid material (plus cross-axis
  cylindrical holes and conical countersinks). Bodies needing several
  extrusion directions for material get a local faceted patch, or the
  faceted fallback.
* **Spheres, tori (edge fillets), free-form blends** are not fitted as
  analytic surfaces; they are patched with exact facets when local, and
  cause a faceted fallback when they dominate.
* Scan input defaults to faceted (`--force-prismatic` overrides); the scan
  repair ladder needs pymeshlab (Linux x86_64).
* The CadQuery script reproduces the recognised extrusion structure; patched
  regions are not in the script.

## Roadmap

A segmentation-first "face-group engine" (regions → plane/cylinder/cone/
sphere/torus fits → sewn B-rep) for multi-direction parts, sphere/torus
features, hole/fillet feature recognition on the solid, and a Fusion 360
script export — see [FIX-PLAN.md](FIX-PLAN.md).

## License

MIT — see [LICENSE](LICENSE).
