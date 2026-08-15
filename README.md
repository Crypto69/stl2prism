# stl2prism

Convert STL or OBJ meshes into **prismatic STEP solids** — clean BREP with true
planes and cylinders you can sketch on, dimension against, and constrain —
replicating the core of Fusion 360's paid "Prismatic" mesh conversion.
Guaranteed output: when a mesh isn't prismatic, the tool falls back to a
faceted (but valid, manifold, coplanar-merged) STEP solid instead of failing.

## Install

```bash
pip install .            # core
pip install .[scan]      # + pymeshlab, for 3D-scan repair (Poisson)
```

## Usage

```bash
stl2prism part.stl                 # -> part.step
stl2prism part.obj                 # OBJ works the same way
stl2prism part.stl out.step --tol 0.05
stl2prism part.obj --units cm      # file is in cm (Fusion's OBJ default); scale to mm
stl2prism scan.stl --force-prismatic   # attempt prismatic on scan input
```

Or from Python:

```python
from stl2prism import run
result = run("part.stl", "part.step")
print(result["mode"], result["metrics"])   # 'prismatic' | 'faceted'
```

Exit code 0 on success. The log reports which mode produced the output and
the measured fidelity (surface deviation, volume error) of prismatic results.

## Web app

A browser UI for the same pipeline: drag an STL or OBJ in, inspect it in 3D,
set the acceptance tolerances, convert, and download the STEP with a
fidelity report (mode, surface deviation vs. your limits, volume error,
face counts and surface types for both files).

### Run locally (dev)

```bash
python -m venv .venv && .venv/bin/pip install -e . fastapi 'uvicorn[standard]' python-multipart
.venv/bin/uvicorn backend.main:app --port 8000     # API
cd frontend && npm install && npm run dev           # UI on :5173, proxies /api
```

### Run as a container

```bash
docker compose up --build        # then open http://localhost:8321
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
that makes general mesh-to-BREP hard.

1. **Prep** (`mesh_prep`) — load, merge, repair. Input is STL or OBJ
   (geometry only: OBJ per-corner normals/UVs are merged away, `.mtl`
   materials are ignored, quads/n-gons are triangulated, and multiple
   objects are combined into one mesh). Neither format records units and
   the pipeline works in mm, so `--units cm|in|m` (UI: "Input units")
   scales the mesh on load. Scan-like input (dense
   tessellation, identified by a low mean dihedral angle) is rebuilt via
   screened Poisson reconstruction and decimated with topology preservation.
   Being non-watertight is treated as *needs repair*, not as *is a scan*: a
   CAD export with an unstitched seam is repaired and still gets the
   prismatic treatment.
2. **Axis discovery** (`extrusion`) — face normals are clustered on the
   Gaussian sphere; the top candidate axes (snapped to global XYZ within
   5°) are each *scored* by the volume fraction of the part that has a
   constant cross-section along them. Best axis wins. This is what lets a
   part whose largest face is tilted still be recognised as an extrusion
   along a different direction.
3. **Slab decomposition** — planar faces perpendicular to the axis vote
   for discrete height levels; between consecutive levels the mesh is
   cross-sectioned (several times per slab, to verify constancy) yielding
   profile polygons with holes.
4. **Profile fitting** (`profile_fit`) — each polygon ring is segmented
   into **lines and circular arcs** by greedy split: fit one primitive to
   a span, split at max-deviation point if over tolerance, recurse.
   Circles use the Taubin algebraic fit (bias-corrected, noise-robust).
   Full-circle rings are detected directly.
5. **Constraint snapping** (GlobFit-lite, `rebuild._global_snap`) —
   near-axis-aligned lines are squared up; radii and centers are clustered
   *globally across all slabs* and snapped to cluster means, then arc
   endpoints are re-projected onto the snapped circles and each ring chain
   re-closed. This is what turns facet-noise families like
   r = 5.242..5.257 into a single r = 5.24 design radius.
6. **Cross-axis features** (`features`) — cylindrical bores not parallel
   to the main axis (e.g. screw holes through a side wall) are recovered
   independently: curved facet regions are clustered, the cylinder axis is
   taken from the null-space of the region's normal covariance (all
   cylinder normals are perpendicular to its axis), radius/center from a
   Taubin fit in the perpendicular plane, verified by radial-normal
   alignment, coaxial fragments merged — then boolean-subtracted.
7. **Rebuild + gate** (`rebuild`, `pipeline`) — slabs are extruded with
   CadQuery/OpenCascade and unioned. The result is *validated against the
   input mesh* (sampled surface deviation + volume error); only if it
   passes (default: p95 <= 0.25 mm, volume within 2%) is the prismatic
   solid written. Otherwise the tool falls back to the faceted converter
   (triangle sewing -> manifold solid -> coplanar-face unification ->
   AP214 STEP with an explicit MANIFOLD_SOLID_BREP).
8. **Multi-body files** — a mesh holding several disconnected bodies (an
   assembly export, a controller with knobs and sticks) is split into
   bodies first; steps 2-7 run per body, and every body is written into
   the *one* STEP as a separate solid, so CAD imports it as multiple
   bodies of one component. The result reports `mixed` when some bodies
   went prismatic and others faceted, with a per-body table. Bodies that
   cannot be closed (under 4 triangles; on scan input also stray blobs
   under 100 triangles and 0.1% of the mesh) are dropped and counted.

### Research basis

The architecture follows the standard two-phase scan-to-BREP paradigm
(segmentation + fitting) established by Schnabel et al.'s Efficient RANSAC
(2007) and surveyed in recent literature; the extrusion-cylinder
decomposition is the classical analogue of Point2Cyl (CVPR 2022); global
constraint snapping follows GlobFit (Li et al.); validation-gated output
with honest fallback is our own addition. Neural pipelines (Point2CAD,
ParseNet, CAD-Recode) were evaluated and rejected for this tool: they
need GPU inference stacks and mostly carry non-commercial licenses.

## Measured results (v0.1)

| part | mode | faces (vs faceted) | p95 dev | max dev | vol err |
|---|---|---|---|---|---|
| servo_bracket_1 | prismatic | 76 (was 470)  | 0.023 mm | 0.10 mm | 0.14% |
| servo_bracket_2 | prismatic | 35 (was 582)  | 0.021 mm | 0.19 mm | 0.36% |
| frame           | prismatic | 70 (was 646)  | 0.103 mm | 1.95 mm | 1.29% |
| Mesh_90p (scan) | faceted   | ~40k          | —        | —       | —      |

## Limitations (v0.1)

* **Single primary axis per body.** One extrusion direction per body (plus
  cross-axis cylindrical holes). Bodies needing several extrusion
  directions for solid material (not just holes) get the faceted fallback.
* **Tapered/lofted features** — gussets, draft angles, chamfered ribs —
  are approximated by their mid-height section. Deviation shows in the
  report; the frame's 1.95 mm max is its countersink cones.
* **Countersinks/cones, spheres, tori, fillets between slabs** are not
  fitted as analytic surfaces yet. Fillets *within* a profile plane are
  captured (they're arcs).
* Scan input defaults to faceted; `--force-prismatic` overrides.

## Roadmap ideas

Multi-region decomposition (per-region axes + boolean assembly), cone
fitting for countersinks, sketch-constraint export (tangency, symmetry),
and emitting the CadQuery build script itself so the output is not just a
STEP but an editable parametric program.

## License

MIT — see [LICENSE](LICENSE).
