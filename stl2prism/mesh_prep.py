"""Mesh loading, repair, and normalization."""
import os

import numpy as np
import trimesh


class PrepError(RuntimeError):
    """Mesh preparation could not produce a usable mesh."""


# Input formats accepted at the user-facing entry points (CLI, web upload).
# trimesh picks the reader from the extension, so keep this list to formats
# it reads without optional dependencies.
SUPPORTED_EXTS = ('.stl', '.obj', '.ply', '.off', '.3mf', '.glb', '.gltf')

# Mesh formats record no units (3MF/glTF nominally do, but exporters are
# inconsistent); the pipeline works in millimetres (all tolerances are mm),
# so input in another unit is scaled on load. Multiply file coordinates by
# this to get mm. Same set as Fusion's Insert Mesh.
UNIT_SCALE = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0, 'in': 25.4, 'ft': 304.8}


def suggest_units(m):
    """Guess the file's unit from its bounding box: a mechanical part is a
    few mm to a metre or so. Returns one of UNIT_SCALE's keys; a hint for
    the UI, never applied silently."""
    ext = float(np.max(m.bounding_box.primitive.extents)) if len(m.vertices) else 0.0
    if ext <= 0:
        return 'mm'
    if ext < 3.0:          # a 3 mm-max part is unlikely; probably metres or inches
        return 'm' if ext < 0.5 else 'in'
    if ext > 5000.0:       # 5 m in mm? more likely a micron/point export — keep mm
        return 'mm'
    return 'mm'


def load_mesh(path, weld_tol=None):
    """Read an STL or OBJ into one clean, geometry-only Trimesh.

    OBJ exporters commonly write per-corner normals (`vn`) and UVs (`vt`);
    trimesh keeps those as split vertices, and the default merge_vertices()
    refuses to merge vertices whose normal/uv differ — so a perfectly closed
    part reads as non-watertight with no face adjacency at all. We only care
    about geometry, so merge on position alone. Multiple `o`/`g` objects are
    concatenated by force='mesh'; a missing .mtl is only a warning.

    Vertices closer than `weld_tol` (default: 1e-6 of the bounding-box
    diagonal, at least 1e-4 model units) are welded. Exporters that
    tessellate faces independently leave cracks of ~1e-5 between faces that
    exact merging cannot close; without welding such a part splits into
    several open bodies.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise PrepError(
            f"unsupported input format '{ext or '(none)'}'; "
            f"expected one of: {', '.join(SUPPORTED_EXTS)}")
    m = trimesh.load(path, force='mesh')
    return clean_mesh(m, weld_tol=weld_tol)


def clean_mesh(m, weld_tol=None):
    """Weld near-duplicate vertices, drop degenerate/duplicate faces."""
    m.merge_vertices(merge_tex=True, merge_norm=True)
    weld_vertices(m, weld_tol)
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    return m


def weld_tolerance(m):
    """Default welding tolerance: relative to size, with an absolute floor."""
    diag = float(np.linalg.norm(m.bounding_box.primitive.extents)) \
        if len(m.vertices) else 0.0
    return max(1e-4, 1e-6 * diag)


def weld_vertices(m, tol=None):
    """Merge vertices within `tol` of each other (union-find over KD-tree
    pairs), in place. Unlike grid rounding this cannot miss pairs that
    straddle a rounding boundary."""
    if len(m.vertices) < 2:
        return
    tol = weld_tolerance(m) if tol is None else float(tol)
    if tol <= 0:
        return
    from scipy.spatial import cKDTree
    V = m.vertices.view(np.ndarray)
    pairs = cKDTree(V).query_pairs(tol, output_type='ndarray')
    if len(pairs) == 0:
        return
    parent = np.arange(len(V))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(len(V))])
    uniq, inv = np.unique(roots, return_inverse=True)
    newV = np.zeros((len(uniq), 3))
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    for k in range(3):
        newV[:, k] = np.bincount(inv, weights=V[:, k], minlength=len(uniq)) / counts
    faces = inv[m.faces]
    m.vertices = newV
    m.faces = faces


# Mean dihedral angle (degrees) below which geometry is treated as a scan.
# Calibrated on the sample set: scans measure 3.4-4.2, CAD exports 11.2-29.2.
# A very finely tessellated CAD export would also sit low here, which is why
# this is paired with a face-count floor rather than used alone.
SCAN_DIHEDRAL_DEG = 8.0
SCAN_MIN_FACES = 50000


def mean_dihedral_deg(m, cap=400000):
    """Mean absolute dihedral angle across face adjacencies, in degrees.

    Scans tessellate smooth surfaces densely, so neighbouring facets differ
    by very little; CAD exports put their facets where the curvature is and
    meet at genuine feature angles.
    """
    a = m.face_adjacency_angles
    if len(a) == 0:
        return float('inf')
    if len(a) > cap:
        a = a[np.linspace(0, len(a) - 1, cap).astype(np.int64)]
    return float(np.degrees(np.abs(a)).mean())


def load_and_prep(path, target_faces=40000, verbose=True,
                  scan_dihedral_deg=SCAN_DIHEDRAL_DEG,
                  scan_min_faces=SCAN_MIN_FACES, units='mm'):
    """Load an STL or OBJ, repair it, and return (mesh, is_scan) in mm.

    `units` names the unit the file's coordinates are in (see UNIT_SCALE);
    the mesh is scaled to mm before anything else looks at it.

    `is_scan` (organic geometry with no analytic surfaces to recover) and
    `needs_repair` (not watertight) are judged separately: a CAD export with
    one unstitched seam needs repair but must still reach the prismatic path.
    """
    m = _load_scaled(path, units, verbose)
    is_scan = classify(m, verbose, scan_dihedral_deg, scan_min_faces)
    return repair(m, is_scan, target_faces, verbose), is_scan


def load_and_prep_bodies(path, target_faces=40000, verbose=True,
                         scan_dihedral_deg=SCAN_DIHEDRAL_DEG,
                         scan_min_faces=SCAN_MIN_FACES, units='mm'):
    """Like load_and_prep, but one prepared `Body` per solid.

    Returns (bodies, is_scan, n_dropped). Bodies are sorted largest first.
    Sliver bodies (see split_bodies) are dropped and counted; a body whose
    scan repair fails is dropped with a log line rather than failing the
    whole file. Internal voids are repaired like their bodies.
    """
    m = _load_scaled(path, units, verbose)
    is_scan = classify(m, verbose, scan_dihedral_deg, scan_min_faces)
    groups, n_dropped = split_bodies(m, is_scan, verbose)
    if len(groups) == 1 and not groups[0].voids:
        return [Body(repair(groups[0].mesh, is_scan, target_faces, verbose))], \
            is_scan, n_dropped

    if is_scan:
        _require_pymeshlab()   # fail once, loudly, not once per body
    total = sum(g.n_faces_total for g in groups)
    bodies = []
    for i, g in enumerate(groups):
        # Share the decimation budget by size, with a floor so small bodies
        # keep enough facets to stay recognisable.
        budget = max(1000, int(target_faces * len(g.mesh.faces) / total))
        if verbose and is_scan:
            print(f"[body {i + 1}/{len(groups)}] repairing {len(g.mesh.faces)} faces "
                  f"(watertight={g.mesh.is_watertight}, budget {budget})")
        try:
            outer = repair(g.mesh, is_scan, budget, verbose)
        except PrepError as e:
            if verbose:
                print(f"[body {i + 1}] repair failed ({e}); body dropped")
            n_dropped += 1
            continue
        voids = []
        for v in g.voids:
            vb = max(1000, int(target_faces * len(v.faces) / total))
            try:
                voids.append(repair(v, is_scan, vb, verbose))
            except PrepError as e:
                if verbose:
                    print(f"[body {i + 1}] void repair failed ({e}); void dropped")
                n_dropped += 1
        bodies.append(Body(outer, voids))
    if not bodies:
        raise PrepError('no body survived preparation')
    return bodies, is_scan, n_dropped


def _load_scaled(path, units, verbose):
    if units not in UNIT_SCALE:
        raise PrepError(f"unknown units '{units}'; "
                        f"expected one of: {', '.join(UNIT_SCALE)}")
    m = load_mesh(path)
    if UNIT_SCALE[units] != 1.0:
        m.apply_scale(UNIT_SCALE[units])
        if verbose:
            print(f"[prep] input units {units}: scaled x{UNIT_SCALE[units]:g} to mm")
    return m


# Fraction of face adjacencies that are exactly coplanar (< 0.06 deg). CAD
# exporters triangulate every planar face, so a CAD export has many (30-66%
# on the CAD samples, 50% on a very fine synthetic export); a scan has few
# but not none (Mesh_90p 4.3%, rc2-clean-controller 3.6%: flat scanned
# regions do produce some pairs within 1e-3 rad). Used with the dihedral
# mean and the face count so a very finely tessellated CAD part is not
# mistaken for a scan. 0.02 misrouted both real scans to the CAD path.
SCAN_MAX_COPLANAR_FRAC = 0.10


def coplanar_fraction(m, cap=400000, tol_rad=1e-3):
    a = m.face_adjacency_angles
    if len(a) == 0:
        return 0.0
    if len(a) > cap:
        a = a[np.linspace(0, len(a) - 1, cap).astype(np.int64)]
    return float((np.abs(a) < tol_rad).mean())


def classify(m, verbose=True, scan_dihedral_deg=SCAN_DIHEDRAL_DEG,
             scan_min_faces=SCAN_MIN_FACES,
             scan_max_coplanar_frac=SCAN_MAX_COPLANAR_FRAC):
    """Decide scan vs CAD export for the whole file (and log the verdict).

    Scan = many faces AND small mean dihedral AND (almost) no exactly
    coplanar facet pairs. Each test alone misfires: a dense CAD export has a
    low dihedral mean but plenty of coplanar pairs; a decimated scan has few
    faces but no coplanar pairs either — that one is treated as CAD, which
    only costs a (gated) prismatic attempt.
    """
    dih = mean_dihedral_deg(m)
    cop = coplanar_fraction(m)
    is_scan = (len(m.faces) > scan_min_faces and dih < scan_dihedral_deg
               and cop < scan_max_coplanar_frac)
    if verbose:
        print(f"[prep] {len(m.faces)} faces, watertight={m.is_watertight}, "
              f"mean dihedral {dih:.1f}deg, coplanar pairs {cop:.1%}, "
              f"treating as {'scan' if is_scan else 'CAD export'}"
              f"{'' if m.is_watertight else ' (needs repair)'}")
    return is_scan


def repair(m, is_scan, target_faces=40000, verbose=True):
    """Close and normalise one body: Poisson/stitch ladder for scans, hole
    filling for leaky CAD exports, consistent outward normals for all."""
    if is_scan:
        m = _poisson_rebuild(m, target_faces, verbose)
    elif not m.is_watertight:
        trimesh.repair.fill_holes(m)
        if not m.is_watertight and verbose:
            print("[prep] warning: mesh still not watertight after repair")
    trimesh.repair.fix_normals(m)
    return m


# A body this small cannot be closed (a tetrahedron is 4 faces); anything
# below is an export artefact, not a part.
MIN_BODY_FACES = 4
# A closed body thinner than this on average (2*volume/area, mm) encloses
# nothing: flattened triangle pairs, zero-thickness decals. CAD exports of
# assemblies are full of them and they cannot become solids.
MIN_BODY_THICKNESS = 1e-3
# Scans shed detached blobs; on scan input a body is also dropped if it is
# both tiny in absolute terms and negligible relative to the whole mesh.
SCAN_SLIVER_FACES = 100
SCAN_SLIVER_FRAC = 0.001


def is_sliver(p, is_scan=False, total_faces=None):
    """True for a body that cannot be a solid: too few faces to close, closed
    but with no volume (see MIN_BODY_THICKNESS), or — scan input only — a
    detached blob that is negligible in both absolute and relative terms."""
    n = len(p.faces)
    if n < MIN_BODY_FACES:
        return True
    if p.is_watertight:
        with np.errstate(divide='ignore', invalid='ignore'):   # zero-volume shells
            vol, area = abs(float(p.volume)), float(p.area)
        if area <= 0 or 2 * vol / area < MIN_BODY_THICKNESS:
            return True
    return bool(is_scan and total_faces and n < SCAN_SLIVER_FACES
                and n < SCAN_SLIVER_FRAC * total_faces)


class Body:
    """One solid to convert: an outer shell plus the shells of any internal
    voids (cavities). Each is a watertight-ish Trimesh; voids are converted
    as positive solids and subtracted from the outer solid."""

    def __init__(self, mesh, voids=None):
        self.mesh = mesh
        self.voids = list(voids or [])

    @property
    def faces(self):
        return self.mesh.faces

    @property
    def is_watertight(self):
        return bool(self.mesh.is_watertight)

    @property
    def n_faces_total(self):
        return len(self.mesh.faces) + sum(len(v.faces) for v in self.voids)

    def volume(self):
        """Enclosed material volume (outer minus voids); NaN if not closed."""
        if not self.mesh.is_watertight:
            return float('nan')
        v = abs(float(self.mesh.volume))
        for h in self.voids:
            if not h.is_watertight:
                return float('nan')
            v -= abs(float(h.volume))
        return v


def split_bodies(m, is_scan, verbose=True):
    """Split into connected bodies, largest first; drop slivers; nest voids.

    Returns (bodies, n_dropped) where each body is a `Body`. CAD input keeps
    every body that could be a closed solid, however small — a washer is a
    part. Scan input additionally sheds blobs under SCAN_SLIVER_FACES that
    are also under SCAN_SLIVER_FRAC of the mesh, since those are repair
    noise, not geometry.

    A shell that lies inside another (odd nesting depth) is a cavity, not a
    part: it is attached to its container as a void so the result is one
    hollow solid rather than two overlapping positive solids. A shell inside
    a cavity (even depth) is a separate part again.
    """
    parts = m.split(only_watertight=False)
    if len(parts) <= 1:
        return [Body(m)], 0
    total = len(m.faces)

    def sliver(p):
        return is_sliver(p, is_scan, total)

    kept = sorted((p for p in parts if not sliver(p)),
                  key=lambda p: len(p.faces), reverse=True)
    dropped = len(parts) - len(kept)
    if not kept:
        raise PrepError('every body is a sliver; nothing to convert')
    bodies = nest_shells(kept)
    if verbose:
        msg = f"[bodies] {len(parts)} connected bodies"
        if dropped:
            # count what was dropped directly: split() can re-process parts,
            # so 'total minus kept' is not exact
            lost = sum(len(p.faces) for p in parts if sliver(p))
            msg += (f"; dropping {dropped} sliver(s) with no volume "
                    f"({lost} faces, {lost / total:.2%} of the mesh)")
        n_voids = sum(len(b.voids) for b in bodies)
        if n_voids:
            msg += f"; {n_voids} internal void(s) attached to their bodies"
        print(msg + f"; converting {len(bodies)}")
    return bodies, dropped


def nest_shells(parts):
    """Group shells into Bodies by containment.

    `parts` are Trimesh shells sorted largest first. Shell j contains shell i
    if j is watertight, j's bounding box contains i's, and a vertex of i is
    inside j. Depth = number of containers; odd depth => void of the deepest
    container; even depth => positive body.
    """
    n = len(parts)
    if n == 1:
        return [Body(parts[0])]
    lo = [p.bounds[0] for p in parts]
    hi = [p.bounds[1] for p in parts]
    containers = [[] for _ in range(n)]
    for j, pj in enumerate(parts):
        if not pj.is_watertight or len(pj.faces) < 4:
            continue
        cand = [i for i in range(n) if i != j
                and np.all(lo[i] >= lo[j] - 1e-9) and np.all(hi[i] <= hi[j] + 1e-9)]
        if not cand:
            continue
        # one probe point per candidate: a vertex of i pushed a hair inward
        # along its own (outward) normal is robust even if surfaces touch
        pts = []
        for i in cand:
            pi = parts[i]
            k = 0
            v = pi.vertices[k] - 1e-6 * pi.vertex_normals[k] \
                if len(pi.vertex_normals) else pi.vertices[k]
            pts.append(v)
        try:
            inside = pj.contains(np.array(pts))
        except Exception:
            continue
        for i, ok in zip(cand, inside):
            if ok:
                containers[i].append(j)
    depth = [len(c) for c in containers]
    bodies = {}
    for i in range(n):
        if depth[i] % 2 == 0:
            bodies[i] = Body(parts[i])
    for i in range(n):
        if depth[i] % 2 == 1:
            # deepest container = the one with the largest depth
            parent = max(containers[i], key=lambda j: depth[j])
            if parent in bodies:
                bodies[parent].voids.append(parts[i])
            else:                      # should not happen; keep as a part
                bodies[i] = Body(parts[i])
    return [bodies[i] for i in sorted(bodies)]


def _pymeshlab_worker(src, dst, target_faces, method, arg):
    """Repair a scan and decimate it, in a child process.

    Isolated because screened Poisson terminates the process on some inputs
    ("Failed to close loop") — and terminates it with status 0, so neither a
    try/except nor an exit code tells you it failed. Only the absence of the
    output file does.
    """
    import pymeshlab
    import trimesh as tm

    mid = dst + '.mid.ply'
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(src)
    if method == 'decimate':
        # Pre-repair reduction only: shrink a huge scan once so every repair
        # rung below works on a fraction of the triangles. preservetopology
        # keeps hole boundaries, so holes stay closable afterwards.
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=arg, preservetopology=True,
            preservenormal=True, planarquadric=True)
        ms.save_current_mesh(dst)
        return
    if method == 'poisson':
        ms.generate_sampling_poisson_disk(samplenum=250000, exactnumflag=False)
        ms.compute_normal_for_point_clouds(k=12)
        ms.generate_surface_reconstruction_screened_poisson(
            depth=arg, samplespernode=3.0)
    else:
        # Direct repair: stitch the scan as-is rather than resurfacing it.
        # Keeps the measured geometry, and has no fragile solver to abort.
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        # `arg` is the hole size cap, in edges. Scans vary hugely in how big
        # their gaps are, so this is retried wider rather than guessed once.
        # selfintersection=True refuses to bridge a hole with overlapping
        # triangles, which keeps the solid BRepCheck-clean but leaves the
        # hardest holes open. 'close_loose' allows them: a closed shell that
        # fails solid-level validation still beats an open one, so it is
        # tried only after the strict pass has had its chance.
        ms.meshing_close_holes(maxholesize=arg,
                               selfintersection=(method == 'close'))
    ms.save_current_mesh(mid)

    # Keep only the largest connected body: both routes leave small detached
    # blobs, and decimating those wastes the face budget.
    p = tm.load(mid)
    p.merge_vertices()
    main = max(p.split(only_watertight=False), key=lambda c: len(c.faces))
    main.export(mid)

    # Quadric decimation with preservetopology=True. trimesh's
    # fast_simplification path is faster but breaks manifoldness, which
    # leaves the sewing stage with hundreds of disjoint shells.
    ms2 = pymeshlab.MeshSet()
    ms2.load_new_mesh(mid)
    ms2.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces, preservetopology=True,
        preservenormal=True, planarquadric=True)
    ms2.save_current_mesh(dst)


# Repair attempts in order. Direct stitching runs first: it preserves the
# measured surface and cannot abort, whereas screened Poisson resurfaces the
# part and fails on some scans regardless of depth. Poisson stays as the
# fallback for scans too broken to stitch.
REPAIR_ATTEMPTS = (('close', 3000), ('close', 100000),
                   ('close_loose', 3000), ('close_loose', 100000),
                   ('poisson', 10), ('poisson', 9), ('poisson', 8))

# Scans above this size are decimated once, before the ladder. Each rung
# loads, repairs, splits and decimates the whole mesh; on a 2.2M-face scan
# that is 2-3 minutes per rung (7 rungs = the better part of half an hour)
# for a result that is 40k faces anyway. Poisson resamples to 250k points
# regardless, and hole closing only looks at boundaries, so neither route
# loses anything meaningful at 300k faces (0.4 mm edges on a 185 mm scan).
PRE_REPAIR_FACES = 300000


def _open_edges(d):
    """Count edges with only one adjacent face — how far from closed a mesh is.

    Ranks near-miss repairs against each other when none of them close fully;
    face count alone cannot tell a nearly-sealed mesh from a badly torn one.
    """
    import trimesh as tm
    return len(tm.grouping.group_rows(d.edges_sorted, require_count=1))


def _require_pymeshlab():
    try:
        import pymeshlab  # noqa: F401
    except ImportError as e:
        # Distinguish "not installed" from "installed but won't load" (e.g.
        # a missing system library) — the remedies are entirely different.
        if e.name == 'pymeshlab':
            raise PrepError(
                "scan input needs pymeshlab, which is an optional dependency; "
                "install it with: pip install 'stl2prism[scan]'") from e
        raise PrepError(
            f"pymeshlab is installed but failed to load ({e}); "
            f"a system library is probably missing") from e


def _poisson_rebuild(m, target_faces, verbose, attempts=REPAIR_ATTEMPTS,
                     pre_faces=PRE_REPAIR_FACES, worker=None):
    """Rebuild a scan as a clean watertight mesh, trying each repair route.

    `worker` is the child-process function (default `_pymeshlab_worker`);
    tests substitute a pymeshlab-free stand-in.
    """
    if worker is None:
        _require_pymeshlab()
        worker = _pymeshlab_worker

    import multiprocessing as mp
    import tempfile, os

    import gc

    ctx = mp.get_context('spawn')
    best, best_open = None, float('inf')
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, 'in.stl')
        n_in = len(m.faces)
        m.export(src)
        # Release the source mesh before the child starts: a 2M-face scan is
        # ~500MB in trimesh, and holding it while pymeshlab loads its own copy
        # doubles peak memory for no reason.
        del m
        gc.collect()
        if pre_faces and n_in > pre_faces:
            pre = os.path.join(td, 'pre.ply')
            proc = ctx.Process(target=worker,
                               args=(src, pre, target_faces, 'decimate', int(pre_faces)))
            proc.start()
            proc.join()
            if os.path.exists(pre):
                src = pre
                if verbose:
                    print(f"[prep] pre-reduced {n_in} -> ~{int(pre_faces)} faces "
                          f"before repair")
            elif verbose:
                print(f"[prep] pre-reduction failed (exit {proc.exitcode}); "
                      f"repairing at full size")
        for i, (method, arg) in enumerate(attempts):
            label = (f'poisson depth {arg}' if method == 'poisson'
                     else f'{method} holes<={arg}')
            dst = os.path.join(td, f'out{i}.ply')
            proc = ctx.Process(target=worker,
                               args=(src, dst, target_faces, method, arg))
            proc.start()
            proc.join()
            # The output file is the only trustworthy success signal: the
            # Poisson solver exits 0 even when it has given up.
            if not os.path.exists(dst):
                if verbose:
                    print(f"[prep] {label} failed (exit {proc.exitcode}); "
                          f"trying next repair route")
                continue
            d = _clean(trimesh.load(dst))
            if d.is_watertight:
                if verbose:
                    print(f"[prep] repair via {label}")
                best = d
                break
            # A non-watertight repair sews into an open shell, so keep looking;
            # hold on to the closest-to-closed effort in case nothing closes.
            open_e = _open_edges(d)
            if verbose:
                print(f"[prep] {label} -> not watertight "
                      f"({open_e} open edges); trying next repair route")
            if best is None or open_e < best_open:
                best, best_open = d, open_e
    if best is None:
        raise PrepError(
            f"every repair route failed "
            f"({', '.join(a[0] for a in attempts)}); "
            f"the scan may be too noisy or too large")
    if verbose:
        print(f"[prep] scan rebuild -> {len(best.faces)} faces, "
              f"watertight={best.is_watertight}")
        if not best.is_watertight:
            print("[prep] warning: no repair route produced a watertight mesh; "
                  "the export will be an open shell, not a closed solid")
    return best


def _clean(d):
    d.merge_vertices()
    d.update_faces(d.nondegenerate_faces())
    d.remove_unreferenced_vertices()
    if not d.is_watertight:
        trimesh.repair.fill_holes(d)
    return d
