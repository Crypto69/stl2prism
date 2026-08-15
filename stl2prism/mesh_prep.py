"""Mesh loading, repair, and normalization."""
import os

import numpy as np
import trimesh


class PrepError(RuntimeError):
    """Mesh preparation could not produce a usable mesh."""


# Input formats accepted at the user-facing entry points (CLI, web upload).
# trimesh picks the reader from the extension, so keep this list to formats
# it reads without optional dependencies.
SUPPORTED_EXTS = ('.stl', '.obj')

# Neither STL nor OBJ records units; the pipeline works in millimetres (all
# tolerances are mm), so input in another unit is scaled on load. Multiply
# file coordinates by this to get mm.
UNIT_SCALE = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0, 'in': 25.4}


def load_mesh(path):
    """Read an STL or OBJ into one clean, geometry-only Trimesh.

    OBJ exporters commonly write per-corner normals (`vn`) and UVs (`vt`);
    trimesh keeps those as split vertices, and the default merge_vertices()
    refuses to merge vertices whose normal/uv differ — so a perfectly closed
    part reads as non-watertight with no face adjacency at all. We only care
    about geometry, so merge on position alone. Multiple `o`/`g` objects are
    concatenated by force='mesh'; a missing .mtl is only a warning.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise PrepError(
            f"unsupported input format '{ext or '(none)'}'; "
            f"expected one of: {', '.join(SUPPORTED_EXTS)}")
    m = trimesh.load(path, force='mesh')
    m.merge_vertices(merge_tex=True, merge_norm=True)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    return m


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
    """Like load_and_prep, but one prepared mesh per connected body.

    Returns (bodies, is_scan, n_dropped). Bodies are sorted largest first.
    Sliver bodies (see split_bodies) are dropped and counted; a body whose
    scan repair fails is dropped with a log line rather than failing the
    whole file. Single-body files take exactly the load_and_prep path.
    """
    m = _load_scaled(path, units, verbose)
    is_scan = classify(m, verbose, scan_dihedral_deg, scan_min_faces)
    parts, n_dropped = split_bodies(m, is_scan, verbose)
    if len(parts) == 1:
        return [repair(parts[0], is_scan, target_faces, verbose)], is_scan, n_dropped

    if is_scan:
        _require_pymeshlab()   # fail once, loudly, not once per body
    total = sum(len(p.faces) for p in parts)
    bodies = []
    for i, p in enumerate(parts):
        # Share the decimation budget by size, with a floor so small bodies
        # keep enough facets to stay recognisable.
        budget = max(1000, int(target_faces * len(p.faces) / total))
        if verbose and is_scan:
            print(f"[body {i + 1}/{len(parts)}] repairing {len(p.faces)} faces "
                  f"(watertight={p.is_watertight}, budget {budget})")
        try:
            bodies.append(repair(p, is_scan, budget, verbose))
        except PrepError as e:
            if verbose:
                print(f"[body {i + 1}] repair failed ({e}); body dropped")
            n_dropped += 1
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


def classify(m, verbose=True, scan_dihedral_deg=SCAN_DIHEDRAL_DEG,
             scan_min_faces=SCAN_MIN_FACES):
    """Decide scan vs CAD export for the whole file (and log the verdict)."""
    dih = mean_dihedral_deg(m)
    is_scan = len(m.faces) > scan_min_faces and dih < scan_dihedral_deg
    if verbose:
        print(f"[prep] {len(m.faces)} faces, watertight={m.is_watertight}, "
              f"mean dihedral {dih:.1f}deg, "
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
        vol, area = abs(float(p.volume)), float(p.area)
        if area <= 0 or 2 * vol / area < MIN_BODY_THICKNESS:
            return True
    return bool(is_scan and total_faces and n < SCAN_SLIVER_FACES
                and n < SCAN_SLIVER_FRAC * total_faces)


def split_bodies(m, is_scan, verbose=True):
    """Split into connected bodies, largest first; drop slivers.

    Returns (bodies, n_dropped). CAD input keeps every body that could be a
    closed solid, however small — a washer is a part. Scan input additionally
    sheds blobs under SCAN_SLIVER_FACES that are also under SCAN_SLIVER_FRAC
    of the mesh, since those are repair noise, not geometry.
    """
    parts = m.split(only_watertight=False)
    if len(parts) <= 1:
        return [m], 0
    total = len(m.faces)

    def sliver(p):
        return is_sliver(p, is_scan, total)

    kept = sorted((p for p in parts if not sliver(p)),
                  key=lambda p: len(p.faces), reverse=True)
    dropped = len(parts) - len(kept)
    if verbose:
        msg = f"[bodies] {len(parts)} connected bodies"
        if dropped:
            lost = total - sum(len(p.faces) for p in kept)
            msg += (f"; dropping {dropped} sliver(s) with no volume "
                    f"({lost} faces, {lost / total:.2%} of the mesh)")
        print(msg + f"; converting {len(kept)}")
    if not kept:
        raise PrepError('every body is a sliver; nothing to convert')
    return kept, dropped


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


def _poisson_rebuild(m, target_faces, verbose, attempts=REPAIR_ATTEMPTS):
    """Rebuild a scan as a clean watertight mesh, trying each repair route."""
    _require_pymeshlab()

    import multiprocessing as mp
    import tempfile, os

    import gc

    ctx = mp.get_context('spawn')
    best, best_open = None, float('inf')
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, 'in.stl')
        m.export(src)
        # Release the source mesh before the child starts: a 2M-face scan is
        # ~500MB in trimesh, and holding it while pymeshlab loads its own copy
        # doubles peak memory for no reason.
        del m
        gc.collect()
        for i, (method, arg) in enumerate(attempts):
            label = (f'poisson depth {arg}' if method == 'poisson'
                     else f'{method} holes<={arg}')
            dst = os.path.join(td, f'out{i}.ply')
            proc = ctx.Process(target=_pymeshlab_worker,
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
