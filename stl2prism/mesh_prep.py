"""Mesh loading, repair, and normalization."""
import numpy as np
import trimesh


def load_and_prep(path, scan_faces_threshold=200000, target_faces=40000, verbose=True):
    """Load an STL, repair it, and return a watertight-ish mesh.

    CAD-exported meshes are passed through with light cleanup.
    Scan-like meshes (huge, noisy, holed) get Poisson reconstruction.
    """
    m = trimesh.load(path, force='mesh')
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()

    is_scan = len(m.faces) > scan_faces_threshold or not m.is_watertight
    if verbose:
        print(f"[prep] {len(m.faces)} faces, watertight={m.is_watertight}, "
              f"treating as {'scan' if is_scan else 'CAD export'}")

    if is_scan and len(m.faces) > scan_faces_threshold:
        m = _poisson_rebuild(m, target_faces, verbose)
    elif not m.is_watertight:
        trimesh.repair.fill_holes(m)
        if not m.is_watertight and verbose:
            print("[prep] warning: mesh still not watertight after repair")

    trimesh.repair.fix_normals(m)
    return m, is_scan


def _poisson_rebuild(m, target_faces, verbose):
    import pymeshlab
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, 'in.stl')
        dst = os.path.join(td, 'out.stl')
        m.export(src)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(src)
        ms.generate_sampling_poisson_disk(samplenum=250000, exactnumflag=False)
        ms.compute_normal_for_point_clouds(k=12)
        ms.generate_surface_reconstruction_screened_poisson(depth=10, samplespernode=3.0)
        ms.save_current_mesh(dst)
        p = trimesh.load(dst)
    p.merge_vertices()
    main = sorted(p.split(only_watertight=False), key=lambda x: len(x.faces), reverse=True)[0]
    d = main.simplify_quadric_decimation(face_count=target_faces)
    d.merge_vertices()
    if verbose:
        print(f"[prep] poisson rebuild -> {len(d.faces)} faces, watertight={d.is_watertight}")
    return d
