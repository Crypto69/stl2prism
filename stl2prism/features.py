"""Detect cylindrical bores whose axis differs from the primary extrusion
axis, and subtract them from the rebuilt solid.

Cylinder recovery per curved region:
  * axis = eigenvector of the face-normal covariance with the smallest
    eigenvalue (all cylinder normals are perpendicular to its axis)
  * center/radius = Taubin circle fit of face centers projected onto the
    plane perpendicular to that axis
  * accept if radial residual is small and normals are consistent
"""
import numpy as np
import trimesh
from .profile_fit import fit_circle_taubin


def find_cross_cylinders(mesh, primary_axis, angle_tol_deg=12.0,
                         max_r=30.0, min_faces=8, resid_tol=0.12,
                         exclude_parallel=True):
    """Return list of dicts {axis, center, r, h0, h1} for cylindrical bores.

    With exclude_parallel=True (default) only cross-axis bores are returned —
    the ones that must be subtracted, since axis-parallel holes are already
    interior rings of the extruded profile. exclude_parallel=False keeps the
    parallel ones too (tagged 'parallel'), which validation needs: a hole is
    a hole regardless of which stage of the rebuild produced it."""
    n = mesh.face_normals
    curved = _curved_face_mask(mesh)
    if curved.sum() < min_faces:
        return []
    adj = mesh.face_adjacency
    both = curved[adj].all(axis=1)
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(np.where(curved)[0])
    G.add_edges_from(adj[both])
    out = []
    for comp in nx.connected_components(G):
        comp = np.array(sorted(comp))
        if len(comp) < min_faces:
            continue
        nn = n[comp]
        # cylinder axis: null direction of the normals
        C = nn.T @ nn
        w, v = np.linalg.eigh(C)
        axis = v[:, 0]
        if w[0] > 0.05 * w[2]:          # normals not planar enough
            continue
        cos = abs(float(axis @ primary_axis))
        parallel = cos > np.cos(np.radians(angle_tol_deg))
        if parallel and exclude_parallel:
            continue                     # parallel to primary: already built
        centers = mesh.triangles_center[comp]
        basis = _plane_basis(axis)
        uv = np.column_stack([centers @ basis[0], centers @ basis[1]])
        fit = fit_circle_taubin(uv)
        if fit is None or fit['dev'] > resid_tol or fit['r'] > max_r:
            continue
        # verify normals point radially
        c3 = fit['center'][0] * basis[0] + fit['center'][1] * basis[1]
        radial = centers - c3 - np.outer(centers @ axis, axis)
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        signed = (radial * nn).sum(axis=1)
        if np.abs(signed).mean() < 0.95:
            continue
        # Outward mesh normals pointing back toward the cylinder axis mean
        # the surface encloses a void: a hole. Pointing away means a boss or
        # fillet — an outer surface, not a mating feature.
        concave = float(signed.mean()) < 0.0
        hs = centers @ axis
        verts = mesh.vertices[np.unique(mesh.faces[comp])]
        vh = verts @ axis
        out.append({'axis': axis, 'center2': fit['center'], 'basis': basis,
                    'r': float(fit['r']), 'parallel': bool(parallel),
                    'concave': concave,
                    'h0': float(vh.min()), 'h1': float(vh.max()),
                    'faces': len(comp)})
    return _merge_coaxial(out)


def _curved_face_mask(mesh, lo_deg=4.0, hi_deg=45.0):
    """Faces participating in gently-curved regions (facet dihedrals in a
    band: bigger than coplanar noise, smaller than sharp edges)."""
    adj = mesh.face_adjacency
    ang = np.degrees(mesh.face_adjacency_angles)
    band = (ang > lo_deg) & (ang < hi_deg)
    mask = np.zeros(len(mesh.faces), bool)
    mask[adj[band].ravel()] = True
    return mask


def _plane_basis(axis):
    x = np.cross([0, 1, 0], axis)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1, 0, 0], axis)
    x /= np.linalg.norm(x)
    y = np.cross(axis, x)
    return x, y


def _merge_coaxial(cyls, center_tol=0.5, r_tol=0.15):
    merged = []
    for c in cyls:
        hit = None
        for m in merged:
            if (abs(abs(float(c['axis'] @ m['axis'])) - 1) < 1e-3 and
                    abs(c['r'] - m['r']) < r_tol and
                    np.linalg.norm(c['center2'] - m['center2']) < center_tol):
                hit = m
                break
        if hit:
            hit['h0'] = min(hit['h0'], c['h0'])
            hit['h1'] = max(hit['h1'], c['h1'])
            hit['faces'] += c['faces']
        else:
            merged.append(dict(c))
    return merged


def subtract_cylinders(solid, cyls, extend=1.0, verbose=True):
    import cadquery as cq
    for c in cyls:
        axis = c['axis']
        bx, by = c['basis']
        c3 = c['center2'][0] * bx + c['center2'][1] * by
        p0 = c3 + axis * (c['h0'] - extend)
        h = (c['h1'] - c['h0']) + 2 * extend
        cut = (cq.Workplane(cq.Plane(origin=tuple(p0), normal=tuple(axis)))
               .circle(c['r']).extrude(h))
        solid = solid.cut(cut)
        if verbose:
            print(f"[feature] cut cylinder r={c['r']:.3f} axis "
                  f"{np.round(axis,2)} depth {h - 2*extend:.2f} "
                  f"({c['faces']} facets)")
    return solid
