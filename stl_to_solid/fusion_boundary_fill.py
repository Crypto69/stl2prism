"""Emit a Fusion 360 script that rebuilds a face-group result with Fusion's
own kernel: every fitted surface (plane / cylinder / cone / sphere) is
recreated slightly oversized as a temporary B-Rep body, and a Boundary Fill
feature picks the cells that are material. Fusion then computes the exact
surface–surface intersection edges — the chord-edge polylines of the STEP
are gone.

Idea borrowed from the ReverseReloaded add-in (nico-schluter, Unlicense):
its users fit surfaces and run Boundary Fill by hand; here both halves are
generated from stl_to_solid's segmentation.

Cell selection is data-driven: the emitter ships a handful of points that
sit just inside the material next to every region (facegroups.export_regions),
and the script keeps every cell that contains one of them. Several cells
can be material when an oversized plane slices through another part of the
body; they are all kept and joined. If no point lands anywhere (pointContainment
unavailable), the cell whose volume is closest to the mesh volume is kept.

Run inside Fusion: put the .py in a folder of its own, Utilities > Add-Ins >
Scripts and Add-Ins > + (Script), choose that folder, Run. Fusion's API works
in centimetres; mm values are divided by 10 at emit time.

Verified in Fusion 360 (2026-08-23) on six synthetic parts — planes,
partial and full cylinders, tilted fillet cylinders, cones, a sphere,
tangent fillets, through-holes: exact CAD face counts, volumes within 0.2%,
clean tangent edges. Fails on parts where the face-group engine falls back
to band clusters: Parasolid cannot resolve the near-coincident sheets and
the main cell never closes. That is an engine limitation, not a script one.
Rolling-ball blends around curved edges are one torus region each since the
torus fit and become a single torus tool; apex cones (a pencil tip) and
tapered pins are single cone regions since the cone merge — an apex-reaching
cone emits one solid createCylinderOrCone tool.
"""
import math
import numpy as np

MAX_REGIONS = 200       # above this Fusion gets slow and the timeline noisy
EXPAND_MM = 1.0         # default oversize, editable at the top of the script


class TooManyRegions(ValueError):
    """No face-group body small enough for the Boundary Fill script."""


def _cm(v, nd=6):
    return round(float(v) / 10.0, nd)


def _cm3(p, nd=6):
    return tuple(_cm(x, nd) for x in p)


def _unit(v):
    return tuple(round(float(x), 7) for x in v)


def _plane_record(rg):
    """('plane', origin, u, v, [(x, y), ...]): the hull as 2-D coordinates in
    an in-plane basis with u x v = normal, counter-clockwise. The script
    rebuilds the corners as origin + x*u + y*v, so they are exactly coplanar
    whatever the rounding — 3-D corners rounded to 10 nm are already too far
    off a tilted plane for a strict kernel to accept the wire as planar."""
    from .facegroups import _plane_basis_vectors
    n = np.asarray(rg['normal'], float)
    n /= np.linalg.norm(n)
    o = np.asarray(rg['point'], float)
    u, v = _plane_basis_vectors(n)
    P = np.asarray(rg['hull'], float) - o
    xy = np.c_[P @ u, P @ v]
    area2 = 0.0
    for i in range(len(xy)):
        a, b = xy[i], xy[(i + 1) % len(xy)]
        area2 += a[0] * b[1] - a[1] * b[0]
    if area2 < 0:
        xy = xy[::-1]
    return ('plane', _cm3(o), _unit(u), _unit(v), [(_cm(x), _cm(y)) for x, y in xy])


TANGENT_DEG = 1.0       # axis within this of the plane: a tangent junction
TANGENT_GAP_MM = 0.05   # ... and the surface within this of touching it
MERGE_DEG = 1.0         # neighbours on the same surface: directions within this
MERGE_GAP_MM = 0.05     # ... and positions/radii within this


def _same_surface(a, b):
    """Two edge-sharing regions whose fits describe the same surface."""
    if a['kind'] != b['kind']:
        return False
    cos_tol = math.cos(math.radians(MERGE_DEG))
    if a['kind'] == 'plane':
        na, nb = np.asarray(a['normal'], float), np.asarray(b['normal'], float)
        if abs(na @ nb) < cos_tol:
            return False
        return abs((np.asarray(b['point']) - np.asarray(a['point'])) @ na) < MERGE_GAP_MM
    if a['kind'] == 'cylinder':
        ax, bx = np.asarray(a['axis'], float), np.asarray(b['axis'], float)
        if abs(ax @ bx) < cos_tol or abs(a['r'] - b['r']) > MERGE_GAP_MM:
            return False
        w = np.asarray(b['point'], float) - np.asarray(a['point'], float)
        return float(np.linalg.norm(w - (w @ ax) * ax)) < MERGE_GAP_MM
    # closed tools (sphere, torus) and cones: the same surface only when the
    # fits coincide — the case of a region split by the engine's pinch repair,
    # whose pieces carry one and the same fit and would otherwise become two
    # coincident whole tools (a kernel failure)
    if a['kind'] == 'cone':
        ax, bx = np.asarray(a['axis'], float), np.asarray(b['axis'], float)
        if ax @ bx < cos_tol or abs(a['half_angle'] - b['half_angle']) > math.radians(MERGE_DEG):
            return False
        return float(np.linalg.norm(np.asarray(b['apex'], float)
                                    - np.asarray(a['apex'], float))) < MERGE_GAP_MM
    if a['kind'] == 'sphere':
        return (abs(a['r'] - b['r']) <= MERGE_GAP_MM
                and float(np.linalg.norm(np.asarray(b['center'], float)
                                         - np.asarray(a['center'], float))) < MERGE_GAP_MM)
    if a['kind'] == 'torus':
        ax, bx = np.asarray(a['axis'], float), np.asarray(b['axis'], float)
        if (abs(ax @ bx) < cos_tol or abs(a['R'] - b['R']) > MERGE_GAP_MM
                or abs(a['r'] - b['r']) > MERGE_GAP_MM):
            return False
        return float(np.linalg.norm(np.asarray(b['center'], float)
                                    - np.asarray(a['center'], float))) < MERGE_GAP_MM
    return False


def _merge_same_surface(regions):
    """Neighbouring regions that sit on the same plane/cylinder with
    slightly different fits become one tool: two parallel sheets a few
    microns apart never cross, and the cell leaks out through the slit.
    Pieces of one cone/sphere/torus fit (pinch repair) become one tool too.
    (The face-group engine keeps them as separate faces in the STEP.)"""
    by_id = {r['id']: r for r in regions}
    parent = {r['id']: r['id'] for r in regions}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in regions:
        for nid, vs in r.get('adjacent', {}).items():
            if nid in by_id and len(vs) >= 2 and _same_surface(r, by_id[nid]):
                parent[find(r['id'])] = find(nid)
    groups = {}
    for r in regions:
        groups.setdefault(find(r['id']), []).append(r)
    out, n_merged = [], 0
    for members in groups.values():
        rep = dict(max(members, key=lambda r: r['area']))
        if len(members) == 1:
            out.append(rep)
            continue
        n_merged += len(members) - 1
        ids = {m['id'] for m in members}
        rep['area'] = sum(m['area'] for m in members)
        rep['inside'] = [p for m in members for p in m.get('inside', [])]
        adj = {}
        for m in members:
            for nid, vs in m.get('adjacent', {}).items():
                if nid not in ids:
                    adj[nid] = sorted(set(adj.get(nid, [])) | set(vs))
        rep['adjacent'] = adj
        if rep['kind'] == 'plane':
            n = np.asarray(rep['normal'], float)
            o = np.asarray(rep['point'], float)
            pts = np.vstack([np.asarray(m['hull'], float) for m in members])
            from .facegroups import _plane_hull
            rep['hull'] = _plane_hull(pts, o, n).tolist()       # onto the kept plane
        elif rep['kind'] in ('cylinder', 'cone'):
            okey = 'point' if rep['kind'] == 'cylinder' else 'apex'
            ax = np.asarray(rep['axis'], float)
            o = np.asarray(rep[okey], float)
            ts, angles = [], []
            xref = np.asarray(rep.get('xref', (1, 0, 0)), float)
            yref = np.cross(ax, xref)
            for m in members:
                om, am = np.asarray(m[okey], float), np.asarray(m['axis'], float)
                for t in (m['t0'], m['t1']):
                    ts.append(float(((om + t * am) - o) @ ax))
                if m.get('a0') is None:
                    angles = None
                if angles is not None:
                    xm = np.asarray(m.get('xref', (1, 0, 0)), float)
                    off = math.atan2(float(xm @ yref), float(xm @ xref))
                    angles += list(np.arange(m['a0'], m['a1'], 0.02) + off) + [m['a1'] + off]
            rep['t0'], rep['t1'] = min(ts), max(ts)
            if angles is None:
                rep['a0'] = rep['a1'] = None
            else:
                ang = np.sort(np.mod(angles, 2 * np.pi))
                gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
                k = int(np.argmax(gaps))
                if gaps[k] < math.radians(20.0):
                    rep['a0'] = rep['a1'] = None
                else:
                    rep['a0'] = round(float(ang[(k + 1) % len(ang)]), 6)
                    rep['a1'] = round(rep['a0'] + float(2 * np.pi - gaps[k]), 6)
        out.append(rep)
    # ids of merged members now point at their representative
    remap = {r['id']: find(r['id']) for r in regions}
    rep_id = {find(r['id']): None for r in regions}
    for r in out:
        rep_id[find(r['id'])] = r['id']
    for r in out:
        adj = {}
        for k, vs in r['adjacent'].items():
            if k in remap and rep_id[remap[k]] != r['id']:
                t = rep_id[remap[k]]
                adj[t] = sorted(set(adj.get(t, [])) | set(vs))
        r['adjacent'] = adj
    return out, n_merged


def _surface_record(rg):
    """One tuple per region for the script's SURFACES table (cm / radians),
    the last element being the minimum extension (cm) its junctions need.
    Curved surfaces carry 9 decimals (0.01 nm): a fillet snapped to exact
    tangency with its plane must stay exact after rounding, or the two
    surfaces miss each other by a few nanometres and the cell leaks."""
    k = rg['kind']
    emin = _cm(rg.get('expand_min', 0.0))        # filled in by _extensions()
    if k == 'plane':
        return _plane_record(rg) + (emin,)
    if k == 'cylinder':
        return ('cyl', _cm3(rg['point'], 9), _unit(rg['axis']), _cm(rg['r'], 9),
                _cm(rg['t0']), _cm(rg['t1']), emin,
                _unit(rg.get('xref', (1, 0, 0))), rg.get('a0'), rg.get('a1'))
    if k == 'cone':
        return ('cone', _cm3(rg['apex'], 9), _unit(rg['axis']),
                round(float(rg['half_angle']), 7), _cm(rg['t0']), _cm(rg['t1']), emin,
                _unit(rg.get('xref', (1, 0, 0))), rg.get('a0'), rg.get('a1'))
    if k == 'sphere':
        return ('sphere', _cm3(rg['center'], 9), _cm(rg['r'], 9))
    if k == 'torus':
        return ('torus', _cm3(rg['center'], 9), _unit(rg['axis']), _cm(rg['R'], 9), _cm(rg['r'], 9))
    raise ValueError(f"unknown region kind {k!r}")


def _snap_tangencies(regions, surfaces, penetrate_mm=0.0):
    """Move cylinders/spheres that are tangent to neighbouring planes or
    cylinders so that they touch them exactly, in the neighbours' *emitted*
    (rounded) position. A fitted fillet typically sits a few microns short
    of its tangent plane; surfaces that never meet leave the cell open.
    Several constraints on one surface are met by alternating projection.
    With `penetrate_mm` > 0 the surfaces are made to cross by that much
    instead of touching exactly (two close intersection lines instead of a
    tangent one — for kernels that drop tangent contacts).
    Returns the set of (id, neighbour id) pairs now tangent; edits
    `surfaces` in place (cm)."""
    pen = penetrate_mm / 10.0
    idx = {r['id']: i for i, r in enumerate(regions)}
    sin_tol = math.sin(math.radians(TANGENT_DEG))
    cos_tol = math.cos(math.radians(TANGENT_DEG))
    planes = {}
    for r, s in zip(regions, surfaces):
        if s[0] == 'plane':
            u, v = np.asarray(s[2], float), np.asarray(s[3], float)
            n = np.cross(u, v)
            planes[r['id']] = (np.asarray(s[1], float), n / np.linalg.norm(n))
    pairs = set()
    rb_of = {r['id']: s[3] for r, s in zip(regions, surfaces) if s[0] == 'cyl'}
    # bigger surfaces first: a fillet snaps to the boss it runs into after
    # the boss itself has been snapped to its plane
    order = sorted(range(len(regions)), key=lambda i: -regions[i]['area'])
    for i in order:
        r, s = regions[i], surfaces[i]
        # a consolidated blend tool is a fit-tol approximation: its tangent
        # neighbours sit up to ~0.1 mm away, so the snap must reach further
        gap_tol = (3.0 if r.get('approx') else 1.0) * TANGENT_GAP_MM / 10.0
        if s[0] == 'torus':
            # a blend torus touches the plane perpendicular to its axis at
            # r from its centre, and the coaxial cylinder it runs into at
            # R = r_cyl +/- r: snap the centre along the axis / onto the
            # cylinder's axis, and the major radius
            c, ax, R, rad = np.asarray(s[1], float), np.asarray(s[2], float), s[3], s[4]
            touched = []
            for nid, vs in r.get('adjacent', {}).items():
                if len(vs) < 2 or nid not in idx:
                    continue
                if nid in planes:
                    o, n = planes[nid]
                    if abs(ax @ n) < cos_tol:
                        continue
                    d = float((c - o) @ n)
                    if abs(abs(d) - rad) <= gap_tol:
                        c = c + (math.copysign(rad - pen, d) - d) * n
                        touched.append(nid)
                    continue
                t = surfaces[idx[nid]]
                if t[0] != 'cyl':
                    continue
                q, bx, rb = np.asarray(t[1], float), np.asarray(t[2], float), t[3]
                if abs(ax @ bx) < cos_tol:
                    continue
                w = (c - q) - ((c - q) @ bx) * bx
                if np.linalg.norm(w) > gap_tol:
                    continue
                for target, want in ((rb + rad, rb + rad - pen), (abs(rb - rad), abs(rb - rad) + pen)):
                    if target > 0 and abs(R - target) <= gap_tol:
                        c = c - w                       # onto the cylinder's axis
                        R = want
                        touched.append(nid)
                        break
            if touched:
                for nid in touched:
                    pairs.add((r['id'], nid))
                    pairs.add((nid, r['id']))
                surfaces[i] = ('torus', tuple(round(float(x), 9) for x in c), s[2],
                               round(float(R), 9), rad)
            continue
        if s[0] == 'cone':
            # a tapered fillet (an approximate cone tool) is tangent to each
            # plane it blends along one ruling: the plane passes through the
            # apex and makes (90 - half) degrees with the axis. Move the apex
            # onto the plane and turn the axis to the exact angle about it,
            # alternating when two planes constrain it. (Cone/cylinder
            # tangencies are not snapped.)
            apex, ax, half = np.asarray(s[1], float), np.asarray(s[2], float), s[3]
            cons = []
            for nid, vs in r.get('adjacent', {}).items():
                if len(vs) < 2 or nid not in planes:
                    continue
                o, n = planes[nid]
                if abs(abs(float(ax @ n)) - math.sin(half)) > sin_tol:
                    continue
                if abs(float((apex - o) @ n)) <= gap_tol:
                    cons.append((o, n, nid))
            if not cons:
                continue
            for _ in range(20):
                for o, n, _nid in cons:
                    apex = apex - float((apex - o) @ n) * n
                    q = float(ax @ n)
                    p = ax - q * n
                    pn = float(np.linalg.norm(p))
                    if pn > 1e-12:
                        ax = math.cos(half) * (p / pn) + math.copysign(math.sin(half), q) * n
            for _o, _n, nid in cons:
                pairs.add((r['id'], nid))
                pairs.add((nid, r['id']))
            xref = np.asarray(s[7], float)
            xref = xref - float(xref @ ax) * ax
            xref = xref / (float(np.linalg.norm(xref)) or 1.0)
            surfaces[i] = (('cone', tuple(round(float(x), 9) for x in apex),
                            tuple(round(float(x), 9) for x in ax)) + tuple(s[3:7])
                           + (tuple(round(float(x), 7) for x in xref),) + tuple(s[8:]))
            continue
        if s[0] not in ('cyl', 'sphere'):
            continue
        p = np.asarray(s[1], float)
        rad = s[3] if s[0] == 'cyl' else s[2]
        ax = np.asarray(s[2], float) if s[0] == 'cyl' else None
        cons = []                                   # (kind, data)
        for nid, vs in r.get('adjacent', {}).items():
            if len(vs) < 2 or nid not in idx:
                continue
            if nid in planes:
                o, n = planes[nid]
                if ax is not None and abs(ax @ n) > sin_tol:
                    continue
                d = float((p - o) @ n)
                if abs(abs(d) - rad) <= gap_tol:
                    cons.append(('plane', o, n, nid))
                continue
            t = surfaces[idx[nid]]
            if t[0] != 'cyl' or regions[idx[nid]]['area'] < r['area']:
                continue
            q, bx, rb = np.asarray(t[1], float), np.asarray(t[2], float), t[3]
            if ax is not None and abs(ax @ bx) < cos_tol:
                continue
            w = (p - q) - ((p - q) @ bx) * bx
            dist = float(np.linalg.norm(w))
            for target in (rad + rb, abs(rad - rb)):
                if target > 0 and abs(dist - target) <= gap_tol:
                    cons.append(('cyl', q, bx, target, nid))
                    break
        if not cons:
            continue
        for _ in range(20):
            for c in cons:
                if c[0] == 'plane':
                    _, o, n, _nid = c
                    d = float((p - o) @ n)
                    p = p + (math.copysign(rad - pen, d) - d) * n
                else:
                    _, q, bx, target, _nid = c
                    w = (p - q) - ((p - q) @ bx) * bx
                    dist = float(np.linalg.norm(w))
                    # outer tangency: axes closer; inner: axes further apart
                    want = target - pen if abs(target - (rad + rb_of[c[-1]])) < 1e-9 else target + pen
                    if dist > 1e-12:
                        p = p + (want - dist) * (w / dist)
        for c in cons:
            pairs.add((r['id'], c[-1]))
            pairs.add((c[-1], r['id']))
        p = tuple(round(float(x), 9) for x in p)
        surfaces[i] = (s[0], p) + tuple(s[2:])
    return pairs


EXTEND_FACTOR = 1.5     # safety on the computed crossing distance
EXTEND_MAX_MM = 5.0     # cap on the extension a junction can demand


def _extensions(regions, surfaces, mesh_vertices, snapped):
    """Minimum extension (cm) per tool so that it crosses every neighbour
    it is not exactly tangent to: two surfaces meeting at angle theta, each
    off the shared vertices by d, cross about (d1 + d2) / sin(theta) away
    from them. Written into the last element of each surface tuple."""
    from .facegroups import DIST, NORM
    V = np.asarray(mesh_vertices, float)
    by_id = {r['id']: r for r in regions}
    params = {}
    for r in regions:
        k = r['kind']
        if k == 'plane':
            params[r['id']] = {'normal': np.asarray(r['normal'], float),
                               'point': np.asarray(r['point'], float)}
        elif k == 'cylinder':
            params[r['id']] = {'axis': np.asarray(r['axis'], float),
                               'point': np.asarray(r['point'], float), 'r': float(r['r'])}
        elif k == 'cone':
            params[r['id']] = {'axis': np.asarray(r['axis'], float),
                               'apex': np.asarray(r['apex'], float),
                               'half_angle': float(r['half_angle'])}
        elif k == 'torus':
            params[r['id']] = {'center': np.asarray(r['center'], float),
                               'axis': np.asarray(r['axis'], float),
                               'R': float(r['R']), 'r': float(r['r'])}
        else:
            params[r['id']] = {'center': np.asarray(r['center'], float), 'r': float(r['r'])}
    for i, r in enumerate(regions):
        need = 0.0
        for nid, vs in r.get('adjacent', {}).items():
            if nid not in by_id or (r['id'], nid) in snapped or len(V) == 0:
                continue
            n2 = by_id[nid]
            P = V[vs]
            da = DIST[r['kind']](P, params[r['id']])
            db = DIST[n2['kind']](P, params[nid])
            na = NORM[r['kind']](P, params[r['id']])
            nb = NORM[n2['kind']](P, params[nid])
            sin = np.sqrt(1.0 - np.clip(np.abs(np.sum(na * nb, axis=1)), 0, 1) ** 2)
            reach = (np.abs(da) + np.abs(db)) / np.maximum(sin, 1e-3)
            need = max(need, float(reach.max()))
        emin = min(EXTEND_MAX_MM, EXTEND_FACTOR * need + 0.05)
        # A blend torus tangent to both this cylinder and a plane (a boss base,
        # a hole mouth) seals the cell between them only through tangent
        # contacts, which Parasolid drops: the cylinder must reach through
        # that plane itself, not stop at the fillet (verified in Fusion:
        # the cell stays open otherwise).
        if r['kind'] == 'cylinder':
            ax = params[r['id']]['axis']
            o = params[r['id']]['point']
            for tid in r.get('adjacent', {}):
                t = by_id.get(tid)
                if t is None or t['kind'] != 'torus' or (r['id'], tid) not in snapped:
                    continue
                for pid in t.get('adjacent', {}):
                    pl = by_id.get(pid)
                    if pl is None or pl['kind'] != 'plane' or (tid, pid) not in snapped:
                        continue
                    n = params[pid]['normal']
                    if abs(float(ax @ n)) < math.cos(math.radians(TANGENT_DEG)):
                        continue
                    tp = float((params[pid]['point'] - o) @ ax)
                    beyond = max(r['t0'] - tp, tp - r['t1'], 0.0)
                    emin = max(emin, min(EXTEND_MAX_MM, beyond + 0.1))
        s = surfaces[i]
        if s[0] == 'plane':
            surfaces[i] = s[:5] + (_cm(emin),) + s[6:]
        elif s[0] in ('cyl', 'cone'):
            surfaces[i] = s[:6] + (_cm(emin),) + s[7:]


def _facet_regions(regions, mesh_vertices, mesh_faces):
    """Regions the engine could not trim on their fitted surface become one
    plane tool per triangle, as the STEP keeps their facets. Their inside
    points stay with the first facet."""
    V = np.asarray(mesh_vertices, float)
    out, n_facets = [], 0
    next_id = max([r['id'] for r in regions] + [0]) + 1
    for r in regions:
        if r.get('built') == 'analytic' or not r.get('faces') or len(V) == 0:
            out.append(r)
            continue
        first = True
        facets = []
        for fi in r['faces']:
            vid = [int(i) for i in mesh_faces[fi]]
            a, b, c = (V[i] for i in vid)
            n = np.cross(b - a, c - a)
            area = 0.5 * float(np.linalg.norm(n))
            if area < 1e-9:
                continue
            n /= np.linalg.norm(n)
            facets.append({'id': next_id, 'kind': 'plane', 'built': 'facet',
                           'normal': n.tolist(), 'point': ((a + b + c) / 3).tolist(),
                           'hull': [a.tolist(), b.tolist(), c.tolist()], 'area': area,
                           'n_faces': 1, 'concave': False, 'resid': 0.0,
                           'inside': r.get('inside', []) if first else [],
                           'adjacent': {}, '_vid': set(vid)})
            next_id += 1
            n_facets += 1
            first = False
        # facets sharing an edge are neighbours (coplanar ones then merge)
        for f1 in facets:
            for f2 in facets:
                if f1 is not f2 and len(f1['_vid'] & f2['_vid']) == 2:
                    f1['adjacent'][f2['id']] = sorted(f1['_vid'] & f2['_vid'])
        for f in facets:
            del f['_vid']
        out.extend(facets)
    return out, n_facets


COINCIDENT_MM = 0.05    # a facet this close to another tool's surface is redundant


def _drop_coincident_facets(regions):
    """Facets (from regions the engine could not trim) that lie on the
    surface of another tool, within that tool's extent, are dropped: two
    sheets coinciding within microns make the kernel's intersection fail,
    and the analytic tool already covers that patch. Typical case: one
    physical cone segmented into two regions with slightly different fits,
    one of which failed to trim."""
    from .facegroups import DIST
    tools = [r for r in regions if r.get('built') != 'facet']
    kept, dropped = [], 0
    for r in regions:
        if r.get('built') != 'facet':
            kept.append(r)
            continue
        P = np.asarray(r['hull'], float)
        redundant = False
        for t in tools:
            k = t['kind']
            if k == 'plane':
                pr = {'normal': np.asarray(t['normal'], float), 'point': np.asarray(t['point'], float)}
                H = np.asarray(t['hull'], float)
                # inside the tool's footprint: within its hull's bounding box (+margin)
                lo, hi = H.min(axis=0) - 0.2, H.max(axis=0) + 0.2
                if not ((P >= lo).all() and (P <= hi).all()):
                    continue
            elif k == 'cylinder':
                pr = {'axis': np.asarray(t['axis'], float), 'point': np.asarray(t['point'], float), 'r': float(t['r'])}
                h = (P - pr['point']) @ pr['axis']
                if h.min() < t['t0'] - 0.2 or h.max() > t['t1'] + 0.2:
                    continue
            elif k == 'cone':
                pr = {'axis': np.asarray(t['axis'], float), 'apex': np.asarray(t['apex'], float), 'half_angle': float(t['half_angle'])}
                h = (P - pr['apex']) @ pr['axis']
                if h.min() < t['t0'] - 0.2 or h.max() > t['t1'] + 0.2:
                    continue
            elif k == 'sphere':
                pr = {'center': np.asarray(t['center'], float), 'r': float(t['r'])}
            elif k == 'torus':
                pr = {'center': np.asarray(t['center'], float), 'axis': np.asarray(t['axis'], float),
                      'R': float(t['R']), 'r': float(t['r'])}
            else:
                continue
            if np.abs(DIST[k](P, pr)).max() < COINCIDENT_MM:
                redundant = True
                break
        if redundant:
            dropped += 1
        else:
            kept.append(r)
    return kept, dropped


def _label(r):
    """A region's id for the script's log, with the engine's ids it was
    made from when they differ ('12<3,4,5>': a consolidated blend tool,
    a pinch piece) — the ids the pipeline's fallback list and verbose
    output use."""
    src = [int(i) for i in r.get('src', [])]
    if src and src != [int(r['id'])]:
        return '{}<{}>'.format(r['id'], ','.join(str(i) for i in src))
    return str(r['id'])


def _body_record(info, name, penetrate_mm=0.0):
    regions, n_facets = _facet_regions(info['regions'], info.get('mesh_vertices', []),
                                       info.get('mesh_faces', []))
    regions, n_dropped = _drop_coincident_facets(regions)
    n_facets -= n_dropped
    regions, n_merged = _merge_same_surface(regions)
    surfaces = [_surface_record(r) for r in regions]
    snapped = _snap_tangencies(regions, surfaces, penetrate_mm)
    n_snapped = len(snapped) // 2
    _extensions(regions, surfaces, info.get('mesh_vertices', []), snapped)
    inside = [_cm3(p) + ('{} {}'.format(_label(r), r['kind']),)
              for r in regions for p in r.get('inside', [])]
    unfitted = sum(1 for r in info['regions'] if r.get('built') != 'analytic')
    return {'name': name, 'volume': round(float(info.get('mesh_volume', 0.0)) / 1000.0, 6),
            'centroid': _cm3(info.get('centroid', (0, 0, 0))),
            'surfaces': surfaces, 'inside': inside, 'unfitted': unfitted,
            'merged': n_merged, 'snapped': n_snapped, 'facets': n_facets,
            # the mesh itself (cm), for the inside-the-part test on cells no
            # inside point reaches (thin wedges at tangent junctions, cells
            # enclosed by other cells)
            'mesh_v': [_cm3(v) for v in info.get('mesh_vertices', [])],
            'mesh_f': [tuple(int(i) for i in f) for f in info.get('mesh_faces', [])]}


def _literal(obj, indent=0, per_line=1):
    """Compact, readable Python literal: one surface per line, mesh arrays
    packed several entries per line."""
    pad = ' ' * indent
    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            n = 6 if k in ('mesh_v', 'mesh_f', 'inside') else 1
            items.append(f"{pad}    {k!r}: {_literal(v, indent + 4, n)}")
        return '{\n' + ',\n'.join(items) + f'\n{pad}}}'
    if isinstance(obj, list):
        if obj and isinstance(obj[0], (list, tuple, dict)):
            rows = [obj[i:i + per_line] for i in range(0, len(obj), per_line)]
            items = [f"{pad}    " + ', '.join(_literal(v, indent + 4) for v in row)
                     for row in rows]
            return '[\n' + ',\n'.join(items) + f'\n{pad}]'
        return repr(obj)
    return repr(obj)


RUNTIME = r'''
import adsk.core, adsk.fusion, traceback, math, time

_app = None


def _progress(msg):
    """A line in the Text Commands palette (Utilities > Add-Ins > Text
    Commands) so a long run shows it is alive; also lets the UI breathe."""
    try:
        _app.log('stlToSolid: ' + msg)
    except Exception:
        pass
    try:
        adsk.doEvents()
    except Exception:
        pass


def _pt(p):
    return adsk.core.Point3D.create(p[0], p[1], p[2])


def _vec(v):
    return adsk.core.Vector3D.create(v[0], v[1], v[2])


def _offset_convex(hull, n, d):
    """Convex polygon (CCW about n) pushed outward by d: each vertex moves
    along the bisector of its two edge normals (mitre, limited)."""
    m = len(hull)
    if m < 3 or d <= 0:
        return list(hull)
    outs = []
    for i in range(m):
        a, b = hull[i], hull[(i + 1) % m]
        e = [b[k] - a[k] for k in range(3)]
        o = [e[1] * n[2] - e[2] * n[1], e[2] * n[0] - e[0] * n[2], e[0] * n[1] - e[1] * n[0]]
        L = math.sqrt(sum(x * x for x in o)) or 1.0
        outs.append([x / L for x in o])
    res = []
    for i in range(m):
        o0, o1 = outs[i - 1], outs[i]
        dot = sum(o0[k] * o1[k] for k in range(3))
        f = d / max(1.0 + dot, 0.2)              # mitre limit ~5x
        res.append([hull[i][k] + f * (o0[k] + o1[k]) for k in range(3)])
    return res


def _expand(size, emin):
    """How far to extend a surface past its region: EXPAND, but never more
    than half the region's own size (a chain of tiny fillet bands extended
    by a full millimetre would cut each other into hundreds of cells), and
    never less than what its shallow-angle junctions need to cross."""
    return max(EXPAND_MIN, emin, min(EXPAND, 0.5 * size))


def _arc_span(r, a0, a1, expand):
    """Angular window (lo, hi) of a cylinder/cone tool: the region's arc
    widened by `expand` along the circumference; None when the region goes
    (nearly) all the way round, meaning the whole circle. Pure arithmetic,
    shared with the OCC dry run (stl_to_solid.bfill_check)."""
    if a0 is None or a1 is None or r <= 0:
        return None
    da = min(math.pi, expand / r)
    if (a1 - a0) + 2 * da >= 2 * math.pi - 0.02:
        return None
    return a0 - da, a1 + da


def _plane_body(tbm, o, u, v, xy, emin):
    n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    hull = [[o[k] + x * u[k] + y * v[k] for k in range(3)] for x, y in xy]
    area2 = 0.0
    for i in range(len(xy)):
        a, b = xy[i], xy[(i + 1) % len(xy)]
        area2 += a[0] * b[1] - a[1] * b[0]
    pts = _offset_convex(hull, n, _expand(math.sqrt(abs(area2) / 2.0), emin))
    m = len(pts)
    lines = [adsk.core.Line3D.create(_pt(pts[i]), _pt(pts[(i + 1) % m])) for i in range(m)]
    wire, _ = tbm.createWireFromCurves(lines)
    if wire is None:
        return None
    return tbm.createFaceFromPlanarWires([wire])


def _arc_or_circle(center, ax, xref, r, span):
    """An arc over the window span = (lo, hi), or the whole circle when
    span is None (see _arc_span)."""
    if span is None:
        return adsk.core.Circle3D.createByCenter(_pt(center), _vec(ax), r)
    return adsk.core.Arc3D.createByCenter(_pt(center), _vec(ax), _vec(xref), r,
                                          span[0], span[1])


def _ruled(tbm, c1, c2):
    w1, _ = tbm.createWireFromCurves([c1])
    w2, _ = tbm.createWireFromCurves([c2])
    if w1 is None or w2 is None:
        return None
    return tbm.createRuledSurface(w1.wires.item(0), w2.wires.item(0))


def _cyl_body(tbm, o, ax, r, t0, t1, emin, xref, a0, a1):
    """Partial cylinder sheet: the region's angular span plus the margin —
    a full tube would cut through the rest of the part and the other bands."""
    expand = _expand(t1 - t0, emin)
    p1 = [o[k] + (t0 - expand) * ax[k] for k in range(3)]
    p2 = [o[k] + (t1 + expand) * ax[k] for k in range(3)]
    span = _arc_span(r, a0, a1, expand)
    return _ruled(tbm, _arc_or_circle(p1, ax, xref, r, span),
                  _arc_or_circle(p2, ax, xref, r, span))


def _cone_body(tbm, apex, ax, half, t0, t1, emin, xref, a0, a1):
    # extension is measured along the surface; along the axis that is
    # expand * cos(half) (a nearly flat cone must not grow into a disc)
    slant = _expand((t1 - t0) / max(math.cos(half), 1e-6), emin)
    expand = slant * math.cos(half)
    s = math.tan(half)
    ta = max(0.0, t0 - expand)                 # never past the apex
    tb = t1 + expand
    r1 = s * ta
    r2 = s * tb
    p1 = [apex[k] + ta * ax[k] for k in range(3)]
    p2 = [apex[k] + tb * ax[k] for k in range(3)]
    if r1 < 0.005:
        # reaches the apex: a solid cone (a ruled sheet cannot end in a point)
        return tbm.createCylinderOrCone(_pt(p1), max(1e-4, r1), _pt(p2), r2)
    # one angular window for both ends (grown at the small end's radius):
    # the rulings stay on the fitted cone, and both ends are arcs or both
    # are circles — a ruled sheet between a circle and an arc fails
    span = _arc_span(r1, a0, a1, slant)
    return _ruled(tbm, _arc_or_circle(p1, ax, xref, r1, span),
                  _arc_or_circle(p2, ax, xref, r2, span))


def _sphere_body(tbm, c, r):
    return tbm.createSphere(_pt(c), r)


def _torus_body(tbm, c, ax, R, r):
    """Whole torus (closed solid, like the sphere): a blend is a small part
    of it, and cell selection keeps only the material side. createTorus in
    Fusion does not honour the centre/axis arguments (2026-08-24: the torus
    lands at the origin in the XY plane, radii correct), so build it there
    on purpose and move it into place with an explicit transform."""
    b = tbm.createTorus(_pt((0.0, 0.0, 0.0)), _vec((0.0, 0.0, 1.0)), R, r)
    if b is None:
        return None
    z = [float(ax[0]), float(ax[1]), float(ax[2])]
    seed = (0.0, 1.0, 0.0) if abs(z[1]) < 0.9 else (1.0, 0.0, 0.0)
    x = [seed[1] * z[2] - seed[2] * z[1], seed[2] * z[0] - seed[0] * z[2],
         seed[0] * z[1] - seed[1] * z[0]]
    n = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5
    x = [v / n for v in x]
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2],
         z[0] * x[1] - z[1] * x[0]]
    m = adsk.core.Matrix3D.create()
    m.setToAlignCoordinateSystems(
        adsk.core.Point3D.create(0.0, 0.0, 0.0),
        adsk.core.Vector3D.create(1.0, 0.0, 0.0),
        adsk.core.Vector3D.create(0.0, 1.0, 0.0),
        adsk.core.Vector3D.create(0.0, 0.0, 1.0),
        _pt(c), _vec(x), _vec(y), _vec(z))
    tbm.transform(b, m)
    return b


def _make_surface(tbm, s):
    if s[0] == 'plane':
        return _plane_body(tbm, s[1], s[2], s[3], s[4], s[5])
    if s[0] == 'cyl':
        return _cyl_body(tbm, s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9])
    if s[0] == 'cone':
        return _cone_body(tbm, s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9])
    if s[0] == 'sphere':
        return _sphere_body(tbm, s[1], s[2])
    if s[0] == 'torus':
        return _torus_body(tbm, s[1], s[2], s[3], s[4])
    return None


class _Mesh:
    """Point-in-mesh test (ray along +z, parity), triangles bucketed on an
    XY grid so a query touches a few dozen triangles."""

    def __init__(self, V, F, n=32):
        self.V, self.F, self.n = V, F, n
        xs = [v[0] for v in V]
        ys = [v[1] for v in V]
        zs = [v[2] for v in V]
        self.lo = (min(xs), min(ys), min(zs))       # bounds: a cheap
        self.hi = (max(xs), max(ys), max(zs))       # "cannot be inside"
        self.x0, self.y0 = min(xs), min(ys)
        self.dx = (max(xs) - self.x0) / n + 1e-9
        self.dy = (max(ys) - self.y0) / n + 1e-9
        self.grid = {}
        for fi, (a, b, c) in enumerate(F):
            P = (V[a], V[b], V[c])
            i0, i1 = self._ix(min(p[0] for p in P)), self._ix(max(p[0] for p in P))
            j0, j1 = self._iy(min(p[1] for p in P)), self._iy(max(p[1] for p in P))
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    self.grid.setdefault((i, j), []).append(fi)

    def _ix(self, x):
        return min(max(int((x - self.x0) / self.dx), 0), self.n - 1)

    def _iy(self, y):
        return min(max(int((y - self.y0) / self.dy), 0), self.n - 1)

    def _parity(self, x, y, z):
        n = 0
        for fi in self.grid.get((self._ix(x), self._iy(y)), ()):
            a, b, c = (self.V[k] for k in self.F[fi])
            d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(d) < 1e-18:
                continue
            l1 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / d
            l2 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / d
            l3 = 1.0 - l1 - l2
            if l1 < 0 or l2 < 0 or l3 < 0:
                continue
            if l1 * a[2] + l2 * b[2] + l3 * c[2] > z:
                n += 1
        return n % 2

    def box_outside(self, lo, hi):
        """Does a box (two corners) lie entirely outside the mesh's bounds?"""
        return any(hi[k] < self.lo[k] or lo[k] > self.hi[k] for k in range(3))

    def contains(self, p):
        # three slightly shifted rays, majority vote: a ray through a mesh
        # edge or vertex would otherwise be counted twice
        e = 1e-5
        votes = (self._parity(p[0], p[1], p[2]) + self._parity(p[0] + e, p[1] + 2 * e, p[2])
                 + self._parity(p[0] - 2 * e, p[1] + e, p[2]))
        return votes >= 2


SLIVER_CM = 0.1   # cm: a cell this small whose centre lies outside the part still gets the face probe


def _cell_is_material(body, pts, pt_hit, mesh, INSIDE):
    """Is this cell material? (hit, how): a probe point lies in it
    ('probe'); else a point of its own interior — the box centre, else the
    centre of mass — lies inside the mesh ('centre'); else, for a crescent
    sliver whose interior points are not its own (a thin curved slice
    between a blend chain's approximate tool and its neighbours: real
    material, 2026-08-24 joystick claw) or a sliver straddling the part's
    skin, a point on one of its faces nudged 0.05 mm inward ('face'). Each
    step is guarded on its own: a Fusion API error skips that step only,
    nothing is read from another cell. bfill_check.check_body runs the
    same procedure on OCC solids."""
    hit = False
    box = None
    try:
        box = body.boundingBox
        for k, p in enumerate(pts):
            if box.contains(p) and body.pointContainment(p) == INSIDE:
                hit = True
                pt_hit[k] = True
    except Exception:
        pass
    if hit:
        return True, 'probe'
    if mesh is None:
        return False, None
    centre, diag = None, 0.0
    try:
        if box is not None:
            lo, hi = box.minPoint, box.maxPoint
            if mesh.box_outside((lo.x, lo.y, lo.z), (hi.x, hi.y, hi.z)):
                return False, None                  # entirely outside the part's bounds
            centre = ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)
            diag = math.sqrt((hi.x - lo.x) ** 2 + (hi.y - lo.y) ** 2 + (hi.z - lo.z) ** 2)
    except Exception:
        centre = None
    inner = None
    try:
        if centre is not None:
            c = adsk.core.Point3D.create(centre[0], centre[1], centre[2])
            if body.pointContainment(c) == INSIDE:
                inner = c
        if inner is None:
            c = body.physicalProperties.centerOfMass
            if body.pointContainment(c) == INSIDE:
                inner = c
    except Exception:
        pass
    if inner is not None:
        if mesh.contains((inner.x, inner.y, inner.z)):
            return True, 'centre'
        if diag > SLIVER_CM:
            return False, None                      # a cell of the outside: decided
    if centre is None:
        return False, None
    try:
        faces = body.faces
        n = min(faces.count, 6)
    except Exception:
        return False, None
    for fi in range(n):
        try:
            p0 = faces.item(fi).pointOnFace
            dx, dy, dz = centre[0] - p0.x, centre[1] - p0.y, centre[2] - p0.z
            nn = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
            p1 = adsk.core.Point3D.create(p0.x + 0.005 * dx / nn,
                                          p0.y + 0.005 * dy / nn,
                                          p0.z + 0.005 * dz / nn)
            if (body.pointContainment(p1) == INSIDE
                    and mesh.contains((p1.x, p1.y, p1.z))):
                return True, 'face'
        except Exception:
            continue
    return False, None


def _select_cells(cells, bd, log):
    """Keep every cell that is material (_cell_is_material: a probe point
    in it, else a point of its own interior inside the mesh, else a face
    probe for slivers); if nothing at all can be tested, the cell whose
    volume is closest to the mesh volume."""
    INSIDE = adsk.fusion.PointContainment.PointInsidePointContainment
    pts = [_pt(p) for p in bd['inside']]
    labels = [p[3] if len(p) > 3 else '?' for p in bd['inside']]
    pt_hit = [False] * len(pts)
    mesh = _Mesh(bd['mesh_v'], bd['mesh_f']) if bd.get('mesh_f') else None
    target_vol = bd['volume']
    n_cells = cells.count
    _progress('{} cells to classify'.format(n_cells))
    kept, vols, by_centre, by_face, kept_vol = 0, [], 0, 0, 0.0
    t_start = time.time()
    for i in range(n_cells):
        if i and i % 25 == 0:
            _progress('cell {}/{} ({} kept so far, {:.0f} s)'.format(
                i, n_cells, kept, time.time() - t_start))
        cell = cells.item(i)
        body = cell.cellBody
        hit, how = _cell_is_material(body, pts, pt_hit, mesh, INSIDE)
        if how == 'centre':
            by_centre += 1
        elif how == 'face':
            by_face += 1
        if hit or kept == 0:
            try:
                vols.append(body.volume)
            except Exception:
                vols.append(0.0)
        else:
            vols.append(0.0)
        cell.isSelected = hit
        if hit:
            kept += 1
            kept_vol += vols[-1]
    if by_face:
        log.append('{} sliver cell(s) kept by a point on their own face'.format(by_face))
    log.append('kept volume {:.3f} cm^3 of {:.3f} (mesh)'.format(kept_vol, target_vol))
    n_miss = sum(1 for h in pt_hit if not h)
    if n_miss:
        miss = {}
        for k, h in enumerate(pt_hit):
            if not h:
                miss[labels[k]] = miss.get(labels[k], 0) + 1
        total = {}
        for lb in labels:
            total[lb] = total.get(lb, 0) + 1
        worst = sorted(miss.items(), key=lambda kv: -kv[1])
        log.append('{} of {} probe points are in no cell: the volume is not enclosed there. '
                   'Regions (id kind, points missed/total):'.format(n_miss, len(pts)))
        log.append('  ' + ', '.join('{} {}/{}'.format(lb, n, total[lb]) for lb, n in worst[:25]))
    if by_centre:
        log.append('{} cell(s) kept by a point of their interior lying inside the mesh'.format(by_centre))
    if kept == 0 and cells.count > 0:
        best = min(range(cells.count), key=lambda i: abs(vols[i] - target_vol))
        cells.item(best).isSelected = True
        kept = 1
        log.append('no inside point hit a cell; kept the cell closest to the mesh volume '
                   '({:.3f} vs {:.3f} cm^3)'.format(vols[best], target_vol))
    log.append('{} of {} cell(s) kept'.format(kept, cells.count))
    return kept


def _find_bad_tools(use, cells_for, log):
    """Which tools make the cell computation fail? First every tool on its
    own (a degenerate sheet), then tools added in chunks to a growing set;
    when a chunk breaks it, that chunk one tool at a time (a bad
    combination, e.g. two overlapping sheets). Every attempt is a cancelled
    boundary fill; the last line in Text Commands says where it stopped."""
    bad, ok = [], []
    for k, nb in enumerate(use):
        _progress('DIAGNOSE: tool {} alone ({}/{})'.format(nb.name, k + 1, len(use)))
        inp, _ = cells_for([nb])
        if inp is not None:
            inp.cancel()
            ok.append(nb)
        else:
            bad.append(nb)
            log.append('tool {} is rejected on its own'.format(nb.name))
    use, ok = ok, []
    chunk = 8
    i = 0
    while i < len(use):
        trial = use[i:i + chunk]
        _progress('DIAGNOSE: adding tools {}..{} of {} ({} bad so far)'.format(
            i, min(i + chunk, len(use)) - 1, len(use), len(bad)))
        inp, _ = cells_for(ok + trial)
        if inp is not None:
            inp.cancel()
            ok += trial
        else:
            for nb in trial:
                _progress('DIAGNOSE: adding {} alone to the {} good ones'.format(nb.name, len(ok)))
                inp, _ = cells_for(ok + [nb])
                if inp is not None:
                    inp.cancel()
                    ok.append(nb)
                else:
                    bad.append(nb)
                    log.append('tool {} breaks the cell computation together with the others'.format(nb.name))
        i += chunk
    return bad


def _build(design, root, tbm, bd, log):
    name = bd['name']
    surfaces = bd['surfaces']
    log.append('--- {}: {} surfaces ({} region(s) kept as {} facets; '
               '{} merged, {} snapped to tangency)'.format(
                   name, len(surfaces), bd['unfitted'], bd.get('facets', 0),
                   bd.get('merged', 0), bd.get('snapped', 0)))
    _progress('building {} surfaces'.format(len(surfaces)))
    tmp, failed = [], 0
    for s in surfaces:
        try:
            b = _make_surface(tbm, s)
        except Exception:
            b = None
        if b is None:
            failed += 1
        else:
            tmp.append(b)
    if failed:
        log.append('{} surface(s) could not be created'.format(failed))
    if not tmp:
        log.append('no surfaces could be created; skipped')
        return []
    # NOTE: no minimum count — closed tools (a sphere, a torus, a cone that
    # reaches its apex) can bound cells with as little as one body, and a
    # simple turned part is 3 tools (2026-08-24: the pencil part was skipped
    # by an old "fewer than 4 sheets cannot close" rule)

    bodies = root.bRepBodies
    added = []
    parametric = design.designType == adsk.fusion.DesignTypes.ParametricDesignType
    base = None
    if parametric:
        base = root.features.baseFeatures.add()
        base.name = name + ' surfaces'
        base.startEdit()
    try:
        for i, b in enumerate(tmp):
            nb = bodies.add(b, base) if base else bodies.add(b)
            added.append(nb)
    finally:
        if base:
            base.finishEdit()
    if base:
        # bodies added while editing are the base feature's *source* bodies
        # and stop being usable when the edit ends; the parametric copies
        # ("result bodies") are what later features must reference
        added = [base.bodies.item(i) for i in range(base.bodies.count)]
    for i, nb in enumerate(added):
        try:
            nb.name = '{}_s{}'.format(name, i)
        except Exception:
            pass
    if not RUN_FILL:
        log.append('RUN_FILL is False: surfaces created, run Boundary Fill by hand')
        return []

    bfills = root.features.boundaryFillFeatures
    newBody = adsk.fusion.FeatureOperations.NewBodyFeatureOperation

    def _cells_for(tool_bodies):
        """Start a boundary fill with these tools; returns (input, cells) or
        (None, None) when Fusion cannot compute the cells (input cancelled)."""
        tools = adsk.core.ObjectCollection.create()
        for nb in tool_bodies:
            tools.add(nb)
        inp = bfills.createInput(tools, newBody)
        try:
            cells = inp.bRepCells
            n = cells.count
            return inp, cells
        except Exception:
            try:
                inp.cancel()
            except Exception:
                pass
            return None, None

    use = [nb for i, nb in enumerate(added) if i not in SKIP]
    _progress('{} surfaces in the design; asking Fusion for the cells (this is the slow step)'.format(len(use)))
    t0 = time.time()
    inp, cells = _cells_for(use)
    _progress('cells computed in {:.0f} s'.format(time.time() - t0) if inp else 'cell computation failed')
    if inp is None:
        log.append('Fusion could not compute the cells from all {} tools'.format(len(use)))
        if DIAGNOSE:
            _progress('DIAGNOSE: testing tools in chunks, each test is a Boundary Fill attempt')
            bad = _find_bad_tools(use, _cells_for, log)
            use = [nb for nb in use if nb not in bad]
            inp, cells = _cells_for(use)
            if inp is None:
                log.append('still failing without the {} tool(s) above; giving up'.format(len(bad)))
                return []
            log.append('retrying without {} tool(s): {}'.format(
                len(bad), ', '.join(b.name for b in bad)))
        else:
            log.append('set DIAGNOSE = True to find the tools Fusion rejects')
            return []
    try:
        kept = _select_cells(cells, bd, log)
        if kept == 0:
            inp.cancel()
            log.append('no cell to keep; Boundary Fill cancelled')
            return []
        inp.isRemoveTools = not KEEP_TOOLS
        _progress('{} cell(s) selected; creating the Boundary Fill feature'.format(kept))
        feat = bfills.add(inp)
    except Exception:
        try:
            inp.cancel()
        except Exception:
            pass
        raise
    if feat is None:
        log.append('Boundary Fill failed')
        return []
    feat.name = name + ' boundary fill'
    out = [feat.bodies.item(i) for i in range(feat.bodies.count)]
    if len(out) > 1:
        # several material cells came out as separate bodies: join them
        combines = root.features.combineFeatures
        tb = adsk.core.ObjectCollection.create()
        for b in out[1:]:
            tb.add(b)
        ci = combines.createInput(out[0], tb)
        ci.operation = adsk.fusion.FeatureOperations.JoinFeatureOperation
        ci.isKeepToolBodies = False
        cf = combines.add(ci)
        out = [cf.bodies.item(i) for i in range(cf.bodies.count)] if cf else out[:1]
    if KEEP_TOOLS:
        for nb in added:
            try:
                nb.isLightBulbOn = False
            except Exception:
                pass
    for b in out:
        b.name = name
        try:
            log.append('{}: {} faces, {:.3f} cm^3 (mesh {:.3f})'.format(
                name, b.faces.count, b.volume, bd['volume']))
        except Exception:
            pass
    return out


def run(context):
    ui = None
    try:
        global _app
        app = adsk.core.Application.get()
        _app = app
        ui = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        if design is None:
            ui.messageBox('stlToSolid: open a design first')
            return
        root = design.rootComponent
        tbm = adsk.fusion.TemporaryBRepManager.get()
        log, made = [], []
        for bd in BODIES:
            made += _build(design, root, tbm, bd, log)
        text = '\n'.join(log)
        try:
            app.log('stlToSolid boundary fill\n' + text)
        except Exception:
            pass
        if not made:
            ui.messageBox('stlToSolid: no solid was built\n\n' + text)
        elif SHOW_SUMMARY:
            ui.messageBox('stlToSolid: {} solid(s) built\n\n{}'.format(len(made), text))
    except:
        if ui:
            ui.messageBox('stlToSolid script failed:\n{}'.format(traceback.format_exc()))
'''


BAND_MAX_BANDS = 3      # more chained blend bands than this: Fusion is expected to fail
ASSESS_BAND_LEN_R = 2.5   # a band, for the heuristic: a curved region shorter than this x radius
ASSESS_BAND_R_TOL = 0.25  # ... next to one of the same kind within this fraction in radius


def assess(build):
    """Will Fusion's Boundary Fill cope with this body? The known failure
    signature is a blend the engine kept as a chain of short cylinder/cone
    bands (tapered corner fillets): their near-coincident sheets defeat the
    cell computation. A band is a curved
    region shorter than its radius with a neighbour of the same kind, a
    similar radius and a nearby axis (length under 2.5 radii). Returns `ok`, `bands`,
    `unfitted`, `tools` and a one-line `reason`."""
    regions = build.get('regions', [])
    by_id = {r['id']: r for r in regions}
    bands = 0
    for r in regions:
        if r['kind'] not in ('cylinder', 'cone'):
            continue
        length = float(r['t1'] - r['t0'])
        rad = float(r['r']) if r['kind'] == 'cylinder' else \
            float(r['t1']) * math.tan(float(r['half_angle']))
        if rad <= 0 or length > ASSESS_BAND_LEN_R * rad:
            continue
        ax = np.asarray(r['axis'], float)
        for nid in r.get('adjacent', {}):
            n = by_id.get(nid)
            if n is None or n['kind'] not in ('cylinder', 'cone'):
                continue
            nrad = float(n['r']) if n['kind'] == 'cylinder' else \
                float(n['t1']) * math.tan(float(n['half_angle']))
            if abs(nrad - rad) > ASSESS_BAND_R_TOL * rad:
                continue
            if abs(float(ax @ np.asarray(n['axis'], float))) > math.cos(math.radians(25)):
                bands += 1
                break
    unfitted = sum(1 for r in regions if r.get('built') != 'analytic')
    tools = len(regions)
    reasons = []
    if bands > BAND_MAX_BANDS:
        reasons.append(f'{bands} blend bands (a torus or tapered fillet the engine could not fit as one surface)')
    if unfitted:
        reasons.append(f'{unfitted} region(s) kept as facets')
    if tools > MAX_REGIONS:
        reasons.append(f'{tools} regions (limit {MAX_REGIONS})')
    ok = not reasons
    return {'ok': ok, 'bands': bands, 'unfitted': unfitted, 'tools': tools,
            'reason': '; '.join(reasons) if reasons else 'all regions fitted cleanly'}


def skipped_bodies(builds, max_regions=MAX_REGIONS):
    """(index, n_regions) of the face-group bodies the script leaves out
    for having more regions than it supports — an outlook must not vouch
    for a script that lacks them."""
    return [(bi, len(info['regions'])) for bi, info in enumerate(builds)
            if info.get('mode') == 'facegroup' and 'regions' in info
            and len(info['regions']) > max_regions]


def emit_boundary_fill_script(builds, stem='part', expand_mm=EXPAND_MM,
                              max_regions=MAX_REGIONS, penetrate_mm=0.0, warning=None):
    """Script text for every face-group body in `builds` (the per-body build
    records pipeline.run collects; prismatic/faceted bodies get a comment).
    `warning` (the pipeline's outlook, when it is a FAIL) goes into the
    docstring header so the file says what the UI says. Raises
    TooManyRegions when no body is small enough."""
    recs, notes = [], []
    n_fg = sum(1 for b in builds if b.get('mode') == 'facegroup' and 'regions' in b)
    for bi, info in enumerate(builds):
        name = stem if len(builds) == 1 else f'{stem}_body{bi + 1}'
        if info.get('mode') != 'facegroup' or 'regions' not in info:
            notes.append(f"# body {bi + 1}: {info.get('mode', '?')} — not a face-group "
                         f"result (import the STEP for it)")
            continue
        n = len(info['regions'])
        if n > max_regions:
            notes.append(f"# body {bi + 1}: {n} regions is above the {max_regions} this "
                         f"script supports; import the STEP for it")
            continue
        recs.append(_body_record(info, name, penetrate_mm))
    if not recs:
        if n_fg:
            raise TooManyRegions(
                f"face-group bodies have more than {max_regions} regions")
        raise ValueError('no face-group body')
    n_surf = sum(len(r['surfaces']) for r in recs)
    warn = [warning] if warning else []
    head = [
        '"""Generated by stlToSolid — rebuilds a face-group result with Fusion 360\'s',
        'Boundary Fill: every fitted surface is recreated slightly oversized as a',
        'temporary body, Boundary Fill keeps the cells that are material, and Fusion',
        'computes the exact edges between the faces. Experimental.',
        '',
        f'{len(recs)} body/bodies, {n_surf} surfaces. Fusion API units are centimetres',
    ] + ([f'WARNING: likely to fail in Fusion — {"; ".join(warn)}.'] if warn else []) + [
        '(values below are mm/10). If a cell fails to close, try EXPAND 0.05 or 0.2;',
        'if Fusion rejects the feature, try KEEP_TOOLS = True (tools stay, hidden)."""',
        '',
        f'# tangent surfaces: {"made to cross by %g mm" % penetrate_mm if penetrate_mm else "exactly tangent"}',
        f'EXPAND = {_cm(expand_mm)}          # cm: how far each surface is extended past its region',
        'EXPAND_MIN = 0.01        # cm: floor for small regions (they get half their size)',
        'KEEP_TOOLS = False       # True: keep the surface bodies (hidden) for debugging',
        'RUN_FILL = True          # False: only create the surfaces (run Boundary Fill by hand)',
        'DIAGNOSE = False         # if Fusion cannot compute the cells, find the tools it rejects (slow, can hang)',
        'SKIP = []                # tool numbers (frame_s<N>) to leave out',
        'SHOW_SUMMARY = True      # message box with the log at the end',
        '',
    ]
    head += notes
    body = ['BODIES = ' + _literal(recs), '']
    return '\n'.join(head + body) + RUNTIME
