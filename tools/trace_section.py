"""Trace one plane section of a mesh the way Fusion's Create Mesh Section
Sketch + Fit Curves to Mesh Section would, and draw the result.

    .venv/bin/python tools/trace_section.py samples/fixed-rc-n2-360.stl \
        --axis z --offset -26 --scale 0.1 --tol 0.08 --join 2.5 --out section.png

`--offset` is measured from the bounding-box centre along the axis (the
way Fusion's section-plane slider works); `--at` gives an absolute
coordinate instead. Open chains (a leaky mesh) stay open, with their free
ends marked, after ends within `--join` mm are joined (0 = draw exactly
what Fusion's Create Mesh Section Sketch draws). Raw section grey, fitted
lines blue, arcs green, splines red; the title carries the counts and the
worst deviation.
"""
import argparse
import os
import sys
import time

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from stl2prism.section_fit import fit_section, prim_points   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mesh')
    ap.add_argument('--axis', choices='xyz', default='z')
    ap.add_argument('--offset', type=float, default=None,
                    help='mm from the bounding-box centre along the axis')
    ap.add_argument('--at', type=float, default=None, help='absolute coordinate, mm')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--tol', type=float, default=0.08)
    ap.add_argument('--join', type=float, default=2.5, help='join free ends closer than this, mm')
    ap.add_argument('--trim', type=float, default=0.0, help='cut slivers thinner than this out of loops, mm')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    m = trimesh.load(a.mesh, force='mesh')
    if a.scale != 1.0:
        m.apply_scale(a.scale)
    k = 'xyz'.index(a.axis)
    c = m.bounding_box.centroid
    at = a.at if a.at is not None else c[k] + (a.offset or 0.0)
    origin = np.zeros(3)
    origin[k] = at
    normal = np.zeros(3)
    normal[k] = 1.0
    t0 = time.time()
    sec = fit_section(m.vertices, m.faces, origin, normal, tol=a.tol, join_mm=a.join, trim_mm=a.trim)
    dt = time.time() - t0
    st = sec['stats']
    print(f"{os.path.basename(a.mesh)} {a.axis}={at:.2f} mm: {st['loops']} loops "
          f"({st['holes']} holes), {st['open']} open chains -> {st['lines']} lines, "
          f"{st['arcs']} arcs, {st['circles']} circles, {st['splines']} splines; "
          f"{st['joins']} gaps joined (largest {st['max_gap']:.2f} mm), {st['trimmed']} slivers trimmed; worst deviation "
          f"{st['dev_max']:.3f} mm; {dt:.2f} s")
    raw_pts = sum(len(o) + sum(len(h) for h in hs) for o, hs in sec['raw'])
    raw_pts += sum(len(q) for q in sec['raw_open'])
    print(f"  raw section: {raw_pts} points")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 8))
    for outer, holes in sec['raw']:
        for xy in [outer] + holes:
            q = np.vstack([xy, xy[:1]])
            ax.plot(q[:, 0], q[:, 1], '-', color='0.75', lw=2.5, zorder=1)
    for xy in sec['raw_open']:
        ax.plot(xy[:, 0], xy[:, 1], '-', color='0.75', lw=2.5, zorder=1)
        ax.plot(xy[[0, -1], 0], xy[[0, -1], 1], 'o', color='#d02020', ms=4, mfc='none', zorder=3)
    col = {'line': '#1f5fd0', 'arc': '#1f9d3a', 'spline': '#d02020'}
    for prims in [q for fo, fh in sec['loops'] for q in [fo] + fh] + sec['open']:
        if isinstance(prims, dict):
            q = prim_points(prims)
            q = np.vstack([q, q[:1]])
            ax.plot(q[:, 0], q[:, 1], '-', color=col['arc'], lw=0.9, zorder=2)
            continue
        for p in prims:
            q = prim_points([p])
            q = np.vstack([q, [p['p1']]]) if not p.get('closed') else np.vstack([q, q[:1]])
            ax.plot(q[:, 0], q[:, 1], '-', color=col[p['type']], lw=0.9, zorder=2)
            ax.plot([p['p0'][0]], [p['p0'][1]], '.', color='k', ms=2, zorder=3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{os.path.basename(a.mesh)} {a.axis}={at:.2f} mm, tol {a.tol}, join {a.join}: "
                 f"{st['lines']} lines (blue), {st['arcs']} arcs (green), {st['circles']} circles, "
                 f"{st['splines']} splines (red); {st['open']} open, {st['joins']} joined; "
                 f"worst dev {st['dev_max']:.3f} mm")
    out = a.out or os.path.splitext(os.path.basename(a.mesh))[0] + f'_{a.axis}{at:+.1f}.png'
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f"  -> {out}")


if __name__ == '__main__':
    main()
