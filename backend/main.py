"""FastAPI app: upload an STL or OBJ, convert it to STEP, report fidelity."""
import os
import re
from typing import Literal

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from stl2prism.mesh_prep import SUPPORTED_EXTS

from . import jobs
from .analysis import sanitize, mesh_stats, body_list

_EXT_RE = re.compile(r'\.(' + '|'.join(e.lstrip('.') for e in SUPPORTED_EXTS)
                     + r')$', re.IGNORECASE)

@asynccontextmanager
async def _lifespan(app):
    os.makedirs(jobs.DATA_DIR, exist_ok=True)
    jobs.cleanup_old()
    yield


app = FastAPI(title='stl2prism', lifespan=_lifespan)


@app.get('/api/version')
def version():
    """Package version plus the git commit / time the image was built from
    (set via Docker build args; 'unknown' when run from a checkout)."""
    from stl2prism import __version__
    return {'version': __version__,
            'commit': os.environ.get('STL2PRISM_BUILD_SHA', 'unknown'),
            'built': os.environ.get('STL2PRISM_BUILD_TIME', 'unknown')}


def _write_preview(src, dst):
    from stl2prism.mesh_prep import load_mesh
    load_mesh(src).export(dst)

# Dev convenience: the Vite dev server runs on another port. In production
# the frontend is served from this same app, so this allows nothing new.
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])

MAX_UPLOAD = int(os.environ.get('STL2PRISM_MAX_UPLOAD', 200 * 1024 * 1024))


class ConvertParams(BaseModel):
    tol: float = Field(0.08, gt=0, le=5,
                       description='profile fit tolerance, mm')
    accept_p95: float = Field(0.25, gt=0, le=10,
                              description='max p95 surface deviation, mm')
    accept_max: float = Field(0.26, gt=0, le=10,
                              description='max single-point deviation, mm')
    accept_hole_max: float = Field(0.10, gt=0, le=10,
                                   description='max bore deviation, mm')
    accept_vol_pct: float = Field(2.0, gt=0, le=50,
                                  description='max volume error, %')
    force_prismatic: bool = False
    # face-group engine: one analytic face per fitted region, tried between
    # the prismatic and faceted routes
    face_groups: bool = True
    reduce_tol: float = Field(0.05, ge=0, le=5,
                              description='faceted output: decimate curved regions within this deviation, mm (0 = off)')
    # STL/OBJ carry no units; this says what the file's numbers mean.
    units: Literal['mm', 'cm', 'm', 'in', 'ft'] = 'mm'
    # Free multiplier on top of `units`, because units only ever enlarge:
    # a cm design exported as mm reads 10x too big and needs 0.1.
    scale: float = Field(1.0, gt=0, le=1e6,
                         description='extra scale factor applied after units')
    # Which shells to convert, as indices into /bodies (largest first).
    # None converts every body, as before.
    bodies: list[int] | None = None
    # 'loft': slice every body along an axis and loft the section outlines
    # (Fusion's Mesh Section Sketch + Loft, automated); 'auto' is the
    # prismatic / face-group / faceted ladder.
    method: Literal['auto', 'loft'] = 'auto'
    slice_mm: float = Field(0.2, gt=0, le=50, description='sliced loft: section spacing, mm')
    slice_axis: Literal['auto', 'x', 'y', 'z'] = 'auto'
    loft_ruled: bool = False


@app.post('/api/jobs')
async def create_job(file: UploadFile):
    m = _EXT_RE.search(file.filename or '')
    if not m:
        raise HTTPException(
            400, f"expected a {' or '.join(SUPPORTED_EXTS)} file")
    # Keep the real extension: trimesh picks its reader from it.
    input_name = 'input.' + m.group(1).lower()
    job_id, d = jobs.new_job()
    size = 0
    with open(os.path.join(d, input_name), 'wb') as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, 'file too large')
            out.write(chunk)
    try:
        stats = await run_in_threadpool(mesh_stats, os.path.join(d, input_name))
        if not input_name.endswith(('.stl', '.obj')):
            # the browser viewer parses STL/OBJ itself; other formats get a
            # server-side STL preview
            await run_in_threadpool(_write_preview, os.path.join(d, input_name),
                                    os.path.join(d, 'preview.stl'))
    except Exception as e:
        raise HTTPException(400, f'could not read mesh: {e}')
    with jobs._lock:
        jobs._jobs[job_id]['filename'] = file.filename
        jobs._jobs[job_id]['input'] = input_name
        jobs._jobs[job_id]['status'] = 'uploaded'
    return {'id': job_id, 'filename': file.filename,
            'input_stats': sanitize(stats)}


@app.post('/api/jobs/{job_id}/convert')
def convert(job_id: str, params: ConvertParams):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, 'unknown job')
    if job['status'] in ('queued', 'running'):
        raise HTTPException(409, 'job already running')
    jobs.cleanup_old()
    jobs.start(job_id, job.get('filename'), params.model_dump())
    return {'id': job_id, 'status': 'running'}


@app.get('/api/jobs/running')
def running():
    """How many conversions are in flight, for deploy.sh to check before it
    replaces the container: a restart kills whatever is converting and
    leaves the user a half-written log and no result."""
    return jobs.running_summary()


@app.post('/api/jobs/{job_id}/cancel')
def cancel(job_id: str):
    """Stop a running conversion. A long file can run for tens of minutes,
    and until now the only way out was to wait or restart the server."""
    what = jobs.cancel(job_id)
    if what == 'unknown':
        raise HTTPException(404, 'unknown job')
    return {'id': job_id, 'status': what}


@app.get('/api/jobs/{job_id}')
def job_state(job_id: str):
    state = jobs.public_state(job_id)
    if state is None:
        raise HTTPException(404, 'unknown job')
    return state


@app.get('/api/jobs/{job_id}/bodies')
async def bodies(job_id: str):
    """Every connected shell of the uploaded mesh, largest first, with the
    triangle -> body map the viewer colours and picks with."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, 'unknown job')
    src = os.path.join(jobs.job_dir(job_id), job.get('input', ''))
    if not job.get('input') or not os.path.exists(src):
        raise HTTPException(404, 'no input file')
    try:
        return sanitize(await run_in_threadpool(body_list, src))
    except Exception as e:
        raise HTTPException(400, f'could not split mesh: {e}')


def _input_path(job_id):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, 'unknown job')
    src = os.path.join(jobs.job_dir(job_id), job.get('input', ''))
    if not job.get('input') or not os.path.exists(src):
        raise HTTPException(404, 'no input file')
    return job, src


_AXES = ('x', 'y', 'z')


@app.get('/api/jobs/{job_id}/section')
async def section(job_id: str, axis: str = 'z', offset: float = 0.0, tol: float = 0.08,
                  units: str = 'mm', scale: float = 1.0, join: float = 2.5,
                  outline: bool = False):
    """One traced section of the uploaded mesh: the plane across `axis`
    at `offset` mm from the bounding-box centre (Fusion's section-plane
    slider), fitted as lines, arcs and splines within `tol`. Open chains
    of a leaky mesh stay open, as in Fusion's mesh section sketch, after
    free ends within `join` mm of each other are joined (0 = never);
    `outline` leaves the open chains out (closed loops only).
    Returns the curves as 3-D polylines (mm, converted frame) for the
    viewer plus the fit statistics."""
    if axis not in _AXES:
        raise HTTPException(400, 'axis must be x, y or z')
    if not (0 < tol <= 5):
        raise HTTPException(400, 'tol must be in (0, 5] mm')
    if not (0 <= join <= 20):
        raise HTTPException(400, 'join must be in [0, 20] mm')
    _, src = _input_path(job_id)
    from .sections import trace
    try:
        return sanitize(await run_in_threadpool(trace, src, axis, offset, tol, units, scale, join,
                                                outline))
    except Exception as e:
        raise HTTPException(400, f'could not trace the section: {e}')


@app.get('/api/jobs/{job_id}/section-script')
async def section_script(job_id: str, axis: str = 'z', offset: float = 0.0, tol: float = 0.08,
                         units: str = 'mm', scale: float = 1.0, join: float = 2.5,
                         outline: bool = False):
    """The same section as a Fusion 360 script: one construction plane
    and one sketch of lines, arcs, circles and fitted splines (Create Mesh
    Section Sketch + Fit Curves to Mesh Section, in one go)."""
    if axis not in _AXES:
        raise HTTPException(400, 'axis must be x, y or z')
    if not (0 <= join <= 20):
        raise HTTPException(400, 'join must be in [0, 20] mm')
    job, src = _input_path(job_id)
    from .sections import trace
    from stl2prism.fusion_export import emit_fusion_sections_script
    try:
        sec = await run_in_threadpool(trace, src, axis, offset, tol, units, scale, join, outline)
    except Exception as e:
        raise HTTPException(400, f'could not trace the section: {e}')
    name = f"section {axis.upper()}={sec['at']:.2f} mm ({offset:+.1f} from centre)"
    text = emit_fusion_sections_script([{'origin': sec['origin'], 'normal': sec['normal'],
                                         'name': name, 'loops': sec['loops'],
                                         'open': sec['open']}])
    stem = _EXT_RE.sub('', job.get('filename') or 'part')
    safe = re.sub(r'[^\w.-]+', '_', stem) or 'part'
    from fastapi.responses import Response
    return Response(text, media_type='text/x-python',
                    headers={'Content-Disposition':
                             f'attachment; filename="{safe}_section_{axis}{offset:+.1f}.py"'})


@app.get('/api/jobs/{job_id}/preview')
def preview(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, 'unknown job')
    path = os.path.join(jobs.job_dir(job_id), 'preview.stl')
    if not os.path.exists(path):
        raise HTTPException(404, 'no preview')
    return FileResponse(path, media_type='model/stl')


@app.get('/api/jobs/{job_id}/script')
def script(job_id: str):
    """The CadQuery script that rebuilds the recognised extrusion structure."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, 'unknown job')
    path = os.path.join(jobs.job_dir(job_id), 'output.py')
    if not os.path.exists(path):
        raise HTTPException(404, 'no script')
    stem = _EXT_RE.sub('', job.get('filename') or 'part')
    safe = re.sub(r'[^\w.-]+', '_', stem) or 'part'
    return FileResponse(path, media_type='text/x-python', filename=f'{safe}.py')


@app.get('/api/jobs/{job_id}/fusion-script')
def fusion_script(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, 'unknown job')
    path = os.path.join(jobs.job_dir(job_id), 'output_fusion.py')
    if not os.path.exists(path):
        raise HTTPException(404, 'no script')
    stem = _EXT_RE.sub('', job.get('filename') or 'part')
    safe = re.sub(r'[^\w.-]+', '_', stem) or 'part'
    return FileResponse(path, media_type='text/x-python', filename=f'{safe}_fusion.py')


@app.get('/api/jobs/{job_id}/fusion-bfill-script')
def fusion_bfill_script(job_id: str):
    """The Fusion 360 Boundary Fill script for face-group results."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, 'unknown job')
    path = os.path.join(jobs.job_dir(job_id), 'output_fusion_bfill.py')
    if not os.path.exists(path):
        raise HTTPException(404, 'no script')
    stem = _EXT_RE.sub('', job.get('filename') or 'part')
    safe = re.sub(r'[^\w.-]+', '_', stem) or 'part'
    return FileResponse(path, media_type='text/x-python', filename=f'{safe}_fusion_bfill.py')


@app.get('/api/jobs/{job_id}/download')
def download(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, 'unknown job')
    path = os.path.join(jobs.job_dir(job_id), 'output.step')
    if not os.path.exists(path):
        raise HTTPException(404, 'no output yet')
    stem = _EXT_RE.sub('', job.get('filename') or 'part')
    safe = re.sub(r'[^\w.-]+', '_', stem) or 'part'
    return FileResponse(path, media_type='application/step',
                        filename=f'{safe}.step')


# Production: serve the built frontend. Registered last so /api wins.
_static = os.environ.get(
    'STL2PRISM_STATIC',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'frontend', 'dist'))
if os.path.isdir(_static):
    app.mount('/', StaticFiles(directory=_static, html=True), name='static')
