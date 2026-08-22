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
from .analysis import sanitize, mesh_stats

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


@app.get('/api/jobs/{job_id}')
def job_state(job_id: str):
    state = jobs.public_state(job_id)
    if state is None:
        raise HTTPException(404, 'unknown job')
    return state


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
