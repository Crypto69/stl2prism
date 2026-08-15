"""FastAPI app: upload an STL or OBJ, convert it to STEP, report fidelity."""
import os
import re

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from stl2prism.mesh_prep import SUPPORTED_EXTS

from . import jobs
from .analysis import sanitize, mesh_stats

_EXT_RE = re.compile(r'\.(' + '|'.join(e.lstrip('.') for e in SUPPORTED_EXTS)
                     + r')$', re.IGNORECASE)

app = FastAPI(title='stl2prism')

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


@app.on_event('startup')
def _startup():
    os.makedirs(jobs.DATA_DIR, exist_ok=True)
    jobs.cleanup_old()


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
        stats = mesh_stats(os.path.join(d, input_name))
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
