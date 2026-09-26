"""FastAPI app: upload an STL or OBJ, convert it to STEP, report fidelity;
or upload a dimensioned drawing (Blueprint) and build it from its labels."""
import json
import logging
import mimetypes
import os
import re
from typing import Optional, Literal

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from stl_to_solid.mesh_prep import SUPPORTED_EXTS

from . import jobs
from . import blueprint as bp
from . import preview as bp_preview
from .analysis import sanitize, mesh_stats, body_list
from .errors import describe

log = logging.getLogger('stltosolid.api')

_EXT_RE = re.compile(r'\.(' + '|'.join(e.lstrip('.') for e in SUPPORTED_EXTS)
                     + r')$', re.IGNORECASE)
_IMG_RE = re.compile(r'\.(' + '|'.join(bp.IMAGE_EXTS) + r')$', re.IGNORECASE)

@asynccontextmanager
async def _lifespan(app):
    os.makedirs(jobs.DATA_DIR, exist_ok=True)
    jobs.cleanup_old()
    yield
    bp_preview.helper.close()


app = FastAPI(title='stlToSolid', lifespan=_lifespan)


# Every failure the browser sees is JSON with one readable 'detail' string:
# the frontend shows that string as-is. Without these, a bug in a route
# came back as a bare "Internal Server Error" and a bad parameter as a
# list of pydantic records the UI printed as "[object Object]".
@app.exception_handler(RequestValidationError)
async def _bad_request(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422,
                        content={'detail': _validation_text(exc)})


@app.exception_handler(Exception)
async def _unexpected(request: Request, exc: Exception):
    log.exception('unhandled error in %s %s', request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={'detail': 'Something went wrong on the server: '
                           f'{describe(exc)} (the server log has the details).'})


def _validation_text(exc):
    """'tol: must be greater than 0' rather than pydantic's record list."""
    parts = []
    for err in exc.errors():
        loc = [str(x) for x in err.get('loc', ()) if x not in ('body', 'query', 'path')]
        msg = err.get('msg', 'invalid value')
        msg = re.sub(r'^(Value|Input|Assertion) error, ', '', msg)
        parts.append(f"{'.'.join(loc) or 'request'}: {msg}")
    return 'Invalid settings: ' + '; '.join(parts) if parts else 'Invalid request.'


@app.get('/api/version')
def version():
    """Package version plus the git commit / time the image was built from
    (set via Docker build args; 'unknown' when run from a checkout)."""
    from stl_to_solid import __version__
    return {'version': __version__,
            'commit': os.environ.get('STLTOSOLID_BUILD_SHA', 'unknown'),
            'built': os.environ.get('STLTOSOLID_BUILD_TIME', 'unknown')}


def _write_preview(src, dst):
    from stl_to_solid.mesh_prep import load_mesh
    load_mesh(src).export(dst)

# Dev convenience: the Vite dev server runs on another port. In production
# the frontend is served from this same app, so this allows nothing new.
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])

MAX_UPLOAD = int(os.environ.get('STLTOSOLID_MAX_UPLOAD', 200 * 1024 * 1024))
# seconds an x-ray script may spend fitting its slices before it gives up
# (the download is one synchronous request; Firefox drops a response after
# 300 s)
XRAY_BUDGET_S = float(os.environ.get('STLTOSOLID_XRAY_BUDGET', 240))


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
    slice_join: float = Field(2.5, ge=0, le=20, description='sliced loft: join loose section ends closer than this, mm')
    slice_trim: float = Field(0.0, ge=0, le=5, description='sliced loft: cut slivers thinner than this, mm (0 = off)')
    slice_from: Optional[float] = Field(None, description='partial loft: start plane, mm from the bounding-box centre')
    slice_range_mm: float = Field(0.0, ge=0, le=2000, description='partial loft: length from the start plane, mm (0 = whole body)')
    slice_range_dir: Literal['+', '-'] = '-'


@app.post('/api/jobs')
async def create_job(file: UploadFile):
    m = _EXT_RE.search(file.filename or '')
    if not m:
        raise HTTPException(
            400, f"That is not a mesh file. Expected one of: {', '.join(SUPPORTED_EXTS)}.")
    # Keep the real extension: trimesh picks its reader from it.
    input_name = 'input.' + m.group(1).lower()
    job_id, d = jobs.new_job()
    try:
        size = 0
        with open(os.path.join(d, input_name), 'wb') as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(
                        413, f'The file is too large: the limit is {MAX_UPLOAD // (1 << 20)} MB.')
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, 'The file is empty.')
        try:
            stats = await run_in_threadpool(mesh_stats, os.path.join(d, input_name))
            if not input_name.endswith(('.stl', '.obj')):
                # the browser viewer parses STL/OBJ itself; other formats get a
                # server-side STL preview
                await run_in_threadpool(_write_preview, os.path.join(d, input_name),
                                        os.path.join(d, 'preview.stl'))
        except Exception as e:
            log.warning('could not read upload %r: %s', file.filename, describe(e))
            raise HTTPException(400, f'Could not read {file.filename or "the file"}: '
                                     f'{describe(e)}')
        if not stats.get('triangles'):
            raise HTTPException(400, f'{file.filename or "The file"} holds no triangles, '
                                     'so there is nothing to convert.')
    except BaseException:
        # a half-written upload or an unreadable mesh must not linger as a
        # job the UI could be pointed at
        jobs.discard(job_id)
        raise
    with jobs._lock:
        jobs._jobs[job_id]['filename'] = file.filename
        jobs._jobs[job_id]['input'] = input_name
        jobs._jobs[job_id]['status'] = 'uploaded'
    return {'id': job_id, 'filename': file.filename,
            'input_stats': sanitize(stats)}


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
        raise HTTPException(404, _UNKNOWN_JOB)
    return {'id': job_id, 'status': what}


@app.get('/api/jobs/{job_id}')
def job_state(job_id: str):
    state = jobs.public_state(job_id)
    if state is None:
        raise HTTPException(404, _UNKNOWN_JOB)
    return state


@app.get('/api/jobs/{job_id}/bodies')
async def bodies(job_id: str):
    """Every connected shell of the uploaded mesh, largest first, with the
    triangle -> body map the viewer colours and picks with."""
    _, src = _input_path(job_id)
    try:
        return sanitize(await run_in_threadpool(body_list, src))
    except Exception as e:
        log.warning('could not split %s into bodies: %s', src, describe(e))
        raise HTTPException(400, f'Could not split the mesh into bodies: {describe(e)}')


_UNKNOWN_JOB = ('This upload is no longer on the server (old uploads are cleared '
                'after a day, and a restart forgets jobs that never finished). '
                'Load the file again.')
_NO_INPUT = ('The uploaded file is no longer on the server. Load it again.')


def _input_path(job_id):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, _UNKNOWN_JOB)
    src = os.path.join(jobs.job_dir(job_id), job.get('input') or '')
    if not job.get('input') or not os.path.exists(src):
        raise HTTPException(404, _NO_INPUT)
    return job, src


@app.post('/api/jobs/{job_id}/convert')
def convert(job_id: str, params: ConvertParams):
    job, _ = _input_path(job_id)
    if job['status'] in ('queued', 'running'):
        raise HTTPException(409, 'This file is already being converted.')
    jobs.cleanup_old()
    try:
        jobs.start(job_id, job.get('filename'), params.model_dump())
    except OSError as e:
        raise HTTPException(500, 'Could not start the conversion: '
                                 f'{describe(e)}')
    return {'id': job_id, 'status': 'running'}



_AXES = ('x', 'y', 'z')


def _check_slice_params(axis, tol, join, trim, step=None):
    if axis not in _AXES:
        raise HTTPException(400, 'axis must be x, y or z')
    if not (0 < tol <= 5):
        raise HTTPException(400, 'tol must be in (0, 5] mm')
    if not (0 <= join <= 20):
        raise HTTPException(400, 'join must be in [0, 20] mm')
    if not (0 <= trim <= 5):
        raise HTTPException(400, 'trim must be in [0, 5] mm')
    if step is not None and not (0 < step <= 50):
        raise HTTPException(400, 'step must be in (0, 50] mm')


def _safe_stem(job):
    stem = _IMG_RE.sub('', _EXT_RE.sub('', job.get('filename') or 'part'))
    return re.sub(r'[^\w.-]+', '_', stem) or 'part'


def _py_attachment(text, filename):
    from fastapi.responses import Response
    return Response(text, media_type='text/x-python',
                    headers={'Content-Disposition': f'attachment; filename="{filename}"'})


@app.get('/api/jobs/{job_id}/loft-axis')
async def loft_axis(job_id: str, units: str = 'mm', scale: float = 1.0, slice_mm: float = 0.2,
                    join: float = 2.5, trim: float = 0.0):
    """Which axis a whole-body sliced loft picks for 'auto' (by slicing
    structure: fewest one-slice runs, then fewest runs plus steep pairs,
    then the longest side), with the per-axis scores. Under a second per
    axis; the view puts the slicing plane on it before the run."""
    _check_slice_params('z', 0.08, join, trim, slice_mm)
    _, src = _input_path(job_id)
    from .sections import loft_axis as _loft_axis
    try:
        return sanitize(await run_in_threadpool(_loft_axis, src, units, scale, slice_mm, join, trim))
    except Exception as e:
        raise HTTPException(400, f'Could not choose the axis: {describe(e)}')


@app.get('/api/jobs/{job_id}/section')
async def section(job_id: str, axis: str = 'z', offset: float = 0.0, tol: float = 0.08,
                  units: str = 'mm', scale: float = 1.0, join: float = 2.5,
                  outline: bool = False, trim: float = 0.0):
    """One traced section of the uploaded mesh: the plane across `axis`
    at `offset` mm from the bounding-box centre (Fusion's section-plane
    slider), fitted as lines, arcs and splines within `tol`. Open chains
    of a leaky mesh stay open, as in Fusion's mesh section sketch, after
    free ends within `join` mm of each other are joined (0 = never);
    `outline` leaves the open chains out (closed loops only); `trim` cuts
    slivers thinner than that many mm out of the closed loops (0 = off).
    Returns the curves as 3-D polylines (mm, converted frame) for the
    viewer plus the fit statistics."""
    _check_slice_params(axis, tol, join, trim)
    _, src = _input_path(job_id)
    from .sections import trace
    try:
        return sanitize(await run_in_threadpool(trace, src, axis, offset, tol, units, scale, join,
                                                outline, trim))
    except Exception as e:
        raise HTTPException(400, f'Could not trace the section: {describe(e)}')


@app.get('/api/jobs/{job_id}/section-script')
async def section_script(job_id: str, axis: str = 'z', offset: float = 0.0, tol: float = 0.08,
                         units: str = 'mm', scale: float = 1.0, join: float = 2.5,
                         outline: bool = False, trim: float = 0.0):
    """The same section as a Fusion 360 script: one construction plane
    and one sketch of lines, arcs, circles and fitted splines (Create Mesh
    Section Sketch + Fit Curves to Mesh Section, in one go)."""
    _check_slice_params(axis, tol, join, trim)
    job, src = _input_path(job_id)
    from .sections import trace
    from stl_to_solid.fusion_export import emit_fusion_sections_script
    try:
        sec = await run_in_threadpool(trace, src, axis, offset, tol, units, scale, join, outline,
                                      trim)
    except Exception as e:
        raise HTTPException(400, f'Could not trace the section: {describe(e)}')
    name = f"section {axis.upper()}={sec['at']:.2f} mm ({offset:+.1f} from centre)"
    text = emit_fusion_sections_script([{'origin': sec['origin'], 'normal': sec['normal'],
                                         'name': name, 'loops': sec['loops'],
                                         'open': sec['open']}])
    return _py_attachment(text, f'{_safe_stem(job)}_section_{axis}{offset:+.1f}.py')


@app.get('/api/jobs/{job_id}/xray')
async def xray(job_id: str, axis: str = 'z', frm: float = Query(0.0, alias='from'),
               to: float = 0.0, step: float = 0.2, units: str = 'mm', scale: float = 1.0):
    """Where an x-ray's planes would go: the offsets (mm from the centre)
    of a stack from `from` to `to` every `step`, the end plane always
    included, both ends kept just inside the part. Cheap (no cut, no
    fit): the count and whether it is over the script's limit."""
    _check_slice_params(axis, 0.08, 0, 0, step)
    _, src = _input_path(job_id)
    from .sections import half_extent, XRAY_MAX
    from stl_to_solid.section_fit import stack_offsets
    try:
        h = await run_in_threadpool(half_extent, src, axis, units, scale)
    except Exception as e:
        raise HTTPException(400, f'Could not read the mesh: {describe(e)}')
    offs = stack_offsets(frm, to, step, h)
    return {'axis': axis, 'from': frm, 'to': to, 'step': step, 'half_extent': h,
            'offsets': offs, 'count': len(offs), 'max': XRAY_MAX, 'over_cap': len(offs) > XRAY_MAX}


@app.get('/api/jobs/{job_id}/xray-script')
async def xray_script(job_id: str, axis: str = 'z', frm: float = Query(0.0, alias='from'),
                      to: float = 0.0, step: float = 0.2, tol: float = 0.08,
                      units: str = 'mm', scale: float = 1.0, join: float = 2.5,
                      outline: bool = False, trim: float = 0.0, extrude: bool = False):
    """A stack of section sketches from `from` to `to` every `step` mm as
    one Fusion 360 script: a construction plane and a fully enclosed
    sketch per slice (lines, arcs, circles, fitted splines), drawn by the
    script with a progress dialog. `extrude` also extrudes every slice to
    the next plane, so the stack comes out as solid slabs: the last slice
    one spacing further but never past the part's far face, and an end
    plane closer than half a spacing to the slice before it is drawn only,
    that slice's slab running through instead (no hair-thin slab). At most
    XRAY_MAX slices and XRAY_BUDGET_S seconds of fitting."""
    _check_slice_params(axis, tol, join, trim, step)
    job, src = _input_path(job_id)
    from .sections import half_extent, trace_stack, XRAY_MAX
    from stl_to_solid.section_fit import stack_offsets, STACK_EDGE_MM
    from stl_to_solid.fusion_export import emit_fusion_xray_script
    try:
        h = await run_in_threadpool(half_extent, src, axis, units, scale)
    except Exception as e:
        raise HTTPException(400, f'Could not read the mesh: {describe(e)}')
    offs = stack_offsets(frm, to, step, h)
    if len(offs) > XRAY_MAX:
        raise HTTPException(400, f'{len(offs)} slices is over the limit of {XRAY_MAX}; raise the '
                                 'spacing or narrow the range')
    try:
        secs = await run_in_threadpool(trace_stack, src, axis, offs, tol, units, scale, join,
                                       outline, trim, XRAY_BUDGET_S)
    except TimeoutError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f'Could not trace the sections: {describe(e)}')
    n = len(secs)
    exts = [None] * n
    if extrude and n > 1:
        # each slice to the next plane (the offsets ascend, so every slab
        # goes towards +axis); the last one a full spacing, like the add-in,
        # but never past the part's far face
        far = secs[-1]['centre'] + secs[-1]['extent'] / 2.0 - STACK_EDGE_MM
        for i in range(n - 1):
            exts[i] = secs[i + 1]['at'] - secs[i]['at']
        exts[-1] = max(0.0, min(step, far - secs[-1]['at']))
        # the end plane is always cut, so the slice before it may be a
        # hair away: draw that end plane but let the previous slab run
        # through to where the end slab would have finished
        if n > 2 and exts[-2] < 0.5 * step:
            exts[-2] += exts[-1]
            exts[-1] = None
        if exts[-1] is not None and exts[-1] < 0.01:
            exts[-1] = None                       # the end plane sits on the face
    sections = []
    for i, sec in enumerate(secs):
        sections.append({'origin': sec['origin'], 'normal': sec['normal'],
                         'name': f"xray {axis.upper()}={sec['at']:.2f} mm ({i + 1}/{n})",
                         'loops': sec['loops'], 'open': sec['open'],
                         'areas_mm2': sec.get('areas_mm2', []), 'extrude_mm': exts[i]})
    notes = []
    n_trim = sum(1 for s in secs if s['stats'].get('trimmed'))
    if n_trim:
        k = sum(s['stats']['trimmed'] for s in secs)
        notes.append(f"{k} sliver{'s' if k != 1 else ''} trimmed in {n_trim} of {n} slices "
                     f"(Trim slivers up to {trim:g} mm); if the part has real walls that thin, "
                     "lower it")
    lo, hi = sorted((frm, to))
    title = f'X-Ray {axis.upper()} {lo:+.1f}..{hi:+.1f} mm every {step:g} mm'
    text = emit_fusion_xray_script(sections, title=title, notes=notes)
    suffix = '_solid' if extrude and n > 1 else ''
    return _py_attachment(text, f'{_safe_stem(job)}_xray_{axis}{lo:+.1f}_{hi:+.1f}_s{step:g}{suffix}.py')


# ---------------------------------------------------------------------------
# Blueprint: a dimensioned drawing (image) -> recipe -> sketches + extrusions

@app.get('/api/blueprints/config')
def blueprint_config():
    """The vision providers the panel can offer, with whether the server
    holds a key for each (never the key itself)."""
    return bp.config()


@app.get('/api/blueprints/models')
async def blueprint_models(provider: Literal['anthropic', 'openai', 'deepseek', 'custom'] = 'anthropic',
                           base_url: Optional[str] = None,
                           x_api_key: Optional[str] = Header(None, alias='X-Api-Key')):
    """The models the given key can use with this provider, newest first,
    for the panel's dropdown (Anthropic: only ones that take images). The
    key travels in the header for this one call, as for /read."""
    preset = bp.PRESETS[provider]
    key = bp.resolve_key(provider, x_api_key)
    if not key and preset['needs_key']:
        raise HTTPException(400, bp.NO_KEY_MESSAGE.format(label=preset['label'], env=preset['env']))
    if provider == 'custom' and not (base_url or '').strip():
        raise HTTPException(400, 'The custom provider needs a base URL to list its models.')
    missing = bp.sdk_available(provider)
    if missing:
        raise HTTPException(503, missing + '.')
    try:
        models = await run_in_threadpool(bp.list_models, provider, key or '', base_url)
    except Exception as e:
        raise HTTPException(400 if getattr(e, 'kind', '') == 'key' else 502,
                            f'Could not list models: {describe(e)}')
    return {'provider': provider, 'models': models, 'default': preset['default_model']}


@app.post('/api/blueprints')
async def create_blueprint(file: UploadFile):
    """Upload a drawing image. Same job directory layout as a mesh upload,
    so the polling, cancel and download routes serve it unchanged."""
    m = _IMG_RE.search(file.filename or '')
    if not m:
        raise HTTPException(400, 'That is not a drawing image. Expected a .jpg, .png or .webp '
                                 '(a mesh goes to Mesh → Solid, Sliced Loft or X-Ray).')
    input_name = 'input.' + m.group(1).lower()
    job_id, d = jobs.new_job()
    try:
        size = 0
        with open(os.path.join(d, input_name), 'wb') as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > bp.MAX_IMAGE:
                    raise HTTPException(
                        413, f'The image is too large: the limit is {bp.MAX_IMAGE // (1 << 20)} MB.')
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, 'The file is empty.')
        try:
            w, h = await run_in_threadpool(bp.verify_image, os.path.join(d, input_name))
        except Exception as e:
            raise HTTPException(400, f'Could not read {file.filename or "the image"}: {describe(e)}')
    except BaseException:
        jobs.discard(job_id)
        raise
    with jobs._lock:
        jobs._jobs[job_id].update(filename=file.filename, input=input_name,
                                  status='uploaded', tool='blueprint')
    return {'id': job_id, 'filename': file.filename, 'width': w, 'height': h}


def _drawing_path(job_id):
    job, src = _input_path(job_id)
    if not _IMG_RE.search(src):
        raise HTTPException(400, 'This job holds a mesh, not a drawing. Blueprint works on an '
                                 'image (.jpg, .png or .webp).')
    return job, src


@app.api_route('/api/blueprints/{job_id}/drawing', methods=['GET', 'HEAD'])
def blueprint_drawing(job_id: str):
    _, src = _drawing_path(job_id)
    return FileResponse(src, media_type=bp.IMAGE_MEDIA[src.rsplit('.', 1)[1].lower()])


class ReadBody(BaseModel):
    provider: Literal['anthropic', 'openai', 'deepseek', 'custom'] = 'anthropic'
    model: Optional[str] = Field(None, max_length=200)
    base_url: Optional[str] = Field(None, max_length=500)
    hints: str = Field('', max_length=2000, description='notes for the reader, e.g. which view is which')
    effort: Literal['low', 'medium', 'high'] = 'high'


@app.post('/api/blueprints/{job_id}/read')
def blueprint_read(job_id: str, body: ReadBody,
                   x_api_key: Optional[str] = Header(None, alias='X-Api-Key')):
    """Read the drawing with a vision model, then build. The key comes in
    the header for this one call (the browser keeps it); with none, the
    server's own key for the provider is used when set."""
    job, src = _drawing_path(job_id)
    if job['status'] in ('reading', 'queued', 'running'):
        raise HTTPException(409, 'This drawing is already being read or built; wait or cancel first.')
    preset = bp.PRESETS[body.provider]
    key = bp.resolve_key(body.provider, x_api_key)
    if not key and preset['needs_key']:
        raise HTTPException(400, bp.NO_KEY_MESSAGE.format(label=preset['label'], env=preset['env']))
    model = (body.model or '').strip() or preset['default_model']
    base_url = (body.base_url or '').strip() or preset['base_url'] or None
    if body.provider == 'custom' and not (model and base_url):
        raise HTTPException(400, 'The custom provider needs both a base URL and a model name.')
    missing = bp.sdk_available(body.provider)
    if missing:
        raise HTTPException(503, missing + '.')
    jobs.cleanup_old()
    from stl_to_solid.blueprint.read_drawing import read_drawing
    with open(src, 'rb') as f:
        image = f.read()
    stem = _safe_stem(job)
    hints = body.hints
    effort = body.effort

    def read_fn():
        return read_drawing(image, body.provider, key or '', model=model, base_url=base_url,
                            hints=hints, timeout=bp.READ_TIMEOUT_S, effort=effort)

    def then_params(res):
        return {'tool': 'blueprint', 'recipe': res['recipe'], 'title': stem,
                'read': {k: v for k, v in res.items() if k not in ('recipe', 'raw_text')}}

    try:
        jobs.start_reading(job_id, job.get('filename'), read_fn, then_params)
    except OSError as e:
        raise HTTPException(500, f'Could not start reading: {describe(e)}')
    return {'id': job_id, 'status': 'reading', 'provider': body.provider, 'model': model}


class BuildBody(BaseModel):
    recipe: dict


@app.post('/api/blueprints/{job_id}/build')
def blueprint_build(job_id: str, body: BuildBody):
    """Build an edited recipe: no model call, the validator first (a
    refused recipe is a 400 listing what is wrong), then the worker."""
    from stl_to_solid.blueprint import normalize, validate
    job, _ = _drawing_path(job_id)
    if job['status'] in ('reading', 'queued', 'running'):
        raise HTTPException(409, 'This drawing is still being read or built; wait or cancel first.')
    recipe = normalize(body.recipe)
    rep = validate(recipe)
    if rep.errors:
        n = len(rep.errors)
        raise HTTPException(400, f'The recipe is not valid ({n} problem{"s" if n != 1 else ""}): '
                                 + '; '.join(rep.errors[:5]) + ('; …' if n > 5 else ''))
    d = jobs.job_dir(job_id)
    read_meta = None
    try:
        with open(os.path.join(d, 'read.json')) as f:
            read_meta = json.load(f)
    except (OSError, ValueError):
        pass
    jobs.cleanup_old()
    try:
        jobs.start(job_id, job.get('filename'),
                   {'tool': 'blueprint', 'recipe': recipe, 'title': _safe_stem(job), 'read': read_meta})
    except OSError as e:
        raise HTTPException(500, f'Could not start the build: {describe(e)}')
    return {'id': job_id, 'status': 'running', 'warnings': rep.warnings}


@app.post('/api/blueprints/{job_id}/preview')
async def blueprint_preview(job_id: str, body: BuildBody):
    """A live look at an edited recipe: the shape only, built by the warm
    preview helper (no STEP, no script, the job's result untouched). Answers
    with the size, volume and warnings, the per-triangle feature index for
    colouring, and the mesh itself (binary STL, base64) so mesh and map
    always belong to the same build."""
    from stl_to_solid.blueprint import normalize, validate
    job, _ = _drawing_path(job_id)
    recipe = normalize(body.recipe)
    rep = validate(recipe)
    if rep.errors:
        n = len(rep.errors)
        raise HTTPException(400, f'The recipe is not valid ({n} problem{"s" if n != 1 else ""}): '
                                 + '; '.join(rep.errors[:5]) + ('; …' if n > 5 else ''))
    d = jobs.job_dir(job_id)
    try:
        ans = await run_in_threadpool(bp_preview.helper.build, recipe, d, 'live')
    except bp_preview.PreviewError as e:
        raise HTTPException(400 if e.kind == 'recipe' else 500, str(e))
    import base64
    ans['stl_b64'] = base64.b64encode(ans.pop('stl_bytes', b'')).decode('ascii')
    ans['recipe'] = recipe
    return sanitize(ans)


@app.api_route('/api/blueprints/{job_id}/live.stl', methods=['GET', 'HEAD'])
def blueprint_live_stl(job_id: str):
    _drawing_path(job_id)
    path = os.path.join(jobs.job_dir(job_id), 'live.stl')
    if not os.path.exists(path):
        raise HTTPException(404, 'No live preview has been built for this drawing yet.')
    return FileResponse(path, media_type='model/stl',
                        headers={'Cache-Control': 'no-store'})


@app.get('/api/blueprints/{job_id}/preview-map')
def blueprint_preview_map(job_id: str):
    """Which feature made each triangle of the built preview (preview.stl),
    for colouring the view by feature after a full build."""
    _drawing_path(job_id)
    path = os.path.join(jobs.job_dir(job_id), 'preview_features.json')
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        raise HTTPException(404, 'This drawing has not been built yet.')
    except ValueError:
        raise HTTPException(500, 'The feature map could not be read; rebuild.')


@app.get('/api/blueprints/{job_id}/recipe')
def blueprint_recipe(job_id: str):
    """The last recipe read or built for this drawing (recipe.json) with
    the read's metadata, so the table can show a recipe whose build was
    refused or is still running."""
    _drawing_path(job_id)
    d = jobs.job_dir(job_id)
    try:
        with open(os.path.join(d, 'recipe.json')) as f:
            recipe = json.load(f)
    except OSError:
        raise HTTPException(404, 'This drawing has not been read yet.')
    except ValueError:
        raise HTTPException(500, 'The saved recipe could not be read; read the drawing again.')
    meta = {}
    try:
        with open(os.path.join(d, 'read.json')) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        pass
    meta.pop('raw_text', None)
    return {'id': job_id, 'recipe': recipe, 'read': meta}


# The file routes also answer HEAD (FastAPI does not add it on its own):
# the UI asks before following a download link, so a file the server no
# longer has is reported in the page instead of saved as a JSON "download".
@app.api_route('/api/jobs/{job_id}/preview', methods=['GET', 'HEAD'])
def preview(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, _UNKNOWN_JOB)
    path = os.path.join(jobs.job_dir(job_id), 'preview.stl')
    if not os.path.exists(path):
        raise HTTPException(404, 'There is no preview for this file.')
    return FileResponse(path, media_type='model/stl')


@app.api_route('/api/jobs/{job_id}/script', methods=['GET', 'HEAD'])
def script(job_id: str):
    """The CadQuery script that rebuilds the recognised extrusion structure."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, _UNKNOWN_JOB)
    path = os.path.join(jobs.job_dir(job_id), 'output.py')
    if not os.path.exists(path):
        raise HTTPException(404, 'This conversion produced no script to download.')
    safe = _safe_stem(job)
    return FileResponse(path, media_type='text/x-python', filename=f'{safe}.py')


@app.api_route('/api/jobs/{job_id}/fusion-script', methods=['GET', 'HEAD'])
def fusion_script(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, _UNKNOWN_JOB)
    path = os.path.join(jobs.job_dir(job_id), 'output_fusion.py')
    if not os.path.exists(path):
        raise HTTPException(404, 'This conversion produced no script to download.')
    safe = _safe_stem(job)
    return FileResponse(path, media_type='text/x-python', filename=f'{safe}_fusion.py')


@app.api_route('/api/jobs/{job_id}/fusion-bfill-script', methods=['GET', 'HEAD'])
def fusion_bfill_script(job_id: str):
    """The Fusion 360 Boundary Fill script for face-group results."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, _UNKNOWN_JOB)
    path = os.path.join(jobs.job_dir(job_id), 'output_fusion_bfill.py')
    if not os.path.exists(path):
        raise HTTPException(404, 'This conversion produced no script to download.')
    safe = _safe_stem(job)
    return FileResponse(path, media_type='text/x-python', filename=f'{safe}_fusion_bfill.py')


@app.api_route('/api/jobs/{job_id}/download', methods=['GET', 'HEAD'])
def download(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, _UNKNOWN_JOB)
    path = os.path.join(jobs.job_dir(job_id), 'output.step')
    if not os.path.exists(path):
        raise HTTPException(404, 'There is no STEP file for this job yet: convert first.')
    safe = _safe_stem(job)
    return FileResponse(path, media_type='application/step',
                        filename=f'{safe}.step')


# Production: serve the built frontend. Registered last so /api wins.
_static = os.environ.get(
    'STLTOSOLID_STATIC',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'frontend', 'dist'))
# The Windows registry can map .js to text/plain, which browsers refuse to
# run as a module; pin the types the built frontend uses.
for _ext, _type in (('.js', 'text/javascript'), ('.mjs', 'text/javascript'),
                    ('.css', 'text/css'), ('.woff2', 'font/woff2'),
                    ('.webp', 'image/webp')):
    mimetypes.add_type(_type, _ext)
if os.path.isdir(_static):
    app.mount('/', StaticFiles(directory=_static, html=True), name='static')
