import { defineStore } from 'pinia'
import { markRaw } from 'vue'
import { errText, friendlyError } from './errors'

// STL/OBJ files carry no units. `units` says what the file's numbers mean;
// the backend scales the mesh to mm on load. Mirrors mesh_prep.UNIT_SCALE.
export const UNITS = [
  { key: 'mm', label: 'millimetres', scale: 1 },
  { key: 'cm', label: 'centimetres', scale: 10 },
  { key: 'in', label: 'inches', scale: 25.4 },
  { key: 'ft', label: 'feet', scale: 304.8 },
  { key: 'm', label: 'metres', scale: 1000 },
]

export const DEFAULT_PARAMS = {
  units: 'mm',
  // Free multiplier on top of the unit. Units only ever enlarge, so a file
  // written 10x too big can only be fixed here, with 0.1.
  scale: 1,
  reduce_tol: 0.05,
  tol: 0.08,
  accept_p95: 0.25,
  accept_max: 0.26,
  accept_hole_max: 0.1,
  accept_vol_pct: 2.0,
  force_prismatic: false,
  face_groups: true,
  // 'auto': the prismatic / face-group / faceted ladder. 'loft': slice the
  // body along an axis and loft the section outlines (Fusion's Create Mesh
  // Section Sketch + Fit Curves + Loft, automated).
  method: 'auto',
  slice_mm: 0.2,
  slice_axis: 'auto',
  loft_ruled: false,
  // the loft cutter and the single-slice view share these: loose ends
  // closer than slice_join mm are joined, slivers thinner than slice_trim
  // mm are cut out (0 = off)
  slice_join: 2.5,
  slice_trim: 0,
  // partial loft: only slice_range_mm from the single-slice plane in this
  // direction (0 = the whole body); slice_from is filled in at submit time
  slice_range_mm: 0,
  slice_range_dir: '-',
}

// the most sketches one x-ray script may hold (backend/sections.XRAY_MAX)
export const XRAY_MAX = 500
// drawing images the Blueprint tool reads (backend/blueprint.IMAGE_EXTS)
export const IMAGE_EXTS = ['jpg', 'jpeg', 'png', 'webp']
// the Blueprint keys and choices live in this browser only
const BP_STORE = 'stltosolid.blueprint'

function loadBp() {
  try {
    return JSON.parse(localStorage.getItem(BP_STORE) || '{}') || {}
  } catch {
    return {}
  }
}
function saveBp(patch) {
  try {
    localStorage.setItem(BP_STORE, JSON.stringify({ ...loadBp(), ...patch }))
  } catch { /* private window or storage off: the choice lasts the session */ }
}
const bpSaved = loadBp()
// planes are kept this far inside the part's ends (section_fit.STACK_EDGE_MM)
const XRAY_EDGE = 1e-3

let pollTimer = null
let liveSeq = 0
let traceSeq = 0
let axisSeq = 0
let xraySeq = 0
// how many polls in a row may fail before the job is given up on: a
// server busy for a second, or a laptop waking up, is not a lost job
const POLL_MAX_MISSES = 8

export const useConvertStore = defineStore('convert', {
  state: () => ({
    // which tool the top toolbar has picked: 'solid' (Mesh -> Solid, the
    // auto ladder), 'loft' (Sliced Loft) or 'xray' (section sketches). It
    // decides what the rail shows; params.method follows it for the API.
    tool: 'solid',
    // idle -> uploading -> ready -> running -> done | error
    status: 'idle',
    jobId: null,
    filename: null,
    inputStats: null,
    params: { ...DEFAULT_PARAMS },
    log: '',
    result: null,
    error: null,
    // server-side view of a running job ('queued' | 'running')
    serverStatus: null,
    // a cancel request is in flight
    cancelling: false,
    // the last run was stopped by the user
    cancelled: false,
    // every connected shell of the upload, largest first (/bodies)
    bodies: [],
    // triangle -> body index, for colouring and picking in the viewer
    triangleBody: null,
    bodiesTruncated: false,
    // why the body list is missing (the file still converts as a whole)
    bodiesError: null,
    // body indices ticked for conversion; empty means "all of them"
    selected: [],
    hovered: -1,
    // single slice (sliced-loft method): where the plane sits along the
    // chosen axis, in mm from the part's centre (Fusion's slider), how far
    // apart two loose ends of a leaky mesh may be and still be joined (mm;
    // 0 draws exactly what Fusion's mesh section draws), and the last
    // traced section from /section
    sliceOffset: 0,
    // closed loops only: leave out the loose open pieces of a leaky mesh
    sliceOutline: false,
    section: null,
    sectionBusy: false,
    sectionError: null,
    // the axis a whole-body loft will pick for 'auto' (from /loft-axis),
    // so the plane shows on it before the run; null until known
    loftAxisAuto: null,
    loftAxisScores: null,
    loftAxisBusy: false,
    // x-ray: a stack of slices from the start plane to the end plane every
    // xraySpacing mm (all mm from the centre along the resolved axis). The
    // slices are traced one after another and every trace stays in the 3D
    // view (xrayTraces, in plane order, of xrayTotal), so the stack shows
    // up slice by slice like an x-ray; xrayTraceMs is the mean time per
    // slice (the estimate); xrayExtrude also extrudes each slice to the next
    xrayFrom: 0,
    xrayTo: 0,
    xraySpacing: 0.2,
    xrayExtrude: false,
    xrayTraces: [],
    xrayTotal: 0,
    xrayBusy: false,
    xrayError: null,
    xrayTraceMs: null,
    // X-Ray's own gap joining and sliver trimming (mm). The loft keeps its
    // values in params: a trim set for a leaky loft must not quietly eat the
    // thin walls of the next part put through the X-Ray
    xrayJoin: 2.5,
    xrayTrim: 0,
    // the part's exact side along one axis, from the server ({job, axis,
    // units, scale, mm}); bbox_mm is rounded to 0.01 mm, and a plane clamped
    // with the rounded half can land just outside the part and cut nothing
    exactExtent: null,
    // Blueprint: the uploaded drawing (served back by /drawing), its pixel
    // size, the providers the server offers (/blueprints/config), the
    // provider, model, URL and per-provider keys this browser keeps, and
    // the last read's recipe plus the build's report (mirrors result.*
    // so the panel binds to one place)
    imageUrl: null,
    imageSize: null,
    blueprintConfig: null,
    provider: bpSaved.provider || 'anthropic',
    providerModel: bpSaved.providerModel || {},     // { provider: model }
    providerBaseUrl: bpSaved.providerBaseUrl || '',
    apiKeys: bpSaved.apiKeys || {},                 // { provider: key }
    hints: '',
    recipe: null,
    recipeRead: null,        // { model, provider, usage, seconds, repaired }
    warnings: [],
    bbox: null,
    volume: null,
    overallCheck: null,
    // the coloured view: the feature list of the shape on the stage and
    // one feature index per triangle of its mesh (from the full build's
    // preview_features.json or a live preview), plus the live preview's
    // mesh itself ({ data, kind, seq }) and its state
    featureList: [],
    featureMap: null,
    liveStl: null,
    liveBusy: false,
    liveError: null,
    liveInfo: null,          // { bbox, volume_mm3, solids, warnings, overall_check } of the live shape
  }),

  getters: {
    busy: (s) => s.status === 'uploading' || s.status === 'running',
    // Blueprint
    isImageJob: (s) => !!s.imageUrl,
    // what the panel's button says while the job runs
    readStatus: (s) => (s.status !== 'running' ? null : s.serverStatus === 'reading' ? 'reading' : 'building'),
    providerPreset: (s) => (s.blueprintConfig?.providers || []).find((p) => p.key === s.provider) || null,
    hasServerKey: (s) => !!s.providerPreset?.server_key,
    apiKey: (s) => s.apiKeys[s.provider] || '',
    modelName: (s) => (s.providerModel[s.provider] ?? s.providerPreset?.default_model ?? ''),
    canRead: (s) =>
      !!s.jobId && !!s.imageUrl && !s.busy && s.status !== 'uploading'
      && (!!s.apiKey || s.hasServerKey || s.providerPreset?.needs_key === false)
      && (s.provider !== 'custom' || (!!s.providerBaseUrl && !!s.modelName)),
    // A file with one body needs no picker at all.
    hasBodyPicker: (s) => s.bodies.length > 1,
    selectedCount: (s) => (s.selected.length || s.bodies.length),
    // file units -> mm, for showing input numbers the way the pipeline sees them
    // file units -> mm: the unit's factor times the free scale, which is
    // what the pipeline applies and so what the previews must use.
    unitScale: (s) =>
      (UNITS.find((u) => u.key === s.params.units)?.scale ?? 1)
      * (Number(s.params.scale) > 0 ? Number(s.params.scale) : 1),
    downloadUrl: (s) =>
      s.status === 'done' && s.result?.ok
        ? `/api/jobs/${s.jobId}/download`
        : null,
    scriptUrl: (s) =>
      s.status === 'done' && s.result?.ok && s.result?.has_script
        ? `/api/jobs/${s.jobId}/script`
        : null,
    // prismatic bodies write it next to the CadQuery script; sliced lofts
    // write it on its own (has_fusion_script), older results only say has_script
    fusionScriptUrl: (s) =>
      s.status === 'done' && s.result?.ok
        && (s.result?.has_fusion_script ?? s.result?.has_script)
        ? `/api/jobs/${s.jobId}/fusion-script`
        : null,
    bfillCheck: (s) => (s.status === 'done' && s.result?.ok ? s.result?.bfill_check || null : null),
    // half the part's side along the resolved slice axis, mm: the slider's range
    sliceHalfExtent: (s) => {
      const a = s.resolvedSliceAxis
      const bb = s.inputStats?.bbox_mm
      if (!a || !bb) return 0
      const e = s.exactExtent
      if (e && e.job === s.jobId && e.axis === a && e.units === s.params.units && e.scale === s.params.scale) {
        return e.mm / 2
      }
      return (bb['xyz'.indexOf(a)] * s.unitScale) / 2
    },
    // the gap and sliver settings of the tool in use
    sliceJoin: (s) => (s.tool === 'xray' ? s.xrayJoin : s.params.slice_join),
    sliceTrim: (s) => (s.tool === 'xray' ? s.xrayTrim : s.params.slice_trim),
    sectionScriptUrl: (s) => {
      const a = s.resolvedSliceAxis
      if (!s.jobId || !a) return null
      return `/api/jobs/${s.jobId}/section-script?${sectionQuery(s, s.sliceOffset)}`
    },
    // The axis the slices are cut across, with 'auto' resolved to the
    // file's longest side the way sliced_loft.axis_index does; null unless
    // a slicing tool (loft, x-ray) is up. Drives the plane in the 3D view.
    resolvedSliceAxis: (s) => {
      if (s.tool !== 'loft' && s.tool !== 'xray') return null
      const a = s.params.slice_axis
      if (a === 'x' || a === 'y' || a === 'z') return a
      // a whole-body loft picks its axis by slicing structure
      // (sliced_loft.choose_axis): the finished run's choice first, else
      // the one /loft-axis worked out for the preview
      const chosen = s.loftAxisChosen
      if (chosen) return chosen
      if (s.tool === 'loft' && s.loftAxisAuto && !(s.params.slice_range_mm > 0)) return s.loftAxisAuto
      const bb = s.inputStats?.bbox_mm
      if (!bb) return null
      return 'xyz'[bb.indexOf(Math.max(...bb))]
    },
    // The axis a finished loft run picked for 'auto' (null before a run,
    // for a partial loft, or when the user fixed the axis).
    loftAxisChosen: (s) => {
      if (s.status !== 'done' || !s.result?.ok || s.tool !== 'loft') return null
      if (s.params.slice_axis !== 'auto') return null
      const lf = s.result?.metrics?.loft
        || (s.result?.bodies || []).find((b) => b.mode === 'loft' && b.metrics?.loft)?.metrics?.loft
      if (!lf || !lf.axis_auto || lf.range) return null
      const a = lf.axis_name
      return a === 'x' || a === 'y' || a === 'z' ? a : null
    },
    fusionBfillScriptUrl: (s) =>
      s.status === 'done' && s.result?.ok && s.result?.has_bfill_script
        ? `/api/jobs/${s.jobId}/fusion-bfill-script`
        : null,
    // What the 3D view draws for the current tool: the slicing planes
    // [{ axis, offset (mm from centre), kind: 'single' | 'start' | 'end' }]
    // and the traced sections that sit on them, in the same order.
    viewPlanes: (s) => {
      const a = s.resolvedSliceAxis
      if (!a) return []
      if (s.tool === 'loft') return [{ axis: a, offset: s.sliceOffset, kind: 'single' }]
      if (s.tool === 'xray') {
        if (s.xrayFrom === s.xrayTo) return [{ axis: a, offset: s.xrayFrom, kind: 'single' }]
        return [{ axis: a, offset: s.xrayFrom, kind: 'start' }, { axis: a, offset: s.xrayTo, kind: 'end' }]
      }
      return []
    },
    viewSections: (s) => {
      if (!s.resolvedSliceAxis) return []
      if (s.tool === 'loft') return s.section ? [s.section] : []
      if (s.tool === 'xray') {
        const n = s.xrayTotal
        return s.xrayTraces.map((sec, i) => ({
          ...sec,
          kind: n === 1 ? 'single' : i === 0 ? 'start' : i === n - 1 ? 'end' : 'mid',
        }))
      }
      return []
    },
    // the traced start plane (the first slice) and end plane (the last,
    // once the whole stack is traced)
    xrayStartSection: (s) => s.xrayTraces[0] || null,
    xrayEndSection: (s) =>
      (s.xrayTotal > 0 && s.xrayTraces.length === s.xrayTotal ? s.xrayTraces[s.xrayTotal - 1] : null),
    // Where the x-ray's planes go: the JS mirror of section_fit.stack_offsets
    // (swap, clamp just inside the part, from + k*step, the end plane always
    // included), so the count is live without a server round trip.
    xrayOffsets: (s) => {
      const h = s.sliceHalfExtent
      if (!(h > 0)) return []
      const lim = Math.max(0, h - XRAY_EDGE)
      let [lo, hi] = [Number(s.xrayFrom) || 0, Number(s.xrayTo) || 0].sort((a, b) => a - b)
      lo = Math.min(Math.max(lo, -lim), lim)
      hi = Math.min(Math.max(hi, -lim), lim)
      const step = Number(s.xraySpacing)
      if (!(step > 0) || hi - lo <= 1e-9) return [lo]
      const n = Math.floor((hi - lo) / step + 1e-9)
      const out = []
      for (let k = 0; k <= n; k++) out.push(lo + k * step)
      if (hi - out[out.length - 1] > 1e-9) out.push(hi)
      return out
    },
    xrayCount: (s) => s.xrayOffsets.length,
    xrayOverCap: (s) => s.xrayCount > XRAY_MAX,
    // seconds the script will take to build, from the last trace's time
    xrayEstimateS: (s) => (s.xrayTraceMs ? (s.xrayCount * s.xrayTraceMs) / 1000 : null),
    xrayScriptUrl: (s) => {
      const a = s.resolvedSliceAxis
      if (!s.jobId || !a || s.xrayCount < 1 || s.xrayOverCap) return null
      const q = sectionQuery(s, 0)
      q.delete('offset')
      q.set('from', String(s.xrayFrom))
      q.set('to', String(s.xrayTo))
      q.set('step', String(s.xraySpacing))
      if (s.xrayExtrude && s.xrayCount > 1) q.set('extrude', 'true')
      return `/api/jobs/${s.jobId}/xray-script?${q}`
    },
  },

  actions: {
    async upload(file) {
      this.stopPolling()
      this.$patch({
        status: 'uploading', jobId: null, inputStats: null,
        log: '', result: null, error: null, filename: file.name,
        loftAxisAuto: null, loftAxisScores: null,
        imageUrl: null, imageSize: null, recipe: null, recipeRead: null, warnings: [],
        bbox: null, volume: null, overallCheck: null,
        featureList: [], featureMap: null, liveStl: null, liveBusy: false, liveError: null, liveInfo: null,
      })
      try {
        const body = new FormData()
        body.append('file', file)
        const res = await fetch('/api/jobs', { method: 'POST', body })
        if (!res.ok) throw new Error(await errText(res))
        const data = await res.json()
        if (!data?.id) throw new Error('the server sent no job id back')
        this.$patch({
          status: 'ready', jobId: data.id, inputStats: data.input_stats,
          bodies: [], triangleBody: null, selected: [], hovered: -1, bodiesError: null,
          sliceOffset: 0, sliceOutline: false, section: null, sectionError: null,
          xrayFrom: 0, xrayTo: 0, xrayExtrude: false, xrayTraces: [], xrayTotal: 0,
          xrayError: null, xrayTraceMs: null, exactExtent: null,
        })
        this.loadBodies()
      } catch (e) {
        this.$patch({ status: 'error', error: `Upload failed: ${friendlyError(e)}` })
      }
    },

    // The shell list is what makes a 72-body file usable: without it the
    // only choice is "convert everything", which on such a file is hours.
    async loadBodies() {
      if (!this.jobId) return
      const job = this.jobId
      try {
        const res = await fetch(`/api/jobs/${job}/bodies`)
        if (!res.ok) throw new Error(await errText(res))
        const d = await res.json()
        if (job !== this.jobId) return          // another file was loaded meanwhile
        this.$patch({
          bodies: d.bodies || [],
          triangleBody: d.triangle_body || null,
          bodiesTruncated: !!d.truncated,
          bodiesError: null,
          // Preselect the bodies worth converting (see DEFAULT_PICK_FRAC):
          // on a controller that is the housing halves, not the 70 screws.
          selected: (d.bodies || []).filter((b) => b.suggested).map((b) => b.index),
        })
      } catch (e) {
        // the picker is optional; conversion still works on the whole file
        if (job === this.jobId) this.bodiesError = friendlyError(e)
      }
    },

    toggleBody(i) {
      const at = this.selected.indexOf(i)
      if (at >= 0) this.selected.splice(at, 1)
      else this.selected.push(i)
    },
    selectAllBodies() { this.selected = this.bodies.map((b) => b.index) },
    selectSuggestedBodies() {
      this.selected = this.bodies.filter((b) => b.suggested).map((b) => b.index)
    },

    async convert() {
      if (!this.jobId) return
      this.$patch({ status: 'running', log: '', result: null, error: null,
                    cancelled: false })
      try {
        const res = await fetch(`/api/jobs/${this.jobId}/convert`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          // Only send a selection when it is a real subset: an empty or
          // complete list means "everything", which the API takes as null.
          body: JSON.stringify({
            ...this.params,
            // a partial loft starts at the single-slice plane
            slice_from: this.params.slice_range_mm > 0 ? this.sliceOffset : null,
            bodies: this.selected.length && this.selected.length < this.bodies.length
              ? [...this.selected].sort((a, b) => a - b)
              : null,
          }),
        })
        if (!res.ok) throw new Error(await errText(res))
        this.startPolling()
      } catch (e) {
        this.$patch({ status: 'error', error: `Could not start the conversion: ${friendlyError(e)}` })
      }
    },

    // A conversion can run for tens of minutes; before this the only way
    // out was to wait it out or restart the server.
    async cancel() {
      if (!this.jobId || this.status !== 'running') return
      this.cancelling = true
      try {
        const res = await fetch(`/api/jobs/${this.jobId}/cancel`, { method: 'POST' })
        if (!res.ok) throw new Error(await errText(res))
        // the poll sees the job stop and reports it; nothing to set here
      } catch (e) {
        this.$patch({ error: `Could not cancel: ${friendlyError(e)}` })
      } finally {
        this.cancelling = false
      }
    },

    startPolling() {
      this.stopPolling()
      const job = this.jobId
      let misses = 0
      let inFlight = false
      pollTimer = setInterval(async () => {
        if (inFlight || job !== this.jobId) return   // a slow answer, or a new file
        inFlight = true
        try {
          const res = await fetch(`/api/jobs/${job}`)
          if (res.status === 404) {
            // the server forgot the job for good (a restart, or the TTL):
            // no point asking again
            throw Object.assign(new Error(await errText(res)), { fatal: true })
          }
          if (!res.ok) throw new Error(await errText(res))
          const s = await res.json()
          misses = 0
          if (job !== this.jobId) return
          this.log = s.log || ''
          this.serverStatus = s.status
          if (s.status === 'uploaded' && this.isImageJob) {
            // the server restarted while the drawing was being read: the
            // registry only knows the upload again
            this.stopPolling()
            this.$patch({ status: 'error',
                          error: 'The server restarted while the drawing was being read. Read it again.' })
            return
          }
          if (s.status === 'done' || s.status === 'error' || s.status === 'cancelled') {
            this.stopPolling()
            this.result = s.result
            if (s.result?.mode === 'blueprint' && s.result?.ok) this.takeBlueprintResult(s.result)
            if (s.status === 'error' && s.result?.failure === 'recipe' && this.isImageJob) {
              // read, but a check failed: show the table so the numbers can be fixed
              this.fetchRecipe()
            }
            if (s.status === 'done' && s.result?.ok) {
              this.status = 'done'
            } else if (s.status === 'cancelled') {
              // stopping on purpose is not a failure: say so plainly and
              // leave the file ready to convert again
              this.status = 'ready'
              this.error = null
              this.cancelled = true
            } else {
              this.status = 'error'
              this.error = s.result?.error
                || 'The conversion failed without saying why. The pipeline log below shows how far it got.'
            }
          }
        } catch (e) {
          misses += 1
          if (!e.fatal && misses < POLL_MAX_MISSES) return   // try again next tick
          if (job !== this.jobId) return
          this.stopPolling()
          this.$patch({
            status: 'error',
            error: e.fatal ? friendlyError(e)
              : `Lost contact with the server while working: ${friendlyError(e)} `
                + 'The job may still be running on the server; once it answers again, '
                + 'run it again or load the file again to start over.',
          })
        } finally {
          inFlight = false
        }
      }, 1000)
    },

    stopPolling() {
      if (pollTimer) clearInterval(pollTimer)
      pollTimer = null
    },

    // The toolbar's choice. The only place params.method is set: the
    // sliced loft is a conversion method, the other two tools use the
    // auto ladder (x-ray never converts at all).
    setTool(t) {
      this.tool = t
      this.params.method = t === 'loft' ? 'loft' : 'auto'
    },

    resetParams() {
      this.params = { ...DEFAULT_PARAMS, method: this.tool === 'loft' ? 'loft' : 'auto' }
      this.xrayJoin = DEFAULT_PARAMS.slice_join
      this.xrayTrim = DEFAULT_PARAMS.slice_trim
    },

    // Remember the part's exact side along `axis` (mm, converted frame), as
    // a /section or /xray answer reports it.
    noteExtent(axis, mm) {
      if (!(mm > 0) || !this.jobId) return
      const e = this.exactExtent
      if (e && e.job === this.jobId && e.axis === axis && e.units === this.params.units
          && e.scale === this.params.scale && e.mm === mm) return
      this.exactExtent = { job: this.jobId, axis, units: this.params.units, scale: this.params.scale, mm }
    },

    // Make sure the half extent on the resolved axis is the server's exact
    // one before the planes are placed (one cheap /xray call; nothing is
    // cut). Any failure leaves the rounded value, which is fine for the
    // count and only ever matters at the very ends of the part.
    async ensureExtent() {
      const a = this.resolvedSliceAxis
      if (!this.jobId || !a) return
      const e = this.exactExtent
      if (e && e.job === this.jobId && e.axis === a && e.units === this.params.units && e.scale === this.params.scale) return
      try {
        const q = new URLSearchParams({ axis: a, from: '0', to: '0', step: '1',
                                        units: this.params.units, scale: String(this.params.scale) })
        const res = await fetch(`/api/jobs/${this.jobId}/xray?${q}`)
        if (!res.ok) return
        const got = await res.json()
        if (got?.half_extent > 0) this.noteExtent(a, got.half_extent * 2)
      } catch (e) {
        /* the rounded value stays */
      }
    },

    // Ask which axis a whole-body loft would pick for 'auto', so the plane
    // in the 3D view sits on it before the run. Cheap; debounced by the
    // caller. Any failure just leaves the longest-side plane.
    async fetchLoftAxis() {
      if (!this.jobId || this.tool !== 'loft' || this.params.slice_axis !== 'auto') return
      const mine = ++axisSeq
      this.loftAxisBusy = true
      try {
        const p = this.params
        const q = new URLSearchParams({
          units: p.units, scale: String(p.scale), slice_mm: String(p.slice_mm),
          join: String(p.slice_join ?? 2.5), trim: String(p.slice_trim ?? 0),
        })
        const res = await fetch(`/api/jobs/${this.jobId}/loft-axis?${q}`)
        if (!res.ok) throw new Error(await errText(res))
        const got = await res.json()
        if (mine !== axisSeq) return
        this.loftAxisAuto = got.axis
        this.loftAxisScores = got.scores
      } catch (e) {
        if (mine === axisSeq) { this.loftAxisAuto = null; this.loftAxisScores = null }
      } finally {
        if (mine === axisSeq) this.loftAxisBusy = false
      }
    },

    // One traced section at `offset` mm from the centre on the resolved
    // axis, as /section returns it. Throws on any failure.
    async fetchSection(offset) {
      const res = await fetch(`/api/jobs/${this.jobId}/section?${sectionQuery(this, offset)}`)
      if (!res.ok) throw new Error(await errText(res))
      return res.json()
    },

    // Trace the x-ray's slices one after another, first to last, keeping
    // every trace so the stack appears slice by slice in the 3D view. Over
    // the script's limit only the two end planes are traced. Debounced by
    // the caller; any change (or stopXray) makes the loop stop at once.
    async traceXray() {
      if (!this.jobId || !this.resolvedSliceAxis) return
      const mine = ++xraySeq
      this.$patch({ xrayTraces: [], xrayTotal: 0, xrayBusy: true, xrayError: null })
      await this.ensureExtent()
      if (mine !== xraySeq) return
      const all = this.xrayOffsets
      const offs = this.xrayOverCap ? [all[0], all[all.length - 1]] : all
      this.xrayTotal = offs.length
      const t0 = performance.now()
      try {
        for (let i = 0; i < offs.length; i++) {
          const sec = await this.fetchSection(offs[i])
          if (mine !== xraySeq) return
          if (sec?.axis && sec.extent) this.noteExtent(sec.axis, sec.extent)
          // a new array each time, so the viewer's watcher sees the change
          this.xrayTraces = [...this.xrayTraces, sec]
          this.xrayTraceMs = (performance.now() - t0) / (i + 1)
        }
      } catch (e) {
        if (mine === xraySeq) this.xrayError = friendlyError(e)
      } finally {
        if (mine === xraySeq) this.xrayBusy = false
      }
    },

    // ---- Blueprint ---------------------------------------------------------

    async loadBlueprintConfig() {
      try {
        const res = await fetch('/api/blueprints/config')
        if (res.ok) this.blueprintConfig = await res.json()
      } catch { /* the panel says it could not ask */ }
    },

    setProvider(p) {
      this.provider = p
      saveBp({ provider: p })
    },
    setKey(k) {
      const keys = { ...this.apiKeys }
      const v = (k || '').trim()
      if (v) keys[this.provider] = v
      else delete keys[this.provider]
      this.apiKeys = keys
      saveBp({ apiKeys: keys })
    },
    setModel(m) {
      const models = { ...this.providerModel, [this.provider]: (m || '').trim() }
      if (!models[this.provider]) delete models[this.provider]
      this.providerModel = models
      saveBp({ providerModel: models })
    },
    setBaseUrl(u) {
      this.providerBaseUrl = (u || '').trim()
      saveBp({ providerBaseUrl: this.providerBaseUrl })
    },

    // POST /api/blueprints: a drawing, not a mesh. Same reset as upload()
    // plus the Blueprint fields; the mesh state is cleared so the rail has
    // nothing stale.
    async uploadImage(file) {
      this.stopPolling()
      this.$patch({
        status: 'uploading', jobId: null, inputStats: null, log: '', result: null, error: null,
        cancelled: false, filename: file.name, imageUrl: null, imageSize: null,
        recipe: null, recipeRead: null, warnings: [], bbox: null, volume: null, overallCheck: null,
        featureList: [], featureMap: null, liveStl: null, liveBusy: false, liveError: null, liveInfo: null,
        bodies: [], triangleBody: null, selected: [], hovered: -1, bodiesError: null,
        section: null, xrayTraces: [], xrayTotal: 0, loftAxisAuto: null, loftAxisScores: null,
      })
      liveSeq++
      try {
        const body = new FormData()
        body.append('file', file)
        const res = await fetch('/api/blueprints', { method: 'POST', body })
        if (!res.ok) throw new Error(await errText(res))
        const d = await res.json()
        if (!d?.id) throw new Error('the server sent no job id back')
        this.$patch({ status: 'ready', jobId: d.id, imageUrl: `/api/blueprints/${d.id}/drawing`,
                      imageSize: { width: d.width, height: d.height } })
        if (!this.blueprintConfig) this.loadBlueprintConfig()
      } catch (e) {
        this.$patch({ status: 'error', error: `Upload failed: ${friendlyError(e)}` })
      }
    },

    // POST /read with the browser's key as a header (never in the URL or body)
    async readDrawing() {
      if (!this.canRead) return
      this.$patch({ status: 'running', serverStatus: 'reading', log: '', result: null, error: null,
                    cancelled: false, warnings: [], bbox: null, volume: null, overallCheck: null })
      try {
        const headers = { 'Content-Type': 'application/json' }
        if (this.apiKey) headers['X-Api-Key'] = this.apiKey
        const res = await fetch(`/api/blueprints/${this.jobId}/read`, {
          method: 'POST', headers,
          body: JSON.stringify({
            provider: this.provider, model: this.modelName || null,
            base_url: this.provider === 'custom' ? this.providerBaseUrl : null,
            hints: this.hints || '',
          }),
        })
        if (!res.ok) throw new Error(await errText(res))
        this.startPolling()
      } catch (e) {
        this.$patch({ status: 'ready', error: `Could not start reading: ${friendlyError(e)}` })
      }
    },

    // POST /build with an edited recipe: validated, then built; no model call
    async rebuild(recipe) {
      if (!this.jobId || this.busy) return
      const before = this.status
      this.$patch({ status: 'running', serverStatus: 'queued', log: '', result: null, error: null,
                    cancelled: false })
      try {
        const res = await fetch(`/api/blueprints/${this.jobId}/build`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ recipe }),
        })
        if (!res.ok) throw new Error(await errText(res))
        this.startPolling()
      } catch (e) {
        // a refused recipe (the validator's list): keep the draft, say why
        this.$patch({ status: before === 'running' ? 'ready' : before,
                      error: `Rebuild refused: ${friendlyError(e)}` })
      }
    },

    async fetchRecipe() {
      if (!this.jobId) return
      const job = this.jobId
      try {
        const res = await fetch(`/api/blueprints/${job}/recipe`)
        if (!res.ok) return
        const d = await res.json()
        if (job !== this.jobId) return
        this.recipe = d.recipe
        this.recipeRead = d.read || null
        this.warnings = [...(d.read?.validation?.errors || []), ...(d.read?.validation?.warnings || [])]
      } catch { /* the table stays as it was */ }
    },

    takeBlueprintResult(r) {
      const rd = r.params?.read || null
      this.$patch({
        recipe: r.recipe, recipeRead: rd,
        warnings: [...(rd?.validation?.warnings || []), ...(r.warnings || [])],
        bbox: r.bbox, volume: r.volume_mm3, overallCheck: r.overall_check || null,
        liveStl: null, liveInfo: null, liveError: null,
      })
      liveSeq++
      this.fetchFeatureMap()
    },

    // the full build's colouring (which feature made each triangle of preview.stl)
    async fetchFeatureMap() {
      if (!this.jobId) return
      const job = this.jobId
      try {
        const res = await fetch(`/api/blueprints/${job}/preview-map?t=${Date.now()}`)
        if (!res.ok) return
        const d = await res.json()
        if (job !== this.jobId) return
        this.$patch({ featureList: d.features || [], featureMap: d.triangle_feature || null,
                      selected: (d.features || []).map((_, i) => i) })
      } catch { /* the view stays grey */ }
    },

    // A live look at the draft: the warm helper builds the shape (well
    // under a second) and the view shows it coloured by feature. Debounced
    // by the panel; a stale answer is dropped.
    async previewLive(recipe) {
      if (!this.jobId || !this.isImageJob) return
      const mine = ++liveSeq
      const job = this.jobId
      this.$patch({ liveBusy: true, liveError: null })
      try {
        const res = await fetch(`/api/blueprints/${job}/preview`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ recipe }),
        })
        if (!res.ok) throw new Error(await errText(res))
        const d = await res.json()
        if (mine !== liveSeq || job !== this.jobId) return
        const stl = await fetch(`${d.stl_url}?t=${Date.now()}`)
        if (!stl.ok) throw new Error(await errText(stl))
        const data = await stl.arrayBuffer()
        if (mine !== liveSeq || job !== this.jobId) return
        this.$patch({
          liveStl: markRaw({ data, kind: 'stl', seq: mine }),
          featureList: d.features || [], featureMap: d.triangle_feature || null,
          selected: (d.features || []).map((_, i) => i),
          liveInfo: { bbox: d.bbox, volume_mm3: d.volume_mm3, solids: d.solids,
                      warnings: d.warnings || [], overall_check: d.overall_check || null },
        })
      } catch (e) {
        if (mine === liveSeq) this.liveError = friendlyError(e)
      } finally {
        if (mine === liveSeq) this.liveBusy = false
      }
    },

    // stop the slice-by-slice trace where it is (the traces so far stay)
    stopXray() {
      xraySeq++
      this.xrayBusy = false
    },

    // the whole part, end to end (the clamp keeps the planes just inside)
    xrayWholePart() {
      const h = this.sliceHalfExtent
      if (!(h > 0)) return
      const lim = Math.max(0, h - XRAY_EDGE)
      this.xrayFrom = -lim
      this.xrayTo = lim
    },

    // Trace the slice at the current offset. Debounced by the caller (the
    // slider fires continuously); a stale answer is dropped.
    async traceSection() {
      if (!this.jobId || !this.resolvedSliceAxis) return
      const mine = ++traceSeq
      this.sectionBusy = true
      try {
        const sec = await this.fetchSection(this.sliceOffset)
        if (mine !== traceSeq) return
        if (sec?.axis && sec.extent) this.noteExtent(sec.axis, sec.extent)
        this.section = sec
        this.sectionError = null
      } catch (e) {
        if (mine === traceSeq) { this.section = null; this.sectionError = friendlyError(e) }
      } finally {
        if (mine === traceSeq) this.sectionBusy = false
      }
    },
  },
})

// The query every section trace and sketch download shares: the plane
// (axis + offset from centre) and the fit settings.
function sectionQuery(s, offset) {
  return new URLSearchParams({
    axis: s.resolvedSliceAxis, offset: String(offset), tol: String(s.params.tol),
    units: s.params.units, scale: String(s.params.scale),
    join: String(s.sliceJoin), outline: s.sliceOutline ? 'true' : 'false',
    trim: String(s.sliceTrim || 0),
  })
}

// Before a plain <a download> link is followed: ask the server whether the
// file is still there. A link to a missing file would otherwise save a
// small JSON error as the "download". Returns null when fine, else the
// message to show.
export async function checkDownload(url) {
  try {
    const res = await fetch(url, { method: 'HEAD' })
    // a route without HEAD must not block the download: let the link go
    if (res.ok || res.status === 405) return null
    // HEAD carries no body; a second request reads the sentence (the
    // answer is a small JSON error, never the file, since HEAD refused)
    const full = await fetch(url)
    return full.ok ? null : await errText(full)
  } catch (e) {
    return friendlyError(e)
  }
}
