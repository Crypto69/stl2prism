import { defineStore } from 'pinia'

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
}

let pollTimer = null
let traceSeq = 0

export const useConvertStore = defineStore('convert', {
  state: () => ({
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
    // body indices ticked for conversion; empty means "all of them"
    selected: [],
    hovered: -1,
    // single slice (sliced-loft method): where the plane sits along the
    // chosen axis, in mm from the part's centre (Fusion's slider), how far
    // apart two loose ends of a leaky mesh may be and still be joined (mm;
    // 0 draws exactly what Fusion's mesh section draws), and the last
    // traced section from /section
    sliceOffset: 0,
    sliceJoin: 2.5,
    // closed loops only: leave out the loose open pieces of a leaky mesh
    sliceOutline: false,
    section: null,
    sectionBusy: false,
    sectionError: null,
  }),

  getters: {
    busy: (s) => s.status === 'uploading' || s.status === 'running',
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
      return (bb['xyz'.indexOf(a)] * s.unitScale) / 2
    },
    sectionScriptUrl: (s) => {
      const a = s.resolvedSliceAxis
      if (!s.jobId || !a) return null
      const q = new URLSearchParams({ axis: a, offset: String(s.sliceOffset), tol: String(s.params.tol),
                                      units: s.params.units, scale: String(s.params.scale),
                                      join: String(s.sliceJoin), outline: s.sliceOutline ? 'true' : 'false' })
      return `/api/jobs/${s.jobId}/section-script?${q}`
    },
    // The axis the sliced loft will cut along, with 'auto' resolved to the
    // file's longest side the way sliced_loft.axis_index does; null when
    // the loft is not the chosen method. Drives the plane in the 3D view.
    resolvedSliceAxis: (s) => {
      if (s.params.method !== 'loft') return null
      const a = s.params.slice_axis
      if (a === 'x' || a === 'y' || a === 'z') return a
      const bb = s.inputStats?.bbox_mm
      if (!bb) return null
      return 'xyz'[bb.indexOf(Math.max(...bb))]
    },
    fusionBfillScriptUrl: (s) =>
      s.status === 'done' && s.result?.ok && s.result?.has_bfill_script
        ? `/api/jobs/${s.jobId}/fusion-bfill-script`
        : null,
  },

  actions: {
    async upload(file) {
      this.stopPolling()
      this.$patch({
        status: 'uploading', jobId: null, inputStats: null,
        log: '', result: null, error: null, filename: file.name,
      })
      try {
        const body = new FormData()
        body.append('file', file)
        const res = await fetch('/api/jobs', { method: 'POST', body })
        if (!res.ok) throw new Error(await errText(res))
        const data = await res.json()
        this.$patch({
          status: 'ready', jobId: data.id, inputStats: data.input_stats,
          bodies: [], triangleBody: null, selected: [], hovered: -1,
          sliceOffset: 0, sliceJoin: 2.5, sliceOutline: false, section: null, sectionError: null,
        })
        this.loadBodies()
      } catch (e) {
        this.$patch({ status: 'error', error: `Upload failed: ${e.message}` })
      }
    },

    // The shell list is what makes a 72-body file usable: without it the
    // only choice is "convert everything", which on such a file is hours.
    async loadBodies() {
      if (!this.jobId) return
      try {
        const res = await fetch(`/api/jobs/${this.jobId}/bodies`)
        if (!res.ok) return
        const d = await res.json()
        this.$patch({
          bodies: d.bodies || [],
          triangleBody: d.triangle_body || null,
          bodiesTruncated: !!d.truncated,
          // Preselect the bodies worth converting (see DEFAULT_PICK_FRAC):
          // on a controller that is the housing halves, not the 70 screws.
          selected: (d.bodies || []).filter((b) => b.suggested).map((b) => b.index),
        })
      } catch (e) { /* the picker is optional; conversion still works */ }
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
            bodies: this.selected.length && this.selected.length < this.bodies.length
              ? [...this.selected].sort((a, b) => a - b)
              : null,
          }),
        })
        if (!res.ok) throw new Error(await errText(res))
        this.startPolling()
      } catch (e) {
        this.$patch({ status: 'error', error: `Convert failed: ${e.message}` })
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
        this.$patch({ error: `Could not cancel: ${e.message}` })
      } finally {
        this.cancelling = false
      }
    },

    startPolling() {
      this.stopPolling()
      pollTimer = setInterval(async () => {
        try {
          const res = await fetch(`/api/jobs/${this.jobId}`)
          if (!res.ok) throw new Error(await errText(res))
          const s = await res.json()
          this.log = s.log || ''
          this.serverStatus = s.status
          if (s.status === 'done' || s.status === 'error' || s.status === 'cancelled') {
            this.stopPolling()
            this.result = s.result
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
              this.error = s.result?.error || 'Conversion failed — see log.'
            }
          }
        } catch (e) {
          this.stopPolling()
          this.$patch({ status: 'error', error: e.message })
        }
      }, 1000)
    },

    stopPolling() {
      if (pollTimer) clearInterval(pollTimer)
      pollTimer = null
    },

    resetParams() {
      this.params = { ...DEFAULT_PARAMS }
    },

    // Trace the slice at the current offset. Debounced by the caller (the
    // slider fires continuously); a stale answer is dropped.
    async traceSection() {
      const a = this.resolvedSliceAxis
      if (!this.jobId || !a) return
      const q = new URLSearchParams({ axis: a, offset: String(this.sliceOffset), tol: String(this.params.tol),
                                      units: this.params.units, scale: String(this.params.scale),
                                      join: String(this.sliceJoin), outline: this.sliceOutline ? 'true' : 'false' })
      const mine = ++traceSeq
      this.sectionBusy = true
      try {
        const res = await fetch(`/api/jobs/${this.jobId}/section?${q}`)
        if (!res.ok) throw new Error(await errText(res))
        const sec = await res.json()
        if (mine !== traceSeq) return
        this.section = sec
        this.sectionError = null
      } catch (e) {
        if (mine === traceSeq) { this.section = null; this.sectionError = e.message }
      } finally {
        if (mine === traceSeq) this.sectionBusy = false
      }
    },
  },
})

async function errText(res) {
  try {
    const j = await res.json()
    return j.detail || res.statusText
  } catch {
    return res.statusText
  }
}
