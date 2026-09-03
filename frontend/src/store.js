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
  reduce_tol: 0.05,
  tol: 0.08,
  accept_p95: 0.25,
  accept_max: 0.26,
  accept_hole_max: 0.1,
  accept_vol_pct: 2.0,
  force_prismatic: false,
  face_groups: true,
}

let pollTimer = null

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
    // every connected shell of the upload, largest first (/bodies)
    bodies: [],
    // triangle -> body index, for colouring and picking in the viewer
    triangleBody: null,
    bodiesTruncated: false,
    // body indices ticked for conversion; empty means "all of them"
    selected: [],
    hovered: -1,
  }),

  getters: {
    busy: (s) => s.status === 'uploading' || s.status === 'running',
    // A file with one body needs no picker at all.
    hasBodyPicker: (s) => s.bodies.length > 1,
    selectedCount: (s) => (s.selected.length || s.bodies.length),
    // file units -> mm, for showing input numbers the way the pipeline sees them
    unitScale: (s) => UNITS.find((u) => u.key === s.params.units)?.scale ?? 1,
    downloadUrl: (s) =>
      s.status === 'done' && s.result?.ok
        ? `/api/jobs/${s.jobId}/download`
        : null,
    scriptUrl: (s) =>
      s.status === 'done' && s.result?.ok && s.result?.has_script
        ? `/api/jobs/${s.jobId}/script`
        : null,
    fusionScriptUrl: (s) =>
      s.status === 'done' && s.result?.ok && s.result?.has_script
        ? `/api/jobs/${s.jobId}/fusion-script`
        : null,
    bfillCheck: (s) => (s.status === 'done' && s.result?.ok ? s.result?.bfill_check || null : null),
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
      this.$patch({ status: 'running', log: '', result: null, error: null })
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

    startPolling() {
      this.stopPolling()
      pollTimer = setInterval(async () => {
        try {
          const res = await fetch(`/api/jobs/${this.jobId}`)
          if (!res.ok) throw new Error(await errText(res))
          const s = await res.json()
          this.log = s.log || ''
          this.serverStatus = s.status
          if (s.status === 'done' || s.status === 'error') {
            this.stopPolling()
            this.result = s.result
            if (s.status === 'done' && s.result?.ok) {
              this.status = 'done'
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
