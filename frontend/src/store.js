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
  }),

  getters: {
    busy: (s) => s.status === 'uploading' || s.status === 'running',
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
        })
      } catch (e) {
        this.$patch({ status: 'error', error: `Upload failed: ${e.message}` })
      }
    },

    async convert() {
      if (!this.jobId) return
      this.$patch({ status: 'running', log: '', result: null, error: null })
      try {
        const res = await fetch(`/api/jobs/${this.jobId}/convert`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.params),
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
