<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { useConvertStore } from '../store'

const store = useConvertStore()

// The verdict is what the user waited for; bring it into view when it lands.
const verdictEl = ref(null)
watch(() => store.status, async (s) => {
  if (s === 'done' || s === 'error') {
    await nextTick()
    verdictEl.value?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
})

const fmtBytes = (n) => {
  if (n == null) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(2)} MB`
}
const fmt = (v, d = 3) => (v == null ? '—' : Number(v).toFixed(d))

// Input stats come back in file units; show them in mm using the units the
// user picked, so the numbers match what the pipeline will actually see.
const inputRows = computed(() => {
  const s = store.inputStats
  if (!s) return []
  const k = store.unitScale
  const L = (v) => (v * k).toFixed(2).replace(/\.?0+$/, '')
  const A = (v) => Math.round(v * k * k).toLocaleString()
  const V = (v) => Math.round(v * k * k * k).toLocaleString()
  return [
    ['Triangles', s.triangles.toLocaleString()],
    ['Vertices', s.vertices.toLocaleString()],
    ['Watertight', s.watertight ? 'yes' : 'no'],
    ['Bodies', s.bodies],
    ['Bounding box', `${s.bbox_mm.map(L).join(' × ')} mm`],
    ...(s.edge_mm
      ? [['Edge length min / mean / max',
          `${(s.edge_mm.min * k).toFixed(3)} / ${(s.edge_mm.mean * k).toFixed(3)} / ${(s.edge_mm.max * k).toFixed(3)} mm`]]
      : []),
    ['Surface area', `${A(s.surface_area_mm2)} mm²`],
    ['Volume', s.volume_mm3 == null ? 'n/a (not watertight)'
                                    : `${V(s.volume_mm3)} mm³`],
    ['File size', fmtBytes(s.file_size)],
  ]
})

// Multi-body runs carry a per-body list; single-body runs do not.
const bodies = computed(() => store.result?.bodies || null)

const verdict = computed(() => {
  const r = store.result
  if (!r?.ok) return null
  if (!bodies.value) {
    if (r.mode === 'prismatic') {
      return { title: 'Prismatic solid', cls: 'prismatic',
               note: 'Clean BREP with true planes and cylinders — every gate below passed.' }
    }
    if (r.mode === 'facegroup') {
      const fg = r.metrics?.fgroup || {}
      const bt = fg.by_type || {}
      const kinds = [['plane', bt.plane], ['cylinder', bt.cylinder], ['cone', bt.cone], ['sphere', bt.sphere]]
        .filter(([, n]) => n > 0).map(([k, n]) => `${n} ${k}${n === 1 ? '' : 's'}`).join(', ')
      return { title: 'Face-group solid', cls: 'facegroup',
               note: `The extrusion fit was rejected, so the mesh was grouped into surface regions and each got a real analytic face (${kinds || 'planes, cylinders, cones, spheres'}) — every gate below passed. No sketch+extrude script for this one.` }
    }
    return { title: 'Faceted solid', cls: 'faceted',
          note: r.is_scan && !r.params?.force_prismatic
            ? 'This looked like a 3D scan (millions of tiny triangles, no flat facets), so the prismatic fit was skipped: the scan was repaired, closed and reduced, and this is an exact faceted copy of it. Tick “Force prismatic” to make the tool try anyway.'
            : 'A prismatic fit wasn’t possible within your limits, so this is an exact faceted copy of the mesh: valid and manifold, but not clean geometry to sketch on.' }
  }
  const m = r.metrics
  const parts = []
  if (m.n_prismatic) parts.push(`${m.n_prismatic} prismatic`)
  if (m.n_facegroup) parts.push(`${m.n_facegroup} face-group`)
  if (m.n_faceted) parts.push(`${m.n_faceted} faceted`)
  if (m.n_failed) parts.push(`${m.n_failed} failed`)
  const dropped = r.n_dropped
    ? ` ${r.n_dropped} sliver${r.n_dropped === 1 ? '' : 's'} (a few stray triangles) dropped.` : ''
  return {
    title: `${r.n_written} bodies · ${parts.join(' · ')}`,
    cls: r.mode,
    note: `The mesh held ${r.n_bodies} separate bodies. Each was converted on its own and all are in the one STEP as separate solids — prismatic where it passed your gates, face-group (analytic faces per region) where the extrusion fit failed, an exact faceted copy where neither worked.${dropped}`,
  }
})

// Per-body rows for multi-body results.
const bodyRows = computed(() =>
  (bodies.value || []).map((b) => ({
    n: b.index + 1,
    faces: b.faces.toLocaleString(),
    mode: b.error ? 'failed' : b.mode,
    detail: b.error ? b.error
      : (b.mode === 'prismatic' || b.mode === 'facegroup')
        ? `p95 ${fmt(b.metrics.dev_p95)} · max ${fmt(b.metrics.dev_max)} mm`
        : `${b.metrics.faces_out.toLocaleString()} faces`
          + (b.metrics.is_solid === false ? ' · open shell' : ''),
  })))

// The inspection card: measured value vs the limit the run was gated on.
const gates = computed(() => {
  const r = store.result
  if (!r?.ok || (r.mode !== 'prismatic' && r.mode !== 'facegroup') || bodies.value) return []
  const m = r.metrics
  const p = r.params
  const rows = [
    { name: 'Surface deviation p95 (mesh → solid)', value: m.dev_p95, limit: p.accept_p95, unit: 'mm' },
    { name: 'Surface deviation max (mesh → solid)', value: m.dev_max, limit: p.accept_max, unit: 'mm' },
  ]
  if (m.symmetric && m.rev_dev_p95 != null) {
    rows.push({ name: 'Surface deviation p95 (solid → mesh)', value: m.rev_dev_p95, limit: p.accept_p95, unit: 'mm' })
    rows.push({ name: 'Surface deviation max (solid → mesh)', value: m.rev_dev_max, limit: p.accept_max, unit: 'mm' })
  }
  if (m.hole_dev_p95 != null) {
    rows.push({ name: `Bore deviation p95 (${m.holes_checked} bore${m.holes_checked === 1 ? '' : 's'})`,
                value: m.hole_dev_p95, limit: p.accept_hole_max, unit: 'mm' })
  }
  if (m.vol_err_pct != null) {
    rows.push({ name: 'Volume error', value: m.vol_err_pct, limit: p.accept_vol_pct, unit: '%' })
  }
  return rows.map((g) => ({ ...g, pass: g.value <= g.limit }))
})

// Extra facts about how the solid was made (patched regions, unverified volume).
const notes = computed(() => {
  const r = store.result
  if (!r?.ok || bodies.value) return []
  const m = r.metrics || {}
  const out = []
  if (r.mode === 'prismatic' && m.patched) {
    out.push(`${m.patches} small region${m.patches === 1 ? '' : 's'} could not be expressed with planes, cylinders or cones and ${m.patches === 1 ? 'was' : 'were'} replaced by exact faceted geometry (about ${(100 * (m.bad_frac || 0)).toFixed(1)}% of the surface). The rest is clean.`)
  }
  if (m.vol_verified === false) {
    out.push('The input mesh was not closed, so the volume check could not run; only the surface deviation was verified.')
  }
  if (r.mode === 'facegroup' && m.fgroup?.unfitted_regions) {
    const fg = m.fgroup
    out.push(`${fg.unfitted_regions} region${fg.unfitted_regions === 1 ? '' : 's'} (${fg.unfitted_faces.toLocaleString()} triangles) matched no plane, cylinder, cone or sphere and ${fg.unfitted_regions === 1 ? 'was' : 'were'} kept as exact facets. The rest is clean.`)
  }
  if (r.mode === 'faceted' && m.reduce?.reduced) {
    out.push(`Curved regions were simplified from ${m.reduce.faces_before.toLocaleString()} to ${m.reduce.faces_after.toLocaleString()} triangles within ${r.params?.reduce_tol} mm.`)
  }
  if (m.voids) {
    out.push(`${m.voids} internal cavit${m.voids === 1 ? 'y was' : 'ies were'} recognised and subtracted, so the solid is hollow like the mesh.`)
  }
  return out
})

const outputRows = computed(() => {
  const r = store.result
  if (!r?.ok) return []
  const o = r.output_stats
  const rows = [['BREP faces', o.faces.toLocaleString()]]
  if (o.solids > 1) rows.push(['Solids', o.solids.toLocaleString()])
  const t = o.surface_types
  const kinds = [
    ['planes', t.planes], ['cylinders', t.cylinders], ['cones', t.cones],
    ['spheres', t.spheres], ['tori', t.tori], ['freeform', t.freeform],
  ].filter(([, n]) => n > 0)
  if (kinds.length) {
    rows.push(['Surface types', kinds.map(([k, n]) => `${n} ${k}`).join(', ')])
  }
  if (bodies.value) {
    // per-body detail lives in the bodies table; only file-level rows here
  } else if (r.mode === 'prismatic' || r.mode === 'facegroup') {
    rows.push(['Mean deviation', `${fmt(r.metrics.dev_mean)} mm`])
    rows.push(['Worst point at', `(${(r.metrics.dev_max_xyz || []).join(', ')}) mm`])
    if (r.metrics.vol_solid != null) {
      rows.push(['Solid volume', `${Math.round(r.metrics.vol_solid).toLocaleString()} mm³`])
    }
  } else {
    rows.push(['Solid volume', r.metrics.volume == null ? '—'
      : `${Math.round(r.metrics.volume).toLocaleString()} mm³`])
  }
  rows.push(['File size', fmtBytes(o.file_size)])
  return rows
})

const reduction = computed(() => {
  const tris = store.inputStats?.triangles
  const faces = store.result?.output_stats?.faces
  if (!tris || !faces || store.result.mode === 'faceted') return null
  return `${tris.toLocaleString()} triangles → ${faces.toLocaleString()} faces`
})
</script>

<template>
  <div class="report">
    <section v-if="inputRows.length">
      <h2 class="micro">Input mesh · {{ store.filename }}</h2>
      <table>
        <tbody>
          <tr v-for="[k, v] in inputRows" :key="k">
            <td>{{ k }}</td><td class="num">{{ v }}</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section v-if="store.status === 'done' && store.result?.ok"
             ref="verdictEl" class="verdict">
      <div class="mode" :class="verdict.cls">{{ verdict.title }}</div>
      <p class="modenote">{{ verdict.note }}</p>

      <table v-if="bodyRows.length" class="gatecard bodies">
        <thead>
          <tr class="micro">
            <th>#</th><th>Triangles</th><th>Result</th><th>Detail</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="b in bodyRows" :key="b.n">
            <td class="num">{{ b.n }}</td>
            <td class="num">{{ b.faces }}</td>
            <td :class="b.mode">{{ b.mode }}</td>
            <td class="num detail">{{ b.detail }}</td>
          </tr>
        </tbody>
      </table>

      <table v-if="gates.length" class="gatecard">
        <thead>
          <tr class="micro">
            <th>Check</th><th>Measured</th><th>Limit</th><th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="g in gates" :key="g.name">
            <td>{{ g.name }}</td>
            <td class="num">{{ fmt(g.value, g.unit === '%' ? 2 : 3) }} {{ g.unit }}</td>
            <td class="num limit">≤ {{ g.limit }} {{ g.unit }}</td>
            <td class="num" :class="g.pass ? 'pass' : 'fail'">
              {{ g.pass ? 'PASS' : 'FAIL' }}
            </td>
          </tr>
        </tbody>
      </table>

      <h2 class="micro">Output STEP</h2>
      <p v-if="reduction" class="reduction num">{{ reduction }}</p>
      <table>
        <tbody>
          <tr v-for="[k, v] in outputRows" :key="k">
            <td>{{ k }}</td><td class="num">{{ v }}</td>
          </tr>
        </tbody>
      </table>

      <ul v-if="notes.length" class="notes">
        <li v-for="(n, i) in notes" :key="i">{{ n }}</li>
      </ul>
      <a class="download" :href="store.downloadUrl" download>Download STEP</a>
      <a v-if="store.scriptUrl" class="download secondary" :href="store.scriptUrl" download>
        Download CadQuery script (.py)
      </a>
      <a v-if="store.fusionScriptUrl" class="download secondary" :href="store.fusionScriptUrl" download>
        Download Fusion 360 script (.py) — experimental
      </a>
      <p v-if="store.scriptUrl" class="hint">
        The script rebuilds the recognised sketches and extrudes as an editable program — change a radius or height and re-run it to get a new STEP.
      </p>
    </section>

    <section v-if="store.error" ref="verdictEl" class="errorbox">
      <h2 class="micro">Failed</h2>
      <p>{{ store.error }}</p>
    </section>

    <details v-if="store.log" class="logbox" :open="store.status === 'running'">
      <summary class="micro">Pipeline log</summary>
      <pre class="num">{{ store.log }}</pre>
    </details>
  </div>
</template>

<style scoped>
.notes { margin: 8px 0; padding-left: 18px; font-size: 13px; opacity: 0.9; }
.notes li { margin: 4px 0; }
.download.secondary { opacity: 0.85; margin-top: 6px; }
.hint { font-size: 12px; opacity: 0.8; }
.report { display: flex; flex-direction: column; gap: 16px; }
h2 { margin-bottom: 6px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
td, th { padding: 4px 0; border-top: 1px solid var(--line); }
td:last-child, th:last-child { text-align: right; }
td:first-child { color: var(--muted); }
th { text-align: left; font-weight: 600; }

.mode {
  display: inline-block;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-size: 13px;
  padding: 4px 10px;
  border-radius: 4px;
  border: 1px solid var(--edge);
  color: var(--edge);
}
.mode.faceted { border-color: var(--muted); color: var(--text); }
.mode.facegroup { border-color: var(--edge); color: var(--text); }
.mode.mixed { border-color: var(--edge); color: var(--text); }
.bodies td:first-child { color: var(--muted); }
.bodies td.prismatic { color: var(--edge); font-weight: 600; }
.bodies td.facegroup { color: var(--edge); }
.bodies td.faceted { color: var(--text); }
.bodies td.failed { color: var(--fail); font-weight: 600; }
.bodies .detail { color: var(--muted); font-size: 12px; word-break: break-word; }
.modenote { color: var(--muted); font-size: 12px; margin: 6px 0 10px; }

.gatecard {
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 6px;
  margin-bottom: 14px;
}
.gatecard td, .gatecard th { padding: 6px 10px; }
.gatecard td:first-child { color: var(--text); }
.limit { color: var(--muted); }
.pass { color: var(--pass); font-weight: 700; }
.fail { color: var(--fail); font-weight: 700; }

.reduction { color: var(--edge); font-size: 13px; margin-bottom: 6px; }

.download {
  display: block;
  text-align: center;
  margin-top: 12px;
  padding: 10px;
  border-radius: 6px;
  background: var(--edge);
  color: var(--ink);
  font-weight: 700;
  text-decoration: none;
}
.download:hover { filter: brightness(1.1); }

.errorbox {
  border: 1px solid var(--fail);
  border-radius: 6px;
  padding: 10px 12px;
}
.errorbox p { color: var(--fail); font-size: 13px; word-break: break-word; }

.logbox summary { cursor: pointer; }
.logbox pre {
  margin-top: 8px;
  font-size: 11px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--muted);
  background: var(--ink);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px;
  max-height: 260px;
  overflow: auto;
}
</style>
