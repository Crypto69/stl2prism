<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useConvertStore, XRAY_MAX } from '../store'
import { errText, friendlyError } from '../errors'
import { summarise } from '../sectionSummary'
import AxisSelect from './AxisSelect.vue'
import PlaneSlider from './PlaneSlider.vue'
import SlicePlaneOptions from './SlicePlaneOptions.vue'

const store = useConvertStore()

// the download is one request that fits every slice; past this it gives up
const BUDGET_S = 240

// The planes follow their sliders at once; the slice-by-slice trace (a
// cut plus a curve fit each) starts when the controls rest. The panel
// mounts when the tool is picked, so the first trace runs at once
// (immediate).
let traceTimer = null
const scheduleTrace = () => {
  clearTimeout(traceTimer)
  traceTimer = setTimeout(() => store.traceXray(), 350)
}
watch(() => [store.xrayFrom, store.xrayTo, store.xraySpacing, store.resolvedSliceAxis, store.params.tol,
             store.params.units, store.params.scale, store.jobId, store.params.slice_join, store.sliceOutline,
             store.params.slice_trim],
      ([, , , axis, , , , job]) => {
        // the old traces belong to the old planes: stop and drop them at once
        store.stopXray()
        store.xrayTraces = []
        store.xrayTotal = 0
        if (!job || !axis) return
        scheduleTrace()
      }, { immediate: true })
onBeforeUnmount(() => { clearTimeout(traceTimer); store.stopXray() })

const single = computed(() => store.xrayFrom === store.xrayTo)
const axisName = computed(() => (store.resolvedSliceAxis || '').toUpperCase())
const range = computed(() => {
  const o = store.xrayOffsets
  return o.length ? [o[0], o[o.length - 1]] : null
})
const fmtS = (s) => (s < 1 ? 'under a second' : s < 90 ? `about ${Math.ceil(s)} s` : `about ${Math.ceil(s / 60)} min`)
const estimate = computed(() => store.xrayEstimateS)
const overBudget = computed(() => estimate.value != null && estimate.value > BUDGET_S)
const startSummary = computed(() => summarise(store.xrayStartSection?.stats, store.sliceOutline))
const endSummary = computed(() => summarise(store.xrayEndSection?.stats, store.sliceOutline))
const canDownload = computed(() => store.xrayScriptUrl && store.xrayStartSection)

// The script is built on demand (every slice is fitted again on the
// server), which takes from under a second to minutes. A plain download
// link would give no sign of that, so the button fetches the file itself,
// shows "Working…" with a spinner until it arrives, then hands it to the
// browser as a download; a server refusal (over the time limit) is shown
// here instead of a failed download.
const downloading = ref(false)
const downloadError = ref(null)
async function download() {
  if (!canDownload.value || downloading.value) return
  downloading.value = true
  downloadError.value = null
  try {
    const res = await fetch(store.xrayScriptUrl)
    if (!res.ok) throw new Error(await errText(res))
    const blob = await res.blob()
    const cd = res.headers.get('content-disposition') || ''
    const name = (cd.match(/filename="?([^";]+)"?/) || [])[1] || 'xray.py'
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = name
    document.body.appendChild(a)
    a.click()
    a.remove()
    setTimeout(() => URL.revokeObjectURL(url), 30000)
  } catch (e) {
    downloadError.value = friendlyError(e)
  } finally {
    downloading.value = false
  }
}
</script>

<template>
  <section class="params">
    <p class="intro">
      A stack of section sketches between two planes, as one Fusion script:
      Create Mesh Section Sketch + Fit Curves to Mesh Section for every
      slice at once. The slices are traced one by one and stay in the 3D
      view as they appear. Start = end gives a single sketch.
    </p>
    <AxisSelect />
    <div v-if="store.sliceHalfExtent > 0" class="slice">
      <PlaneSlider v-model="store.xrayFrom" id="xray_from" label="Start plane" />
      <PlaneSlider v-model="store.xrayTo" id="xray_to" label="End plane" />
      <div class="row">
        <label for="xray_step">
          Spacing
          <span class="unit num">mm</span>
        </label>
        <span class="pair">
          <button class="link" type="button" @click="store.xrayWholePart()">Whole part</button>
          <input
            id="xray_step" type="number" step="0.1" min="0.05" max="50"
            v-model.number="store.xraySpacing"
            :class="{ touched: store.xraySpacing !== 0.2 }"
          />
        </span>
      </div>
      <p v-if="range" class="hint num">
        <template v-if="single">1 slice at {{ range[0].toFixed(2) }} mm along {{ axisName }}</template>
        <template v-else>
          {{ store.xrayCount }} slices from {{ range[0].toFixed(2) }} to {{ range[1].toFixed(2) }} mm
          along {{ axisName }}<template v-if="estimate != null && !store.xrayOverCap"> · {{ fmtS(estimate) }} to build</template>
        </template>
      </p>
      <p v-if="store.xrayOverCap" class="hint warn">
        {{ store.xrayCount }} slices is over the limit of {{ XRAY_MAX }} — raise the spacing or
        narrow the range.
      </p>
      <p v-else-if="overBudget" class="hint warn">
        That is over the {{ BUDGET_S / 60 }}-minute limit for one download: raise the spacing,
        narrow the range, or tick Outline only (about 5× faster on organic sections).
      </p>
      <SlicePlaneOptions :tol="true" />
      <label class="check">
        <input type="checkbox" v-model="store.xrayExtrude" :disabled="store.xrayCount < 2" />
        <span>
          Extrude each slice to the next (solid slabs)
          <span class="help">Every sketch is extruded up to the next plane and joined to the slab before it, so the stack comes out as a stepped solid Fusion builds without fail, where a Loft between two complex profiles folds over. Holes are filled unless Outline only is off and the hole loops are drawn; Outline only gives the cleanest slabs on a leaky mesh. Needs at least two slices.</span>
        </span>
      </label>
      <p v-if="store.xrayBusy" class="hint">
        <template v-if="single">tracing the plane…</template>
        <template v-else>
          tracing slice {{ store.xrayTraces.length + 1 }} of {{ store.xrayTotal }}…
          <button class="link" type="button" @click="store.stopXray()">stop</button>
        </template>
      </p>
      <p v-else-if="store.xrayError" class="hint warn">Could not trace: {{ store.xrayError }}</p>
      <p v-else-if="store.xrayTotal && store.xrayTraces.length < store.xrayTotal" class="hint">
        stopped after {{ store.xrayTraces.length }} of {{ store.xrayTotal }} slices
        <button class="link" type="button" @click="store.traceXray()">trace the rest</button>
      </p>
      <template v-if="startSummary">
        <p class="hint num">{{ single ? '' : 'start: ' }}{{ startSummary }}</p>
        <p v-if="!single && endSummary" class="hint num">end: {{ endSummary }}</p>
      </template>
      <button
        v-if="canDownload" type="button" class="dl press"
        :class="{ busy: downloading }" :disabled="downloading"
        @click="download"
      >
        <template v-if="downloading">
          <span class="spin edge" aria-hidden="true"></span> Working… fitting {{ store.xrayCount }} slice{{ store.xrayCount === 1 ? '' : 's' }}
        </template>
        <template v-else>
          Download Fusion sketches (.py) — {{ store.xrayCount }} sketch{{ store.xrayCount === 1 ? '' : 'es' }}{{ store.xrayExtrude && store.xrayCount > 1 ? ', extruded' : '' }}
        </template>
      </button>
      <span v-else class="dl off">
        {{ store.xrayOverCap ? 'Too many slices to download' : 'Download Fusion sketches (.py)' }}
      </span>
      <p v-if="downloadError" class="hint warn">Could not build the script: {{ downloadError }}</p>
      <p class="hint">
        One fully enclosed sketch per slice on its own construction plane,
        named <span class="num">xray {{ axisName }}=… mm (k/N)</span>. Put the .py in an
        empty folder, then in Fusion Utilities → Add-Ins → Scripts and
        Add-Ins → + → choose that folder → Run. The script shows a progress
        dialog and can be cancelled part-way.
      </p>
    </div>
  </section>
</template>

<style scoped>
.pair { display: inline-flex; align-items: center; gap: 10px; }
/* a button that visibly presses: it sinks and fills on the click, then
   turns into the working state */
.press {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  transition: transform 0.08s ease, background 0.12s ease, box-shadow 0.12s ease;
}
.press:not(:disabled):hover { box-shadow: 0 0 0 1px var(--edge) inset; }
.press:not(:disabled):active {
  transform: scale(0.96);
  background: rgba(90, 210, 234, 0.22);
}
.press.busy {
  cursor: progress;
  background: rgba(90, 210, 234, 0.12);
  animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(90, 210, 234, 0); }
  50% { box-shadow: 0 0 0 3px rgba(90, 210, 234, 0.25); }
}
</style>
