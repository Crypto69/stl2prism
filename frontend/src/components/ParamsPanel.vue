<script setup>
import { computed, ref, watch } from 'vue'
import { useConvertStore, DEFAULT_PARAMS, UNITS } from '../store'

const store = useConvertStore()

const fields = [
  {
    key: 'tol', label: 'Fit tolerance', unit: 'mm', step: 0.01,
    blurb: 'The converter redraws each outline of your part using clean straight lines and arcs. This sets how far that redrawn outline is allowed to stray from the original mesh.',
    lower: 'Follows every tiny bump — keeps more detail, but makes more faces and can mistake 3D-print roughness for real features.',
    higher: 'Smooths over small bumps — a simpler, cleaner part, but very small features may get rounded away.',
  },
  {
    key: 'accept_p95', label: 'Surface deviation (p95)', unit: 'mm', step: 0.01,
    blurb: 'After rebuilding, the tool measures thousands of points and checks how far the new solid is from the original surface. 95% of the surface must be closer than this. It is the main quality bar: fail it, and you get the exact faceted copy instead of a clean solid.',
    lower: 'Demands a near-perfect match — more parts will fall back to the faceted copy.',
    higher: 'Accepts a rougher match — more parts convert to clean solids, but they may visibly differ from the original.',
  },
  {
    key: 'accept_max', label: 'Surface deviation (max)', unit: 'mm', step: 0.01,
    blurb: 'The same check, but for the single worst spot anywhere on the part. A taper, chamfer, or rounded corner is usually what breaks this limit.',
    lower: 'Even one slightly-off spot rejects the clean solid.',
    higher: 'Lets one small area be off (say, a taper the tool can’t model yet) while the rest stays accurate.',
  },
  {
    key: 'accept_hole_max', label: 'Bore deviation (max)', unit: 'mm', step: 0.01,
    blurb: 'Round holes get their own, stricter budget. A hole that is even slightly the wrong size means a screw or pin will not fit, so hole error is judged separately from the rest of the surface.',
    lower: 'Holes must be almost exactly the right size.',
    higher: 'Tolerates hole-size error — only sensible when the holes are cosmetic and nothing has to fit in them.',
  },
  {
    key: 'accept_vol_pct', label: 'Volume error', unit: '%', step: 0.1,
    blurb: 'A final sanity check: the rebuilt part must contain about the same amount of material as the original. It catches big mistakes, like a pocket that got filled in or a chunk that went missing.',
    lower: 'Stricter overall shape check.',
    higher: 'More forgiving — rarely needs changing either way.',
  },
  {
    key: 'reduce_tol', label: 'Faceted reduce tolerance', unit: 'mm', step: 0.01,
    blurb: 'Only used for the faceted copy (when a clean solid is not possible, or for scans). Curved regions are simplified — fewer triangles — as long as the simplified surface stays within this distance of the original. Like Fusion’s “Reduce by tolerance”. Set to 0 to keep every triangle.',
    lower: 'Keeps more triangles; the faceted copy is closer to the original mesh.',
    higher: 'Fewer triangles and a smaller STEP; small details may be smoothed.',
  },
]

const suggestion = computed(() => store.inputStats?.units_suggestion || null)
// Set when the mesh, read as mm, is an implausible size for a part. Shown
// before conversion because the whole run happens at the wrong scale
// otherwise, and that only becomes obvious in CAD afterwards.
const sizeWarning = computed(() => store.inputStats?.unit_warning || null)
// The longest side as the pipeline will read it, so a wrong scale is
// visible here rather than after a twelve-minute conversion.
const scaledLongest = computed(() => {
  const bb = store.inputStats?.bbox_mm
  if (!bb) return null
  return (Math.max(...bb) * store.unitScale).toFixed(1)
})
const longest = computed(() =>
  store.inputStats ? Math.max(...store.inputStats.bbox_mm).toFixed(2) : '')
const openInfo = ref(null)
const toggleInfo = (key) => { openInfo.value = openInfo.value === key ? null : key }
const isDefault = (key) => store.params[key] === DEFAULT_PARAMS[key]
const isLoft = computed(() => store.params.method === 'loft')
// Single slice: the plane follows the slider at once; the trace (a cut
// plus a curve fit, a few seconds on a big mesh) runs when the slider
// rests. Any change of axis, tolerance, scale or gap joining re-traces too.
let traceTimer = null
const scheduleTrace = () => {
  clearTimeout(traceTimer)
  traceTimer = setTimeout(() => store.traceSection(), 350)
}
watch(() => [store.sliceOffset, store.resolvedSliceAxis, store.params.tol, store.params.units,
             store.params.scale, store.jobId, store.sliceJoin],
      ([off, axis, , , , job]) => {
        // the old outline belongs to the old plane: drop it at once, so the
        // view never shows a trace that does not sit on the plane
        store.section = null
        if (!job || !axis) return
        scheduleTrace()
      })
const sliceStep = computed(() => {
  const h = store.sliceHalfExtent
  return h > 50 ? 0.5 : h > 10 ? 0.1 : 0.05
})
const sectionSummary = computed(() => {
  const st = store.section?.stats
  if (!st) return ''
  const parts = []
  if (st.lines) parts.push(`${st.lines} lines`)
  if (st.arcs) parts.push(`${st.arcs} arcs`)
  if (st.circles) parts.push(`${st.circles} circle${st.circles === 1 ? '' : 's'}`)
  if (st.splines) parts.push(`${st.splines} spline${st.splines === 1 ? '' : 's'}`)
  let head = `${st.loops} outline${st.loops === 1 ? '' : 's'}`
  if (st.open) head += `, ${st.open} open curve${st.open === 1 ? '' : 's'}`
  let tail = ''
  if (st.joins) tail = `; ${st.joins} gap${st.joins === 1 ? '' : 's'} joined (largest ${st.max_gap.toFixed(2)} mm)`
  return `${head}: ${parts.join(', ')}${tail}; worst miss ${st.dev_max.toFixed(3)} mm`
})
// Axis choices with the file's side along each, so X / Y / Z mean
// something before the plane appears in the 3D view, and 'auto' says
// which side it will pick.
const axisOptions = computed(() => {
  const bb = store.inputStats?.bbox_mm
  const k = store.unitScale
  const len = (i) => (bb ? `${(bb[i] * k).toFixed(1)} mm` : '')
  const longest = bb ? 'XYZ'[bb.indexOf(Math.max(...bb))] : null
  return [
    { key: 'auto', label: longest ? `auto · longest side (${longest}, ${len('XYZ'.indexOf(longest))})` : 'auto · longest side' },
    { key: 'x', label: bb ? `X · ${len(0)}` : 'X' },
    { key: 'y', label: bb ? `Y · ${len(1)}` : 'Y' },
    { key: 'z', label: bb ? `Z · ${len(2)}` : 'Z' },
  ]
})
</script>

<template>
  <section class="params">
    <header class="head">
      <h2 class="micro">Input units</h2>
    </header>
    <div class="row">
      <label for="units">
        The file's numbers are in
        <button
          class="info num"
          :aria-expanded="openInfo === 'units'"
          aria-label="What does Input units do?"
          @click="toggleInfo('units')"
        >i</button>
      </label>
      <select id="units" v-model="store.params.units"
              :class="{ touched: !isDefault('units') }">
        <option v-for="u in UNITS" :key="u.key" :value="u.key">
          {{ u.key }} · {{ u.label }}
        </option>
      </select>
    </div>
    <div class="row">
      <label for="scale">
        …then multiplied by
        <button
          class="info num"
          :aria-expanded="openInfo === 'scale'"
          aria-label="What does the scale factor do?"
          @click="toggleInfo('scale')"
        >i</button>
      </label>
      <input
        id="scale" type="number" step="0.1" min="0.000001"
        v-model.number="store.params.scale"
        :class="{ touched: !isDefault('scale') }"
      />
    </div>
    <p v-if="scaledLongest" class="hint">
      Longest side as converted: <b class="num">{{ scaledLongest }} mm</b>
    </p>
    <div v-if="openInfo === 'scale'" class="explain">
      <p>The unit above can only make a mesh bigger (cm is ×10, inches ×25.4). Some exports are simply written at the wrong scale — a part designed in centimetres but saved as millimetres reads ten times too big — and the only fix is to multiply by a fraction. Put that here: 0.1 shrinks by ten.</p>
      <p><span class="dir">Leave it at 1</span> unless the size above looks wrong.</p>
    </div>
    <div v-if="openInfo === 'units'" class="explain">
      <p>Mesh files do not say what unit they are in — they are just numbers. Every program guesses. This tool works in millimetres, so tell it what the file meant and it scales the mesh before doing anything else. All the limits below are in mm.</p>
      <p><span class="dir">Tip:</span> Fusion 360 assumes centimetres for OBJ. If a part looks 10× too small here but right in Fusion, pick cm.</p>
    </div>
    <p v-if="sizeWarning" class="warn">
      <b>Check the size.</b>
      Read as millimetres this part is {{ sizeWarning.size_mm }} mm across, which is
      {{ sizeWarning.too === 'big' ? 'very large' : 'very small' }} for a part.
      <template v-if="sizeWarning.unit">
        Its numbers look like <b>{{ sizeWarning.unit }}</b> — that would make it
        {{ sizeWarning.would_be }} mm.
        <button class="link" @click="store.params.units = sizeWarning.unit">Use {{ sizeWarning.unit }}</button>
      </template>
      <template v-else-if="sizeWarning.scale">
        At ×{{ sizeWarning.scale }} it would be {{ sizeWarning.would_be }} mm.
        <button class="link" @click="store.params.scale = sizeWarning.scale">
          Use ×{{ sizeWarning.scale }}
        </button>
      </template>
    </p>
    <p v-if="suggestion && suggestion !== store.params.units" class="hint">
      This file looks like it might be in <b>{{ suggestion }}</b> (its longest side is
      {{ longest }} units). <button class="link" @click="store.params.units = suggestion">Use {{ suggestion }}</button>
    </p>

    <header class="head gate">
      <h2 class="micro">Method</h2>
    </header>
    <div class="row">
      <label for="method">
        Build the solid by
        <button
          class="info num"
          :aria-expanded="openInfo === 'method'"
          aria-label="What does Method do?"
          @click="toggleInfo('method')"
        >i</button>
      </label>
      <select id="method" v-model="store.params.method"
              :class="{ touched: !isDefault('method') }">
        <option value="auto">Prismatic / face-group (auto)</option>
        <option value="loft">Sliced loft</option>
      </select>
    </div>
    <div v-if="openInfo === 'method'" class="explain">
      <p><span class="dir">Auto</span> looks for the design intent: an extrusion direction, flat faces, cylinders, cones, spheres. Best for machined or CAD-designed parts; it gives real planes and cylinders you can measure and sketch on.</p>
      <p><span class="dir">Sliced loft</span> does what you would do by hand in Fusion with Create Mesh Section Sketch, Fit Curves to Mesh Section and Loft: it cuts the body into thin slices along one axis, redraws each outline as a smooth curve and lofts the stack into one smooth surface per run of slices. Best for organic shells — controller housings, handles, caps — where the auto method has nothing flat or round to find. A hole drilled sideways through the part comes out smeared, because no slice sees it as a circle.</p>
    </div>
    <template v-if="isLoft">
      <div class="row">
        <label for="slice_mm">
          Slice spacing
          <span class="unit num">mm</span>
        </label>
        <input
          id="slice_mm" type="number" step="0.1" min="0.05" max="50"
          v-model.number="store.params.slice_mm"
          :class="{ touched: !isDefault('slice_mm') }"
        />
      </div>
      <div class="row">
        <label for="slice_axis">Slice along</label>
        <select id="slice_axis" v-model="store.params.slice_axis"
                :class="{ touched: !isDefault('slice_axis') }">
          <option v-for="o in axisOptions" :key="o.key" :value="o.key">{{ o.label }}</option>
        </select>
      </div>
      <p class="hint">
        The 3D view shows the chosen axis and the slicing plane:
        <span class="ax x">X</span> red, <span class="ax y">Y</span> green,
        <span class="ax z">Z</span> blue, as in Fusion. Slices are cut across
        that axis, so the plane you see is one slice.
      </p>
      <div v-if="store.sliceHalfExtent > 0" class="slice">
        <div class="row">
          <label for="slice_offset">
            Single slice
            <span class="unit num">mm from centre</span>
          </label>
          <input
            id="slice_offset_num" type="number" :step="sliceStep"
            :min="-store.sliceHalfExtent" :max="store.sliceHalfExtent"
            v-model.number="store.sliceOffset"
          />
        </div>
        <input
          id="slice_offset" class="slider" type="range" :step="sliceStep"
          :min="-store.sliceHalfExtent" :max="store.sliceHalfExtent"
          v-model.number="store.sliceOffset"
          :aria-label="`Slice position along ${(store.resolvedSliceAxis || '').toUpperCase()}`"
        />
        <p class="hint">
          Move the slider and the plane in the 3D view follows; the traced
          outline is drawn on it a moment later. This is Fusion's Create Mesh
          Section Sketch + Fit Curves to Mesh Section, for one plane.
        </p>
        <div class="row">
          <label for="slice_join">
            Join gaps up to
            <span class="unit num">mm</span>
          </label>
          <input
            id="slice_join" type="number" step="0.1" min="0" max="20"
            v-model.number="store.sliceJoin"
          />
        </div>
        <p class="hint">
          A leaky mesh (loose surface patches) cuts into pieces; loose ends
          closer than this are joined so the outline closes. 0 draws exactly
          what Fusion's Create Mesh Section Sketch draws, open pieces and all.
        </p>
        <p v-if="store.sectionBusy" class="hint">tracing…</p>
        <p v-else-if="store.sectionError" class="hint warn">Could not trace: {{ store.sectionError }}</p>
        <p v-else-if="sectionSummary" class="hint num">{{ sectionSummary }}</p>
        <a v-if="store.section && store.sectionScriptUrl" class="dl" :href="store.sectionScriptUrl" download>
          Download Fusion sketch of this slice (.py)
        </a>
      </div>
      <label class="check loft">
        <input type="checkbox" v-model="store.params.loft_ruled" />
        <span>
          Ruled loft
          <span class="help">Straight faces between neighbouring slices (one face per pair) instead of one smooth face per run. Exact within a hair at 0.2 mm spacing and never overshoots, but many more faces. The smooth loft is used by default and falls back to ruled on its own where a run's smooth surface comes out wrong.</span>
        </span>
      </label>
      <p class="hint">
        Slices are cut every {{ store.params.slice_mm }} mm; the loft itself goes through at most about 60 of them per run, and the checks below still measure the result against every triangle of the mesh. Steps (flat faces across the axis) and changes in the outline count split the stack into runs, so a shoulder stays a real flat face.
      </p>
    </template>

    <header class="head gate">
      <h2 class="micro">Acceptance gate</h2>
      <button class="reset" @click="store.resetParams()">Reset defaults</button>
    </header>
    <p v-if="!isLoft" class="intro">
      A result is only written if it passes every limit below — the prismatic
      fit first, then the face-group engine. Otherwise you get a faceted
      (exact but unclean) STEP instead.
    </p>
    <p v-else class="intro">
      With the sliced loft the limits below are measured and shown, not
      enforced: you chose the method, so its solid is written whenever it
      can be built. Only a loft that cannot be built at all falls back to a
      faceted copy.
    </p>

    <div v-for="f in fields" :key="f.key" class="field">
      <div class="row">
        <label :for="f.key">
          {{ f.label }}
          <span class="unit num">{{ f.unit }}</span>
          <button
            class="info num"
            :aria-expanded="openInfo === f.key"
            :aria-label="`What does ${f.label} do?`"
            @click="toggleInfo(f.key)"
          >i</button>
        </label>
        <input
          :id="f.key" type="number" :step="f.step" min="0.01"
          v-model.number="store.params[f.key]"
          :class="{ touched: !isDefault(f.key) }"
        />
      </div>
      <div v-if="openInfo === f.key" class="explain">
        <p>{{ f.blurb }}</p>
        <p><span class="dir">Lower it:</span> {{ f.lower }}</p>
        <p><span class="dir">Raise it:</span> {{ f.higher }}</p>
      </div>
    </div>

    <label class="check">
      <input type="checkbox" v-model="store.params.force_prismatic" />
      <span>
        Force prismatic on scan-like meshes
        <span class="help">Dense organic scans (millions of tiny triangles) normally skip straight to the faceted copy, because they rarely have flat faces and true cylinders to recover. Tick this to make the tool try anyway.</span>
      </span>
    </label>

    <label class="check">
      <input type="checkbox" v-model="store.params.face_groups" />
      <span>
        Face-group engine
        <span class="help">When the extrusion fit is rejected, group the mesh into surface regions and give each one a real plane, cylinder, cone or sphere face (like Fusion's "Prismatic" mesh conversion) before falling back to a faceted copy. Untick to skip it.</span>
      </span>
    </label>
  </section>
</template>

<style scoped>
.params { display: flex; flex-direction: column; gap: 10px; }
.warn {
  font-size: 12px;
  line-height: 1.5;
  color: var(--text);
  background: rgba(240, 173, 78, 0.10);
  border: 1px solid rgba(240, 173, 78, 0.45);
  border-left-width: 3px;
  border-radius: 4px;
  padding: 8px 10px;
  margin-top: 8px;
}
.warn .link { color: var(--edge); }
.hint { font-size: 12px; opacity: 0.85; }
.link { background: none; border: none; color: var(--accent, #5ad2ea); cursor: pointer; text-decoration: underline; padding: 0; font: inherit; }
.head { display: flex; justify-content: space-between; align-items: baseline; }
.reset {
  background: none;
  border: none;
  color: var(--edge);
  font-size: 12px;
  padding: 0;
}
.reset:hover { text-decoration: underline; }
.intro, .help { color: var(--muted); font-size: 12px; }
.field { border-top: 1px solid var(--line); padding-top: 8px; }
.row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}
label { font-weight: 500; }
.unit { color: var(--muted); font-size: 12px; margin-left: 2px; }
input.touched, select.touched { border-color: var(--edge); }
.head.gate { border-top: 1px solid var(--line); padding-top: 12px; margin-top: 4px; }

.info {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  margin-left: 6px;
  border-radius: 50%;
  border: 1px solid var(--muted);
  background: none;
  color: var(--muted);
  font-size: 11px;
  font-style: italic;
  line-height: 1;
  vertical-align: 1px;
}
.info:hover, .info[aria-expanded='true'] {
  border-color: var(--edge);
  color: var(--edge);
}

.explain {
  margin-top: 8px;
  padding: 8px 10px;
  border-left: 2px solid var(--edge);
  background: var(--panel-2);
  border-radius: 0 4px 4px 0;
  font-size: 12px;
  color: var(--text);
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.explain .dir { color: var(--edge); font-weight: 600; }

.check {
  display: flex;
  gap: 8px;
  border-top: 1px solid var(--line);
  padding-top: 10px;
  align-items: flex-start;
}
.check input { margin-top: 3px; accent-color: var(--edge); }
.check .help { display: block; font-weight: 400; }
.check.loft { border-top: none; padding-top: 4px; }
.ax { font-weight: 700; }
.slice { border-top: 1px solid var(--line); padding-top: 8px; display: flex; flex-direction: column; gap: 6px; }
.slice .slider { width: 100%; accent-color: var(--edge); }
.slice .dl {
  display: block; text-align: center; padding: 7px; border-radius: 6px;
  border: 1px solid var(--edge); color: var(--edge); text-decoration: none; font-size: 12px;
}
.slice .dl:hover { background: var(--panel-2); }
.hint.warn { color: var(--warn, #b26a00); }
.ax.x { color: #e5484d; }
.ax.y { color: #46b95a; }
.ax.z { color: #4c8dff; }
</style>
