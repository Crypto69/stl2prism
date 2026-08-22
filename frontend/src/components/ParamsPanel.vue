<script setup>
import { computed, ref } from 'vue'
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
const longest = computed(() =>
  store.inputStats ? Math.max(...store.inputStats.bbox_mm).toFixed(2) : '')
const openInfo = ref(null)
const toggleInfo = (key) => { openInfo.value = openInfo.value === key ? null : key }
const isDefault = (key) => store.params[key] === DEFAULT_PARAMS[key]
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
    <div v-if="openInfo === 'units'" class="explain">
      <p>Mesh files do not say what unit they are in — they are just numbers. Every program guesses. This tool works in millimetres, so tell it what the file meant and it scales the mesh before doing anything else. All the limits below are in mm.</p>
      <p><span class="dir">Tip:</span> Fusion 360 assumes centimetres for OBJ. If a part looks 10× too small here but right in Fusion, pick cm.</p>
    </div>
    <p v-if="suggestion && suggestion !== store.params.units" class="hint">
      This file looks like it might be in <b>{{ suggestion }}</b> (its longest side is
      {{ longest }} units). <button class="link" @click="store.params.units = suggestion">Use {{ suggestion }}</button>
    </p>

    <header class="head gate">
      <h2 class="micro">Acceptance gate</h2>
      <button class="reset" @click="store.resetParams()">Reset defaults</button>
    </header>
    <p class="intro">
      A result is only written if it passes every limit below — the prismatic
      fit first, then the face-group engine. Otherwise you get a faceted
      (exact but unclean) STEP instead.
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
</style>
