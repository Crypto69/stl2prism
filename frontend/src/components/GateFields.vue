<script setup>
import { ref } from 'vue'
import { useConvertStore, DEFAULT_PARAMS } from '../store'

// `enforced`: the auto ladder only writes a clean solid that passes every
// limit; the sliced loft measures and shows them but writes anyway.
defineProps({ enforced: { type: Boolean, default: true } })

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

const openInfo = ref(null)
const toggleInfo = (key) => { openInfo.value = openInfo.value === key ? null : key }
const isDefault = (key) => store.params[key] === DEFAULT_PARAMS[key]
</script>

<template>
  <header class="head gate">
    <h2 class="micro">Acceptance gate</h2>
    <button class="reset" @click="store.resetParams()">Reset defaults</button>
  </header>
  <p v-if="enforced" class="intro">
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
</template>
