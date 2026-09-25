<script setup>
import { useConvertStore, DEFAULT_PARAMS } from '../store'

// How a slice is traced: gap joining, sliver trimming, outline only, and
// (for a tool without the acceptance gate) the fit tolerance itself.
defineProps({ tol: { type: Boolean, default: false } })

const store = useConvertStore()
const isDefault = (key) => store.params[key] === DEFAULT_PARAMS[key]
</script>

<template>
  <div v-if="tol" class="row">
    <label for="slice_tol">
      Fit tolerance
      <span class="unit num">mm</span>
    </label>
    <input
      id="slice_tol" type="number" step="0.01" min="0.01" max="5"
      v-model.number="store.params.tol"
      :class="{ touched: !isDefault('tol') }"
    />
  </div>
  <p v-if="tol" class="hint">
    How far the redrawn lines, arcs and splines may stray from the mesh.
    0.08 follows a clean CAD export; 0.15 hides the layer lines of a 3D
    print.
  </p>
  <div class="row">
    <label for="slice_join">
      Join gaps up to
      <span class="unit num">mm</span>
    </label>
    <input
      id="slice_join" type="number" step="0.1" min="0" max="20"
      v-model.number="store.params.slice_join"
      :class="{ touched: !isDefault('slice_join') }"
    />
  </div>
  <p class="hint">
    A leaky mesh (loose surface patches) cuts into pieces; loose ends
    closer than this are joined so the outline closes. 0 draws exactly
    what Fusion's Create Mesh Section Sketch draws, open pieces and all.
    The loft's cutter uses the same setting.
  </p>
  <div class="row">
    <label for="slice_trim">
      Trim slivers up to
      <span class="unit num">mm</span>
    </label>
    <input
      id="slice_trim" type="number" step="0.1" min="0" max="5"
      v-model.number="store.params.slice_trim"
      :class="{ touched: !isDefault('slice_trim') }"
    />
  </div>
  <p class="hint">
    Where the mesh has a double skin, the outline goes out and back along
    the same path and Fusion makes thin sliver profiles there. Slivers
    narrower than this are cut out of the loop before fitting. 0 leaves
    the outline exactly as traced; 0.3 is a good start.
  </p>
  <label class="check">
    <input type="checkbox" v-model="store.sliceOutline" />
    <span>
      Outline only
      <span class="help">Draw just the outer outline: loose open pieces and inner loops (holes, islands) are left out, so the sketch is one clean profile to extrude. The summary still says how many were skipped.</span>
    </span>
  </label>
</template>
