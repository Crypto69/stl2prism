<script setup>
import { computed } from 'vue'
import { useConvertStore, DEFAULT_PARAMS } from '../store'

const store = useConvertStore()

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
  <div class="row">
    <label for="slice_axis">Slice along</label>
    <select id="slice_axis" v-model="store.params.slice_axis"
            :class="{ touched: store.params.slice_axis !== DEFAULT_PARAMS.slice_axis }">
      <option v-for="o in axisOptions" :key="o.key" :value="o.key">{{ o.label }}</option>
    </select>
  </div>
  <p class="hint">
    The 3D view shows the chosen axis and the slicing plane:
    <span class="ax x">X</span> red, <span class="ax y">Y</span> green,
    <span class="ax z">Z</span> blue, as in Fusion. Slices are cut across
    that axis, so a plane you see is one slice.
  </p>
</template>
