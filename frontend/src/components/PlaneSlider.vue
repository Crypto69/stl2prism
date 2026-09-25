<script setup>
import { computed } from 'vue'
import { useConvertStore } from '../store'

// A plane's position along the resolved axis: a number box and a slider
// over the part's whole side, in mm from the box centre (Fusion's slider).
const props = defineProps({
  modelValue: { type: Number, default: 0 },
  id: { type: String, required: true },
  label: { type: String, default: 'Plane' },
})
const emit = defineEmits(['update:modelValue'])

const store = useConvertStore()
const step = computed(() => {
  const h = store.sliceHalfExtent
  return h > 50 ? 0.5 : h > 10 ? 0.1 : 0.05
})
const set = (v) => { if (Number.isFinite(v)) emit('update:modelValue', v) }
</script>

<template>
  <div class="row">
    <label :for="id + '_num'">
      {{ label }}
      <span class="unit num">mm from centre</span>
    </label>
    <input
      :id="id + '_num'" type="number" :step="step"
      :min="-store.sliceHalfExtent" :max="store.sliceHalfExtent"
      :value="modelValue"
      @input="set($event.target.valueAsNumber)"
    />
  </div>
  <input
    :id="id" class="slider" type="range" :step="step"
    :min="-store.sliceHalfExtent" :max="store.sliceHalfExtent"
    :value="modelValue"
    :aria-label="`${label} along ${(store.resolvedSliceAxis || '').toUpperCase()}`"
    @input="set($event.target.valueAsNumber)"
  />
</template>
