<script setup>
import { computed, ref } from 'vue'
import { useConvertStore, DEFAULT_PARAMS, UNITS } from '../store'

const store = useConvertStore()

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
  </section>
</template>
