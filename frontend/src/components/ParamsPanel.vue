<script setup>
import { useConvertStore, DEFAULT_PARAMS } from '../store'

const store = useConvertStore()

const fields = [
  { key: 'tol', label: 'Fit tolerance', unit: 'mm', step: 0.01,
    help: 'How closely lines and arcs must follow the mesh profile. Smaller = more detail kept, more faces.' },
  { key: 'accept_p95', label: 'Surface deviation (p95)', unit: 'mm', step: 0.01,
    help: '95% of the surface must be within this distance of the original mesh.' },
  { key: 'accept_max', label: 'Surface deviation (max)', unit: 'mm', step: 0.01,
    help: 'No single point may deviate more than this.' },
  { key: 'accept_hole_max', label: 'Bore deviation (max)', unit: 'mm', step: 0.01,
    help: 'Tighter budget for cylindrical holes — a wrong hole size stops parts fitting.' },
  { key: 'accept_vol_pct', label: 'Volume error', unit: '%', step: 0.1,
    help: 'The rebuilt solid volume must match the mesh within this percentage.' },
]

const isDefault = (key) => store.params[key] === DEFAULT_PARAMS[key]
</script>

<template>
  <section class="params">
    <header class="head">
      <h2 class="micro">Acceptance gate</h2>
      <button class="reset" @click="store.resetParams()">Reset defaults</button>
    </header>
    <p class="intro">
      The prismatic result is only written if it passes every limit below.
      Otherwise you get a faceted (exact but unclean) STEP instead.
    </p>

    <div v-for="f in fields" :key="f.key" class="field">
      <div class="row">
        <label :for="f.key">
          {{ f.label }}
          <span class="unit num">{{ f.unit }}</span>
        </label>
        <input
          :id="f.key" type="number" :step="f.step" min="0.01"
          v-model.number="store.params[f.key]"
          :class="{ touched: !isDefault(f.key) }"
        />
      </div>
      <p class="help">{{ f.help }}</p>
    </div>

    <label class="check">
      <input type="checkbox" v-model="store.params.force_prismatic" />
      <span>
        Force prismatic on scan-like meshes
        <span class="help">Dense organic scans normally skip straight to faceted output.</span>
      </span>
    </label>
  </section>
</template>

<style scoped>
.params { display: flex; flex-direction: column; gap: 10px; }
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
input.touched { border-color: var(--edge); }
.help { margin-top: 2px; }
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
