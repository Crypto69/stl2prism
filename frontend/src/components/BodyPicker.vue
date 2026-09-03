<script setup>
import { computed } from 'vue'
import { useConvertStore } from '../store'

const store = useConvertStore()

// Volumes and sizes arrive in file units; show them the way the pipeline
// will read them, so a wrongly-set unit is visible here rather than after
// a long conversion.
const scale = computed(() => store.unitScale)
const rows = computed(() => store.bodies.map((b) => ({
  ...b,
  mm: b.size.map((v) => v * scale.value),
  volMm3: b.volume == null ? null : b.volume * scale.value ** 3,
  on: store.selected.includes(b.index),
})))

// Must match MeshViewer.bodyColor: same golden-angle walk, same
// saturation and lightness, or the legend lies about the scene.
function bodyHue(i) {
  return `hsl(${((i * 0.381966) % 1) * 360} 62% 62%)`
}

function fmtSize(mm) {
  return mm.map((v) => (v >= 100 ? v.toFixed(0) : v.toFixed(1))).join(' × ')
}
function fmtVol(v) {
  if (v == null) return 'open'
  if (v >= 1000) return `${(v / 1000).toFixed(1)} cm³`
  return `${v.toFixed(v < 10 ? 2 : 0)} mm³`
}
</script>

<template>
  <section v-if="store.hasBodyPicker" class="panel">
    <header>
      <h2>Bodies</h2>
      <span class="micro count">{{ store.selectedCount }} of {{ store.bodies.length }}</span>
    </header>
    <p class="micro help">
      This file holds {{ store.bodies.length }} separate bodies. Tick the ones to
      convert, or click them in the view. Converting only what you need is much faster.
    </p>
    <div class="actions">
      <button class="link micro" @click="store.selectSuggestedBodies()">Largest only</button>
      <button class="link micro" @click="store.selectAllBodies()">All</button>
    </div>
    <ul
      class="list"
      @mouseleave="store.hovered = -1"
    >
      <li
        v-for="r in rows"
        :key="r.index"
        :class="{ on: r.on, hot: store.hovered === r.index }"
        @mouseenter="store.hovered = r.index"
        @click="store.toggleBody(r.index)"
      >
        <input type="checkbox" :checked="r.on" tabindex="-1" @click.stop="store.toggleBody(r.index)" />
        <span class="swatch" :style="{ background: bodyHue(r.index) }"></span>
        <span class="name">Body {{ r.index + 1 }}</span>
        <span class="num size">{{ fmtSize(r.mm) }} mm</span>
        <span class="num vol" :class="{ open: r.volume == null }">{{ fmtVol(r.volMm3) }}</span>
      </li>
    </ul>
    <p v-if="store.bodiesTruncated" class="micro help">
      Only the largest bodies are listed; the rest are too small to matter.
    </p>
  </section>
</template>

<style scoped>
.panel { display: flex; flex-direction: column; gap: 8px; min-height: 0; }
header { display: flex; align-items: baseline; gap: 8px; }
h2 { font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase; margin: 0; }
.count { margin-left: auto; color: var(--edge); }
.help { color: var(--muted); line-height: 1.45; }
.actions { display: flex; gap: 12px; }
.link {
  background: none; border: none; color: var(--edge); padding: 0;
  text-decoration: underline; cursor: pointer;
}
.list {
  list-style: none; margin: 0; padding: 0;
  max-height: 260px; overflow-y: auto;
  border: 1px solid var(--line); border-radius: 6px;
}
li {
  display: grid;
  grid-template-columns: auto auto 1fr auto;
  grid-template-areas: 'check swatch name vol' 'check swatch size size';
  align-items: center;
  gap: 2px 8px;
  padding: 6px 10px;
  border-bottom: 1px solid var(--line);
  cursor: pointer;
}
li:last-child { border-bottom: none; }
li:hover, li.hot { background: rgba(90, 210, 234, 0.08); }
li.on { background: rgba(90, 210, 234, 0.05); }
input { grid-area: check; }
.swatch {
  grid-area: swatch; width: 10px; height: 10px; border-radius: 2px;
  opacity: 0.35;
}
li.on .swatch { opacity: 1; }
.name { grid-area: name; font-weight: 600; }
.size { grid-area: size; color: var(--muted); font-size: 11px; }
.vol { grid-area: vol; color: var(--edge); font-size: 11px; }
.vol.open { color: var(--muted); font-style: italic; }
</style>
