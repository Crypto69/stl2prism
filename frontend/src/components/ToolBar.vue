<script setup>
import { useConvertStore } from '../store'

const store = useConvertStore()
// "New project": the parent owns the hidden file input, so it opens it
const emit = defineEmits(['new'])

// One button per tool. The rail shows that tool's controls; the 3D view
// follows (a slicing plane for the loft, start and end planes for x-ray).
const TOOLS = [
  {
    key: 'solid', label: 'Mesh → Solid',
    title: 'Mesh → Solid: find the design intent (an extrusion direction, flat faces, cylinders, cones, spheres) and write a clean STEP. Best for machined or CAD-designed parts.',
  },
  {
    key: 'loft', label: 'Sliced Loft',
    title: 'Sliced Loft: cut the body into thin slices along one axis, redraw each outline as a smooth curve and loft the stack into a solid. Best for organic shells.',
  },
  {
    key: 'xray', label: 'X-Ray',
    title: 'X-Ray: a stack of section sketches between two planes, as a Fusion script. Like Create Mesh Section Sketch + Fit Curves, for every slice at once.',
  },
  {
    key: 'blueprint', label: 'Blueprint',
    title: 'Blueprint: read a dimensioned drawing (front / top / side views) into named parameters and sketch + extrude features; edit the numbers, rebuild, download a parametric Fusion script and a STEP. The picture is sent to the vision provider you pick, with your key.',
  },
]
</script>

<template>
  <nav class="tools" aria-label="Tool">
    <button class="new" title="New project: start over with another mesh file or drawing" @click="emit('new')">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9z" /><path d="M13 3v6h6M12 12v6M9 15h6" /></svg>
      <span>New project</span>
    </button>
    <span class="sep" />
    <button
      v-for="t in TOOLS" :key="t.key"
      :aria-pressed="store.tool === t.key"
      :class="{ on: store.tool === t.key }"
      :title="t.title"
      @click="store.setTool(t.key)"
    >
      <svg v-if="t.key === 'solid'" viewBox="0 0 24 24"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z" /><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" /></svg>
      <svg v-else-if="t.key === 'loft'" viewBox="0 0 24 24"><path d="M5 6c3-2 11-2 14 0M5 6c-1 2 0 4 2 5M19 6c1 2 0 4-2 5M7 11c3 2 7 2 10 0M5 17c3-2 11-2 14 0M5 17c-1-2 0-4 2-6M19 17c1-2 0-4-2-6" /></svg>
      <svg v-else-if="t.key === 'xray'" viewBox="0 0 24 24"><rect x="4" y="3" width="16" height="18" rx="1.5" /><path d="M7 8h10M7 12h10M7 16h10" stroke-dasharray="2 1.6" /></svg>
      <svg v-else viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="1.5" /><path d="M7 16V8h5a2.5 2.5 0 0 1 0 5H7M15 8v8" /><path d="M3 9h2M3 15h2M19 9h2M19 15h2" stroke-dasharray="1.5 1.5" /></svg>
      <span>{{ t.label }}</span>
    </button>
  </nav>
</template>

<style scoped>
/* the same look as the viewer's bottom toolbar, laid flat in the header */
.tools {
  display: flex;
  align-items: center;
  gap: 2px;
  padding: 3px;
  background: rgba(20, 23, 28, 0.85);
  border: 1px solid var(--line);
  border-radius: 6px;
}
.tools button {
  height: 28px;
  padding: 0 10px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: none;
  border: none;
  border-radius: 4px;
  color: var(--muted);
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}
.tools button:hover { color: var(--text); background: var(--panel-2); }
.tools button.on { color: var(--edge); background: var(--panel-2); }
.tools .sep { width: 1px; height: 18px; background: var(--line); margin: 0 3px; }
.tools svg {
  width: 18px;
  height: 18px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}
@media (max-width: 640px) {
  .tools button span { display: none; }
  .tools button { padding: 0 7px; }
}
</style>
