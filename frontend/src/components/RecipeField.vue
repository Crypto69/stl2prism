<script setup>
// One recipe number: a plain value (22.5) or an expression over the
// parameters ("body_w/2"). Typed as text so both fit in one box; a
// number is stored as a number, anything else as the expression string
// (the server's validator names a bad one on Rebuild).
import { computed } from 'vue'

const props = defineProps({
  modelValue: { type: [Number, String], default: 0 },
  // the value the server built from, for the "changed" outline
  original: { type: [Number, String], default: undefined },
  size: { type: Number, default: 8 },
})
const emit = defineEmits(['update:modelValue'])

const text = computed(() => (props.modelValue == null ? '' : String(props.modelValue)))
const touched = computed(() => props.original !== undefined && String(props.original) !== text.value)

function commit(ev) {
  const raw = (ev.target.value || '').trim()
  if (raw === '') return
  const n = Number(raw)
  emit('update:modelValue', Number.isFinite(n) && /^[-+]?\d*\.?\d+(e[-+]?\d+)?$/i.test(raw) ? n : raw)
}
</script>

<template>
  <input
    type="text" class="num field" :class="{ touched }" :value="text" :size="size"
    spellcheck="false" autocomplete="off"
    @change="commit" @keydown.enter="commit"
  />
</template>

<style scoped>
.field {
  font-family: var(--font-data);
  background: var(--ink);
  border: 1px solid var(--line);
  border-radius: 4px;
  color: var(--text);
  padding: 3px 6px;
  font-size: 12px;
  width: auto;
  min-width: 56px;
}
.field.touched { border-color: var(--edge); }
</style>
