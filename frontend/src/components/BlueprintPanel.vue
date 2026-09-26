<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useConvertStore, checkDownload } from '../store'
import { swatch } from '../colors'
import RecipeField from './RecipeField.vue'

const store = useConvertStore()
onMounted(() => { if (!store.blueprintConfig) store.loadBlueprintConfig() })

// ---- provider, model, key: kept in this browser --------------------------------
const keyDraft = ref(store.apiKey)
const showKey = ref(false)
const modelDraft = ref(store.modelName)
const urlDraft = ref(store.providerBaseUrl)
watch(() => store.provider, () => {
  keyDraft.value = store.apiKey
  modelDraft.value = store.modelName
})
watch(() => store.blueprintConfig, () => { if (!modelDraft.value) modelDraft.value = store.modelName })
const saveKey = () => store.setKey(keyDraft.value)
const clearKey = () => { keyDraft.value = ''; store.setKey('') }
const saveModel = () => store.setModel(modelDraft.value)
const saveUrl = () => store.setBaseUrl(urlDraft.value)
const preset = computed(() => store.providerPreset)
const providers = computed(() => store.blueprintConfig?.providers || [])

// ---- the editable recipe ----------------------------------------------------------
// A working copy of the recipe the server last read or built. `dirty`
// when it differs; Rebuild sends it, Revert drops the edits.
const draft = ref(null)
const clone = (x) => (x == null ? null : JSON.parse(JSON.stringify(x)))
watch(() => store.recipe, (r) => { draft.value = clone(r) }, { immediate: true })
const dirty = computed(() => !!draft.value && JSON.stringify(draft.value) !== JSON.stringify(store.recipe))
const revert = () => { draft.value = clone(store.recipe) }

// Every edit redraws the shape after a short rest: the warm helper builds
// it in well under a second, so which number moves what is visible at
// once. The full Rebuild (STEP + Fusion script) stays a button.
let liveTimer = null
watch(draft, (d) => {
  clearTimeout(liveTimer)
  if (!d || !dirty.value || store.busy) return
  liveTimer = setTimeout(() => store.previewLive(d), 500)
}, { deep: true })
onBeforeUnmount(() => clearTimeout(liveTimer))

// which feature the pointer is on, in the list or in the view
const hover = (i) => { store.hovered = i }
const unhover = () => { store.hovered = -1 }
const liveOverall = computed(() => {
  const oc = store.liveInfo?.overall_check
  if (!oc?.expected) return null
  const worst = Math.max(...(oc.dev_pct || []).filter((d) => d != null).map(Math.abs), 0)
  return `${store.liveInfo.bbox.size.map((v) => fmt(v, 2)).join(' × ')} mm · worst ${fmt(worst, 1)} % off the drawing`
})
const rebuild = () => { if (draft.value) store.rebuild(draft.value) }
const orig = (fi, path) => {
  // the server's value at the same place, for the changed-outline
  let o = store.recipe?.features?.[fi]
  for (const k of path) { o = o?.[k] }
  return o
}
const origParam = (name) => store.recipe?.params?.find((p) => p.name === name)?.value
const SHAPE_KEYS = {
  rect: [['center', 'u'], ['center', 'v'], ['w'], ['h'], ['corner_radius']],
  circle: [['center', 'u'], ['center', 'v'], ['d']],
  slot: [['p1', 'u'], ['p1', 'v'], ['p2', 'u'], ['p2', 'v'], ['width']],
}
const keyLabel = (path) => path.join('.').replace('center.', 'c').replace('corner_radius', 'r')
const getAt = (obj, path) => path.reduce((o, k) => (o == null ? o : o[k]), obj)
const setAt = (obj, path, v) => {
  let o = obj
  for (const k of path.slice(0, -1)) { if (o[k] == null) o[k] = {}; o = o[k] }
  o[path[path.length - 1]] = v
}
const planeWord = { XY: 'XY (top view)', XZ: 'XZ (front view)', YZ: 'YZ (side view)' }

// ---- report + downloads -----------------------------------------------------------
const fmt = (v, d = 2) => (v == null || Number.isNaN(Number(v)) ? '—' : Number(v).toFixed(d))
const bboxText = computed(() => (store.bbox?.size ? store.bbox.size.map((v) => fmt(v, 2)).join(' × ') : null))
const overallText = computed(() => {
  const oc = store.overallCheck
  if (!oc?.expected) return null
  const want = oc.expected.map((v) => (v == null ? '?' : fmt(v, 2))).join(' × ')
  const worst = Math.max(...(oc.dev_pct || []).filter((d) => d != null).map(Math.abs), 0)
  return `${want} on the drawing · worst ${fmt(worst, 1)} % off`
})
const downloadError = ref(null)
async function guardDownload(ev, url) {
  if (!url) return
  ev.preventDefault()
  downloadError.value = null
  const why = await checkDownload(url)
  if (why) { downloadError.value = `Could not download: ${why}`; return }
  const a = document.createElement('a')
  a.href = url
  a.download = ''
  document.body.appendChild(a)
  a.click()
  a.remove()
}
watch(() => store.jobId, () => { downloadError.value = null })
const readLabel = computed(() => {
  if (store.readStatus === 'reading') return 'Reading the drawing…'
  if (store.readStatus === 'building') return 'Building…'
  return store.recipe ? 'Read the drawing again' : 'Read drawing'
})
</script>

<template>
  <section class="params">
    <p class="intro">
      Drop or paste a drawing with front / top / side views and dimension
      labels. A vision model reads the labels into named parameters and a
      short list of sketch + extrude features; the part is built from those
      numbers, so fix any it misread in the table and rebuild. The picture is
      sent to the provider you pick with your key; the key stays in this browser.
    </p>

    <!-- 1. provider + key -->
    <div class="field">
      <div class="row">
        <label for="bp_provider">Vision model</label>
        <select id="bp_provider" :value="store.provider" @change="store.setProvider($event.target.value)">
          <option v-for="p in providers" :key="p.key" :value="p.key">{{ p.label }}</option>
          <option v-if="!providers.length" value="anthropic">Anthropic (Claude)</option>
        </select>
      </div>
      <div class="row">
        <label for="bp_model">Model</label>
        <input
          id="bp_model" type="text" class="num wide" v-model="modelDraft" spellcheck="false" autocomplete="off"
          :placeholder="preset?.default_model || 'model name'" :class="{ touched: modelDraft !== (preset?.default_model || '') }"
          @blur="saveModel" @keydown.enter="saveModel"
        />
      </div>
      <div v-if="store.provider === 'custom'" class="row">
        <label for="bp_url">Base URL</label>
        <input
          id="bp_url" type="text" class="num wide" v-model="urlDraft" spellcheck="false" autocomplete="off"
          placeholder="http://localhost:11434/v1" @blur="saveUrl" @keydown.enter="saveUrl"
        />
      </div>
      <div class="row">
        <label for="bp_key">API key <span class="help">saved in this browser only</span></label>
        <span class="pair">
          <input
            id="bp_key" :type="showKey ? 'text' : 'password'" class="num wide" v-model="keyDraft"
            :placeholder="preset?.needs_key === false ? 'optional' : 'paste your key'"
            autocomplete="off" spellcheck="false" @blur="saveKey" @keydown.enter="saveKey"
          />
          <button class="link" type="button" @click="showKey = !showKey">{{ showKey ? 'hide' : 'show' }}</button>
          <button v-if="keyDraft" class="link" type="button" @click="clearKey">clear</button>
        </span>
      </div>
      <p class="hint">
        <template v-if="store.blueprintConfig === null">Asking the server which providers it offers…</template>
        <template v-else-if="store.hasServerKey">The server has its own {{ preset?.label }} key; yours is used first when set.</template>
        <template v-else-if="preset?.needs_key === false">No key needed for a local server; one is sent if you set it.</template>
        <template v-else-if="!store.apiKey">A key is needed here to read a drawing (or set {{ preset?.env }} on the server).</template>
        <template v-else>Key saved for {{ preset?.label }}.</template>
        <template v-if="preset?.note"> {{ preset.note }}</template>
      </p>
    </div>

    <!-- 2. hints + read -->
    <div class="field">
      <label for="bp_hints">Notes for the reader <span class="help">optional: which view is which, what to ignore</span></label>
      <textarea id="bp_hints" v-model="store.hints" rows="2" maxlength="2000"
                placeholder="e.g. the view with the round boss is the top view; ignore the logo" />
    </div>
    <button
      type="button" class="dl press" :class="{ busy: store.readStatus }"
      :disabled="!store.canRead" @click="store.readDrawing()"
    >
      <span v-if="store.readStatus" class="spin edge" aria-hidden="true"></span> {{ readLabel }}
    </button>
    <button
      v-if="store.status === 'running'" type="button" class="cancel"
      :disabled="store.cancelling" @click="store.cancel()"
    >{{ store.cancelling ? 'Stopping…' : 'Cancel' }}</button>
    <p v-if="!store.jobId || !store.isImageJob" class="hint">Drop or paste a drawing image first.</p>
    <p v-if="store.error" class="hint warn">{{ store.error }}</p>
    <p v-if="store.cancelled" class="hint">Stopped. The drawing is still here.</p>

    <!-- 3. warnings -->
    <ul v-if="store.warnings.length" class="warn list">
      <li v-for="(w, i) in store.warnings" :key="i">{{ w }}</li>
    </ul>

    <!-- 4. the recipe: parameters, then features -->
    <template v-if="draft">
      <header class="head gate">
        <h2 class="micro">Parameters (mm)</h2>
        <button v-if="dirty" class="reset" type="button" @click="revert">Revert</button>
      </header>
      <table class="recipe">
        <thead class="micro"><tr><th>Name</th><th>Value</th><th>From</th></tr></thead>
        <tbody>
          <tr v-for="p in draft.params" :key="p.name">
            <td class="num name">{{ p.name }}<span v-if="p.inferred" class="badge" title="deduced, not read from a label">?</span></td>
            <td>
              <input type="number" step="0.1" class="val" v-model.number="p.value"
                     :class="{ touched: p.value !== origParam(p.name) }" />
            </td>
            <td class="from">{{ p.source || '—' }}</td>
          </tr>
        </tbody>
      </table>

      <h2 class="micro">Features <span class="help">colours match the 3D view; hover to light one up</span></h2>
      <ol class="features">
        <li v-for="(f, fi) in draft.features" :key="f.id || fi"
            :class="{ lit: store.hovered === fi }"
            @mouseenter="hover(fi)" @mouseleave="unhover">
          <div class="row top">
            <span class="fname">
              <span class="swatch" :style="{ background: swatch(fi) }"></span>
              <b>{{ f.name || f.id }}</b>
              <span class="badge op" :class="f.op">{{ f.op.replace('_', ' ') }}</span>
              <span v-if="f.source?.inferred" class="badge" title="some numbers deduced">?</span>
              <span v-if="f.confidence < 0.7" class="badge low" :title="'confidence ' + fmt(f.confidence, 2)">low</span>
            </span>
            <span class="num plane">{{ planeWord[f.plane] || f.plane }}</span>
          </div>
          <div class="kv">
            <span class="k">offset</span>
            <RecipeField :model-value="f.offset" :original="orig(fi, ['offset'])" @update:model-value="f.offset = $event" />
            <span class="k">dir</span>
            <select class="mini" v-model="f.direction" :class="{ touched: f.direction !== orig(fi, ['direction']) }"
                    title="'+' along the plane's normal (XY: up), '-' the other way, symmetric both ways">
              <option value="+">+</option>
              <option value="-">−</option>
              <option value="symmetric">sym</option>
            </select>
            <template v-if="!f.through_all">
              <span class="k">distance</span>
              <RecipeField :model-value="f.distance" :original="orig(fi, ['distance'])" @update:model-value="f.distance = $event" />
            </template>
            <label v-if="f.op === 'cut'" class="tick" :class="{ touched: f.through_all !== orig(fi, ['through_all']) }"
                   title="cut the whole part along the normal, whatever is in line">
              <input type="checkbox" v-model="f.through_all" /> through all
            </label>
          </div>
          <div v-for="(s, si) in f.shapes" :key="si" class="kv shape">
            <span class="kind">{{ s.kind }}</span>
            <template v-if="SHAPE_KEYS[s.kind]">
              <template v-for="path in SHAPE_KEYS[s.kind]" :key="path.join('.')">
                <span class="k">{{ keyLabel(path) }}</span>
                <RecipeField
                  :model-value="getAt(s, path) ?? 0" :original="orig(fi, ['shapes', si, ...path])" :size="6"
                  @update:model-value="setAt(s, path, $event)"
                />
              </template>
            </template>
            <span v-else class="k">{{ s.points?.length || 0 }} points</span>
          </div>
        </li>
      </ol>
      <p v-if="store.liveBusy" class="hint"><span class="spin edge" aria-hidden="true"></span> redrawing…</p>
      <p v-else-if="store.liveError" class="hint warn">Could not redraw: {{ store.liveError }}</p>
      <p v-else-if="dirty && liveOverall" class="hint num">live: {{ liveOverall }}</p>
      <ul v-if="dirty && store.liveInfo?.warnings?.length" class="warn list">
        <li v-for="(w, i) in store.liveInfo.warnings" :key="i">{{ w }}</li>
      </ul>
      <button type="button" class="dl press" :disabled="!dirty || store.busy" @click="rebuild">
        {{ dirty ? 'Rebuild with these values (STEP + Fusion script)' : 'Rebuild (nothing changed)' }}
      </button>
      <p class="hint">
        A value may be a number or an expression over the parameters, like
        <span class="num">body_w/2</span>. Rebuild runs the checks and the build
        again without asking the model.
      </p>
    </template>

    <!-- 5. report + downloads -->
    <template v-if="store.status === 'done' && store.result?.ok && store.isImageJob">
      <header class="head gate"><h2 class="micro">Built</h2></header>
      <table class="report">
        <tbody>
          <tr><td>Size</td><td class="num">{{ bboxText }} mm</td></tr>
          <tr v-if="overallText"><td>Drawing says</td><td class="num">{{ overallText }}</td></tr>
          <tr><td>Volume</td><td class="num">{{ store.volume != null ? Math.round(store.volume).toLocaleString() : '—' }} mm³</td></tr>
          <tr><td>Bodies</td><td class="num">{{ store.result.solids }}</td></tr>
          <tr v-if="store.recipeRead?.model">
            <td>Read by</td>
            <td class="num">
              {{ store.recipeRead.model }}
              <template v-if="store.recipeRead.usage"> · {{ store.recipeRead.usage.input_tokens }} in / {{ store.recipeRead.usage.output_tokens }} out</template>
              <template v-if="store.recipeRead.seconds"> · {{ fmt(store.recipeRead.seconds, 0) }} s</template>
            </td>
          </tr>
        </tbody>
      </table>
      <a class="dl" :href="store.fusionScriptUrl" download @click="guardDownload($event, store.fusionScriptUrl)">
        Download Fusion 360 script (.py) — parametric
      </a>
      <a class="dl" :href="store.downloadUrl" download @click="guardDownload($event, store.downloadUrl)">
        Download STEP
      </a>
      <p v-if="downloadError" class="hint warn">{{ downloadError }}</p>
      <p class="hint">
        Put the .py in an empty folder, then in Fusion Utilities → Add-Ins →
        Scripts and Add-Ins → + → choose that folder → Run, in a new parametric
        design. Every parameter becomes a User Parameter (Modify → Change
        Parameters), every feature a dimensioned sketch and an Extrude.
      </p>
    </template>
    <details v-if="store.log" class="logbox" :open="store.status === 'running'">
      <summary class="micro">Log</summary>
      <pre class="num">{{ store.log }}</pre>
    </details>
  </section>
</template>

<style scoped>
.pair { display: inline-flex; align-items: center; gap: 8px; }
.wide { width: 150px; }
.field label { display: block; }
.field textarea {
  width: 100%;
  margin-top: 6px;
  font: inherit;
  font-size: 12px;
  background: var(--ink);
  border: 1px solid var(--line);
  border-radius: 4px;
  color: var(--text);
  padding: 5px 8px;
  resize: vertical;
}
.list { margin: 0; padding-left: 18px; display: flex; flex-direction: column; gap: 4px; }
.recipe, .report { width: 100%; border-collapse: collapse; font-size: 12px; }
.recipe th { text-align: left; font-weight: 500; padding: 2px 6px 4px 0; }
.recipe td, .report td { padding: 3px 6px 3px 0; vertical-align: middle; }
.recipe td.name { white-space: nowrap; }
.recipe .val { width: 72px; padding: 3px 6px; font-size: 12px; }
.recipe .from { color: var(--muted); font-size: 11px; line-height: 1.3; }
.report td:first-child { color: var(--muted); white-space: nowrap; }
.badge {
  display: inline-block;
  margin-left: 6px;
  padding: 0 5px;
  border-radius: 8px;
  border: 1px solid var(--line);
  color: var(--muted);
  font-size: 10px;
  font-weight: 600;
  line-height: 15px;
  vertical-align: 1px;
}
.badge.low { border-color: var(--warn, #f0ad4e); color: var(--warn, #f0ad4e); }
.badge.op { text-transform: uppercase; letter-spacing: 0.04em; }
.badge.op.cut { border-color: var(--fail); color: var(--fail); }
.badge.op.join, .badge.op.new_body { border-color: var(--edge); color: var(--edge); }
.features { margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 6px; }
.features li { border-top: 1px solid var(--line); padding: 6px 4px 2px; margin: 0 -4px; border-radius: 4px; display: flex; flex-direction: column; gap: 4px; }
.features li.lit { background: var(--panel-2); }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.hint .spin { display: inline-block; vertical-align: -2px; margin-right: 6px; }
.features .top { align-items: baseline; }
.features .fname { font-size: 12px; }
.features .plane { color: var(--muted); font-size: 11px; white-space: nowrap; }
.kv { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; font-size: 12px; }
.kv .k { color: var(--muted); font-size: 11px; margin-left: 4px; }
.kv .k:first-child { margin-left: 0; }
.kv.shape { padding-left: 8px; }
.kv .kind { font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
.kv select.mini { width: auto; padding: 2px 4px; font-size: 12px; max-width: 64px; }
.kv .tick { display: inline-flex; align-items: center; gap: 4px; font-size: 11px; color: var(--muted); margin-left: 4px; }
.kv .tick input { accent-color: var(--edge); margin: 0; }
.kv .tick.touched { color: var(--edge); }
.logbox pre {
  margin-top: 6px;
  max-height: 180px;
  overflow: auto;
  font-size: 11px;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--muted);
}
.cancel {
  padding: 9px;
  border-radius: 6px;
  border: 1px solid var(--line);
  background: none;
  color: var(--muted);
  font-weight: 600;
  font-size: 13px;
}
.cancel:not(:disabled):hover { color: var(--fail); border-color: var(--fail); }
.cancel:disabled { opacity: 0.6; cursor: default; }
</style>
