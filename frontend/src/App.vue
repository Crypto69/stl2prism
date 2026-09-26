<script setup>
import { computed, markRaw, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'
import MeshViewer from './components/MeshViewer.vue'
import BodyPicker from './components/BodyPicker.vue'
import ToolBar from './components/ToolBar.vue'
import SetupPanel from './components/SetupPanel.vue'
import SolidPanel from './components/SolidPanel.vue'
import LoftPanel from './components/LoftPanel.vue'
import XRayPanel from './components/XRayPanel.vue'
import BlueprintPanel from './components/BlueprintPanel.vue'
import ReportPanel from './components/ReportPanel.vue'
import { useConvertStore, IMAGE_EXTS } from './store'
import { appError, clearAppError, friendlyError } from './errors'

const store = useConvertStore()
const buffer = shallowRef(null)
const dragOver = ref(false)
const fileInput = ref(null)

const MESH_ACCEPT = ['stl', 'obj', 'ply', 'off', '3mf', 'glb', 'gltf']
// a drawing image goes to Blueprint, a mesh to the other tools
const ACCEPT = [...MESH_ACCEPT, ...IMAGE_EXTS]
const ACCEPT_ATTR = ACCEPT.map((e) => '.' + e).join(',')
// Blueprint's stage: the drawing itself, or the 3D preview of the build
const stageView = ref('drawing')
// a file is being read (the browser parses STL/OBJ itself; other formats
// wait for the server's preview): the drop zone says so meanwhile
const loading = ref(false)
const loadingName = ref('')

// Which build is running: package version + git commit + build time from
// /api/version, so a tester can match the browser to a commit at a glance.
const build = ref(null)
onMounted(async () => {
  window.addEventListener('paste', onPaste)
  try {
    const res = await fetch('/api/version')
    if (res.ok) build.value = await res.json()
  } catch (e) { /* badge is optional */ }
})
onBeforeUnmount(() => window.removeEventListener('paste', onPaste))

// Cmd/Ctrl+V with a picture on the clipboard loads it as a drawing (not
// when pasting into a text field). Clipboard images arrive as "image.png"
// with no extension we route on, so they get a name.
function onPaste(e) {
  const t = e.target
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
  const items = [...(e.clipboardData?.items || [])]
  const it = items.find((i) => i.kind === 'file' && i.type.startsWith('image/'))
  let file = it ? it.getAsFile() : e.clipboardData?.files?.[0]
  if (!file || !file.type.startsWith('image/')) return
  e.preventDefault()
  const ext = (file.type.split('/')[1] || 'png').replace('jpeg', 'jpg')
  if (!/\.[a-z0-9]+$/i.test(file.name || '')) {
    file = new File([file], `pasted-drawing-${Date.now()}.${ext}`, { type: file.type })
  }
  takeFile(file)
}

// A finished Blueprint build has a preview.stl: show it (a rebuild
// rewrites the file, so the fetch is never cached).
watch(() => [store.status, store.jobId], async ([s, job]) => {
  if (s !== 'done' || !store.isImageJob || !store.result?.ok || !job) return
  try {
    const res = await fetch(`/api/jobs/${job}/preview?t=${Date.now()}`)
    if (!res.ok) throw new Error(`the server sent no preview (${res.status})`)
    const data = await res.arrayBuffer()
    if (job !== store.jobId) return
    buffer.value = markRaw({ data, kind: 'stl' })
    stageView.value = '3d'
    previewError.value = null
  } catch (e) {
    previewError.value = `No 3D preview (${friendlyError(e)}). The STEP and the script still download.`
  }
})

// why the 3D preview is missing while the file itself is fine
const previewError = ref(null)

async function takeFile(file) {
  if (!file) return
  const kind = (file.name || '').split('.').pop().toLowerCase()
  if (!ACCEPT.includes(kind)) {
    store.$patch({
      status: 'error',
      error: `${file.name || 'That file'} is not one of ${ACCEPT.map((e) => '.' + e).join(', ')}.`,
    })
    return
  }
  if (store.busy) store.stopPolling()
  loading.value = true
  loadingName.value = file.name
  previewError.value = null
  try {
    if (IMAGE_EXTS.includes(kind)) {
      // a drawing: Blueprint's, whatever tool was up
      buffer.value = null
      stageView.value = 'drawing'
      if (store.tool !== 'blueprint') store.setTool('blueprint')
      await store.uploadImage(file)
      return
    }
    // a mesh dropped while Blueprint is up belongs to the mesh tools
    if (store.tool === 'blueprint') store.setTool('solid')
    if (kind === 'stl' || kind === 'obj') {
      let data
      try {
        // reading can fail when the file moved, is locked, or is too big
        // for the browser to hold at once
        data = await file.arrayBuffer()
      } catch (e) {
        store.$patch({ status: 'error',
                       error: `Could not read ${file.name}: ${friendlyError(e)}` })
        return
      }
      buffer.value = markRaw({ data, kind })
      await store.upload(file)
      return
    }
    // other formats: the server converts them to an STL preview on upload
    buffer.value = null
    await store.upload(file)
    if (store.jobId) {
      try {
        const res = await fetch(`/api/jobs/${store.jobId}/preview`)
        if (!res.ok) throw new Error(`the server sent no preview (${res.status})`)
        buffer.value = markRaw({ data: await res.arrayBuffer(), kind: 'stl' })
      } catch (e) {
        // the preview is optional: the file still converts
        previewError.value = `No 3D preview for this file (${friendlyError(e)}). It can still be converted.`
      }
    }
  } catch (e) {
    store.$patch({ status: 'error', error: `Could not load ${file.name}: ${friendlyError(e)}` })
  } finally {
    loading.value = false
  }
}

function onDrop(e) {
  dragOver.value = false
  takeFile(e.dataTransfer?.files?.[0])
}

// The rail's heading: which tool the controls below belong to
const TOOL_HEAD = {
  solid: { title: 'Mesh → Solid', blurb: 'Find the design intent and write a clean STEP solid.' },
  loft: { title: 'Sliced Loft', blurb: 'Slice along an axis and loft the outlines into a smooth solid.' },
  xray: { title: 'X-Ray', blurb: 'Section sketches between two planes, as a Fusion script.' },
  blueprint: { title: 'Blueprint', blurb: 'A dimensioned drawing read into parametric sketches and extrusions.' },
}
const toolHead = computed(() => TOOL_HEAD[store.tool] || TOOL_HEAD.solid)

const canConvert = computed(
  () => store.jobId && !store.busy && store.status !== 'uploading',
)
</script>

<template>
  <div class="shell">
    <header class="topbar">
      <div class="wordmark">
        <span class="stl">STL</span><span class="arrow">▸</span><span class="solid">SOLID</span>
      </div>
      <p class="tag micro">Mesh in · solid out</p>
      <ToolBar @new="fileInput.click()" />
      <p v-if="build" class="build micro" :title="'built ' + build.built">
        v{{ build.version }} · {{ build.commit }}
        <span v-if="build.built !== 'unknown'" class="when">· {{ build.built }}</span>
      </p>
    </header>

    <div v-if="appError.message" class="banner" role="alert">
      <div class="text">
        <p>{{ appError.message }}</p>
        <details v-if="appError.detail">
          <summary class="micro">details</summary>
          <pre class="num">{{ appError.detail }}</pre>
        </details>
      </div>
      <button class="close" title="Dismiss" @click="clearAppError()">×</button>
    </div>

    <main class="grid">
      <div
        class="stage"
        :class="{ over: dragOver }"
        @dragover.prevent="dragOver = true"
        @dragleave="dragOver = false"
        @drop.prevent="onDrop"
      >
        <div v-if="store.isImageJob" class="stagetabs" role="tablist">
          <button :class="{ on: stageView === 'drawing' }" role="tab" @click="stageView = 'drawing'">Drawing</button>
          <button :class="{ on: stageView === '3d' }" :disabled="!buffer" role="tab"
                  :title="buffer ? '' : 'the 3D preview appears once the part is built'"
                  @click="stageView = '3d'">3D</button>
        </div>
        <img
          v-if="store.isImageJob && stageView === 'drawing'"
          class="drawing" :src="store.imageUrl" :alt="store.filename || 'the drawing'"
        />
        <MeshViewer
          v-if="buffer && (!store.isImageJob || stageView === '3d')"
          :buffer="buffer"
          :unit-scale="store.unitScale"
          :triangle-body="store.triangleBody"
          :selected="store.selected"
          :hovered="store.hovered"
          :planes="store.viewPlanes"
          :sections="store.viewSections"
          @pick="store.toggleBody($event)"
          @hover="store.hovered = $event"
          @error="previewError = $event"
        />
        <p v-if="previewError && (buffer || store.jobId)" class="preview-note micro">
          {{ previewError }}
        </p>
        <div v-if="!buffer && !store.isImageJob" class="dropzone">
          <div class="prism-mark" aria-hidden="true">
            <svg viewBox="0 0 120 100" width="120" height="100">
              <path d="M60 8 L112 82 L8 82 Z" fill="none"
                    stroke="var(--edge)" stroke-width="1.5" />
              <path d="M60 8 L60 82 M60 8 L34 82 M60 8 L86 82"
                    stroke="var(--line)" stroke-width="1" />
            </svg>
          </div>
          <template v-if="loading">
            <p class="big working"><span class="spin edge" aria-hidden="true"></span> Working… reading {{ loadingName }}</p>
          </template>
          <template v-else>
            <p class="big">Drop an STL, OBJ, PLY, OFF, 3MF or GLB here</p>
            <p class="sub">or a dimensioned drawing (JPG / PNG, or paste one) for Blueprint — or</p>
            <button class="browse" @click="fileInput.click()">Choose a file</button>
          </template>
        </div>
        <input
          ref="fileInput" type="file" :accept="ACCEPT_ATTR" hidden
          @change="takeFile($event.target.files[0]); $event.target.value = ''"
        />
      </div>

      <aside class="rail">
        <header class="toolhead">
          <h1>{{ toolHead.title }}</h1>
          <p class="micro">{{ toolHead.blurb }}</p>
        </header>
        <!-- Blueprint has no mesh: no body picker, no units, its own buttons and report -->
        <template v-if="store.tool !== 'blueprint'">
          <BodyPicker />
          <p v-if="store.bodiesError" class="bodies-note micro">
            The body list could not be built ({{ store.bodiesError }}), so the whole
            file will be converted as one selection.
          </p>
          <SetupPanel />
        </template>
        <SolidPanel v-if="store.tool === 'solid'" />
        <LoftPanel v-else-if="store.tool === 'loft'" />
        <BlueprintPanel v-else-if="store.tool === 'blueprint'" />
        <XRayPanel v-else />
        <!-- x-ray never converts and Blueprint has its own buttons: no Convert button here -->
        <template v-if="store.tool !== 'xray' && store.tool !== 'blueprint'">
          <button
            class="convert"
            :disabled="!canConvert"
            @click="store.convert()"
          >
            <span v-if="store.status === 'running'" class="spin" aria-hidden="true"></span>
            {{ store.status === 'running'
               ? (store.serverStatus === 'queued' ? 'Waiting in queue…' : 'Converting…')
               : store.status === 'uploading' ? 'Uploading…'
               : 'Convert to STEP' }}
          </button>
          <button
            v-if="store.status === 'running'"
            class="cancel"
            :disabled="store.cancelling"
            @click="store.cancel()"
          >{{ store.cancelling ? 'Stopping…' : 'Cancel' }}</button>
          <p v-if="store.cancelled" class="cancelled micro">
            Conversion cancelled. Your selection is still here — convert again
            when you are ready.
          </p>
        </template>
        <ReportPanel v-if="store.tool !== 'blueprint'" />
      </aside>
    </main>
  </div>
</template>

<style scoped>
.shell { height: 100%; display: flex; flex-direction: column; }

/* the app-wide error banner: what the global handlers caught */
.banner {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 10px 20px;
  background: rgba(255, 107, 94, 0.12);
  border-bottom: 1px solid var(--fail);
  color: var(--fail);
  font-size: 13px;
}
.banner .text { flex: 1; min-width: 0; }
.banner p { margin: 0; word-break: break-word; }
.banner details { margin-top: 4px; }
.banner summary { cursor: pointer; }
.banner pre {
  margin-top: 4px;
  max-height: 160px;
  overflow: auto;
  font-size: 11px;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--muted);
}
.banner .close {
  background: none;
  border: 1px solid var(--fail);
  color: var(--fail);
  border-radius: 4px;
  width: 26px;
  height: 26px;
  font-size: 16px;
  line-height: 1;
  cursor: pointer;
  flex: 0 0 auto;
}

/* Blueprint's stage: the drawing, and the Drawing | 3D switch */
.drawing {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: var(--ink);
  display: block;
}
.stagetabs {
  position: absolute;
  top: 12px;
  left: 14px;
  z-index: 3;
  display: flex;
  gap: 2px;
  padding: 3px;
  background: rgba(20, 23, 28, 0.85);
  border: 1px solid var(--line);
  border-radius: 6px;
}
.stagetabs button {
  height: 26px;
  padding: 0 10px;
  background: none;
  border: none;
  border-radius: 4px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.stagetabs button:hover:not(:disabled) { color: var(--text); background: var(--panel-2); }
.stagetabs button.on { color: var(--edge); background: var(--panel-2); }
.stagetabs button:disabled { opacity: 0.5; cursor: default; }
.stage:has(.stagetabs) .preview-note { top: 46px; }

.preview-note {
  position: absolute;
  top: 12px;
  left: 14px;
  right: 14px;
  z-index: 2;
  text-transform: none;
  letter-spacing: 0;
  font-size: 12px;
  color: var(--warn, #f0ad4e);
  background: rgba(20, 23, 28, 0.85);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 6px 10px;
  pointer-events: none;
}
.bodies-note {
  text-transform: none;
  letter-spacing: 0;
  font-size: 12px;
  color: var(--warn, #f0ad4e);
  line-height: 1.45;
}

.topbar {
  display: flex;
  align-items: baseline;
  gap: 16px;
  padding: 12px 20px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
}
.topbar > nav { align-self: center; margin-left: 8px; }
.wordmark {
  font-weight: 800;
  font-size: 17px;
  letter-spacing: 0.14em;
  font-stretch: 125%;
}
.build {
  margin-left: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: var(--edge);
  opacity: 0.85;
  white-space: nowrap;
}
.build .when { opacity: 0.7; }
.wordmark .arrow { color: var(--edge); margin: 0 4px; }
.wordmark .solid { color: var(--edge); }

.grid {
  flex: 1;
  min-height: 0;
  display: grid;
  /* The rail holds the controls and must stay legible at any width, so it
     keeps a floor and the stage takes what is left (minmax(0,1fr), not
     1fr: a grid track's default min-content floor would let the canvas
     push the rail off screen instead of shrinking). */
  grid-template-columns: minmax(0, 1fr) clamp(300px, 28vw, 380px);
}

.stage {
  position: relative;
  min-height: 0;
  border-right: 1px solid var(--line);
}
.stage.over::after {
  content: 'Drop to load';
  position: absolute;
  inset: 10px;
  display: grid;
  place-items: center;
  border: 2px dashed var(--edge);
  border-radius: 8px;
  color: var(--edge);
  font-weight: 700;
  background: rgba(90, 210, 234, 0.06);
  pointer-events: none;
}

.dropzone {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
.big { font-size: 22px; font-weight: 700; }
.big.working { display: flex; align-items: center; gap: 12px; color: var(--edge); }
.big.working .spin { width: 20px; height: 20px; border-width: 3px; }
.sub { color: var(--muted); font-size: 12px; }
.browse {
  background: none;
  border: 1px solid var(--edge);
  color: var(--edge);
  border-radius: 6px;
  padding: 8px 18px;
  font-weight: 600;
}
.browse:hover { background: rgba(90, 210, 234, 0.1); }


.rail {
  padding: 16px;
  min-height: 0;
  overflow-y: auto;
  background: var(--panel);
  display: flex;
  flex-direction: column;
  gap: 16px;
}
/* A flex column shrinks its children to fit by default, which squeezed the
   body list to nothing and overlapped the panel headings. The rail scrolls
   instead: every child keeps the height its content needs. */
.rail > * { flex: 0 0 auto; }

/* the tool's name over its controls, in the edge cyan so it reads first */
.toolhead {
  padding-bottom: 12px;
  border-bottom: 2px solid var(--edge);
}
.toolhead h1 {
  font-size: 20px;
  font-weight: 800;
  letter-spacing: 0.04em;
  color: var(--edge);
  line-height: 1.2;
}
.toolhead .micro { margin-top: 4px; text-transform: none; letter-spacing: 0; font-size: 12px; }

.convert {
  padding: 12px;
  border-radius: 6px;
  border: none;
  background: var(--edge);
  color: var(--ink);
  font-weight: 800;
  font-size: 14px;
  letter-spacing: 0.04em;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
.convert:disabled { background: var(--line); color: var(--muted); cursor: default; }

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
.cancelled { color: var(--muted); line-height: 1.45; }
.convert:not(:disabled):hover { filter: brightness(1.1); }


/* Narrow: stack, and let the viewer shrink rather than the controls. The
   stage takes a share of the height with a floor; the rail keeps its own
   scroll so every control stays reachable. */
@media (max-width: 900px) {
  .grid {
    grid-template-columns: minmax(0, 1fr);
    grid-template-rows: minmax(180px, 40vh) minmax(0, 1fr);
  }
  .stage { border-right: none; border-bottom: 1px solid var(--line); }
}
@media (max-width: 900px) and (max-height: 560px) {
  .grid { grid-template-rows: minmax(140px, 32vh) minmax(0, 1fr); }
}
</style>
