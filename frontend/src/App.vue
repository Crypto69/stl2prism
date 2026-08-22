<script setup>
import { computed, markRaw, onMounted, ref, shallowRef } from 'vue'
import MeshViewer from './components/MeshViewer.vue'
import ParamsPanel from './components/ParamsPanel.vue'
import ReportPanel from './components/ReportPanel.vue'
import { useConvertStore } from './store'

const store = useConvertStore()
const buffer = shallowRef(null)
const dragOver = ref(false)
const fileInput = ref(null)

const ACCEPT = ['stl', 'obj', 'ply', 'off', '3mf', 'glb', 'gltf']

// Which build is running: package version + git commit + build time from
// /api/version, so a tester can match the browser to a commit at a glance.
const build = ref(null)
onMounted(async () => {
  try {
    const res = await fetch('/api/version')
    if (res.ok) build.value = await res.json()
  } catch (e) { /* badge is optional */ }
})

async function takeFile(file) {
  const kind = file?.name.split('.').pop().toLowerCase()
  if (!file || !ACCEPT.includes(kind)) {
    store.$patch({
      status: 'error',
      error: `That is not one of ${ACCEPT.map((e) => '.' + e).join(', ')}.`,
    })
    return
  }
  if (kind === 'stl' || kind === 'obj') {
    buffer.value = markRaw({ data: await file.arrayBuffer(), kind })
    store.upload(file)
    return
  }
  // other formats: the server converts them to an STL preview on upload
  buffer.value = null
  await store.upload(file)
  if (store.jobId) {
    try {
      const res = await fetch(`/api/jobs/${store.jobId}/preview`)
      if (res.ok) buffer.value = markRaw({ data: await res.arrayBuffer(), kind: 'stl' })
    } catch (e) { /* preview is optional */ }
  }
}

function onDrop(e) {
  dragOver.value = false
  takeFile(e.dataTransfer?.files?.[0])
}

const canConvert = computed(
  () => store.jobId && !store.busy && store.status !== 'uploading',
)
</script>

<template>
  <div class="shell">
    <header class="topbar">
      <div class="wordmark">
        <span class="stl">STL</span><span class="arrow">▸</span><span class="prism">PRISM</span>
      </div>
      <p class="tag micro">Mesh in · machined solid out</p>
      <p v-if="build" class="build micro" :title="'built ' + build.built">
        v{{ build.version }} · {{ build.commit }}
        <span v-if="build.built !== 'unknown'" class="when">· {{ build.built }}</span>
      </p>
    </header>

    <main class="grid">
      <div
        class="stage"
        :class="{ over: dragOver }"
        @dragover.prevent="dragOver = true"
        @dragleave="dragOver = false"
        @drop.prevent="onDrop"
      >
        <MeshViewer v-if="buffer" :buffer="buffer" :unit-scale="store.unitScale" />
        <div v-else class="dropzone">
          <div class="prism-mark" aria-hidden="true">
            <svg viewBox="0 0 120 100" width="120" height="100">
              <path d="M60 8 L112 82 L8 82 Z" fill="none"
                    stroke="var(--edge)" stroke-width="1.5" />
              <path d="M60 8 L60 82 M60 8 L34 82 M60 8 L86 82"
                    stroke="var(--line)" stroke-width="1" />
            </svg>
          </div>
          <p class="big">Drop an STL, OBJ, PLY, OFF, 3MF or GLB here</p>
          <p class="sub">or</p>
          <button class="browse" @click="fileInput.click()">Choose a file</button>
        </div>
        <button
          v-if="buffer"
          class="replace micro"
          @click="fileInput.click()"
        >Replace file</button>
        <input
          ref="fileInput" type="file" accept=".stl,.obj,.ply,.off,.3mf,.glb,.gltf" hidden
          @change="takeFile($event.target.files[0]); $event.target.value = ''"
        />
      </div>

      <aside class="rail">
        <ParamsPanel />
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
        <ReportPanel />
      </aside>
    </main>
  </div>
</template>

<style scoped>
.shell { height: 100%; display: flex; flex-direction: column; }

.topbar {
  display: flex;
  align-items: baseline;
  gap: 16px;
  padding: 12px 20px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
}
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
.wordmark .prism { color: var(--edge); }

.grid {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 1fr 360px;
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

.replace {
  position: absolute;
  top: 12px;
  right: 14px;
  background: rgba(20, 23, 28, 0.75);
  border: 1px solid var(--line);
  border-radius: 4px;
  color: var(--muted);
  padding: 4px 10px;
}
.replace:hover { color: var(--text); border-color: var(--muted); }

.rail {
  padding: 16px;
  overflow-y: auto;
  background: var(--panel);
  display: flex;
  flex-direction: column;
  gap: 16px;
}

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
.convert:not(:disabled):hover { filter: brightness(1.1); }

.spin {
  width: 13px;
  height: 13px;
  border: 2px solid rgba(20, 23, 28, 0.3);
  border-top-color: var(--ink);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

@media (max-width: 900px) {
  .grid { grid-template-columns: 1fr; grid-template-rows: 55vh auto; }
  .stage { border-right: none; border-bottom: 1px solid var(--line); }
}
</style>
