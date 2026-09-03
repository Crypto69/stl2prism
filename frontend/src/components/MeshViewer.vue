<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { STLLoader } from 'three/addons/loaders/STLLoader.js'
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js'
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

const props = defineProps({
  // { data: ArrayBuffer, kind: 'stl' | 'obj' } (non-reactive)
  buffer: { type: Object, default: null },
  // file units -> mm; the callout shows real-world size, the geometry is
  // rendered as-is (the camera fits it either way)
  unitScale: { type: Number, default: 1 },
  // triangle index -> body index, from /api/jobs/{id}/bodies. When set,
  // each body is drawn in its own colour and can be clicked.
  triangleBody: { type: Array, default: null },
  // body indices currently ticked for conversion
  selected: { type: Array, default: () => [] },
  // body index under the cursor in the list, highlighted in the scene
  hovered: { type: Number, default: -1 },
})
const emit = defineEmits(['pick', 'hover'])

const host = ref(null)
const rawSize = ref(null)  // in file units
const dims = computed(() =>
  rawSize.value && rawSize.value.map((v) => (v * props.unitScale).toFixed(1)))

let renderer, scene, camera, controls, mesh, grid, frameId, resizeObs
let raycaster, pointer, triToBody = null, faceStart = null

// One hue per body, walked by the golden angle so neighbours never share a
// colour. Unselected bodies keep the hue but drop to a low saturation, so
// what will be converted reads at a glance without hiding the rest.
const PICKED = new THREE.Color(), MUTED = new THREE.Color(), HOVER = new THREE.Color(0xffffff)
function bodyColor(i, isSelected, isHovered) {
  if (isHovered) return HOVER
  const hue = (i * 0.381966) % 1
  return isSelected
    ? PICKED.setHSL(hue, 0.62, 0.62)
    : MUTED.setHSL(hue, 0.10, 0.30)
}

onMounted(() => {
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x14171c)

  camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000)
  camera.position.set(80, 60, 80)

  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  host.value.appendChild(renderer.domElement)

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true

  scene.add(new THREE.HemisphereLight(0xdde4ee, 0x2a2f38, 1.1))
  const key = new THREE.DirectionalLight(0xffffff, 1.4)
  key.position.set(1, 2, 1.5)
  scene.add(key)
  const rim = new THREE.DirectionalLight(0x5ad2ea, 0.25)
  rim.position.set(-2, -1, -1)
  scene.add(rim)

  raycaster = new THREE.Raycaster()
  pointer = new THREE.Vector2()
  renderer.domElement.addEventListener('pointerdown', onPointerDown)
  renderer.domElement.addEventListener('pointerup', onPointerUp)
  renderer.domElement.addEventListener('pointermove', onPointerMove)
  renderer.domElement.addEventListener('pointerleave', () => emit('hover', -1))

  resizeObs = new ResizeObserver(resize)
  resizeObs.observe(host.value)
  resize()
  animate()
  if (props.buffer) loadMesh(props.buffer)
})

watch(() => props.buffer, (b) => { if (b) loadMesh(b) })

// OBJLoader yields a Group of Meshes (one per object/material), each with
// whatever attributes the file had (normal, uv, color). Reduce that to one
// position-only geometry so the rest of the viewer treats it like an STL.
function parseObj(data) {
  const group = new OBJLoader().parse(new TextDecoder().decode(data))
  const parts = []
  group.traverse((o) => {
    if (!o.isMesh || !o.geometry?.getAttribute('position')) return
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', o.geometry.getAttribute('position'))
    parts.push(g)
  })
  if (!parts.length) throw new Error('OBJ contains no faces')
  return parts.length === 1 ? parts[0] : mergeGeometries(parts, false)
}

// Paint every triangle from its body's colour. The geometry is
// non-indexed (STL always, OBJ after the merge above), so triangle t owns
// vertices 3t..3t+2 and one pass over the array is enough.
function paintBodies() {
  if (!mesh || !triToBody) return
  const pos = mesh.geometry.getAttribute('position')
  const nTri = pos.count / 3
  let attr = mesh.geometry.getAttribute('color')
  if (!attr || attr.count !== pos.count) {
    attr = new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3)
    mesh.geometry.setAttribute('color', attr)
  }
  const picked = new Set(props.selected)
  for (let t = 0; t < nTri; t++) {
    const b = triToBody[t]
    const c = bodyColor(b, picked.has(b), b === props.hovered)
    for (let k = 0; k < 3; k++) attr.setXYZ(t * 3 + k, c.r, c.g, c.b)
  }
  attr.needsUpdate = true
  mesh.material.vertexColors = true
  mesh.material.color.set(0xffffff)
  mesh.material.needsUpdate = true
}

// Which body is under the pointer, or -1.
function bodyAt(ev) {
  if (!mesh || !triToBody || !raycaster) return -1
  const r = renderer.domElement.getBoundingClientRect()
  pointer.set(((ev.clientX - r.left) / r.width) * 2 - 1,
              -((ev.clientY - r.top) / r.height) * 2 + 1)
  raycaster.setFromCamera(pointer, camera)
  const hit = raycaster.intersectObject(mesh, false)[0]
  if (!hit || hit.faceIndex == null) return -1
  const b = triToBody[hit.faceIndex]
  return b == null ? -1 : b
}

// A drag that orbits must not also toggle a body, so only a press and
// release in nearly the same place counts as a click.
let downAt = null
function onPointerDown(ev) { downAt = [ev.clientX, ev.clientY] }
function onPointerUp(ev) {
  if (!downAt) return
  const moved = Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1])
  downAt = null
  if (moved > 4) return
  const b = bodyAt(ev)
  if (b >= 0) emit('pick', b)
}
let hoverRaf = 0
function onPointerMove(ev) {
  if (!triToBody || hoverRaf) return
  hoverRaf = requestAnimationFrame(() => {
    hoverRaf = 0
    const b = bodyAt(ev)
    if (b !== props.hovered) emit('hover', b)
    renderer.domElement.style.cursor = b >= 0 ? 'pointer' : ''
  })
}

watch(() => props.triangleBody, (t) => {
  triToBody = t && t.length ? t : null
  if (!triToBody && mesh) {
    mesh.geometry.deleteAttribute('color')
    mesh.material.vertexColors = false
    mesh.material.color.set(0x9aa7b5)
    mesh.material.needsUpdate = true
  }
  paintBodies()
}, { immediate: true })
watch(() => [props.selected, props.hovered], paintBodies, { deep: true })

function loadMesh({ data, kind }) {
  if (mesh) {
    scene.remove(mesh)
    mesh.geometry.dispose()
    mesh.material.dispose()
    mesh = null
  }
  if (grid) { scene.remove(grid); grid.dispose(); grid = null }

  let geo
  try {
    geo = kind === 'obj' ? parseObj(data) : new STLLoader().parse(data)
  } catch {
    rawSize.value = null
    return
  }
  geo.computeBoundingBox()
  const bb = geo.boundingBox
  const size = new THREE.Vector3()
  bb.getSize(size)
  rawSize.value = [size.x, size.y, size.z]

  // Assume Z-up (STL convention; CAD OBJ exports usually match). The
  // viewport floor is Y-up. Center on the floor. A Y-up OBJ (e.g. from
  // Blender) previews rotated, which is cosmetic: the pipeline finds
  // the extrusion axis itself.
  const center = new THREE.Vector3()
  bb.getCenter(center)
  geo.translate(-center.x, -center.y, -bb.min.z)
  geo.computeVertexNormals()

  const mat = new THREE.MeshStandardMaterial({
    color: 0x9aa7b5, metalness: 0.25, roughness: 0.55, flatShading: true,
  })
  mesh = new THREE.Mesh(geo, mat)
  mesh.rotation.x = -Math.PI / 2
  scene.add(mesh)
  paintBodies()

  const span = Math.max(size.x, size.y, size.z)
  grid = new THREE.GridHelper(span * 3, 30, 0x3a4350, 0x242a33)
  scene.add(grid)

  const d = span * 1.6
  camera.position.set(d, d * 0.75, d)
  camera.near = span / 100
  camera.far = span * 20
  camera.updateProjectionMatrix()
  controls.target.set(0, size.z / 2, 0)
  controls.update()
}

function resize() {
  if (!host.value) return
  const w = host.value.clientWidth
  const h = host.value.clientHeight
  if (!w || !h) return
  renderer.setSize(w, h)
  camera.aspect = w / h
  camera.updateProjectionMatrix()
}

function animate() {
  frameId = requestAnimationFrame(animate)
  controls.update()
  renderer.render(scene, camera)
}

onBeforeUnmount(() => {
  cancelAnimationFrame(frameId)
  cancelAnimationFrame(hoverRaf)
  renderer?.domElement.removeEventListener('pointerdown', onPointerDown)
  renderer?.domElement.removeEventListener('pointerup', onPointerUp)
  renderer?.domElement.removeEventListener('pointermove', onPointerMove)
  resizeObs?.disconnect()
  controls?.dispose()
  renderer?.dispose()
})
</script>

<template>
  <div class="viewer" ref="host">
    <div v-if="dims" class="callout num">
      {{ dims[0] }} × {{ dims[1] }} × {{ dims[2] }} mm
    </div>
    <div class="hint micro">
      drag to rotate · scroll to zoom<span v-if="triangleBody"> · click a body to select it</span>
    </div>
  </div>
</template>

<style scoped>
.viewer {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
}
.viewer :deep(canvas) { display: block; }
.callout {
  position: absolute;
  top: 12px;
  left: 14px;
  font-size: 13px;
  color: var(--edge);
  background: rgba(20, 23, 28, 0.75);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 4px 10px;
}
.hint {
  position: absolute;
  bottom: 10px;
  right: 14px;
  pointer-events: none;
}
</style>
