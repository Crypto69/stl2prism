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
})

const host = ref(null)
const rawSize = ref(null)  // in file units
const dims = computed(() =>
  rawSize.value && rawSize.value.map((v) => (v * props.unitScale).toFixed(1)))

let renderer, scene, camera, controls, mesh, grid, frameId, resizeObs

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
    <div class="hint micro">drag to rotate · scroll to zoom</div>
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
