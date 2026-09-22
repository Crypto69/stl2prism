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
  // 'x' | 'y' | 'z' when the sliced loft is chosen: the viewer draws a
  // labelled XYZ triad (Fusion's colours) and a translucent slicing
  // plane across that axis, so the dropdown's letter has a picture
  sliceAxis: { type: String, default: null },
  // where the single slice sits: mm (converted units) from the box centre
  sliceOffset: { type: Number, default: 0 },
  // the traced section from /section: { polylines: [[[x,y,z]...]...] } in
  // converted mm, drawn as bright curves on the plane
  section: { type: Object, default: null },
})
const emit = defineEmits(['pick', 'hover'])

const host = ref(null)
const rawSize = ref(null)  // in file units
const dims = computed(() =>
  rawSize.value && rawSize.value.map((v) => (v * props.unitScale).toFixed(1)))

let renderer, scene, camera, controls, mesh, grid, frameId, resizeObs
let raycaster, pointer, triToBody = null, faceStart = null
// `frame` is the part's own frame (file Z up -> scene Y up): the mesh and
// every helper hang off it, so the mesh can be hidden on its own
let frame = null
// helpers drawn in the part's frame: the triad, the slice plane, the trace
let triad = null, slicePlane = null, localBox = null, sectionLines = null
// view tools (Fusion's bottom toolbar): which mouse tool the left button
// drives, and whether the mesh is shown
const tool = ref('orbit')          // 'orbit' | 'pan' | 'zoom'
const modelVisible = ref(true)
const gridVisible = ref(true)      // the floor grid the part sits on
const TOOL_BUTTON = { orbit: THREE.MOUSE.ROTATE, pan: THREE.MOUSE.PAN, zoom: THREE.MOUSE.DOLLY }
const TOOL_CURSOR = { orbit: '', pan: 'grab', zoom: 'ns-resize' }
// the ViewCube: its own little scene, drawn into a corner of the same canvas
let cubeScene, cubeCamera, cube, cubeFaces = [], cubeHover = -1, snapAnim = null
const CUBE_PX = 132, CUBE_MARGIN = 12
let meshShift = null   // file-frame translation applied to the geometry

// Fusion's axis colours: X red, Y green, Z blue
const AXIS_COLOR = { x: 0xe5484d, y: 0x46b95a, z: 0x4c8dff }

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
  // zoom towards the point under the cursor (wheel, and the Zoom tool's
  // drag), as Fusion does; right-drag pans and the wheel zooms in every tool
  controls.zoomToCursor = true
  applyTool()

  frame = new THREE.Group()
  frame.rotation.x = -Math.PI / 2
  scene.add(frame)

  scene.add(new THREE.HemisphereLight(0xdde4ee, 0x2a2f38, 1.1))
  const key = new THREE.DirectionalLight(0xffffff, 1.4)
  key.position.set(1, 2, 1.5)
  scene.add(key)
  const rim = new THREE.DirectionalLight(0x5ad2ea, 0.25)
  rim.position.set(-2, -1, -1)
  scene.add(rim)

  buildViewCube()

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

// The left mouse button does what the chosen tool says; OrbitControls
// reads the mapping on every press.
function applyTool() {
  if (!controls) return
  controls.mouseButtons.LEFT = TOOL_BUTTON[tool.value]
  if (renderer) renderer.domElement.style.cursor = TOOL_CURSOR[tool.value]
}
watch(tool, applyTool)

function setModelVisible(on) {
  modelVisible.value = on
  if (mesh) mesh.visible = on
  if (!on && props.hovered >= 0) emit('hover', -1)
}

function setGridVisible(on) {
  gridVisible.value = on
  if (grid) grid.visible = on
}

// Which body is under the pointer, or -1. A hidden mesh is not pickable
// (the Raycaster itself does not look at visibility).
function bodyAt(ev) {
  if (!mesh || !mesh.visible || !triToBody || !raycaster) return -1
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
function onPointerDown(ev) {
  downAt = [ev.clientX, ev.clientY]
  // three r170's _handleMouseDownDolly passes clientX twice, so the Zoom
  // tool's drag would anchor at the wrong height; this listener runs after
  // OrbitControls' own and sets the anchor properly
  if (tool.value === 'zoom' && ev.button === 0 && controls._updateZoomParameters) {
    controls._updateZoomParameters(ev.clientX, ev.clientY)
  }
  if (tool.value === 'pan' && ev.button === 0) renderer.domElement.style.cursor = 'grabbing'
}
function onPointerUp(ev) {
  if (tool.value === 'pan') renderer.domElement.style.cursor = TOOL_CURSOR.pan
  if (!downAt) return
  const moved = Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1])
  downAt = null
  if (moved > 4) return
  const f = cubeFaceAt(ev)
  if (f >= 0) { snapToFace(f); return }
  const b = bodyAt(ev)
  if (b >= 0) emit('pick', b)
}
let hoverRaf = 0
function onPointerMove(ev) {
  const f = cubeFaceAt(ev)
  if (f !== cubeHover) {
    cubeHover = f
    cubeFaces.forEach((m, i) => { m.color.set(i === f ? 0x5ad2ea : 0xffffff) })
  }
  if (f >= 0) { renderer.domElement.style.cursor = 'pointer'; return }
  if (!triToBody || hoverRaf) {
    if (!triToBody) renderer.domElement.style.cursor = downAt && tool.value === 'pan' ? 'grabbing' : TOOL_CURSOR[tool.value]
    return
  }
  hoverRaf = requestAnimationFrame(() => {
    hoverRaf = 0
    const b = bodyAt(ev)
    if (b !== props.hovered) emit('hover', b)
    renderer.domElement.style.cursor = b >= 0 ? 'pointer'
      : downAt && tool.value === 'pan' ? 'grabbing' : TOOL_CURSOR[tool.value]
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
    if (triad) { frame.remove(triad); triad = null }
    if (slicePlane) { frame.remove(slicePlane); slicePlane = null }
    if (sectionLines) { frame.remove(sectionLines); sectionLines = null }
    frame.remove(mesh)
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
  // remember the shift before translating: geo.translate() also moves
  // geo.boundingBox (bb), so anything read from bb afterwards is already
  // in the local frame
  meshShift = new THREE.Vector3(-center.x, -center.y, -bb.min.z)
  geo.translate(-center.x, -center.y, -bb.min.z)
  geo.computeBoundingBox()
  localBox = geo.boundingBox.clone()
  geo.computeVertexNormals()

  const mat = new THREE.MeshStandardMaterial({
    color: 0x9aa7b5, metalness: 0.25, roughness: 0.55, flatShading: true,
  })
  mesh = new THREE.Mesh(geo, mat)
  mesh.visible = modelVisible.value
  frame.add(mesh)
  paintBodies()
  drawTriad()
  drawSlicePlane()
  drawSection()

  const span = Math.max(size.x, size.y, size.z)
  grid = new THREE.GridHelper(span * 3, 30, 0x3a4350, 0x242a33)
  grid.visible = gridVisible.value
  scene.add(grid)

  camera.near = span / 100
  camera.far = span * 20
  camera.updateProjectionMatrix()
  // a fresh load is a fit from the usual three-quarter view
  camera.position.set(span, span * 0.75, span)
  controls.target.set(0, size.z / 2, 0)
  controls.update()
  zoomToFit(false)
}

// Frame the whole part: keep the view direction, move the orbit target to
// the part's centre and back off until its bounding sphere fills the
// narrower side of the view. Animated unless `animate` is false.
function zoomToFit(animate = true) {
  if (!mesh) return
  const geo = mesh.geometry
  if (!geo.boundingSphere) geo.computeBoundingSphere()
  frame.updateMatrixWorld(true)
  const centre = geo.boundingSphere.center.clone().applyMatrix4(mesh.matrixWorld)
  const radius = geo.boundingSphere.radius * 1.12
  const fovV = THREE.MathUtils.degToRad(camera.fov)
  const fovH = 2 * Math.atan(Math.tan(fovV / 2) * camera.aspect)
  const dist = radius / Math.sin(Math.min(fovV, fovH) / 2)
  const from = camera.position.clone().sub(controls.target)
  const dir = from.clone().normalize()
  if (!animate) {
    controls.target.copy(centre)
    camera.position.copy(centre).addScaledVector(dir, dist)
    camera.lookAt(centre)
    controls.update()
    return
  }
  snapAnim = { from: dir, to: dir.clone(), fromDist: from.length(), toDist: dist,
               fromTarget: controls.target.clone(), toTarget: centre,
               t0: performance.now(), ms: 320 }
}

function textSprite(text, color) {
  const c = document.createElement('canvas')
  c.width = c.height = 64
  const g = c.getContext('2d')
  g.font = 'bold 44px system-ui, sans-serif'
  g.textAlign = 'center'
  g.textBaseline = 'middle'
  g.fillStyle = '#' + color.toString(16).padStart(6, '0')
  g.fillText(text, 32, 34)
  const tex = new THREE.CanvasTexture(c)
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }))
  return sp
}

// Labelled XYZ arrows at the file's box corner, in the file's own frame
// (children of the rotated frame group, so file Z points up like the part).
function drawTriad() {
  if (triad) { frame?.remove(triad); triad = null }
  if (!mesh || !localBox) return
  const size = new THREE.Vector3()
  localBox.getSize(size)
  const span = Math.max(size.x, size.y, size.z)
  const L = span * 0.35
  const origin = new THREE.Vector3(localBox.min.x - span * 0.08, localBox.min.y - span * 0.08, 0)
  triad = new THREE.Group()
  const dirs = { x: new THREE.Vector3(1, 0, 0), y: new THREE.Vector3(0, 1, 0), z: new THREE.Vector3(0, 0, 1) }
  for (const k of ['x', 'y', 'z']) {
    const on = props.sliceAxis === k
    const arrow = new THREE.ArrowHelper(dirs[k], origin, L, AXIS_COLOR[k], L * 0.14, L * 0.07)
    arrow.line.material.linewidth = 2
    if (props.sliceAxis && !on) {
      arrow.line.material.transparent = arrow.cone.material.transparent = true
      arrow.line.material.opacity = arrow.cone.material.opacity = 0.45
    }
    triad.add(arrow)
    const label = textSprite(k.toUpperCase(), AXIS_COLOR[k])
    label.position.copy(origin).addScaledVector(dirs[k], L * 1.12)
    const ls = span * (on ? 0.09 : 0.065)
    label.scale.set(ls, ls, 1)
    triad.add(label)
  }
  frame.add(triad)
}

// One translucent slice across the chosen axis, in that axis's colour, at
// the slider's position: the plane the single-slice trace is cut on.
function drawSlicePlane() {
  if (slicePlane) {
    frame?.remove(slicePlane)
    slicePlane.traverse((o) => { o.geometry?.dispose(); o.material?.dispose?.() })
    slicePlane = null
  }
  if (!mesh || !localBox || !props.sliceAxis) return
  const k = props.sliceAxis
  const size = new THREE.Vector3()
  const center = new THREE.Vector3()
  localBox.getSize(size)
  localBox.getCenter(center)
  const pad = 1.15
  // plane geometry lies in XY facing +Z; rotate it to face the axis
  const w = k === 'x' ? size.y : size.x
  const h = k === 'z' ? size.y : size.z
  slicePlane = new THREE.Group()
  // the plane sits at the slider's offset (converted mm -> file units),
  // clamped to the box
  const off = (props.sliceOffset || 0) / (props.unitScale || 1)
  const f = Math.max(-0.5, Math.min(0.5, size[k] > 0 ? off / size[k] : 0))
  const geo = new THREE.PlaneGeometry(w * pad, h * pad)
  const mat = new THREE.MeshBasicMaterial({
    color: AXIS_COLOR[k], transparent: true, opacity: 0.3,
    side: THREE.DoubleSide, depthWrite: false,
  })
  const pl = new THREE.Mesh(geo, mat)
  if (k === 'x') pl.rotation.y = Math.PI / 2
  else if (k === 'y') pl.rotation.x = Math.PI / 2
  pl.position.copy(center)
  pl.position[k] = center[k] + f * size[k]
  slicePlane.add(pl)
  const edge = new THREE.LineSegments(new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({ color: AXIS_COLOR[k], transparent: true, opacity: 0.9 }))
  edge.rotation.copy(pl.rotation)
  edge.position.copy(pl.position)
  slicePlane.add(edge)
  frame.add(slicePlane)
}

// The traced outline: bright lines in the file frame (converted mm back
// to file units, then the same shift the geometry got).
function drawSection() {
  if (sectionLines) {
    frame?.remove(sectionLines)
    sectionLines.traverse((o) => { o.geometry?.dispose(); o.material?.dispose?.() })
    sectionLines = null
  }
  if (!mesh || !meshShift || !props.section?.polylines?.length || !props.sliceAxis) return
  const k = 1 / (props.unitScale || 1)
  sectionLines = new THREE.Group()
  const mat = new THREE.LineBasicMaterial({ color: 0xffd166, depthTest: false, transparent: true, opacity: 0.95 })
  for (const poly of props.section.polylines) {
    const pts = poly.map(([x, y, z]) => new THREE.Vector3(x * k + meshShift.x, y * k + meshShift.y, z * k + meshShift.z))
    sectionLines.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat))
  }
  sectionLines.renderOrder = 10
  frame.add(sectionLines)
}

watch(() => props.sliceAxis, () => { drawTriad(); drawSlicePlane(); drawSection() })
watch(() => props.sliceOffset, drawSlicePlane)
watch(() => props.section, drawSection)

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
  if (snapAnim) stepSnap()
  controls.update()
  renderer.render(scene, camera)
  drawViewCube()
}

// ---- ViewCube -----------------------------------------------------------
// Fusion's cube: it turns with the view, a click on a face snaps the view
// to that face, and dragging on it orbits (dragging anywhere does). The
// labels are in the part's own frame: the file's Z is up, so TOP is +Z,
// FRONT looks from the file's -Y side. The mesh is rotated x -90 deg in
// the scene, so file (x, y, z) sits at world (x, z, -y).
const CUBE_FACES = [
  // [world normal, label, file axis it faces]
  [[1, 0, 0], 'RIGHT'], [[-1, 0, 0], 'LEFT'],
  [[0, 1, 0], 'TOP'], [[0, -1, 0], 'BOTTOM'],
  [[0, 0, 1], 'FRONT'], [[0, 0, -1], 'BACK'],
]

function faceTexture(label) {
  const c = document.createElement('canvas')
  c.width = c.height = 128
  const g = c.getContext('2d')
  g.fillStyle = '#e8ecf1'
  g.fillRect(0, 0, 128, 128)
  g.strokeStyle = '#9aa7b5'
  g.lineWidth = 3
  g.strokeRect(1.5, 1.5, 125, 125)
  g.fillStyle = '#2b3138'
  g.font = 'bold 26px system-ui, sans-serif'
  g.textAlign = 'center'
  g.textBaseline = 'middle'
  g.fillText(label, 64, 66)
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  return t
}

function buildViewCube() {
  cubeScene = new THREE.Scene()
  cubeCamera = new THREE.PerspectiveCamera(35, 1, 0.1, 20)
  cubeScene.add(new THREE.AmbientLight(0xffffff, 1.6))
  const dl = new THREE.DirectionalLight(0xffffff, 0.6)
  dl.position.set(1, 2, 1.5)
  cubeScene.add(dl)
  // three's BoxGeometry material order: +x, -x, +y, -y, +z, -z = CUBE_FACES
  // the texture is multiplied by the colour: light faces, cyan on hover
  cubeFaces = CUBE_FACES.map(([, label]) =>
    new THREE.MeshLambertMaterial({ map: faceTexture(label), color: 0xffffff }))
  cube = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), cubeFaces)
  cubeScene.add(cube)
  cubeScene.add(new THREE.LineSegments(new THREE.EdgesGeometry(cube.geometry),
    new THREE.LineBasicMaterial({ color: 0x14171c })))
  // the part's axes from the cube's near-bottom-left corner: file X red
  // (world +x), file Y green (world -z), file Z blue (world +y)
  const o = new THREE.Vector3(-0.5, -0.5, 0.5)
  const L = 0.9
  const axes = [
    [new THREE.Vector3(1, 0, 0), AXIS_COLOR.x, 'X'],
    [new THREE.Vector3(0, 0, -1), AXIS_COLOR.y, 'Y'],
    [new THREE.Vector3(0, 1, 0), AXIS_COLOR.z, 'Z'],
  ]
  for (const [d, col, name] of axes) {
    const g = new THREE.BufferGeometry().setFromPoints([o, o.clone().addScaledVector(d, L)])
    cubeScene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: col })))
    const label = textSprite(name, col)
    label.position.copy(o).addScaledVector(d, L + 0.16)
    label.scale.set(0.28, 0.28, 1)
    cubeScene.add(label)
  }
}

function drawViewCube() {
  if (!cubeScene || !controls) return
  const dir = camera.position.clone().sub(controls.target).normalize()
  cubeCamera.position.copy(dir).multiplyScalar(3.0)
  cubeCamera.up.copy(camera.up)
  cubeCamera.lookAt(0, 0, 0)
  const h = renderer.domElement.clientHeight
  renderer.autoClear = false
  renderer.clearDepth()
  renderer.setScissorTest(true)
  renderer.setViewport(CUBE_MARGIN, CUBE_MARGIN, CUBE_PX, CUBE_PX)
  renderer.setScissor(CUBE_MARGIN, CUBE_MARGIN, CUBE_PX, CUBE_PX)
  renderer.render(cubeScene, cubeCamera)
  renderer.setScissorTest(false)
  renderer.setViewport(0, 0, renderer.domElement.clientWidth, h)
  renderer.autoClear = true
}

// Index of the cube face under the pointer, or -1 (outside the corner or
// off the cube).
function cubeFaceAt(ev) {
  if (!cube || !renderer) return -1
  const r = renderer.domElement.getBoundingClientRect()
  const x = ev.clientX - r.left
  const y = r.height - (ev.clientY - r.top)          // from the bottom, like the viewport
  if (x < CUBE_MARGIN || x > CUBE_MARGIN + CUBE_PX || y < CUBE_MARGIN || y > CUBE_MARGIN + CUBE_PX) return -1
  const nx = ((x - CUBE_MARGIN) / CUBE_PX) * 2 - 1
  const ny = ((y - CUBE_MARGIN) / CUBE_PX) * 2 - 1
  const rc = new THREE.Raycaster()
  rc.setFromCamera(new THREE.Vector2(nx, ny), cubeCamera)
  const hit = rc.intersectObject(cube, false)[0]
  if (!hit) return -1
  const n = hit.face.normal
  return CUBE_FACES.findIndex(([v]) => v[0] === Math.round(n.x) && v[1] === Math.round(n.y) && v[2] === Math.round(n.z))
}

// Snap the view to a face: keep the distance, swing the camera to sit on
// the face's normal, eased over a third of a second. Straight above or
// below gets a hair of tilt so the camera's up vector stays meaningful;
// the tilt's sign puts the part's +Y (world -z) at the top of the screen
// for both TOP and BOTTOM, as Fusion does.
function snapToFace(i) {
  const [v] = CUBE_FACES[i]
  const dir = new THREE.Vector3(...v)
  if (v[1] !== 0) dir.z = 1e-3 * (v[1] > 0 ? 1 : -1)
  dir.normalize()
  const from = camera.position.clone().sub(controls.target)
  const dist = from.length()
  snapAnim = { from: from.normalize(), to: dir, fromDist: dist, toDist: dist,
               fromTarget: controls.target.clone(), toTarget: controls.target.clone(),
               t0: performance.now(), ms: 320 }
}

// One eased step of a view move: direction, distance and orbit target all
// travel together (a face snap keeps the last two, a fit keeps the first).
function stepSnap() {
  const k = Math.min(1, (performance.now() - snapAnim.t0) / snapAnim.ms)
  const e = 1 - Math.pow(1 - k, 3)
  const d = snapAnim.from.clone().lerp(snapAnim.to, e).normalize()
  const dist = snapAnim.fromDist + (snapAnim.toDist - snapAnim.fromDist) * e
  controls.target.copy(snapAnim.fromTarget).lerp(snapAnim.toTarget, e)
  camera.position.copy(controls.target).addScaledVector(d, dist)
  camera.lookAt(controls.target)
  if (k >= 1) snapAnim = null
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
    <div v-if="dims" class="tools" role="toolbar" aria-label="View tools">
      <button :class="{ on: tool === 'orbit' }" title="Orbit: left-drag turns the view" @click="tool = 'orbit'">
        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="6.5" /><path d="M3 12a9 9 0 0 1 3.5-7.1M21 12a9 9 0 0 1-3.5 7.1" /><path d="M6.5 2.5v2.6H3.9M17.5 21.5v-2.6h2.6" /></svg>
      </button>
      <button :class="{ on: tool === 'pan' }" title="Pan: left-drag slides the view" @click="tool = 'pan'">
        <svg viewBox="0 0 24 24"><path d="M12 3v18M3 12h18" /><path d="M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3" /></svg>
      </button>
      <button :class="{ on: tool === 'zoom' }" title="Zoom: left-drag up to zoom in, down to zoom out, at the point you press" @click="tool = 'zoom'">
        <svg viewBox="0 0 24 24"><circle cx="10.5" cy="10.5" r="6.5" /><path d="M15.5 15.5L21 21M7.5 10.5h6M10.5 7.5v6" /></svg>
      </button>
      <span class="sep" />
      <button title="Zoom to fit the whole part" @click="zoomToFit()">
        <svg viewBox="0 0 24 24"><path d="M3 8V4h4M21 8V4h-4M3 16v4h4M21 16v4h-4" /><rect x="7.5" y="7.5" width="9" height="9" stroke-dasharray="2 2" /></svg>
      </button>
      <button :class="{ off: !modelVisible }" :title="modelVisible ? 'Hide the model (the slice trace stays)' : 'Show the model'" @click="setModelVisible(!modelVisible)">
        <svg v-if="modelVisible" viewBox="0 0 24 24"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></svg>
        <svg v-else viewBox="0 0 24 24"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12z" /><circle cx="12" cy="12" r="3" /><path d="M4 20L20 4" /></svg>
      </button>
      <button :class="{ off: !gridVisible }" :title="gridVisible ? 'Hide the floor grid' : 'Show the floor grid'" @click="setGridVisible(!gridVisible)">
        <svg viewBox="0 0 24 24"><path d="M3 9h18M3 15h18M9 3v18M15 3v18" /><rect x="3" y="3" width="18" height="18" rx="1" /><path v-if="!gridVisible" d="M4 20L20 4" /></svg>
      </button>
    </div>
    <div class="hint micro">
      scroll zooms at the cursor · cube snaps the view<span v-if="triangleBody"> · click a body to select it</span><span v-if="sliceAxis"> · plane = the slice across {{ sliceAxis.toUpperCase() }}<span v-if="section"> · yellow = its trace</span></span>
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
  left: 160px;            /* clear of the ViewCube */
  text-align: right;
  pointer-events: none;
}
.tools {
  position: absolute;
  bottom: 34px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 2px;
  padding: 3px;
  background: rgba(20, 23, 28, 0.85);
  border: 1px solid var(--line);
  border-radius: 6px;
}
.tools button {
  width: 32px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: none;
  border-radius: 4px;
  color: var(--muted);
  cursor: pointer;
  padding: 0;
}
.tools button:hover { color: var(--text); background: var(--panel-2); }
.tools button.on { color: var(--edge); background: var(--panel-2); }
.tools button.off { color: var(--warn, #b26a00); }
.tools svg {
  width: 18px;
  height: 18px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.tools .sep { width: 1px; height: 18px; background: var(--line); margin: 0 3px; }
</style>
