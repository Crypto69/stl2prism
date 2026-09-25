<script setup>
import { useConvertStore } from '../store'
import GateFields from './GateFields.vue'

const store = useConvertStore()
</script>

<template>
  <section class="params">
    <GateFields :enforced="true" />

    <label class="check">
      <input type="checkbox" v-model="store.params.force_prismatic" />
      <span>
        Force prismatic on scan-like meshes
        <span class="help">Dense organic scans (millions of tiny triangles) normally skip straight to the faceted copy, because they rarely have flat faces and true cylinders to recover. Tick this to make the tool try anyway.</span>
      </span>
    </label>

    <label class="check">
      <input type="checkbox" v-model="store.params.face_groups" />
      <span>
        Face-group engine
        <span class="help">When the extrusion fit is rejected, group the mesh into surface regions and give each one a real plane, cylinder, cone or sphere face (like Fusion's "Prismatic" mesh conversion) before falling back to a faceted copy. Untick to skip it.</span>
      </span>
    </label>
  </section>
</template>
