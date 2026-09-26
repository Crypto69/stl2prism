// One hue per body or feature, walked by the golden angle so neighbours
// never share a colour. Must match MeshViewer.bodyColor (same walk, same
// saturation and lightness), or a legend lies about the scene.
export function hueOf(i) {
  return ((i * 0.381966) % 1) * 360
}

export function swatch(i) {
  return `hsl(${hueOf(i)} 62% 62%)`
}
