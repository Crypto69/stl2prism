// One line describing a traced section from its /section stats, for the
// loft's single slice and the x-ray's start and end planes alike.
export function summarise(st, outline) {
  if (!st) return ''
  const parts = []
  if (st.lines) parts.push(`${st.lines} lines`)
  if (st.arcs) parts.push(`${st.arcs} arcs`)
  if (st.circles) parts.push(`${st.circles} circle${st.circles === 1 ? '' : 's'}`)
  if (st.splines) parts.push(`${st.splines} spline${st.splines === 1 ? '' : 's'}`)
  let head = `${st.loops} outline${st.loops === 1 ? '' : 's'}`
  if (st.open) head += `, ${st.open} open curve${st.open === 1 ? '' : 's'}${outline ? ' skipped' : ''}`
  if (st.inner) head += `, ${st.inner} inner loop${st.inner === 1 ? '' : 's'} skipped`
  let tail = ''
  if (st.joins) tail = `; ${st.joins} gap${st.joins === 1 ? '' : 's'} joined (largest ${st.max_gap.toFixed(2)} mm)`
  if (st.trimmed) tail += `; ${st.trimmed} sliver${st.trimmed === 1 ? '' : 's'} trimmed`
  return `${head}: ${parts.join(', ') || 'nothing'}${tail}; worst miss ${st.dev_max.toFixed(3)} mm`
}
