// One place for turning failures into sentences a person can act on, and
// for the app-wide error banner that catches what nothing else did.
import { reactive } from 'vue'

// { message, detail } or null; App.vue shows it as a banner
export const appError = reactive({ message: null, detail: null })

export function showAppError(message, detail = null) {
  appError.message = message
  appError.detail = detail ? String(detail).slice(0, 600) : null
}

export function clearAppError() {
  appError.message = null
  appError.detail = null
}

// A fetch() that never reached the server rejects with a TypeError whose
// text depends on the browser ("Failed to fetch", "Load failed",
// "NetworkError when attempting to fetch resource"). None of those tells
// the user what to do.
export function friendlyError(e) {
  if (!e) return 'Unknown error.'
  if (e.name === 'AbortError') return 'The request was cancelled.'
  const m = String(e.message || e)
  if (e instanceof TypeError && /fetch|network|load failed|connection/i.test(m)) {
    return 'Could not reach the server. If the app is still open, wait a moment and try again; otherwise start it again.'
  }
  if (/QuotaExceeded|out of memory|allocation failed/i.test(m)) {
    return 'The browser ran out of memory. Close other tabs or windows and load the file again.'
  }
  return m
}

// The 'detail' of an error response, as one sentence. FastAPI sends a
// string for the errors this app raises; a validation error used to be a
// list of records, which printed as "[object Object]".
export async function errText(res) {
  let detail = null
  try {
    const j = await res.json()
    detail = j?.detail ?? j?.message ?? null
  } catch {
    detail = null
  }
  if (Array.isArray(detail)) {
    detail = detail.map((d) => {
      if (typeof d === 'string') return d
      const loc = (d?.loc || []).filter((x) => !['body', 'query', 'path'].includes(x)).join('.')
      return loc ? `${loc}: ${d?.msg || 'invalid value'}` : d?.msg || JSON.stringify(d)
    }).join('; ')
  } else if (detail && typeof detail === 'object') {
    detail = JSON.stringify(detail)
  }
  if (detail) return String(detail)
  return statusText(res)
}

function statusText(res) {
  switch (res.status) {
    case 404: return 'The server has no record of this job any more. Load the file again.'
    case 413: return 'The file is too large for the server.'
    case 502: case 503: case 504:
      return 'The server is not answering. If the app is still open, wait a moment and try again.'
    case 500: return 'Something went wrong on the server. The server log has the details.'
    default: return res.statusText ? `${res.statusText} (${res.status})` : `Request failed (${res.status})`
  }
}
