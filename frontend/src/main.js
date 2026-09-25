import { createApp } from 'vue'
import { createPinia } from 'pinia'
import '@fontsource-variable/archivo'
import './style.css'
import App from './App.vue'
import { showAppError, friendlyError } from './errors'

const app = createApp(App)

// A component that throws while rendering would otherwise leave the page
// blank or half-drawn with only the console to say why. Log it, and show a
// banner the user can dismiss; the rest of the app keeps working.
app.config.errorHandler = (err, instance, info) => {
  console.error('[stlToSolid]', info, err)
  showAppError(`Something went wrong in the app (${info}): ${friendlyError(err)}`,
               err?.stack)
}
window.addEventListener('error', (ev) => {
  // resource errors (a font that failed to load) have no error object
  if (!ev.error) return
  console.error('[stlToSolid]', ev.error)
  showAppError(`Something went wrong in the app: ${friendlyError(ev.error)}`, ev.error?.stack)
})
window.addEventListener('unhandledrejection', (ev) => {
  console.error('[stlToSolid]', ev.reason)
  showAppError(`Something went wrong in the app: ${friendlyError(ev.reason)}`,
               ev.reason?.stack)
  ev.preventDefault()
})

app.use(createPinia()).mount('#app')
