import { useEffect, useRef } from 'react'
import template from './fillyTemplate.html?raw'
import { initFillyApp } from './fillyApp.js'

export default function App() {
  const appRootRef = useRef(null)

  useEffect(() => {
    if (!appRootRef.current) return

    appRootRef.current.innerHTML = template
    initFillyApp()
  }, [])

  return <div ref={appRootRef} />
}
