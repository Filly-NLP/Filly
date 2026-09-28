const API_BASE_URL = (import.meta.env?.VITE_API_BASE_URL || '/api/v1').replace(/\/$/, '')

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...options.headers,
    },
  })

  const raw = await response.text()
  let payload = null
  if (raw) {
    try {
      payload = JSON.parse(raw)
    } catch {
      payload = null
    }
  }

  if (!response.ok) {
    const detail = payload?.detail
    const message = typeof detail === 'string'
      ? detail
      : Array.isArray(detail)
        ? detail.map((item) => item.msg || String(item)).join('; ')
        : `FILLY API request failed (${response.status})`
    throw new Error(message)
  }

  if (!payload || typeof payload !== 'object') {
    throw new Error('The FILLY API returned an invalid response.')
  }
  return payload
}

function postText(path, text) {
  return requestJson(path, {
    method: 'POST',
    body: JSON.stringify({ text }),
  })
}

export function analyzeText(text) {
  return postText('/analyze', text)
}

export function normalizeText(text) {
  return postText('/normalize', text)
}

export function correctGrammar(text) {
  return postText('/gec', text)
}

export function healthCheck() {
  return requestJson('/health')
}
