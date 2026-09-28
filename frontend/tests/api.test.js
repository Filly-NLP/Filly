import test from 'node:test'
import assert from 'node:assert/strict'
import {
  analyzeText,
  correctGrammar,
  healthCheck,
  normalizeText,
} from '../src/api/fillyApi.js'

test('analysis client posts the exact text to the versioned API', async () => {
  const previousFetch = globalThis.fetch
  let request
  globalThis.fetch = async (url, options) => {
    request = { url, options }
    return new Response(JSON.stringify({ original_text: '  aq  ' }), { status: 200 })
  }

  try {
    const result = await analyzeText('  aq  ')
    assert.equal(request.url, '/api/v1/analyze')
    assert.equal(request.options.method, 'POST')
    assert.equal(request.options.headers['Content-Type'], 'application/json')
    assert.deepEqual(JSON.parse(request.options.body), { text: '  aq  ' })
    assert.equal(result.original_text, '  aq  ')
  } finally {
    globalThis.fetch = previousFetch
  }
})

test('module clients target their dedicated API routes', async () => {
  const previousFetch = globalThis.fetch
  const paths = []
  globalThis.fetch = async (url) => {
    paths.push(url)
    return new Response('{}', { status: 200 })
  }

  try {
    await normalizeText('text')
    await correctGrammar('text')
    await healthCheck()
    assert.deepEqual(paths, ['/api/v1/normalize', '/api/v1/gec', '/api/v1/health'])
  } finally {
    globalThis.fetch = previousFetch
  }
})

test('API errors expose backend detail for the UI error state', async () => {
  const previousFetch = globalThis.fetch
  globalThis.fetch = async () => new Response(JSON.stringify({ detail: 'Model unavailable' }), { status: 503 })

  try {
    await assert.rejects(analyzeText('test'), /Model unavailable/)
  } finally {
    globalThis.fetch = previousFetch
  }
})
