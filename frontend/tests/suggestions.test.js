import test from 'node:test'
import assert from 'node:assert/strict'
import {
  applyAcceptedSuggestionsSafely,
  applySuggestionsSafely,
  codePointOffsetToUtf16,
  normalizeSuggestions,
} from '../src/suggestions.js'

test('converts Python code-point offsets to UTF-16 offsets after astral characters', () => {
  assert.equal(codePointOffsetToUtf16('🙂aq', 1), 2)
  assert.equal(codePointOffsetToUtf16('🙂aq', 3), 4)
  assert.equal(codePointOffsetToUtf16('🙂aq', 4), null)
})

test('keeps suggestions only when the original surface matches at the converted range', () => {
  const suggestions = normalizeSuggestions('🙂 aq', [
    { id: 'valid', start: 2, end: 4, original: 'aq', replacement: 'ako', source: 'normalization', tag: 'NORMALIZATION' },
    { id: 'stale', start: 2, end: 4, original: 'q', replacement: 'ako', source: 'normalization' },
    { id: 'invalid', start: -1, end: 1, original: '🙂', replacement: 'hi', source: 'gec' },
  ])

  assert.equal(suggestions.length, 1)
  assert.equal(suggestions[0].start, 3)
  assert.equal(suggestions[0].type, 'norm')
})

test('distinguishes normalization, GEC, and combined alignment sources', () => {
  const suggestions = normalizeSuggestions('abc', [
    { id: 'norm', start: 0, end: 1, original: 'a', replacement: 'A', source: 'normalization' },
    { id: 'gec', start: 1, end: 2, original: 'b', replacement: 'B', source: 'gec' },
    { id: 'combined', start: 2, end: 3, original: 'c', replacement: 'C', source: 'combined' },
  ])
  assert.deepEqual(suggestions.map(({ type }) => type), ['norm', 'gram', 'combined'])
})

test('accepts multiple current suggestions from the end of the text safely', () => {
  const result = applySuggestionsSafely('aq nsa', [
    { start: 0, end: 2, original: 'aq', replacement: 'ako', status: 'pending' },
    { start: 3, end: 6, original: 'nsa', replacement: 'nasa', status: 'pending' },
  ])
  assert.deepEqual(result, { ok: true, text: 'ako nasa' })
})

test('exportable text applies only accepted suggestions after partial review', () => {
  const suggestions = [
    { start: 0, end: 2, original: 'aq', replacement: 'ako', status: 'accepted' },
    { start: 3, end: 6, original: 'nsa', replacement: 'nasa', status: 'ignored' },
    { start: 7, end: 9, original: 'kc', replacement: 'kasi', status: 'pending' },
  ]

  assert.deepEqual(applyAcceptedSuggestionsSafely('aq nsa kc', suggestions), {
    ok: true,
    text: 'ako nsa kc',
  })
})

test('accept all applies every suggestion together from the analyzed source', () => {
  const suggestions = [
    { start: 0, end: 2, original: 'aq', replacement: 'ako', status: 'accepted' },
    { start: 3, end: 6, original: 'nsa', replacement: 'nasa', status: 'accepted' },
  ]

  assert.deepEqual(applyAcceptedSuggestionsSafely('aq nsa', suggestions), {
    ok: true,
    text: 'ako nasa',
  })
})

test('rejects accept operations when an offset is stale or suggestions overlap', () => {
  assert.equal(applySuggestionsSafely('changed', [
    { start: 0, end: 2, original: 'aq', replacement: 'ako', status: 'pending' },
  ]).ok, false)
  assert.equal(applySuggestionsSafely('abcd', [
    { start: 0, end: 3, original: 'abc', replacement: 'x', status: 'pending' },
    { start: 2, end: 4, original: 'cd', replacement: 'y', status: 'pending' },
  ]).ok, false)
})
