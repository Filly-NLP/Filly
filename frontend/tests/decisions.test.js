import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { deriveResolvedOutput, normalizeSuggestions, segmentText } from '../src/suggestions.js'
import { formatSuggestionSource } from '../src/presentation.js'

const original = '🙂 aq kumain ng mabilis daw siya siya.'
function fixture() {
  return normalizeSuggestions(original, [
    { id: 'norm', start: 2, end: 4, original: 'aq', replacement: 'Ako', source: 'combined' },
    { id: 'ng', start: 12, end: 14, original: 'ng', replacement: 'nang', source: 'gec' },
    { id: 'daw', start: 23, end: 26, original: 'daw', replacement: 'raw', source: 'gec' },
    { id: 'delete', start: 31, end: 36, original: ' siya', replacement: '', source: 'gec' },
  ])
}
test('preview, Accept, Ignore and Accept All replay from Unicode source', () => {
  const suggestions = fixture()
  assert.equal(suggestions.length, 4)
  const full = '🙂 Ako kumain nang mabilis raw siya.'
  assert.equal(deriveResolvedOutput(original, suggestions).text, full)
  suggestions[1].status = 'accepted'
  assert.match(deriveResolvedOutput(original, suggestions).text, /nang/)
  suggestions[1].status = 'ignored'
  assert.match(deriveResolvedOutput(original, suggestions).text, /ng mabilis/)
  suggestions.forEach(s => { s.status = 'accepted' })
  assert.equal(deriveResolvedOutput(original, suggestions).text, full)
})
test('mixed decisions and different click orders produce identical output', () => {
  const run = (order) => {
    const suggestions = fixture()
    order.forEach(([id, status]) => { suggestions.find(s => s.id === id).status = status })
    return deriveResolvedOutput(original, suggestions).text
  }
  const decisions = [['ng', 'accepted'], ['daw', 'ignored'], ['delete', 'accepted']]
  assert.equal(run(decisions), '🙂 Ako kumain nang mabilis daw siya.')
  assert.equal(run(decisions), run(decisions.toReversed()))
})
test('replay rejects stale and overlapping edits and preserves segmented source', () => {
  const suggestions = fixture()
  assert.equal(deriveResolvedOutput('edited', suggestions).ok, false)
  assert.equal(deriveResolvedOutput(original, [...suggestions, suggestions[0]]).ok, false)
  assert.equal(segmentText(original, suggestions).map(s => s.text).join(''), original)
})
test('source labels preserve actual combined provenance', () => {
  assert.equal(formatSuggestionSource('normalization'), 'Normalization')
  assert.equal(formatSuggestionSource('gec'), 'Grammar')
  assert.equal(formatSuggestionSource('combined'), 'Normalization + Grammar')
})
test('output export and rendering share resolved state, highlights and scrolling are wired', () => {
  const app = readFileSync(new URL('../src/fillyApp.js', import.meta.url), 'utf8')
  const css = readFileSync(new URL('../src/index.css', import.meta.url), 'utf8')
  assert.match(app, /escH\(resolvedOutputText\)/)
  assert.match(app, /currentAnalysisText === editor.value \? resolvedOutputText : ''/)
  assert.equal((app.match(/const text\s*=\s*getOutputPlainText\(\)/g) || []).length, 2)
  assert.match(app, /suggestion.status !== 'pending'/)
  assert.match(app, /span.classList.toggle\('selected'/)
  assert.match(css, /\.original-highlight.selected\{/)
  assert.match(css, /\.recs-panel\{[^}]*height:calc\(var\(--editor-panel-height\)/)
  assert.match(css, /\.recs-body\{[^}]*min-height:0;[^}]*overflow-y:auto/)
})
