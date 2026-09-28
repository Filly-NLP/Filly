export function codePointOffsetToUtf16(text, codePointOffset) {
  if (!Number.isSafeInteger(codePointOffset) || codePointOffset < 0) return null

  let points = 0
  let utf16Offset = 0
  for (const character of text) {
    if (points === codePointOffset) return utf16Offset
    points += 1
    utf16Offset += character.length
  }
  return points === codePointOffset ? utf16Offset : null
}

function suggestionType(source) {
  const normalizedSource = String(source || '').toLowerCase()
  if (normalizedSource.includes('combined')) return 'combined'
  return normalizedSource.includes('norm') ? 'norm' : 'gram'
}

export function normalizeSuggestions(text, suggestions) {
  if (!Array.isArray(suggestions)) return []

  const candidates = []
  const usedIds = new Set()
  suggestions.forEach((item, index) => {
    if (!item || typeof item !== 'object') return
    if (typeof item.original !== 'string' || typeof item.replacement !== 'string') return

    const start = codePointOffsetToUtf16(text, item.start)
    const end = codePointOffsetToUtf16(text, item.end)
    if (start === null || end === null || end < start) return
    if (text.slice(start, end) !== item.original) return

    const baseId = item.id == null || String(item.id).length === 0
      ? `suggestion-${index}`
      : String(item.id)
    let id = baseId
    let suffix = 1
    while (usedIds.has(id)) id = `${baseId}-${suffix++}`
    usedIds.add(id)

    candidates.push({
      id,
      start,
      end,
      original: item.original,
      replacement: item.replacement,
      source: String(item.source || ''),
      tag: item.tag == null ? '' : String(item.tag),
      type: suggestionType(item.source),
      status: 'pending',
    })
  })

  candidates.sort((left, right) => left.start - right.start || left.end - right.end)
  const accepted = []
  let previousEnd = -1
  let previousStart = -1
  for (const candidate of candidates) {
    if (candidate.start < previousEnd || candidate.start === previousStart) continue
    accepted.push(candidate)
    previousStart = candidate.start
    previousEnd = candidate.end
  }
  return accepted
}

export function segmentText(text, suggestions) {
  const segments = []
  let cursor = 0

  for (const suggestion of suggestions) {
    if (suggestion.start > cursor) {
      segments.push({ text: text.slice(cursor, suggestion.start), suggestionId: null })
    }
    if (suggestion.start < cursor || text.slice(suggestion.start, suggestion.end) !== suggestion.original) continue
    segments.push({
      text: suggestion.original,
      suggestionId: suggestion.id,
      insertion: suggestion.start === suggestion.end,
    })
    cursor = suggestion.end
  }

  if (cursor < text.length) segments.push({ text: text.slice(cursor), suggestionId: null })
  if (segments.length === 0 && text.length === 0) return []
  return segments
}

function applySelectedSuggestionsSafely(text, suggestions) {
  const selected = suggestions
    .slice()
    .sort((left, right) => left.start - right.start || left.end - right.end)

  let previousEnd = -1
  let previousStart = -1
  for (const suggestion of selected) {
    if (!Number.isSafeInteger(suggestion.start) || !Number.isSafeInteger(suggestion.end)
      || suggestion.start < 0 || suggestion.end < suggestion.start || suggestion.end > text.length
      || suggestion.start < previousEnd || suggestion.start === previousStart
      || text.slice(suggestion.start, suggestion.end) !== suggestion.original) {
      return { ok: false, text, reason: 'Suggestions no longer match the editor text.' }
    }
    previousStart = suggestion.start
    previousEnd = suggestion.end
  }

  let updatedText = text
  for (const suggestion of selected.reverse()) {
    updatedText = updatedText.slice(0, suggestion.start)
      + suggestion.replacement
      + updatedText.slice(suggestion.end)
  }
  return { ok: true, text: updatedText }
}

export function applySuggestionsSafely(text, suggestions) {
  return applySelectedSuggestionsSafely(
    text,
    suggestions.filter((suggestion) => suggestion.status === 'pending'),
  )
}

export function applyAcceptedSuggestionsSafely(text, suggestions) {
  return applySelectedSuggestionsSafely(
    text,
    suggestions.filter((suggestion) => suggestion.status === 'accepted'),
  )
}
