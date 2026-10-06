export function formatSuggestionSource(source) {
  if (source === 'combined') return 'Normalization + Grammar'
  if (source === 'normalization') return 'Normalization'
  return 'Grammar'
}
