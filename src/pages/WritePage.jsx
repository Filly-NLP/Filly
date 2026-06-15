import { useState, useRef } from 'react'
import styles from './WritePage.module.css'

export default function WritePage() {
  const [title, setTitle] = useState('Untitled')
  const [text, setText] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const fileInputRef = useRef(null)

  const wordCount = text.trim() === '' ? 0 : text.trim().split(/\s+/).length
  const MAX_WORDS = 250
  const isFormal = text.trim().length > 0 && suggestions.length === 0 && !isAnalyzing

  const handleSave = () => {
    alert('Saved. Backend integration is still pending.')
  }

  const handleUpload = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => setText(ev.target.result)
    reader.readAsText(file)
  }

  const analyzeText = (nextText) => {
    if (!nextText.trim()) {
      setSuggestions([])
      return
    }

    setIsAnalyzing(true)
    window.clearTimeout(analyzeText.timer)
    analyzeText.timer = window.setTimeout(() => {
      setSuggestions([])
      setIsAnalyzing(false)
    }, 500)
  }

  return (
    <div className={styles.layout}>
      <section className={styles.editorCard}>
        <div className={styles.editorHeader}>
          <input
            className={styles.titleInput}
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            spellCheck={false}
            aria-label="Document title"
          />
          <div className={styles.headerActions}>
            <button className={styles.btnUpload} onClick={() => fileInputRef.current?.click()} type="button">
              <UploadIcon /> Upload File
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt,.doc,.docx"
              className={styles.hiddenInput}
              onChange={handleUpload}
            />
            <button className={styles.btnSave} onClick={handleSave} type="button">
              <SaveIcon /> Save
            </button>
          </div>
        </div>

        <textarea
          className={styles.textarea}
          placeholder="Start typing..."
          value={text}
          onChange={(e) => {
            const nextText = e.target.value
            setText(nextText)
            analyzeText(nextText)
          }}
          spellCheck={false}
        />

        <div className={styles.editorFooter}>
          <span className={wordCount > MAX_WORDS ? styles.overLimit : ''}>
            {wordCount} / {MAX_WORDS} words
          </span>
        </div>
      </section>

      <aside className={styles.recommendPanel}>
        <h2 className={styles.panelTitle}>Recommendations</h2>

        <div className={styles.suggestionsMeta}>
          Suggestions ({suggestions.length})
        </div>

        {suggestions.length === 0 ? (
          <div className={styles.emptyState}>
            <p>No suggestions</p>
            {(isFormal || !text.trim()) && <p>Your text looks formal!</p>}
            {isAnalyzing && <p>Checking text...</p>}
          </div>
        ) : (
          <ul className={styles.suggestionList}>
            {suggestions.map((s, i) => (
              <SuggestionCard key={i} suggestion={s} />
            ))}
          </ul>
        )}
      </aside>
    </div>
  )
}

function SuggestionCard({ suggestion }) {
  const [dismissed, setDismissed] = useState(false)
  if (dismissed) return null
  return (
    <li className={styles.suggestionCard}>
      <p className={styles.suggestionOriginal}>{suggestion.original}</p>
      <p className={styles.suggestionFormal}>{suggestion.formal}</p>
      <div className={styles.suggestionActions}>
        <button className={styles.btnAccept} type="button">Accept</button>
        <button className={styles.btnIgnore} onClick={() => setDismissed(true)} type="button">Ignore</button>
      </div>
    </li>
  )
}

function UploadIcon() {
  return (
    <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="m17 8-5-5-5 5" />
      <path d="M12 3v12" />
    </svg>
  )
}

function SaveIcon() {
  return (
    <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2Z" />
      <path d="M17 21v-8H7v8" />
      <path d="M7 3v5h8" />
    </svg>
  )
}
