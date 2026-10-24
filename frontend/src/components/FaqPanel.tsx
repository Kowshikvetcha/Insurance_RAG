import { useMemo, useState } from 'react'
import { FAQ } from '../faq'

interface Props {
  onPick: (question: string) => void
  disabled?: boolean
  compact?: boolean // inside the composer popover: shorter list
}

/** Common questions grouped by category, with tabs and a search box. */
export function FaqPanel({ onPick, disabled = false, compact = false }: Props) {
  const [category, setCategory] = useState(FAQ[0].id)
  const [query, setQuery] = useState('')

  const results = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return FAQ.find((c) => c.id === category)?.questions ?? []
    return FAQ.flatMap((c) => c.questions).filter((s) => s.toLowerCase().includes(q))
  }, [category, query])

  return (
    <div className={`faq${compact ? ' compact' : ''}`}>
      <div className="faq-head">
        <div className="faq-tabs" role="tablist" aria-label="Question categories">
          {FAQ.map((c) => (
            <button
              key={c.id}
              role="tab"
              aria-selected={!query && category === c.id}
              className={`faq-tab${!query && category === c.id ? ' active' : ''}`}
              title={c.hint}
              onClick={() => {
                setCategory(c.id)
                setQuery('')
              }}
            >
              {c.label}
            </button>
          ))}
        </div>
        <input
          className="faq-search"
          type="search"
          placeholder="Search questions…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search common questions"
        />
      </div>
      <ul className="faq-list">
        {results.map((q) => (
          <li key={q}>
            <button className="faq-q" disabled={disabled} onClick={() => onPick(q)}>
              <span>{q}</span>
              <span className="faq-go" aria-hidden>
                →
              </span>
            </button>
          </li>
        ))}
        {results.length === 0 && <li className="muted small faq-empty">No match. Type your own question below.</li>}
      </ul>
    </div>
  )
}
