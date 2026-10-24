import { useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Citation } from '../api'
import { figureUrl } from '../api'
import { FaqPanel } from './FaqPanel'

export interface Message {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  pending?: boolean
  error?: string
}

interface Props {
  sessionId: string | null
  messages: Message[]
  onAsk: (q: string) => void
  busy: boolean
}

export function ChatWindow({ sessionId, messages, onAsk, busy }: Props) {
  const [text, setText] = useState('')
  const [open, setOpen] = useState<Citation | null>(null)
  const [faqOpen, setFaqOpen] = useState(false)
  const end = useRef<HTMLDivElement>(null)

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const submit = (q: string) => {
    const question = q.trim()
    if (!question || busy || !sessionId) return
    onAsk(question)
    setText('')
    setFaqOpen(false)
  }

  return (
    <div className="chat">
      <div className="messages">
        {!sessionId && (
          <div className="empty">
            <h2>Understand a health policy before you buy it</h2>
            <p className="muted">
              Upload the policy wording or brochure. Ask about coverage, limits, waiting periods and exclusions. Every
              answer is grounded in the document and cites its page.
            </p>
            <h3 className="faq-title">Questions you can ask</h3>
            <FaqPanel onPick={submit} disabled />
          </div>
        )}
        {sessionId && messages.length === 0 && (
          <div className="empty">
            <h2>Ask anything about this policy</h2>
            <p className="muted">Pick a common question or type your own.</p>
            <FaqPanel onPick={submit} disabled={busy} />
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="bubble">
              {m.role === 'assistant' ? (
                <div className="md">
                  <Markdown remarkPlugins={[remarkGfm]}>{m.content || (m.pending ? '…' : '')}</Markdown>
                </div>
              ) : (
                m.content
              )}
              {m.error && <div className="error">{m.error}</div>}
              {m.citations && m.citations.length > 0 && (
                <div className="citations">
                  {m.citations.map((c) => (
                    <button key={c.source} className="chip cite" onClick={() => setOpen(c)} title={c.snippet}>
                      <b>{c.source}</b> {c.type === 'table' ? 'Table' : c.type === 'figure' ? 'Figure' : 'Text'} · {c.page_label || c.section}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={end} />
      </div>

      {faqOpen && sessionId && messages.length > 0 && (
        <div className="faq-popover">
          <div className="faq-popover-head">
            <strong>Common questions</strong>
            <button className="ghost" onClick={() => setFaqOpen(false)} aria-label="Close common questions">
              ✕
            </button>
          </div>
          <FaqPanel onPick={submit} disabled={busy} compact />
        </div>
      )}

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault()
          submit(text)
        }}
      >
        <textarea
          value={text}
          placeholder={sessionId ? 'Ask about coverage, limits, waiting periods…' : 'Upload a document first'}
          disabled={!sessionId}
          rows={2}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit(text)
            }
          }}
        />
        {sessionId && messages.length > 0 && (
          <button
            type="button"
            className={`secondary${faqOpen ? ' active' : ''}`}
            onClick={() => setFaqOpen((v) => !v)}
            aria-expanded={faqOpen}
            title="Browse common questions"
          >
            Common questions
          </button>
        )}
        <button type="submit" disabled={!sessionId || busy || !text.trim()}>
          {busy ? '…' : 'Ask'}
        </button>
      </form>

      {open && sessionId && <SourceViewer citation={open} sessionId={sessionId} onClose={() => setOpen(null)} />}
    </div>
  )
}

function SourceViewer({ citation, sessionId, onClose }: { citation: Citation; sessionId: string; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="overlay" onClick={onClose}>
      <div className="viewer" onClick={(e) => e.stopPropagation()}>
        <header>
          <div>
            <div className="muted small">
              Source {citation.source} · {citation.page_label}
            </div>
            <strong>{citation.title || citation.section}</strong>
            {citation.section && citation.section !== citation.title && <div className="muted small">{citation.section}</div>}
          </div>
          <button className="ghost" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>
        {citation.figure_id && <img className="figure" src={figureUrl(sessionId, citation.figure_id)} alt={citation.title} />}
        <div className="md table-scroll">
          {citation.type === 'table' ? (
            <Markdown remarkPlugins={[remarkGfm]}>{citation.content}</Markdown>
          ) : (
            <pre className="source-text">{citation.content}</pre>
          )}
        </div>
      </div>
    </div>
  )
}
