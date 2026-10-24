import { useEffect, useRef, useState } from 'react'
import type { ChatTurn, JobState, Report } from './api'
import { fetchReport, streamChat, uploadDocument, watchJob } from './api'
import type { Message } from './components/ChatWindow'
import { ChatWindow } from './components/ChatWindow'
import { IngestionReport } from './components/IngestionReport'
import { UploadPanel } from './components/UploadPanel'

const TOKEN_FLUSH_MS = 60

export default function App() {
  const [job, setJob] = useState<JobState | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [report, setReport] = useState<Report | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [busy, setBusy] = useState(false)
  const stopWatching = useRef<(() => void) | null>(null)

  useEffect(() => () => stopWatching.current?.(), [])

  const onFile = async (file: File) => {
    stopWatching.current?.()
    setUploadError(null)
    setSessionId(null)
    setReport(null)
    setMessages([])
    setJob({ id: '', filename: file.name, status: 'queued', stage: 'queued', message: 'Uploading', percent: 0, session_id: null, error: null })
    try {
      const { job_id } = await uploadDocument(file)
      stopWatching.current = watchJob(job_id, (j) => {
        setJob(j)
        if (j.status === 'done' && j.session_id) {
          const sid = j.session_id
          setSessionId(sid)
          fetchReport(sid).then(setReport, (e) => setUploadError(String(e.message ?? e)))
        }
      })
    } catch (e) {
      setJob(null)
      setUploadError(e instanceof Error ? e.message : String(e))
    }
  }

  const patchLast = (fn: (m: Message) => Message) =>
    setMessages((ms) => [...ms.slice(0, -1), fn(ms[ms.length - 1])])

  // Tokens arrive many times a second. Buffer them and apply at most every TOKEN_FLUSH_MS,
  // so the answer re-renders (and re-parses its markdown) a few times a second, not per token.
  const tokenBuffer = useRef('')
  const flushTimer = useRef<number | null>(null)
  const flushTokens = () => {
    if (flushTimer.current !== null) {
      window.clearTimeout(flushTimer.current)
      flushTimer.current = null
    }
    const text = tokenBuffer.current
    if (!text) return
    tokenBuffer.current = ''
    patchLast((m) => ({ ...m, content: m.content + text }))
  }
  const queueToken = (t: string) => {
    tokenBuffer.current += t
    if (flushTimer.current === null) flushTimer.current = window.setTimeout(flushTokens, TOKEN_FLUSH_MS)
  }

  const onAsk = async (question: string) => {
    if (!sessionId) return
    const history: ChatTurn[] = messages.filter((m) => !m.error && m.content).map(({ role, content }) => ({ role, content }))
    setMessages((ms) => [...ms, { role: 'user', content: question }, { role: 'assistant', content: '', pending: true }])
    setBusy(true)
    try {
      await streamChat(sessionId, question, history, {
        onToken: queueToken,
        onCitations: (c) => {
          flushTokens()
          patchLast((m) => ({ ...m, citations: c }))
        },
        onError: (err) => {
          flushTokens()
          patchLast((m) => ({ ...m, error: err }))
        },
      })
    } catch (e) {
      flushTokens()
      patchLast((m) => ({ ...m, error: e instanceof Error ? e.message : String(e) }))
    } finally {
      flushTokens()
      patchLast((m) => ({ ...m, pending: false }))
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <header className="brand">
          <span className="logo">◆</span> Policy Analyser
        </header>
        <UploadPanel job={job} error={uploadError} onFile={onFile} />
        {report && sessionId && <IngestionReport report={report} sessionId={sessionId} />}
        <p className="muted small disclaimer">
          Answers come only from your document. This tool explains policy terms and does not give purchase advice.
        </p>
      </aside>
      <main>
        <ChatWindow sessionId={sessionId} messages={messages} onAsk={onAsk} busy={busy} />
      </main>
    </div>
  )
}
