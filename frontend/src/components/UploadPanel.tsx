import { useEffect, useRef, useState } from 'react'
import type { JobState } from '../api'

const STAGES = [
  { key: 'parsing', label: 'Reading pages: layout, tables & OCR' },
  { key: 'tables', label: 'Normalising tables' },
  { key: 'figures', label: 'Reading charts & figures' },
  { key: 'enrichment', label: 'Definitions & summaries' },
  { key: 'embedding', label: 'Building the search index' },
]

interface Props {
  job: JobState | null
  error: string | null
  onFile: (file: File) => void
}

function fmt(seconds: number): string {
  const s = Math.max(0, Math.round(seconds))
  const m = Math.floor(s / 60)
  return `${m}:${String(s % 60).padStart(2, '0')}`
}

/** Elapsed time since the job started, plus a countdown that ticks locally between server updates. */
function useJobClock(job: JobState | null, busy: boolean) {
  const [now, setNow] = useState(() => Date.now())
  const [startedAt, setStartedAt] = useState<number | null>(null)
  const [eta, setEta] = useState<{ seconds: number; at: number } | null>(null)

  // The clock starts when an upload begins (busy goes true) and survives the upload → job handover.
  useEffect(() => {
    if (busy) {
      setStartedAt(Date.now())
      setEta(null)
    }
  }, [busy])

  // Re-anchor the countdown whenever the server sends a fresh estimate; tick locally in between.
  const etaSeconds = job?.eta_seconds
  const version = job?.version
  useEffect(() => {
    if (etaSeconds != null) setEta({ seconds: etaSeconds, at: Date.now() })
  }, [etaSeconds, version])

  useEffect(() => {
    if (!busy) return
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [busy])

  const elapsed = startedAt ? (now - startedAt) / 1000 : 0
  const remaining = eta ? eta.seconds - (now - eta.at) / 1000 : null
  return { elapsed, remaining }
}

export function UploadPanel({ job, error, onFile }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const busy = job?.status === 'queued' || job?.status === 'running'
  const uploading = busy && !job?.id
  const stageIdx = job ? STAGES.findIndex((s) => s.key === job.stage) : -1
  const { elapsed, remaining } = useJobClock(job, busy)

  const pick = (files: FileList | null) => {
    const f = files?.[0]
    if (f) onFile(f)
  }

  let timeLeft = 'Estimating time…'
  if (uploading) timeLeft = 'Uploading…'
  else if (remaining != null) timeLeft = remaining > 1 ? `About ${fmt(remaining)} left` : 'Almost done…'

  return (
    <section className="panel">
      <div
        className={`dropzone${dragging ? ' dragging' : ''}${busy ? ' disabled' : ''}`}
        onClick={() => !busy && input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          if (!busy) pick(e.dataTransfer.files)
        }}
        role="button"
        tabIndex={0}
      >
        <input ref={input} type="file" accept=".pdf,.docx" hidden onChange={(e) => pick(e.target.files)} />
        <strong>{job?.filename ?? 'Upload a policy document'}</strong>
        <span className="muted">{busy ? 'Processing your document…' : 'Drop a PDF or DOCX here, or click to browse'}</span>
      </div>

      {job && busy && (
        <div className="progress" aria-live="polite">
          <div className="progress-top">
            <span className="pct">{job.percent}%</span>
            <span className="eta">{timeLeft}</span>
          </div>
          <div className="bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={job.percent}>
            <div className="fill" style={{ width: `${Math.max(job.percent, 3)}%` }} />
          </div>
          <ol className="stages">
            {STAGES.map((s, i) => (
              <li key={s.key} className={i < stageIdx ? 'done' : i === stageIdx ? 'active' : ''}>
                <span className="stage-icon" aria-hidden>
                  {i < stageIdx ? '✓' : i === stageIdx ? <span className="spinner" /> : '○'}
                </span>
                <span>
                  {s.label}
                  {i === stageIdx && <span className="stage-detail">{job.message}</span>}
                </span>
              </li>
            ))}
          </ol>
          <div className="muted small">Elapsed {fmt(elapsed)}. The first upload also loads the AI models.</div>
        </div>
      )}
      {(error || job?.error) && <div className="error">{error ?? job?.error}</div>}
    </section>
  )
}
