import { useRef, useState } from 'react'
import type { JobState } from '../api'

const STAGES = [
  { key: 'parsing', label: 'Layout, tables & OCR' },
  { key: 'tables', label: 'Normalising tables' },
  { key: 'figures', label: 'Reading charts & figures' },
  { key: 'enrichment', label: 'Definitions & summaries' },
  { key: 'embedding', label: 'Embedding' },
]

interface Props {
  job: JobState | null
  error: string | null
  onFile: (file: File) => void
}

export function UploadPanel({ job, error, onFile }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const busy = job?.status === 'queued' || job?.status === 'running'
  const stageIdx = job ? STAGES.findIndex((s) => s.key === job.stage) : -1

  const pick = (files: FileList | null) => {
    const f = files?.[0]
    if (f) onFile(f)
  }

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
        <span className="muted">{busy ? 'Processing…' : 'Drop a PDF or DOCX here, or click to browse'}</span>
      </div>

      {job && job.status !== 'done' && (
        <div className="progress">
          <div className="bar">
            <div className="fill" style={{ width: `${job.percent}%` }} />
          </div>
          <ol className="stages">
            {STAGES.map((s, i) => (
              <li key={s.key} className={i < stageIdx ? 'done' : i === stageIdx ? 'active' : ''}>
                {s.label}
              </li>
            ))}
          </ol>
          <div className="muted small">{job.message}</div>
        </div>
      )}
      {(error || job?.error) && <div className="error">{error ?? job?.error}</div>}
    </section>
  )
}
