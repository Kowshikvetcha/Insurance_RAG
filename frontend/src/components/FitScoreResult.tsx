import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Citation, DimensionResult, FitScore } from '../api'

export interface FitRun {
  stage: string
  citations: Citation[]
  dimensions: DimensionResult[]
  score: FitScore | null
  reasoning: string
  error: string | null
  pending: boolean
}

const REMARK_PLUGINS = [remarkGfm]

/** Colour band for a 1–10 value, matching the verdict bands in backend/app/fit_score.py. */
const tone = (v: number | null) => (v === null ? 'none' : v >= 7 ? 'good' : v >= 4 ? 'mid' : 'poor')

function ScoreDial({ score }: { score: FitScore }) {
  const r = 52
  const circumference = 2 * Math.PI * r
  const filled = score.score === null ? 0 : (score.score / 10) * circumference
  return (
    <div className={`fit-dial ${tone(score.score)}`}>
      <svg viewBox="0 0 120 120" aria-hidden>
        <circle cx="60" cy="60" r={r} className="track" />
        <circle cx="60" cy="60" r={r} className="arc" strokeDasharray={`${filled} ${circumference}`} transform="rotate(-90 60 60)" />
      </svg>
      <div className="fit-dial-value">
        <strong>{score.score === null ? '–' : score.score}</strong>
        <span className="muted small">out of 10</span>
      </div>
    </div>
  )
}

interface Props {
  run: FitRun
  onCite: (c: Citation) => void
  onEdit: () => void
}

export function FitScoreResult({ run, onCite, onEdit }: Props) {
  const bySource = new Map(run.citations.map((c) => [c.source, c]))
  const cite = (n: number) => {
    const c = bySource.get(n)
    return (
      c && (
        <button key={n} className="chip cite" onClick={() => onCite(c)} title={c.snippet}>
          <b>{n}</b> {c.page_label || c.section}
        </button>
      )
    )
  }
  // Show dimensions in the score's (rubric) order once known, arrival order before that.
  const order = run.score?.breakdown.map((b) => b.key)
  const dims = order ? [...run.dimensions].sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key)) : run.dimensions

  return (
    <div className="fit-result">
      <div className="fit-head">
        {run.score ? (
          <>
            <ScoreDial score={run.score} />
            <div>
              <h2 className="fit-band">{run.score.band}</h2>
              <div className="muted small">
                Confidence: <b>{run.score.confidence}</b>
                {run.score.capped && ' · capped because of a deal-breaker'}
              </div>
              {run.score.deal_breakers.length > 0 && (
                <div className="error">Deal-breakers: {run.score.deal_breakers.join(', ')}</div>
              )}
            </div>
          </>
        ) : (
          <div className="fit-stage">
            {run.pending && <span className="spinner" />} {run.pending ? run.stage || 'Starting…' : 'Could not finish scoring'}
            <span className="muted small"> · {run.dimensions.length} area(s) assessed</span>
          </div>
        )}
        <button className="secondary fit-edit" onClick={onEdit} disabled={run.pending}>
          Edit profile
        </button>
      </div>

      {run.error && <div className="error">{run.error}</div>}

      {(run.reasoning || (run.score && run.pending)) && (
        <section className="fit-reasoning panel">
          <h3>Why this score</h3>
          <div className={`md${run.pending ? ' streaming' : ''}`}>
            {run.reasoning ? (
              <Markdown remarkPlugins={REMARK_PLUGINS}>{run.reasoning}</Markdown>
            ) : (
              <span className="typing" aria-label="Writing">
                <i />
                <i />
                <i />
              </span>
            )}
          </div>
        </section>
      )}

      <div className="fit-dims">
        {dims.map((d) => (
          <article key={d.key} className={`fit-dim ${tone(d.rating)}`}>
            <header>
              <strong>{d.label}</strong>
              <span className="fit-rating">{d.rating === null ? (d.status === 'failed' ? 'Error' : 'Not specified') : `${d.rating}/10`}</span>
            </header>
            <div className="bar">
              <div className="fit-fill" style={{ width: `${(d.rating ?? 0) * 10}%` }} />
            </div>
            <div className="muted small">
              Weight {d.weight}
              {d.deal_breaker && <b className="fit-flag"> · deal-breaker</b>}
            </div>
            {d.verdict && <p className="fit-verdict">{d.verdict}</p>}
            {d.evidence.length > 0 && (
              <ul className="fit-evidence">
                {d.evidence.map((e, i) => (
                  <li key={i}>
                    {e.text} {cite(e.source)}
                  </li>
                ))}
              </ul>
            )}
          </article>
        ))}
      </div>

      {run.score && run.score.not_specified.length > 0 && (
        <p className="muted small">Not specified in the policy: {run.score.not_specified.join(', ')}. These were left out of the score.</p>
      )}
      <p className="muted small disclaimer">
        This score rates how well the document's terms match the profile you entered. It is not a recommendation to buy,
        and it does not consider price or other insurers. Confirm important terms with the insurer.
      </p>
    </div>
  )
}
