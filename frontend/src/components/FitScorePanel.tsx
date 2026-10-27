import { useEffect, useRef, useState } from 'react'
import type { Citation, HealthProfile } from '../api'
import { streamFitScore } from '../api'
import { SourceViewer } from './ChatWindow'
import { EMPTY_PROFILE, FitScoreForm } from './FitScoreForm'
import type { FitRun } from './FitScoreResult'
import { FitScoreResult } from './FitScoreResult'

const TOKEN_FLUSH_MS = 60

/** Policy fit: the health profile form, then the streamed score. Mount it with key={sessionId}
 * so a new upload starts clean; the profile is never written anywhere but this state. */
export function FitScorePanel({ sessionId }: { sessionId: string }) {
  const [profile, setProfile] = useState<HealthProfile>(EMPTY_PROFILE)
  const [run, setRun] = useState<FitRun | null>(null)
  const [editing, setEditing] = useState(true)
  const [open, setOpen] = useState<Citation | null>(null)
  const [rejected, setRejected] = useState<string | null>(null)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => () => abort.current?.abort(), [])

  // Same buffering as the chat: re-render the reasoning a few times a second, not per token.
  const tokenBuffer = useRef('')
  const flushTimer = useRef<number | null>(null)
  const patch = (fn: (r: FitRun) => FitRun) => setRun((r) => (r ? fn(r) : r))
  const flushTokens = () => {
    if (flushTimer.current !== null) {
      window.clearTimeout(flushTimer.current)
      flushTimer.current = null
    }
    const text = tokenBuffer.current
    if (!text) return
    tokenBuffer.current = ''
    patch((r) => ({ ...r, reasoning: r.reasoning + text }))
  }

  const start = async (p: HealthProfile) => {
    abort.current?.abort()
    const controller = new AbortController()
    abort.current = controller
    setProfile(p)
    setRejected(null)
    setEditing(false)
    let started = false // true once the server accepted the request and began streaming
    setRun({ stage: '', citations: [], dimensions: [], score: null, reasoning: '', error: null, pending: true })
    try {
      await streamFitScore(
        sessionId,
        p,
        {
          onStage: (stage) => {
            started = true
            patch((r) => ({ ...r, stage }))
          },
          onCitations: (citations) => patch((r) => ({ ...r, citations })),
          onDimension: (d) => patch((r) => ({ ...r, dimensions: [...r.dimensions, d] })),
          onScore: (score) => patch((r) => ({ ...r, score })),
          onToken: (t) => {
            tokenBuffer.current += t
            if (flushTimer.current === null) flushTimer.current = window.setTimeout(flushTokens, TOKEN_FLUSH_MS)
          },
          onError: (error) => {
            if (started) return patch((r) => ({ ...r, error }))
            // Rejected before any work (e.g. validation): go back to the form and show why.
            setRejected(error)
            setEditing(true)
          },
        },
        controller.signal,
      )
    } catch (e) {
      if (!controller.signal.aborted) patch((r) => ({ ...r, error: e instanceof Error ? e.message : String(e) }))
    } finally {
      flushTokens()
      patch((r) => ({ ...r, pending: false }))
    }
  }

  return (
    <div className="fit">
      <div className="fit-scroll">
        <div className="fit-inner">
          {editing || !run ? (
            <>
              <h2>How well does this policy fit you?</h2>
              <p className="muted">
                Tell us about the people to be covered. Each relevant area of the policy is rated 1–10 against your
                profile, with page citations, and combined into one score.
              </p>
              <FitScoreForm key={rejected ?? ''} initial={profile} busy={run?.pending ?? false} serverError={rejected} onSubmit={start} />
            </>
          ) : (
            <FitScoreResult run={run} onCite={setOpen} onEdit={() => setEditing(true)} />
          )}
        </div>
      </div>
      {open && <SourceViewer citation={open} sessionId={sessionId} onClose={() => setOpen(null)} />}
    </div>
  )
}
