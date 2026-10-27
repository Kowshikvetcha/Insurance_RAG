export const API = '/api'

export interface JobState {
  id: string
  filename: string
  status: 'queued' | 'running' | 'done' | 'error'
  stage: string
  message: string
  percent: number
  session_id: string | null
  error: string | null
  eta_seconds?: number | null // rough estimate of time left, from the backend
  version?: number
}

export interface TableInfo {
  id: string
  title: string
  pages: string
  rows: number
  cols: number
  merged_from: number
  footnotes: number
  preview: string
  summary: string
}

export interface FigureInfo {
  id: string
  page: number | null
  caption: string
  status: 'described' | 'skipped' | 'caption/OCR text only' | 'no description'
  reason: string
  has_image: boolean
}

export interface Report {
  filename: string
  parser: string
  pages: number
  ocr_pages: number[]
  tables: TableInfo[]
  table_fragments: number
  tables_summarised_by_llm: number
  figures: FigureInfo[]
  figures_described_by_llm: number
  definitions: number
  definition_terms: string[]
  sections: number
  parents: number
  search_units: Record<string, number>
  warnings: string[]
  timings: Record<string, number>
}

export interface Citation {
  source: number
  type: 'text' | 'table' | 'figure'
  pages: number[]
  page_label: string
  section: string
  title: string
  snippet: string
  content: string
  figure_id: string | null
}

export interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
}

async function errorText(res: Response): Promise<string> {
  try {
    const body = await res.json()
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail)) {
      // FastAPI validation errors: [{loc: ['body', 'profile', 'age'], msg}] → "age: Input should be …"
      return body.detail
        .map((d: { loc?: unknown[]; msg?: string }) => `${String(d.loc?.[d.loc.length - 1] ?? 'input').replace(/_/g, ' ')}: ${d.msg}`)
        .join('; ')
    }
    return JSON.stringify(body.detail)
  } catch {
    return `${res.status} ${res.statusText}`
  }
}

export async function uploadDocument(file: File): Promise<{ job_id: string; filename: string }> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API}/documents`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await errorText(res))
  return res.json()
}

export function watchJob(jobId: string, onUpdate: (job: JobState) => void): () => void {
  const es = new EventSource(`${API}/documents/jobs/${jobId}/events`)
  es.addEventListener('progress', (e) => {
    const job = JSON.parse((e as MessageEvent).data) as JobState
    onUpdate(job)
    if (job.status === 'done' || job.status === 'error') es.close()
  })
  es.onerror = () => {
    // Connection dropped: fall back to a single status fetch so the UI doesn't hang.
    es.close()
    fetch(`${API}/documents/jobs/${jobId}`).then((r) => (r.ok ? r.json() : null)).then((j) => j && onUpdate(j))
  }
  return () => es.close()
}

export async function fetchReport(sessionId: string): Promise<Report> {
  const res = await fetch(`${API}/documents/${sessionId}/report`)
  if (!res.ok) throw new Error(await errorText(res))
  return res.json()
}

export function figureUrl(sessionId: string, figureId: string): string {
  return `${API}/documents/${sessionId}/figures/${figureId}.png`
}

export interface ChatHandlers {
  onQuery?: (q: string) => void
  onToken: (t: string) => void
  onCitations: (c: Citation[]) => void
  onError: (msg: string) => void
}

/** POST JSON and dispatch each SSE event (EventSource can't POST, so we read the body ourselves).
 * Returns the HTTP error text when the request is rejected before streaming starts. */
async function postSse(path: string, body: unknown, onEvent: (event: string, payload: unknown) => void, signal?: AbortSignal): Promise<string | null> {
  const res = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) return errorText(res)
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
    let sep: number
    while ((sep = buffer.indexOf('\n\n')) >= 0) {
      const raw = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      let event = 'message'
      const data: string[] = []
      for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''))
      }
      if (data.length) onEvent(event, JSON.parse(data.join('\n')))
    }
  }
  return null
}

/** POST /chat. Events: query, token…, citations, error, done. */
export async function streamChat(sessionId: string, question: string, history: ChatTurn[], h: ChatHandlers, signal?: AbortSignal) {
  const failed = await postSse('/chat', { session_id: sessionId, question, history }, (event, payload) => {
    if (event === 'token') h.onToken(payload as string)
    else if (event === 'query') h.onQuery?.(payload as string)
    else if (event === 'citations') h.onCitations(payload as Citation[])
    else if (event === 'error') h.onError(payload as string)
  }, signal)
  if (failed) h.onError(failed)
}

// ----------------------------------------------------------------- policy fit score

export type Member = 'self' | 'spouse' | 'children' | 'parents'

/** Mirrors `HealthProfile` in backend/app/schemas.py. Kept in memory only, never stored. */
export interface HealthProfile {
  age: number
  gender: 'female' | 'male' | 'other' | 'unspecified'
  city_tier: 'metro' | 'tier2' | 'tier3' | 'unspecified'
  covered_members: Member[]
  desired_sum_insured: number | null // rupees
  annual_budget: number | null // rupees
  conditions: string[]
  medications: string
  past_surgeries: string
  hospitalised_last_2y: boolean
  smoker: boolean
  alcohol: 'none' | 'occasional' | 'regular'
  height_cm: number | null
  weight_kg: number | null
  hazardous_activity: string
  planning_pregnancy: boolean
  planned_procedures: string[]
  wants_opd: boolean
  wants_dental: boolean
  wants_ayush: boolean
  avoid_copay: boolean
  avoid_room_rent_limit: boolean
}

export interface DimensionResult {
  key: string
  label: string
  weight: number
  status: 'rated' | 'not_specified' | 'failed'
  rating: number | null
  deal_breaker: boolean
  verdict: string
  evidence: { source: number; text: string }[]
  sources: number[]
}

export interface FitScore {
  score: number | null
  band: string
  confidence: 'High' | 'Medium' | 'Low'
  rated_weight_share: number
  capped: boolean
  deal_breakers: string[]
  not_specified: string[]
  breakdown: { key: string; label: string; weight: number; rating: number | null; share: number }[]
}

export interface FitHandlers {
  onStage: (s: string) => void
  onCitations: (c: Citation[]) => void
  onDimension: (d: DimensionResult) => void
  onScore: (s: FitScore) => void
  onToken: (t: string) => void
  onError: (msg: string) => void
}

/** POST /fit-score. Events: stage, citations, dimension…, score, token…, error, done. */
export async function streamFitScore(sessionId: string, profile: HealthProfile, h: FitHandlers, signal?: AbortSignal) {
  const failed = await postSse('/fit-score', { session_id: sessionId, profile }, (event, payload) => {
    if (event === 'stage') h.onStage(payload as string)
    else if (event === 'citations') h.onCitations(payload as Citation[])
    else if (event === 'dimension') h.onDimension(payload as DimensionResult)
    else if (event === 'score') h.onScore(payload as FitScore)
    else if (event === 'token') h.onToken(payload as string)
    else if (event === 'error') h.onError(payload as string)
  }, signal)
  if (failed) h.onError(failed)
}
