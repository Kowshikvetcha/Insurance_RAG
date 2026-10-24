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
    return typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
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

/** POST /chat and parse the SSE stream (EventSource can't POST, so we read the body ourselves). */
export async function streamChat(sessionId: string, question: string, history: ChatTurn[], h: ChatHandlers, signal?: AbortSignal) {
  const res = await fetch(`${API}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, question, history }),
    signal,
  })
  if (!res.ok || !res.body) {
    h.onError(await errorText(res))
    return
  }
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
      if (!data.length) continue
      const payload = JSON.parse(data.join('\n'))
      if (event === 'token') h.onToken(payload)
      else if (event === 'query') h.onQuery?.(payload)
      else if (event === 'citations') h.onCitations(payload)
      else if (event === 'error') h.onError(payload)
    }
  }
}
