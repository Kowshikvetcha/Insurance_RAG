import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Report } from '../api'
import { figureUrl } from '../api'

export function IngestionReport({ report, sessionId }: { report: Report; sessionId: string }) {
  const units = Object.values(report.search_units).reduce((a, b) => a + b, 0)
  const described = report.figures.filter((f) => f.status !== 'skipped')
  return (
    <section className="panel report">
      <h3>What we extracted</h3>
      <div className="stats">
        <Stat label="Pages" value={report.pages} />
        <Stat label="Tables" value={report.tables.length} />
        <Stat label="Figures" value={described.length} />
        <Stat label="Definitions" value={report.definitions} />
        <Stat label="OCR pages" value={report.ocr_pages.length} />
        <Stat label="Search units" value={units} />
      </div>

      {report.warnings.length > 0 && (
        <ul className="warnings">
          {report.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}

      {report.tables.length > 0 && (
        <>
          <h4>Tables</h4>
          <p className="muted small">Check that the columns and rows look right. Answers about limits come from these.</p>
          {report.tables.map((t) => (
            <details key={t.id} className="item">
              <summary>
                <span className="title">{t.title}</span>
                <span className="muted small">
                  {t.pages} · {t.rows}×{t.cols}
                  {t.merged_from > 1 ? ` · joined from ${t.merged_from} pages` : ''}
                  {t.footnotes ? ` · ${t.footnotes} footnote(s)` : ''}
                </span>
              </summary>
              <div className="md table-scroll">
                <Markdown remarkPlugins={[remarkGfm]}>{t.preview}</Markdown>
              </div>
              {t.rows > 8 && <div className="muted small">Showing the first 8 of {t.rows} rows.</div>}
            </details>
          ))}
        </>
      )}

      {described.length > 0 && (
        <>
          <h4>Figures</h4>
          {described.map((f) => (
            <details key={f.id} className="item">
              <summary>
                <span className="title">{f.caption || `Figure on page ${f.page}`}</span>
                <span className="muted small">p. {f.page} · {f.status}</span>
              </summary>
              {f.has_image && <img className="figure" src={figureUrl(sessionId, f.id)} alt={f.caption || 'figure'} />}
            </details>
          ))}
        </>
      )}

      {report.definition_terms.length > 0 && (
        <details className="item">
          <summary>
            <span className="title">Defined terms</span>
            <span className="muted small">{report.definitions}</span>
          </summary>
          <div className="terms">
            {report.definition_terms.map((t) => (
              <span key={t} className="chip">
                {t}
              </span>
            ))}
          </div>
        </details>
      )}
      <div className="muted small">Processed in {report.timings.total}s</div>
    </section>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="stat">
      <div className="value">{value}</div>
      <div className="label">{label}</div>
    </div>
  )
}
