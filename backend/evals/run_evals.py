"""Offline evaluation suite: run a golden question set through the real pipeline and score it.

    cd backend
    .venv/Scripts/python -m evals.run_evals                      # fixture policy + golden set
    .venv/Scripts/python -m evals.run_evals --limit 5 --no-ragas  # quick smoke run

Per question it records:
  * our retrieval metrics: hit@k (an expected page is among the retrieved sources) and whether a
    "not in document" question was correctly refused (or an answerable one wrongly refused);
  * the live judge's metrics (app/evaluation.py): groundedness, context/answer relevance, confidence;
  * RAGAS 0.4 metrics (unless --no-ragas): faithfulness, answer relevancy, context precision and
    context recall against the reference answer.

Writes evals/reports/<run>.md and .csv. When Langfuse keys are configured it also uploads the run as
a dataset experiment ("dataset run"), so runs can be compared side by side in the Langfuse UI.
Uses the providers in backend/.env; with LLM_PROVIDER=fake only the retrieval metrics mean anything.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app import observability as obs  # noqa: E402
from app.config import get_settings, judge_target  # noqa: E402
from app.evaluation import evaluate_answer, is_refusal  # noqa: E402
from app.ingestion.pipeline import ingest  # noqa: E402
from app.providers import get_embeddings, get_judge_llm, get_llm, get_reranker  # noqa: E402
from app.rag import answer_stream, build_context, retrieve  # noqa: E402

DEFAULT_DATASET = BACKEND / "evals" / "datasets" / "sample_policy.jsonl"
DEFAULT_PDF = BACKEND / "tests" / "fixtures" / "sample_policy.pdf"
REPORTS = BACKEND / "evals" / "reports"

RAGAS_COLUMNS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
JUDGE_COLUMNS = ["groundedness", "context_relevance", "answer_relevance", "confidence"]


def load_dataset(path: Path, limit: int | None) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return items[:limit] if limit else items


async def run_questions(items: list[dict], session, settings, llm, reranker, judge) -> list[dict]:
    rows = []
    for i, item in enumerate(items, start=1):
        question = item["question"]
        t0 = time.perf_counter()
        answer = "".join([p async for kind, p in answer_stream(session, question, [], llm, reranker, settings) if kind == "token"])
        latency = time.perf_counter() - t0
        # Same deterministic retrieval the answer used, to get the exact contexts shown to the LLM.
        hits = retrieve(session, question, reranker, settings)
        excerpts, _, citations = build_context(session, question, hits)
        hits = hits[: len(citations)]
        pages = sorted({p for h in hits for p in h.parent.pages})
        expected = set(item.get("expected_pages") or [])
        refused = is_refusal(answer)
        live = await evaluate_answer(question, excerpts, answer, hits, judge)
        rows.append({
            "id": item["id"], "type": item["type"], "question": question, "reference": item["reference"],
            "answer": answer, "contexts": [h.parent.text for h in hits], "retrieved_pages": pages,
            "hit_at_k": (bool(expected & set(pages)) if expected else None),
            "refusal_correct": (refused if item["type"] == "not_in_doc" else not refused),
            "latency_s": round(latency, 2),
            "groundedness": live.groundedness, "context_relevance": live.context_relevance,
            "answer_relevance": live.answer_relevance, "confidence": live.confidence,
            "confidence_label": live.confidence_label, "judge_error": live.judge_error,
        })
        print(f"[{i}/{len(items)}] {item['id']} hit={rows[-1]['hit_at_k']} refusal_ok={rows[-1]['refusal_correct']} "
              f"conf={live.confidence} ({latency:.1f}s)", flush=True)
    return rows


def run_ragas(rows: list[dict], settings) -> None:
    """Adds RAGAS metric columns to rows in place."""
    from evals.ragas_compat import install

    install()
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.run_config import RunConfig

    dataset = EvaluationDataset.from_list([
        {"user_input": r["question"], "response": r["answer"], "retrieved_contexts": r["contexts"] or [""], "reference": r["reference"]}
        for r in rows
    ])
    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=LangchainLLMWrapper(get_judge_llm(settings)),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings(settings)),
        show_progress=True,
        # Few workers: RAGAS defaults to 16, which hits provider rate limits and times jobs out (NaN scores).
        run_config=RunConfig(max_workers=4, timeout=240),
    )
    frame = result.to_pandas()
    names = {"faithfulness": "faithfulness", "answer_relevancy": "answer_relevancy",
             "llm_context_precision_with_reference": "context_precision", "context_recall": "context_recall"}
    for row, (_, rec) in zip(rows, frame.iterrows()):
        for src, dst in names.items():
            value = rec.get(src)
            row[dst] = None if value is None or value != value else round(float(value), 4)  # NaN -> None


def mean(values) -> float | None:
    """Mean of numeric/bool values (True=1), ignoring None."""
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return round(statistics.fmean(vals), 4) if vals else None


def summarise(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["hit_at_k"] is not None]
    summary = {
        "questions": len(rows),
        "hit_rate_at_k": mean([r["hit_at_k"] for r in answerable]),
        "refusal_accuracy": mean([r["refusal_correct"] for r in rows]),
        "mean_latency_s": mean([r["latency_s"] for r in rows]),
        "judge_errors": sum(1 for r in rows if r.get("judge_error")),
    }
    for col in JUDGE_COLUMNS + RAGAS_COLUMNS:
        summary[col] = mean([r.get(col) for r in rows if r.get(col) is not None])
    by_type = {}
    for t in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == t]
        by_type[t] = {"n": len(sub), "hit_rate_at_k": mean([r["hit_at_k"] for r in sub if r["hit_at_k"] is not None]),
                      "groundedness": mean([r["groundedness"] for r in sub if r["groundedness"] is not None]),
                      "faithfulness": mean([r.get("faithfulness") for r in sub if r.get("faithfulness") is not None])}
    summary["by_type"] = by_type
    return summary


def write_reports(run_name: str, rows: list[dict], summary: dict, settings) -> tuple[Path, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = REPORTS / f"{run_name}.csv", REPORTS / f"{run_name}.md"
    cols = ["id", "type", "question", "hit_at_k", "refusal_correct", "latency_s", *JUDGE_COLUMNS, "confidence_label",
            *RAGAS_COLUMNS, "retrieved_pages", "judge_error", "answer", "reference"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: "–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = [
        f"# Evaluation run `{run_name}`", "",
        f"LLM `{settings.llm_provider}:{settings.llm_model}` · judge `{':'.join(judge_target(settings))}` · embeddings `{settings.embed_provider}:{settings.embed_model}` · "
        f"reranker `{settings.rerank_provider}` · final_k {settings.final_k}", "",
        "## Summary", "", "| Metric | Value |", "| --- | --- |",
    ]
    lines += [f"| {k} | {fmt(v)} |" for k, v in summary.items() if k != "by_type"]
    lines += ["", "## By question type", "", "| Type | n | hit@k | groundedness | faithfulness |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {t} | {s['n']} | {fmt(s['hit_rate_at_k'])} | {fmt(s['groundedness'])} | {fmt(s['faithfulness'])} |"
              for t, s in summary["by_type"].items()]
    lines += ["", "## Per question", "",
              "| id | type | hit@k | refusal ok | groundedness | ctx rel | ans rel | confidence | faithfulness | ctx precision | ctx recall |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['id']} | {r['type']} | {fmt(r['hit_at_k'])} | {fmt(r['refusal_correct'])} | {fmt(r['groundedness'])} | "
              f"{fmt(r['context_relevance'])} | {fmt(r['answer_relevance'])} | {fmt(r['confidence'])} | "
              f"{fmt(r.get('faithfulness'))} | {fmt(r.get('context_precision'))} | {fmt(r.get('context_recall'))} |" for r in rows]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, csv_path


def upload_to_langfuse(dataset_name: str, run_name: str, rows: list[dict], settings) -> str | None:
    """Upsert the golden items as a Langfuse dataset and record this run as an experiment on it.
    The task returns the precomputed answer, so no LLM call is repeated."""
    if not obs.configure(settings):
        return None
    from langfuse import Evaluation, get_client

    client = get_client(public_key=settings.langfuse_public_key)
    try:
        client.create_dataset(name=dataset_name, description="Golden questions for the synthetic sample policy")
    except Exception:  # noqa: BLE001 - already exists
        pass
    for r in rows:
        client.create_dataset_item(dataset_name=dataset_name, id=f"{dataset_name}-{r['id']}",
                                   input={"question": r["question"]}, expected_output={"answer": r["reference"]},
                                   metadata={"type": r["type"]})
    by_question = {r["question"]: r for r in rows}

    def task(*, item, **_):
        return by_question[item.input["question"]]["answer"]

    def evaluator(*, input, output, expected_output, metadata, **_):
        r = by_question[input["question"]]
        evals = [Evaluation(name=k, value=float(r[k])) for k in JUDGE_COLUMNS + RAGAS_COLUMNS if r.get(k) is not None]
        if r["hit_at_k"] is not None:
            evals.append(Evaluation(name="hit_at_k", value=bool(r["hit_at_k"]), data_type="BOOLEAN"))
        evals.append(Evaluation(name="refusal_correct", value=bool(r["refusal_correct"]), data_type="BOOLEAN"))
        return evals

    dataset = client.get_dataset(dataset_name)
    result = dataset.run_experiment(name=run_name, task=task, evaluators=[evaluator], max_concurrency=4,
                                    metadata={"llm": settings.llm_model, "judge": judge_target(settings)[1],
                                              "embeddings": settings.embed_model, "final_k": str(settings.final_k)})
    obs.flush()
    return getattr(result, "dataset_run_url", None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-ragas", action="store_true", help="skip RAGAS metrics (no extra LLM calls)")
    ap.add_argument("--no-langfuse", action="store_true", help="do not upload the run to Langfuse")
    ap.add_argument("--name", default=None, help="run name (default: timestamp)")
    args = ap.parse_args()

    # Live per-answer scoring is off here: this suite scores every answer itself.
    settings = get_settings().model_copy(update={"eval_enabled": False})
    real_llm = settings.llm_provider != "fake"
    run_name = args.name or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    items = load_dataset(args.dataset, args.limit)
    if not args.pdf.exists():
        from tests.fixtures.make_sample_policy import make

        make()

    print(f"Ingesting {args.pdf.name} ...", flush=True)
    session = ingest(args.pdf.read_bytes(), args.pdf.name, settings, lambda *a: None)
    llm, reranker = get_llm(settings), get_reranker(settings)
    judge = get_judge_llm(settings) if real_llm else None
    rows = asyncio.run(run_questions(items, session, settings, llm, reranker, judge))

    if args.no_ragas or not real_llm:
        print("Skipping RAGAS" + ("" if real_llm else " (LLM_PROVIDER=fake: answers are echoed excerpts)"))
    else:
        print("Running RAGAS ...", flush=True)
        run_ragas(rows, settings)

    summary = summarise(rows)
    md, csv_path = write_reports(run_name, rows, summary, settings)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_type"}, indent=2))
    print(f"Report: {md}\nCSV:    {csv_path}")
    if not args.no_langfuse:
        url = upload_to_langfuse(args.dataset.stem, run_name, rows, settings)
        print(f"Langfuse dataset run: {url}" if url else "Langfuse: not configured, skipped")


if __name__ == "__main__":
    main()
