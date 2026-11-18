"""Evaluate frozen, passage-level references independently of answer generation."""

from dataclasses import dataclass
import json
from math import log2
from pathlib import Path
import platform
import re
from statistics import mean, median, quantiles
from time import perf_counter

from .corpus import Chunk
from .generation import Generator
from .retrieval import MODES, Mode, SearchIndex


@dataclass(frozen=True)
class Reference:
    source: str
    anchor: str


@dataclass(frozen=True)
class Case:
    id: str
    kind: str
    question: str
    reference_answer: str
    evidence: tuple[Reference, ...]


def cases_from(path: Path) -> list[Case]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    return [
        Case(entry["id"], entry["kind"], entry["question"], entry["reference_answer"],
             tuple(Reference(**reference) for reference in entry["evidence"]))
        for entry in entries
    ]


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def evidence_map(cases: list[Case], chunks: list[Chunk]) -> dict[str, list[set[str]]]:
    by_source = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source.id, []).append((chunk.id, normalize(chunk.text)))
    matches = {}
    missing = []
    for case in cases:
        groups = []
        for reference in case.evidence:
            anchor = normalize(reference.anchor)
            ids = {
                chunk_id for chunk_id, text in by_source.get(reference.source, [])
                if f" {anchor} " in f" {text} "
            }
            if not ids:
                missing.append(f"{case.id}: {reference.source} · {reference.anchor}")
            groups.append(ids)
        matches[case.id] = groups
    if missing:
        raise ValueError("References without a matching passage:\n" + "\n".join(missing))
    return matches


def metrics(result: list[str], targets: list[set[str]]) -> tuple[float, float, float]:
    matched = set()
    first = 0.0
    dcg = 0.0
    for rank, chunk_id in enumerate(result, 1):
        new = {i for i, ids in enumerate(targets) if chunk_id in ids} - matched
        if new:
            first = first or 1 / rank
            dcg += 1 / log2(rank + 1)
            matched.update(new)
    ideal = sum(1 / log2(rank + 1) for rank in range(1, min(5, len(targets)) + 1))
    return len(matched) / len(targets), first, dcg / ideal if ideal else 0.0


def evaluate(index: SearchIndex, cases: list[Case], generator: Generator | None, answer_mode: Mode) -> dict:
    targets = evidence_map(cases, index.chunks)
    comparison = {}
    for mode in MODES:
        latencies = []
        results = []
        for case in cases:
            start = perf_counter()
            hits = index.search(case.question, mode=mode, top_k=5)
            latencies.append((perf_counter() - start) * 1000)
            if targets[case.id]:
                recall, mrr, ndcg = metrics([hit.chunk.id for hit in hits], targets[case.id])
                results.append((case.id, recall, mrr, ndcg))
        comparison[mode] = {
            "recall@5": round(mean(row[1] for row in results), 4),
            "mrr@5": round(mean(row[2] for row in results), 4),
            "ndcg@5": round(mean(row[3] for row in results), 4),
            "p50_ms": round(median(latencies), 2),
            "p95_ms": round(quantiles(latencies, n=20, method="inclusive")[18], 2),
            "incomplete": [case_id for case_id, recall, _, _ in results if recall < 1],
        }

    outcome = {
        "cases": len(cases),
        "answerable": sum(bool(case.evidence) for case in cases),
        "unanswerable": sum(not case.evidence for case in cases),
        "passages": len(index.chunks),
        "embeddings": index.dense.name,
        "reranker": index.reranker_name,
        "runtime": f"Python {platform.python_version()} · {platform.system()} {platform.machine()}",
        "retrieval": comparison,
    }
    if generator is not None:
        rows = []
        for case in cases:
            hits = index.search(case.question, mode=answer_mode, top_k=5)
            answer = generator.answer(case.question, hits, index)
            citations = [hits[number - 1].chunk.id for number in answer.numbers]
            evidence = targets[case.id]
            rows.append({
                "id": case.id,
                "mode": answer.mode,
                "correct_abstention": answer.abstained == (not evidence),
                "abstained": answer.abstained,
                "cited_annotated_evidence": round(metrics(citations, evidence)[0], 4) if evidence else None,
                "answer": answer.text,
                "citations": [hits[number - 1].chunk.url for number in answer.numbers],
                "provider_fallback": answer.note is not None,
            })
        outcome["answers"] = {
            "retrieval": answer_mode,
            "provider": generator.provider,
            "model": generator.model,
            "abstention_accuracy": round(mean(row["correct_abstention"] for row in rows), 4),
            "answered_in_scope": sum(not row["abstained"] for case, row in zip(cases, rows) if case.evidence),
            "abstained_out_of_scope": sum(row["abstained"] for case, row in zip(cases, rows) if not case.evidence),
            "cited_annotated_evidence_recall": round(mean(
                row["cited_annotated_evidence"] for row in rows if row["cited_annotated_evidence"] is not None
            ), 4),
            "provider_fallbacks": sum(row["provider_fallback"] for row in rows),
            "cases": rows,
        }
    return outcome
