"""Repeatable corpus, indexing, search and evaluation commands."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import faiss

from .corpus import ROOT, Corpus
from .evaluation import cases_from, evaluate
from .generation import Generator
from .retrieval import DEFAULT_MODE, MODES, SearchIndex
from .settings import Settings


def output_json(data: dict, path: Path | None = None) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m doclens", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Check upstream document, example and license checksums")
    commands.add_parser("fetch", help="Restore missing files from the frozen upstream source inventory")
    index = commands.add_parser("index", help="Export citable passages and their FAISS vectors")
    index.add_argument("--output", type=Path, default=Path(".cache/index"))
    benchmark = commands.add_parser("evaluate", help="Compare four retrieval methods on frozen questions")
    benchmark.add_argument("--cases", type=Path, default=ROOT / "evaluation.json")
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument("--skip-answers", action="store_true")
    benchmark.add_argument("--answer-mode", choices=MODES, default=DEFAULT_MODE)
    for command in ("search", "ask"):
        search = commands.add_parser(command, help=f"{command.capitalize()} without starting the web server")
        search.add_argument("question")
        search.add_argument("--mode", choices=MODES, default=DEFAULT_MODE)
        search.add_argument("--top-k", type=int, choices=range(1, 13), metavar="1–12", default=5)
        search.add_argument("--topic", choices=("python", "fastapi", "transformers", "rag"))
        search.add_argument("--collection")
        search.add_argument("--version")
    for command in (index, benchmark):
        command.add_argument("--strategy", choices=("heading", "recursive"), default="heading")
        command.add_argument("--target", type=int, choices=range(120, 801), metavar="120–800", default=360)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = Corpus()
    if args.command == "verify":
        print(f"Verified {corpus.verify()} files across {len(corpus.sources)} pinned pages.")
        return
    if args.command == "fetch":
        print(f"Restored {corpus.fetch()} upstream files. Verified {corpus.verify()} known assets.")
        return

    start = perf_counter()
    strategy = getattr(args, "strategy", "heading")
    target = getattr(args, "target", 360)
    settings = Settings.from_env()
    index = SearchIndex(corpus.chunks(strategy, target), settings)
    build_seconds = round(perf_counter() - start, 3)
    if args.command == "index":
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "passages.jsonl").open("w", encoding="utf-8") as file:
            for chunk in index.chunks:
                file.write(json.dumps({**asdict(chunk), "url": chunk.url}, ensure_ascii=False) + "\n")
        faiss.write_index(index.dense.index, str(args.output / "vectors.faiss"))
        output_json({
            "strategy": strategy, "target": target, "documents": len(corpus.sources),
            "passages": len(index.chunks), "embeddings": index.dense.name,
        }, args.output / "metadata.json")
        print(f"Exported {len(index.chunks)} passages and FAISS vectors to {args.output}.")
    elif args.command == "evaluate":
        generator = None if args.skip_answers else Generator(settings)
        report = evaluate(index, cases_from(args.cases), generator, args.answer_mode)
        output_json({
            "strategy": strategy, "target": target, "index_build_seconds": build_seconds, **report,
        }, args.output)
        if args.output:
            print(f"Wrote evaluation to {args.output}.")
    else:
        hits = index.search(args.question, args.mode, args.top_k, args.topic, args.collection, args.version)
        evidence = [
            {"number": number, "score": hit.score, "id": hit.chunk.id, "title": hit.chunk.source.title,
             "heading": hit.chunk.heading, "url": hit.chunk.url, "text": hit.chunk.text}
            for number, hit in enumerate(hits, 1)
        ]
        payload = {"question": args.question, "mode": args.mode, "evidence": evidence}
        if args.command == "ask":
            payload["answer"] = asdict(Generator(settings).answer(args.question, hits, index))
        output_json(payload)


if __name__ == "__main__":
    main()
