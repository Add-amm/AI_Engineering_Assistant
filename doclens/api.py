"""FastAPI service and browser entry point."""

from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints

from .corpus import Corpus, Source
from .generation import Generator
from .retrieval import DEFAULT_MODE, Hit, Mode, SearchIndex
from .settings import Provider, Settings

WEB = Path(__file__).resolve().parent / "web"
Topic = Literal["python", "fastapi", "transformers", "rag"]


class SearchRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=1000)]
    mode: Mode = DEFAULT_MODE
    top_k: Annotated[int, Field(ge=1, le=12)] = 5
    topic: Topic | None = None
    collection: Annotated[str, StringConstraints(min_length=1, max_length=60)] | None = None
    version: Annotated[str, StringConstraints(min_length=1, max_length=60)] | None = None


class Citation(BaseModel):
    number: int
    id: str
    source_id: str
    title: str
    heading: tuple[str, ...]
    topic: Topic
    collection: str
    version: str
    url: str


class Passage(Citation):
    text: str
    score: float
    start: int
    end: int


class SearchResponse(BaseModel):
    mode: Mode
    elapsed_ms: float
    evidence: list[Passage]


class AskResponse(SearchResponse):
    answer: str
    citations: list[Citation]
    abstained: bool
    generation: Provider
    note: str | None


def passage(number: int, hit: Hit) -> Passage:
    chunk = hit.chunk
    source = chunk.source
    return Passage(
        number=number, id=chunk.id, source_id=source.id, title=source.title,
        heading=chunk.heading, topic=source.collection.topic, collection=source.collection.id,
        version=source.collection.version, url=chunk.url, text=chunk.text, score=hit.score,
        start=chunk.start, end=chunk.end,
    )


def source_record(source: Source) -> dict:
    return {
        "id": source.id, "title": source.title, "topic": source.collection.topic,
        "collection": source.collection.id, "version": source.collection.version,
        "revision": source.collection.revision, "license": source.collection.license,
        "format": source.format, "url": source.pinned_url, "official_url": source.url,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(service: FastAPI):
        config = settings or Settings.from_env()
        corpus = Corpus()
        index = SearchIndex(corpus.chunks(), config)
        service.state.corpus = corpus
        service.state.index = index
        service.state.generator = Generator(config)
        yield

    service = FastAPI(
        title="DocLens", version="1.0.0",
        description="Search pinned official docs and generate answers with passage-level citations.",
        lifespan=lifespan,
    )

    @service.get("/health")
    def health(request: Request) -> dict:
        state = request.app.state
        return {
            "status": "ok", "documents": len(state.corpus.sources),
            "passages": len(state.index.chunks), "topics": 4,
            "default_mode": DEFAULT_MODE, "embeddings": state.index.dense.name,
            "reranker": state.index.reranker_name, "generation": state.generator.provider,
            "model": state.generator.model,
        }

    @service.get("/sources")
    def sources(request: Request) -> dict:
        corpus = request.app.state.corpus
        counts = Counter(source.collection.id for source in corpus.sources)
        return {
            "retrieved_on": corpus.retrieved_on,
            "collections": [
                {"id": group.id, "topic": group.topic, "title": group.title,
                 "description": group.description, "version": group.version,
                 "revision": group.revision, "license": group.license, "url": group.url,
                 "count": counts[group.id]}
                for group in corpus.collections.values()
            ],
            "documents": [source_record(source) for source in corpus.sources],
        }

    def retrieve(data: SearchRequest, index: SearchIndex) -> tuple[list[Hit], list[Passage]]:
        hits = index.search(data.question, data.mode, data.top_k, data.topic, data.collection, data.version)
        return hits, [passage(number, hit) for number, hit in enumerate(hits, 1)]

    @service.post("/search", response_model=SearchResponse)
    def search(data: SearchRequest, request: Request) -> SearchResponse:
        start = perf_counter()
        _, evidence = retrieve(data, request.app.state.index)
        return SearchResponse(mode=data.mode, elapsed_ms=round((perf_counter() - start) * 1000, 2),
                              evidence=evidence)

    @service.post("/ask", response_model=AskResponse)
    def ask(data: SearchRequest, request: Request) -> AskResponse:
        start = perf_counter()
        state = request.app.state
        hits, evidence = retrieve(data, state.index)
        result = state.generator.answer(data.question, hits, state.index)
        return AskResponse(
            mode=data.mode, elapsed_ms=round((perf_counter() - start) * 1000, 2),
            evidence=evidence, answer=result.text, abstained=result.abstained,
            citations=[Citation.model_validate(evidence[number - 1].model_dump()) for number in result.numbers],
            generation=result.mode, note=result.note,
        )

    @service.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})

    service.mount("/static", StaticFiles(directory=WEB), name="static")
    return service


app = create_app()
