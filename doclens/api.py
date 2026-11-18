"""FastAPI service and browser entry point."""

from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
import re
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
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=1000)]
CONTINUATION = re.compile(r"(?:continue|tell me more|why|how so|how does (?:it|this|that) work)[?!. ]*", re.I)
FOLLOW_UP = re.compile(
    r"^(?:and\b|what about\b|tell me more\b|(?:continue|why|how so)[?!. ]*$|et\b)"
    r"|\b(?:cela|ça|the (?:answer|example) above)\b"
    r"|\b(?:does|did|could|would|should|will|is|are)\s+(?:it|this|that|they)"
    r"\s+(?:work|run|support|use|called|apply|differ)\b"
    r"|\b(?:explain|test|use|using|compare|with|about)\s+(?:it|this|that|them|those)"
    r"(?:[?!. ]*$|\s+(?:with|in|for|as|to)\b)"
    r"|^(?:(?:can|could|would) you\s+)?(?:(?:show|give|provide)\s+(?:me\s+)?)?"
    r"(?:(?:a|an|the|some)\s+)?(?:code\s+example|example|examples|snippet|more details)"
    r"(?:\s+please)?[?!. ]*$",
    re.I,
)


class SearchRequest(BaseModel):
    question: Question
    mode: Mode = DEFAULT_MODE
    top_k: Annotated[int, Field(ge=1, le=12)] = 5
    topic: Topic | None = None
    collection: Annotated[str, StringConstraints(min_length=1, max_length=60)] | None = None
    version: Annotated[str, StringConstraints(min_length=1, max_length=60)] | None = None


class ConversationTurn(BaseModel):
    question: Question
    answer: Annotated[str, StringConstraints(max_length=2500)]


class AskRequest(SearchRequest):
    history: list[ConversationTurn] = Field(default_factory=list, max_length=6)


def retrieval_query(question: str, history: list[ConversationTurn]) -> str:
    if not history or not FOLLOW_UP.search(question):
        return question
    anchor = next(
        (turn.question for turn in reversed(history) if not FOLLOW_UP.search(turn.question)),
        history[-1].question,
    )
    if CONTINUATION.fullmatch(question):
        return anchor
    # Keep the current question intact and the combined query within the search API limit.
    prefix = anchor[:1000 - len(question) - 1].rstrip() if len(question) < 999 else ""
    return f"{prefix}\n{question}" if prefix else question


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
    retrieval_query: str


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
    def ask(data: AskRequest, request: Request) -> AskResponse:
        start = perf_counter()
        state = request.app.state
        query = retrieval_query(data.question, data.history)
        hits, evidence = retrieve(data.model_copy(update={"question": query}), state.index)
        result = state.generator.answer(
            data.question, hits, state.index, retrieval_query=query,
            history=[turn.model_dump() for turn in data.history],
        )
        return AskResponse(
            mode=data.mode, elapsed_ms=round((perf_counter() - start) * 1000, 2),
            evidence=evidence, answer=result.text, abstained=result.abstained,
            citations=[Citation.model_validate(evidence[number - 1].model_dump()) for number in result.numbers],
            generation=result.mode, note=result.note, retrieval_query=query,
        )

    @service.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})

    service.mount("/static", StaticFiles(directory=WEB), name="static")
    return service


app = create_app()
