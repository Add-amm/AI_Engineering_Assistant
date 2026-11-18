"""Independent BM25 and dense retrievers, rank fusion, and candidate reranking."""

from collections import Counter
from dataclasses import dataclass
from math import log
import re
from typing import Literal

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from snowballstemmer import stemmer

from .corpus import FENCED_CODE, Chunk
from .settings import Settings

Mode = Literal["bm25", "dense", "hybrid", "reranked"]
MODES: tuple[Mode, ...] = ("bm25", "dense", "hybrid", "reranked")
DEFAULT_MODE: Mode = "reranked"
LEXEME = re.compile(r"[^\W\d_]\w*", re.UNICODE)
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|_+")
HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
STOP_WORDS = ENGLISH_STOP_WORDS | {
    "tell", "explain", "example", "examples", "different", "difference", "work", "works",
}
TOPIC_HINTS = {
    "python": re.compile(r"\b(?:asyncio|cpython|dataclass(?:es)?)\b", re.I),
    "fastapi": re.compile(r"\bfastapi\b", re.I),
    "transformers": re.compile(r"\b(?:(?<!sentence )transformers|bert|gpt-2|lora|hugging face)\b", re.I),
    "rag": re.compile(r"\b(?:rag|retrieval.augmented generation|semantic search|sentence transformers|langchain|vector stores?)\b", re.I),
}


def question_topics(question: str) -> set[str]:
    return {topic for topic, pattern in TOPIC_HINTS.items() if pattern.search(question)}


def terms(text: str) -> list[str]:
    tokens = []
    for original in LEXEME.findall(text):
        parts = CAMEL_BOUNDARY.split(original)
        variants = [original] if len(parts) == 1 else [original, *parts]
        tokens.extend(
            word for variant in variants
            if len(word := variant.casefold()) > 1 and (word not in STOP_WORDS or variant in HTTP_METHODS)
        )
    return stemmer("english").stemWords(tokens)


def query_terms(question: str) -> list[str]:
    words = terms(question)
    if re.search(r"\b(?:endpoints?|routes?)\b", question, re.I):
        # FastAPI uses "path operation" throughout its docs.
        words.extend(word for word in terms("path operation") if word not in words)
    return words


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class DenseRetriever:
    def __init__(self, texts: list[str], tokenized: list[list[str]], settings: Settings):
        self.encoder = None
        if settings.embeddings == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError("Install the semantic extra to enable Sentence Transformers.") from error
            self.encoder = SentenceTransformer(
                settings.embedding_model,
                revision=settings.embedding_revision,
                local_files_only=settings.local_models_only,
                trust_remote_code=False,
                device="cpu",
            )
            self.name = settings.embedding_model
            values = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        else:
            self.name = "TF-IDF + LSA"
            self.vectorizer = TfidfVectorizer(
                tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=(1, 2),
                min_df=2, max_features=18_000, sublinear_tf=True,
            )
            sparse = self.vectorizer.fit_transform(" ".join(words) for words in tokenized)
            components = min(128, sparse.shape[0], sparse.shape[1] - 1)
            self.projection = TruncatedSVD(n_components=components, n_iter=5, random_state=42)
            values = self.projection.fit_transform(sparse)
        normalized = np.ascontiguousarray(values, dtype=np.float32)
        faiss.normalize_L2(normalized)
        self.index = faiss.IndexFlatIP(normalized.shape[1])
        self.index.add(normalized)

    def scores(self, question: str, tokens: list[str]) -> np.ndarray:
        if self.encoder is not None:
            vector = self.encoder.encode([question], normalize_embeddings=True, show_progress_bar=False)
        else:
            vector = self.projection.transform(self.vectorizer.transform([" ".join(tokens)]))
        query = np.ascontiguousarray(vector, dtype=np.float32)
        faiss.normalize_L2(query)
        scores, positions = self.index.search(query, self.index.ntotal)
        aligned = np.zeros(self.index.ntotal, dtype=np.float32)
        aligned[positions[0]] = scores[0]
        return aligned


class SearchIndex:
    def __init__(self, chunks: list[Chunk], settings: Settings):
        if not chunks:
            raise ValueError("The corpus has no searchable passages.")
        self.chunks = chunks
        texts = [chunk.search_text for chunk in chunks]
        tokenized = [terms(text) for text in texts]
        self.position = {chunk.id: i for i, chunk in enumerate(chunks)}
        self.topics = [chunk.source.collection.topic for chunk in chunks]
        self.words = [set(tokens) for tokens in tokenized]
        self.headings = [set(terms(" ".join((chunk.source.title, *chunk.heading)))) for chunk in chunks]
        self.prose_words = [set(terms(FENCED_CODE.sub("", chunk.text))) for chunk in chunks]
        self.bigrams = [set(zip(tokens, tokens[1:])) for tokens in tokenized]
        self.frequency = Counter(token for tokens in self.words for token in tokens)
        self.lexical = BM25Okapi(tokenized)
        self.dense = DenseRetriever(texts, tokenized, settings)
        self.cross_encoder = None
        if settings.rerank_model:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise RuntimeError("Install the semantic extra to enable cross-encoder reranking.") from error
            self.cross_encoder = CrossEncoder(
                settings.rerank_model,
                revision=settings.rerank_revision,
                local_files_only=settings.local_models_only,
                trust_remote_code=False,
                device="cpu",
            )
            if self.cross_encoder.model.config.num_labels != 1:
                raise ValueError("The configured cross-encoder must return one relevance score per passage.")
        self.reranker_name = settings.rerank_model or "Term and phrase coverage"

    def weights(self, tokens: list[str]) -> dict[str, float]:
        size = len(self.chunks)
        return {token: 1 + log((size + 1) / (self.frequency[token] + 1)) for token in set(tokens)}

    def search(
        self, question: str, mode: Mode = DEFAULT_MODE, top_k: int = 5,
        topic: str | None = None, collection: str | None = None, version: str | None = None,
    ) -> list[Hit]:
        tokens = query_terms(question)
        if not tokens:
            return []
        eligible = [
            i for i, chunk in enumerate(self.chunks)
            if (topic is None or chunk.source.collection.topic == topic)
            and (collection is None or chunk.source.collection.id == collection)
            and (version is None or chunk.source.collection.version == version)
        ]
        if not eligible:
            return []
        pool_size = max(30, top_k * 3)
        lexical, dense = [], []
        lexical_scores = None
        if mode != "dense":
            lexical_scores = self.lexical.get_scores(tokens)
            lexical = self.rank(lexical_scores, eligible, pool_size, 0)
        if mode != "bm25":
            dense = self.rank(self.dense.scores(question, tokens), eligible, pool_size, 0.02)
        if mode == "bm25":
            ordered = lexical
        elif mode == "dense":
            ordered = dense
        else:
            fused = Counter()
            for ranking in (lexical, dense):
                for rank, (index, _) in enumerate(ranking, 1):
                    fused[index] += 1 / (60 + rank)
            if mode == "reranked":
                scopes = question_topics(question) if len(eligible) == len(self.chunks) else set()
                if len(scopes) > 1:
                    for scope in sorted(scopes):
                        scoped = [i for i in eligible if self.topics[i] == scope]
                        for i, _ in self.rank(lexical_scores, scoped, 4, 0):
                            fused.setdefault(i, 1 / (60 + pool_size))
                else:
                    scopes = set()
                candidates = sorted(fused.items(), key=lambda hit: (-hit[1], hit[0]))
                if self.cross_encoder is not None and not scopes:
                    candidates = candidates[:max(24, top_k * 2)]
                ordered = self.rerank(question, tokens, candidates)
                ordered = self.select(ordered, tokens, scopes, top_k)
            else:
                ordered = sorted(fused.items(), key=lambda hit: (-hit[1], hit[0]))
        return [Hit(self.chunks[i], float(score)) for i, score in ordered[:top_k]]

    @staticmethod
    def rank(
        scores: np.ndarray, eligible: list[int], limit: int, threshold: float,
    ) -> list[tuple[int, float]]:
        ranked = sorted(
            ((i, float(scores[i])) for i in eligible if scores[i] > threshold),
            key=lambda hit: (-hit[1], hit[0]),
        )
        return ranked[:limit]

    def rerank(
        self, question: str, tokens: list[str], candidates: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        if not candidates:
            return []
        if self.cross_encoder is not None:
            pairs = [(question, self.chunks[i].search_text) for i, _ in candidates]
            scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
            return sorted(
                ((candidate[0], float(score)) for candidate, score in zip(candidates, scores)),
                key=lambda hit: (-hit[1], hit[0]),
            )
        weights = self.weights(tokens)
        scale = sum(weights.values())
        phrases = set(zip(tokens, tokens[1:]))
        ranked = []
        for i, fusion in candidates:
            overlap = sum(weights[word] for word in weights.keys() & self.words[i]) / scale
            title_overlap = sum(weights[word] for word in weights.keys() & self.headings[i]) / scale
            prose_overlap = sum(weights[word] for word in weights.keys() & self.prose_words[i]) / scale
            phrase_overlap = len(self.bigrams[i] & phrases) / max(len(phrases), 1)
            score = 48 * fusion + 1.4 * overlap + 0.65 * title_overlap + 0.45 * prose_overlap + 0.8 * phrase_overlap
            ranked.append((i, score))
        return sorted(ranked, key=lambda hit: (-hit[1], hit[0]))

    def select(
        self, ranking: list[tuple[int, float]], tokens: list[str], scopes: set[str], top_k: int,
    ) -> list[tuple[int, float]]:
        # Spread the evidence budget across explicitly mentioned topics.
        selected = []
        if scopes:
            rounds = min(2, max(1, top_k // len(scopes)))
            for _ in range(rounds):
                unseen = scopes.copy()
                for i, score in ranking:
                    if self.topics[i] in unseen and (i, score) not in selected:
                        selected.append((i, score))
                        unseen.remove(self.topics[i])
        if self.cross_encoder is not None:
            return selected + [hit for hit in ranking if hit not in selected]

        weights = self.weights(tokens)
        total = sum(weights.values())
        used = Counter(self.chunks[i].source.id for i, _ in selected)
        seen = set().union(*(self.words[i] for i, _ in selected))
        remaining = [item for item in ranking if item not in selected]
        while remaining and len(selected) < top_k:
            def priority(hit):
                i, score = hit
                new = (weights.keys() & self.words[i]) - seen
                novelty = sum(weights[word] for word in new) / total
                return score + 0.45 * novelty - 0.26 * used[self.chunks[i].source.id]

            best = max(remaining, key=priority)
            remaining.remove(best)
            selected.append(best)
            i = best[0]
            used[self.chunks[i].source.id] += 1
            seen.update(self.words[i])
        return selected

    def enough_evidence(self, question: str, hits: list[Hit]) -> bool:
        """Require support in a passage as well as coverage across the candidate set."""
        weights = self.weights(query_terms(question))
        if not hits or not weights:
            return False
        top = [self.words[self.position[hit.chunk.id]] for hit in hits[:4]]
        in_corpus = {word for word in weights if word in self.frequency}
        available = set.union(*top) & weights.keys()
        total = sum(weights.values())
        combined = sum(weights[word] for word in available) / total
        best = max(sum(weights[word] for word in weights.keys() & words) / total for words in top)
        return (
            len(in_corpus) >= min(2, len(weights))
            and len(available) >= min(2, len(weights))
            and combined >= 0.48
            and best >= 0.28
        )
