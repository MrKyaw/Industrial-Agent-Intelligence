"""Dependency-free hybrid retrieval and context packing.

Retrieval is not the product here, but grounded context is. The important
properties are: every chunk has a stable citation id, untrusted text is fenced
and neutralised, and context never silently exceeds its character budget.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .guardrails import INJECTION_PATTERNS
from .ontology import Ontology
from .util import approx_tokens

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}


class RetrievalError(ValueError):
    """Raised for malformed documents or retrieval configuration."""


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    tags: tuple[str, ...] = ()
    source: str = ""
    trust: str = "reference"
    site_id: str | None = None

    def __post_init__(self) -> None:
        if not self.doc_id or not self.title:
            raise RetrievalError("document id and title are required")
        if self.trust not in {"reference", "untrusted"}:
            raise RetrievalError("trust must be 'reference' or 'untrusted'")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int
    tags: tuple[str, ...]
    source: str
    trust: str
    site_id: str | None = None


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    matched_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "text": self.chunk.text,
            "score": round(self.score, 4),
            "tags": list(self.chunk.tags),
            "source": self.chunk.source,
            "trust": self.chunk.trust,
        }


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        if token in STOPWORDS or len(token) < 2:
            continue
        tokens.append(token[:-1] if len(token) > 3 and token.endswith("s") else token)
    return tokens


def chunk_document(document: Document, *, max_chars: int = 900) -> list[Chunk]:
    if max_chars < 200:
        raise RetrievalError("max_chars must be at least 200")
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", document.text) if block.strip()]
    if not paragraphs:
        return []
    chunks: list[Chunk] = []
    current: list[str] = []
    current_size = 0
    position = 0

    def flush() -> None:
        nonlocal current, current_size, position
        if not current:
            return
        body = "\n\n".join(current).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}#{position}",
                doc_id=document.doc_id,
                title=document.title,
                text=body,
                position=position,
                tags=document.tags,
                source=document.source,
                trust=document.trust,
                site_id=document.site_id,
            )
        )
        position += 1
        current = []
        current_size = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush()
            for start in range(0, len(paragraph), max_chars):
                current = [paragraph[start : start + max_chars]]
                flush()
            continue
        if current_size + len(paragraph) + 2 > max_chars:
            flush()
        current.append(paragraph)
        current_size += len(paragraph) + 2
    flush()
    return chunks


class BM25Index:
    """Classic BM25 over in-memory chunks."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise RetrievalError("BM25 parameters out of range")
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._term_freqs: list[dict[str, int]] = []
        self._lengths: list[int] = []
        self._doc_freq: dict[str, int] = {}
        self._avg_length = 0.0

    def add(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            tokens = tokenize(f"{chunk.title} {chunk.text} {' '.join(chunk.tags)}")
            if not tokens:
                continue
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            self._chunks.append(chunk)
            self._term_freqs.append(frequencies)
            self._lengths.append(len(tokens))
            for term in frequencies:
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
        total = sum(self._lengths)
        self._avg_length = total / len(self._lengths) if self._lengths else 0.0

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalHit]:
        if limit <= 0:
            raise RetrievalError("limit must be positive")
        terms = tokenize(query)
        if not terms or not self._chunks:
            return []
        total_docs = len(self._chunks)
        scores: list[tuple[float, int, list[str]]] = []
        for index, frequencies in enumerate(self._term_freqs):
            score = 0.0
            matched: list[str] = []
            length = self._lengths[index]
            for term in terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                doc_freq = self._doc_freq[term]
                idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self._avg_length or 1.0)
                )
                score += idf * (frequency * (self.k1 + 1)) / denominator
                matched.append(term)
            if score > 0:
                scores.append((score, index, matched))
        scores.sort(key=lambda item: (-item[0], self._chunks[item[1]].chunk_id))
        return [
            RetrievalHit(
                chunk=self._chunks[index],
                score=score,
                matched_terms=tuple(sorted(set(matched))),
            )
            for score, index, matched in scores[:limit]
        ]

    def __len__(self) -> int:
        return len(self._chunks)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Parse a tiny ``key: value`` front-matter block without a YAML dependency."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise RetrievalError(f"malformed front matter line: {line!r}")
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    if end_index is None:
        raise RetrievalError("unterminated front matter")
    body = "\n".join(lines[end_index + 1 :]).strip()
    return meta, body


def load_knowledge_dir(directory: str | Path) -> list[Document]:
    path = Path(directory)
    if not path.is_dir():
        raise RetrievalError(f"knowledge directory not found: {path}")
    documents: list[Document] = []
    seen: set[str] = set()
    for file_path in sorted(path.rglob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        meta, body = parse_front_matter(text)
        doc_id = meta.get("id", file_path.stem)
        if doc_id in seen:
            raise RetrievalError(f"duplicate document id: {doc_id}")
        seen.add(doc_id)
        tags = tuple(
            tag.strip() for tag in meta.get("tags", "").split(",") if tag.strip()
        )
        documents.append(
            Document(
                doc_id=doc_id,
                title=meta.get("title", file_path.stem.replace("-", " ").title()),
                text=body,
                tags=tags,
                source=str(file_path.relative_to(path)),
                trust=meta.get("trust", "reference"),
                site_id=meta.get("site_id") or None,
            )
        )
    return documents


class HybridRetriever:
    """BM25 plus ontology-aware boosts and site filtering."""

    def __init__(self, documents: Iterable[Document], ontology: Ontology | None = None) -> None:
        self.documents = list(documents)
        self.ontology = ontology
        self.index = BM25Index()
        chunks: list[Chunk] = []
        for document in self.documents:
            chunks.extend(chunk_document(document))
        self.chunks = chunks
        self.index.add(chunks)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        site_id: str | None = None,
    ) -> list[RetrievalHit]:
        expanded = query
        entity_names: set[str] = set()
        if self.ontology is not None:
            for node in self.ontology.nodes():
                if node.id.lower() in query.lower() or node.name.lower() in query.lower():
                    entity_names.add(node.name.lower())
                    expanded += f" {node.name} {' '.join(node.tags)}"
        hits = self.index.search(expanded, limit=max(limit * 4, limit))
        boosted: list[RetrievalHit] = []
        for hit in hits:
            if site_id is not None and hit.chunk.site_id not in {None, site_id}:
                continue
            score = hit.score
            haystack = f"{hit.chunk.title} {hit.chunk.text}".lower()
            if any(name in haystack for name in entity_names):
                score += 0.8
            if hit.chunk.trust == "reference":
                score += 0.05
            boosted.append(
                RetrievalHit(chunk=hit.chunk, score=score, matched_terms=hit.matched_terms)
            )
        boosted.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return boosted[:limit]


def sanitize_for_prompt(text: str) -> tuple[str, list[str]]:
    """Neutralise instruction-like content while preserving the rest of the text."""
    attempts: list[str] = []
    sanitized = text
    for pattern in INJECTION_PATTERNS:
        for match in pattern.finditer(sanitized):
            attempts.append(match.group(0))
        sanitized = pattern.sub("[neutralized instruction-like text]", sanitized)
    return sanitized, attempts


@dataclass
class PackedContext:
    text: str
    hits: list[RetrievalHit]
    injection_attempts: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return approx_tokens(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tokens": self.tokens,
            "citations": [hit.chunk.chunk_id for hit in self.hits],
            "injection_attempts": self.injection_attempts,
        }


class ContextPacker:
    """Pack retrieved chunks into a fenced, budgeted context block."""

    def __init__(self, *, max_chars: int = 5000) -> None:
        if max_chars < 500:
            raise RetrievalError("max_chars must be at least 500")
        self.max_chars = max_chars

    def pack(self, hits: Iterable[RetrievalHit]) -> PackedContext:
        sections: list[str] = []
        used: list[RetrievalHit] = []
        injections: list[str] = []
        budget = self.max_chars
        for hit in hits:
            body, attempts = sanitize_for_prompt(hit.chunk.text)
            injections.extend(attempts)
            trust = "UNTRUSTED" if hit.chunk.trust == "untrusted" else "REFERENCE"
            header = f"[{hit.chunk.chunk_id}] {hit.chunk.title} ({trust})"
            section = f"{header}\n{body}"
            if len(section) + 2 > budget:
                if not used:
                    section = section[: max(0, budget - 2)]
                else:
                    break
            sections.append(section)
            used.append(hit)
            budget -= len(section) + 2
            if budget <= 0:
                break
        text = (
            "<untrusted_context>\n"
            + "\n\n".join(sections)
            + "\n</untrusted_context>"
            if sections
            else ""
        )
        return PackedContext(text=text, hits=used, injection_attempts=injections)
