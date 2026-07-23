"""RAG-based policy retriever for grounding GenAI explanations.

Phase 6: Vectorizes policy documents from docs/policy_corpus/ and
retrieves the most relevant snippets given a natural-language query.
Supports ChromaDB (optional) with an in-memory TF-IDF / keyword fallback.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class PolicySnippet:
    content: str
    source: str
    section: str

    def __str__(self) -> str:
        return f"[{self.source} §{self.section}] {self.content}"


# ---------------------------------------------------------------------------
# Markdown corpus parser
# ---------------------------------------------------------------------------

_SECTION_HEADING_RE = re.compile(r"^##\s+(\d+\.\d+)\s+(.+)$")
_SUB_ITEM_RE = re.compile(r"^(\d+\.\d+)\s+(.+)$")


def _parse_policy_file(path: Path) -> list[PolicySnippet]:
    """Parse a policy markdown file into a list of PolicySnippet."""
    source = str(path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    snippets: list[PolicySnippet] = []
    current_section: str | None = None

    for line in lines:
        heading_match = _SECTION_HEADING_RE.match(line)
        if heading_match:
            section_id = heading_match.group(1)
            title = heading_match.group(2).strip()
            current_section = section_id
            snippets.append(PolicySnippet(
                content=title,
                source=source,
                section=section_id,
            ))
            continue

        sub_match = _SUB_ITEM_RE.match(line)
        if sub_match:
            sub_id = sub_match.group(1)
            content = sub_match.group(2).strip()
            snippets.append(PolicySnippet(
                content=content,
                source=source,
                section=sub_id,
            ))

    return snippets


def _load_corpus(corpus_path: str | Path) -> list[PolicySnippet]:
    """Load all markdown files from the corpus directory."""
    path = Path(corpus_path)
    if not path.exists() or not path.is_dir():
        return []

    all_snippets: list[PolicySnippet] = []
    for md_file in sorted(path.glob("*.md")):
        all_snippets.extend(_parse_policy_file(md_file))
    return all_snippets


# ---------------------------------------------------------------------------
# Scoring helpers (in-memory fallback)
# ---------------------------------------------------------------------------

def _simple_keyword_score(query: str, documents: list[str]) -> list[float]:
    """Score documents by counting how many query tokens appear in each."""
    query_tokens = set(query.lower().split())
    scores: list[float] = []
    for doc in documents:
        doc_lower = doc.lower()
        score = sum(1 for t in query_tokens if t in doc_lower)
        scores.append(float(score))
    return scores


def _tfidf_score(query: str, documents: list[str]) -> list[float] | None:
    """Score documents using TF-IDF cosine similarity (scikit-learn)."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return None

    if not documents:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        query_vec = vectorizer.transform([query])
    except ValueError:
        return _simple_keyword_score(query, documents)

    # cosine similarity = dot product on unit-normalized vectors
    scores = (tfidf_matrix @ query_vec.T).toarray().flatten().tolist()
    return scores


def _score_query_against_snippets(
    query: str,
    snippets: list[PolicySnippet],
) -> list[tuple[float, PolicySnippet]]:
    """Return list of (score, snippet) pairs sorted descending."""
    if not snippets:
        return []

    documents = [s.content for s in snippets]

    scores = _tfidf_score(query, documents)
    if scores is None:
        scores = _simple_keyword_score(query, documents)

    scored = list(zip(scores, snippets))
    scored.sort(key=lambda x: (-x[0], x[1].section))
    return scored


# ---------------------------------------------------------------------------
# ChromaDB integration (optional)
# ---------------------------------------------------------------------------

class _ChromaStore(Protocol):
    """Minimal protocol for what we need from ChromaDB."""

    def query(
        self, query_texts: list[str], n_results: int
    ) -> dict:
        ...


def _chromadb_available() -> bool:
    try:
        import chromadb  # noqa: F401
        return True
    except ImportError:
        return False


def _init_chromadb(
    snippets: list[PolicySnippet],
) -> _ChromaStore | None:
    """Initialise a local ChromaDB collection from snippets."""
    try:
        import chromadb
    except ImportError:
        return None

    client = chromadb.Client()
    try:
        collection = client.get_collection("policy_corpus")
    except Exception:
        collection = client.create_collection("policy_corpus")

    if collection.count() > 0:
        return collection

    if not snippets:
        return collection

    ids = []
    documents = []
    metadatas = []
    for i, s in enumerate(snippets):
        ids.append(str(i))
        documents.append(s.content)
        metadatas.append({"source": s.source, "section": s.section})

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    return collection


def _chromadb_retrieve(
    collection: _ChromaStore,
    query: str,
    snippets: list[PolicySnippet],
    top_k: int,
) -> list[PolicySnippet]:
    """Retrieve from ChromaDB and remap back to snippet objects."""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, len(snippets)),
        )
    except Exception:
        return _in_memory_retrieve(query, snippets, top_k)

    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    id_set = set(ids)
    retrieved = [s for s in snippets if str(id_set.intersection({str(i) for i, _ in enumerate(snippets)}))]
    # Fallback: build from metadatas
    if not retrieved and metadatas:
        for m in metadatas:
            for s in snippets:
                if s.source == m["source"] and s.section == m["section"]:
                    retrieved.append(s)
                    break

    if retrieved:
        return retrieved[:top_k]

    # Fallback to in-memory if ChromaDB returned nothing useful
    return _in_memory_retrieve(query, snippets, top_k)


# ---------------------------------------------------------------------------
# In-memory retriever
# ---------------------------------------------------------------------------

def _in_memory_retrieve(
    query: str,
    snippets: list[PolicySnippet],
    top_k: int,
) -> list[PolicySnippet]:
    """In-memory retrieval using TF-IDF or keyword matching."""
    scored = _score_query_against_snippets(query, snippets)
    # Only return results with a positive relevance score
    return [s for score, s in scored if score > 0][:top_k]


# ---------------------------------------------------------------------------
# PolicyRetriever
# ---------------------------------------------------------------------------

class PolicyRetriever:
    """Retrieves policy snippets relevant to a natural-language query.

    Supports ChromaDB (optional) with an automatic in-memory fallback
    using TF-IDF (scikit-learn optional) or simple keyword matching.
    """

    def __init__(
        self,
        corpus_path: str | Path = "docs/policy_corpus",
        use_chromadb: bool = False,
    ):
        self._snippets = _load_corpus(corpus_path)

        if use_chromadb and _chromadb_available():
            self._chroma_collection = _init_chromadb(self._snippets)
        else:
            self._chroma_collection = None

    @property
    def snippets(self) -> list[PolicySnippet]:
        return list(self._snippets)

    def retrieve(self, query: str, top_k: int = 3) -> list[PolicySnippet]:
        if not query.strip():
            return []

        if self._chroma_collection is not None:
            return _chromadb_retrieve(
                self._chroma_collection, query, self._snippets, top_k,
            )

        return _in_memory_retrieve(query, self._snippets, top_k)


# ---------------------------------------------------------------------------
# Narrative enrichment
# ---------------------------------------------------------------------------

def enrich_narrative_with_policy(
    narrative: str,
    snippets: list[PolicySnippet],
) -> str:
    """Append policy citations to a narrative string."""
    if not snippets:
        return narrative

    citations: list[str] = []
    seen: set[str] = set()
    for s in snippets:
        key = f"{s.source}§{s.section}"
        if key not in seen:
            seen.add(key)
            citations.append(f"  - {s}")

    if not citations:
        return narrative

    return narrative + "\n\n**Policy References:**\n" + "\n".join(citations)
