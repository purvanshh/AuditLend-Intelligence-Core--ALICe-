"""Tests for the RAG-based policy retriever (Phase 6)."""

from __future__ import annotations

from pathlib import Path

from ml.explain.policy_rag import (
    PolicyRetriever,
    enrich_narrative_with_policy,
)


def _write_policy(tmp_path, filename: str, content: str) -> Path:
    d = tmp_path / "docs" / "policy_corpus"
    d.mkdir(parents=True, exist_ok=True)
    p = d / filename
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# PolicyRetriever initialisation
# ---------------------------------------------------------------------------


def test_initializes_from_corpus_directory(tmp_path) -> None:
    _write_policy(tmp_path, "TEST.md", (
        "# Test Policy\n"
        "## 1.0 Test Section\n"
        "1.1 This is the first rule.\n"
        "1.2 This is the second rule.\n"
    ))
    retriever = PolicyRetriever(corpus_path=str(tmp_path / "docs" / "policy_corpus"))
    assert len(retriever.snippets) >= 2


def test_initializes_with_empty_corpus(tmp_path) -> None:
    d = tmp_path / "empty_corpus"
    d.mkdir(parents=True, exist_ok=True)
    retriever = PolicyRetriever(corpus_path=str(d))
    assert retriever.snippets == []


# ---------------------------------------------------------------------------
# retrieve() basic
# ---------------------------------------------------------------------------


def test_retrieve_returns_empty_for_empty_corpus(tmp_path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    retriever = PolicyRetriever(corpus_path=str(d))
    assert retriever.retrieve("DTI") == []


def test_retrieve_returns_empty_for_empty_query(tmp_path) -> None:
    _write_policy(tmp_path, "TEST.md", "## 1.0 Test\n1.1 Some rule.\n")
    retriever = PolicyRetriever(corpus_path=str(tmp_path / "docs" / "policy_corpus"))
    assert retriever.retrieve("") == []


def test_retrieve_returns_top_k(tmp_path) -> None:
    _write_policy(tmp_path, "TEST.md", (
        "# Policy\n"
        "## 1.0 A\n1.1 DTI threshold is 35%.\n"
        "## 2.0 B\n2.1 Credit score minimum is 600.\n"
        "## 3.0 C\n3.1 DTI for unsecured loans is 50% max.\n"
    ))
    retriever = PolicyRetriever(corpus_path=str(tmp_path / "docs" / "policy_corpus"))
    results = retriever.retrieve("DTI threshold", top_k=2)
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# Relevance ordering
# ---------------------------------------------------------------------------


def test_retrieve_sorted_by_relevance(tmp_path) -> None:
    _write_policy(tmp_path, "TEST.md", (
        "# Policy\n"
        "## 1.0 DTI Rules\n"
        "1.1 DTI maximum for automatic approval is 35%.\n"
        "## 2.0 Credit Score\n"
        "2.1 Credit score minimum is 600.\n"
    ))
    retriever = PolicyRetriever(corpus_path=str(tmp_path / "docs" / "policy_corpus"))
    results = retriever.retrieve("DTI automatic approval")
    assert len(results) >= 1
    assert "DTI" in results[0].content or "35%" in results[0].content


def test_retrieve_exact_keyword_match(tmp_path) -> None:
    _write_policy(tmp_path, "TEST.md", (
        "# Policy\n"
        "## 1.0 DTI\n"
        "1.1 Maximum DTI for automatic approval: 35%\n"
        "## 2.0 GST\n"
        "2.1 GST non-compliance caps risk score at 54\n"
    ))
    retriever = PolicyRetriever(corpus_path=str(tmp_path / "docs" / "policy_corpus"))
    results = retriever.retrieve("GST non-compliance")
    assert any("GST" in r.content for r in results)


def test_retrieve_partial_match(tmp_path) -> None:
    _write_policy(tmp_path, "TEST.md", (
        "# Policy\n"
        "## 1.0 Income\n"
        "1.1 Monthly income minimum is 25000.\n"
    ))
    retriever = PolicyRetriever(corpus_path=str(tmp_path / "docs" / "policy_corpus"))
    results = retriever.retrieve("monthly salary income")
    assert len(results) >= 1


def test_retrieve_non_existent_query_returns_empty(tmp_path) -> None:
    _write_policy(tmp_path, "TEST.md", (
        "# Policy\n"
        "## 1.0 DTI\n"
        "1.1 DTI threshold 35%.\n"
    ))
    retriever = PolicyRetriever(corpus_path=str(tmp_path / "docs" / "policy_corpus"))
    results = retriever.retrieve("quantum computing superconducting")
    assert len(results) == 0


# ---------------------------------------------------------------------------
# Multiple policy documents
# ---------------------------------------------------------------------------


def test_multiple_policy_documents(tmp_path) -> None:
    _write_policy(tmp_path, "CREDIT.md", (
        "# Credit Policy\n"
        "## 1.0 Scoring\n"
        "1.1 Risk score uses weighted components.\n"
    ))
    _write_policy(tmp_path, "GOVERNANCE.md", (
        "# Governance\n"
        "## 2.0 Approvals\n"
        "2.1 Board approval needed for weight changes > 10%.\n"
    ))
    retriever = PolicyRetriever(corpus_path=str(tmp_path / "docs" / "policy_corpus"))
    assert len(retriever.snippets) >= 2
    sources = {s.source for s in retriever.snippets}
    assert len(sources) >= 2


def test_retrieve_from_multiple_documents(tmp_path) -> None:
    _write_policy(tmp_path, "CREDIT.md", (
        "# Credit Policy\n"
        "## 1.0 DTI\n"
        "1.1 DTI threshold 35%.\n"
    ))
    _write_policy(tmp_path, "GOVERNANCE.md", (
        "# Governance\n"
        "## 2.0 Rules\n"
        "2.1 Rule versions are immutable dataclasses.\n"
    ))
    retriever = PolicyRetriever(corpus_path=str(tmp_path / "docs" / "policy_corpus"))
    results = retriever.retrieve("immutable dataclasses", top_k=5)
    assert len(results) >= 1
    assert "immutable" in results[0].content.lower()


# ---------------------------------------------------------------------------
# In-memory fallback when ChromaDB is not available
# ---------------------------------------------------------------------------


def test_in_memory_fallback_no_chromadb(tmp_path) -> None:
    _write_policy(tmp_path, "TEST.md", (
        "# Policy\n"
        "## 1.0 Test\n"
        "1.1 Sample rule content.\n"
    ))
    retriever = PolicyRetriever(
        corpus_path=str(tmp_path / "docs" / "policy_corpus"),
        use_chromadb=False,
    )
    results = retriever.retrieve("sample rule")
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# enrich_narrative_with_policy
# ---------------------------------------------------------------------------


def test_enrich_narrative_with_policy_appends_citations(tmp_path) -> None:
    from ml.explain.policy_rag import PolicySnippet

    narrative = "Your application was declined due to high DTI."
    snippets = [
        PolicySnippet(
            content="Maximum DTI for automatic approval: 35%",
            source="docs/policy_corpus/CREDIT_POLICY.md",
            section="4.1",
        ),
    ]
    enriched = enrich_narrative_with_policy(narrative, snippets)
    assert narrative in enriched
    assert "Policy References" in enriched
    assert "4.1" in enriched
    assert "CREDIT_POLICY.md" in enriched


def test_enrich_narrative_with_policy_empty_snippets() -> None:
    narrative = "Your application was approved."
    enriched = enrich_narrative_with_policy(narrative, [])
    assert enriched == narrative


def test_enrich_narrative_with_policy_deduplicates(tmp_path) -> None:
    from ml.explain.policy_rag import PolicySnippet

    narrative = "Declined due to DTI and GST non-compliance."
    snippets = [
        PolicySnippet(content="DTI threshold 35%", source="policies.md", section="4.1"),
        PolicySnippet(content="DTI threshold 35%", source="policies.md", section="4.1"),
    ]
    enriched = enrich_narrative_with_policy(narrative, snippets)
    assert enriched.count("4.1") == 1


# ---------------------------------------------------------------------------
# Real corpus integration (uses actual docs/policy_corpus/)
# ---------------------------------------------------------------------------


def test_real_corpus_loads_successfully() -> None:
    retriever = PolicyRetriever()
    assert len(retriever.snippets) > 0


def test_real_corpus_retrieve_dti_policy() -> None:
    retriever = PolicyRetriever()
    results = retriever.retrieve("DTI threshold for unsecured loans", top_k=3)
    assert len(results) >= 1
    assert any("DTI" in r.content or "dti" in r.content.lower() for r in results)


def test_real_corpus_retrieve_gst_rule() -> None:
    retriever = PolicyRetriever()
    results = retriever.retrieve("GST compliance risk score cap")
    assert len(results) >= 1


def test_real_corpus_retrieve_manual_review_criteria() -> None:
    retriever = PolicyRetriever()
    results = retriever.retrieve("manual review confidence")
    assert len(results) >= 1
