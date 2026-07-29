"""Tests for RAG-based policy retriever (ml/explain/policy_rag.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml.explain.policy_rag import (
    PolicyRetriever,
    PolicySnippet,
    _chromadb_available,
    _in_memory_retrieve,
    _load_corpus,
    _parse_policy_file,
    _score_query_against_snippets,
    _simple_keyword_score,
    _tfidf_score,
    enrich_narrative_with_policy,
)


# ---------------------------------------------------------------------------
# _simple_keyword_score
# ---------------------------------------------------------------------------


class TestSimpleKeywordScore:
    def test_exact_match_scores_higher(self):
        docs = ["credit score is important", "income must be verified"]
        scores = _simple_keyword_score("credit score", docs)
        assert scores[0] > scores[1]

    def test_no_match_scores_zero(self):
        docs = ["income must be verified"]
        scores = _simple_keyword_score("credit score", docs)
        assert scores[0] == 0.0

    def test_empty_documents(self):
        scores = _simple_keyword_score("query", [])
        assert scores == []

    def test_case_insensitive(self):
        docs = ["CREDIT SCORE matters"]
        scores = _simple_keyword_score("credit score", docs)
        assert scores[0] > 0


# ---------------------------------------------------------------------------
# _tfidf_score
# ---------------------------------------------------------------------------


class TestTfidfScore:
    def test_returns_list_of_floats(self):
        docs = ["credit score is important for lending", "income determines eligibility"]
        scores = _tfidf_score("credit score lending", docs)
        if scores is not None:  # sklearn may or may not be installed
            assert len(scores) == 2
            assert all(isinstance(s, float) for s in scores)

    def test_empty_documents_returns_empty(self):
        scores = _tfidf_score("query", [])
        if scores is not None:
            assert scores == []


# ---------------------------------------------------------------------------
# _parse_policy_file
# ---------------------------------------------------------------------------


class TestParsePolicyFile:
    def _write_policy(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "policy.md"
        p.write_text(content, encoding="utf-8")
        return p

    def test_parses_section_heading(self, tmp_path):
        content = "## 1.1 Minimum Credit Score\nSome text\n"
        path = self._write_policy(tmp_path, content)
        snippets = _parse_policy_file(path)
        assert any(s.section == "1.1" for s in snippets)

    def test_parses_sub_item(self, tmp_path):
        content = "## 2.0 Income Verification\n2.1 Minimum monthly income must be 25000\n"
        path = self._write_policy(tmp_path, content)
        snippets = _parse_policy_file(path)
        assert any(s.section == "2.1" for s in snippets)

    def test_source_set_to_file_path(self, tmp_path):
        path = self._write_policy(tmp_path, "## 1.0 Policy\n")
        snippets = _parse_policy_file(path)
        for s in snippets:
            assert s.source == str(path)

    def test_empty_file_returns_empty_list(self, tmp_path):
        path = self._write_policy(tmp_path, "")
        snippets = _parse_policy_file(path)
        assert snippets == []

    def test_non_matching_lines_ignored(self, tmp_path):
        content = "This is a plain line\nAnother plain line\n"
        path = self._write_policy(tmp_path, content)
        snippets = _parse_policy_file(path)
        assert snippets == []


# ---------------------------------------------------------------------------
# _load_corpus
# ---------------------------------------------------------------------------


class TestLoadCorpus:
    def test_nonexistent_dir_returns_empty(self, tmp_path):
        snippets = _load_corpus(tmp_path / "nonexistent")
        assert snippets == []

    def test_file_not_dir_returns_empty(self, tmp_path):
        f = tmp_path / "not_a_dir.md"
        f.write_text("content")
        snippets = _load_corpus(f)
        assert snippets == []

    def test_loads_markdown_files(self, tmp_path):
        (tmp_path / "policy1.md").write_text("## 1.1 Rule One\n")
        (tmp_path / "policy2.md").write_text("## 2.1 Rule Two\n")
        snippets = _load_corpus(tmp_path)
        assert len(snippets) >= 2

    def test_non_md_files_ignored(self, tmp_path):
        (tmp_path / "policy.md").write_text("## 1.1 Rule\n")
        (tmp_path / "policy.txt").write_text("## 1.2 NotRule\n")
        snippets = _load_corpus(tmp_path)
        sections = [s.section for s in snippets]
        assert "1.1" in sections
        # txt file should not be loaded
        assert not any("NotRule" in s.content for s in snippets)

    def test_loads_real_policy_corpus(self):
        """Smoke test: load actual docs/policy_corpus if it exists."""
        snippets = _load_corpus("docs/policy_corpus")
        # If the corpus exists, we should get some snippets
        # If it doesn't exist, load returns [] without error
        assert isinstance(snippets, list)


# ---------------------------------------------------------------------------
# _score_query_against_snippets
# ---------------------------------------------------------------------------


class TestScoreQueryAgainstSnippets:
    def test_returns_empty_for_no_snippets(self):
        result = _score_query_against_snippets("query", [])
        assert result == []

    def test_returns_sorted_by_score_descending(self):
        snippets = [
            PolicySnippet("unrelated content here", "src", "1.1"),
            PolicySnippet("credit score minimum requirement", "src", "1.2"),
        ]
        scored = _score_query_against_snippets("credit score", snippets)
        if len(scored) >= 2:
            # Best match should come first
            assert scored[0][0] >= scored[1][0]

    def test_score_is_float(self):
        snippets = [PolicySnippet("credit score", "src", "1.1")]
        scored = _score_query_against_snippets("credit", snippets)
        assert all(isinstance(s, float) for s, _ in scored)


# ---------------------------------------------------------------------------
# PolicyRetriever
# ---------------------------------------------------------------------------


class TestPolicyRetriever:
    def _make_corpus(self, tmp_path: Path) -> Path:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "policy.md").write_text(
            "## 1.1 Credit Score\n"
            "1.2 Minimum credit score must be 650\n"
            "## 2.1 Income\n"
            "2.2 Monthly income verification required\n",
            encoding="utf-8",
        )
        return corpus

    def test_init_loads_snippets(self, tmp_path):
        corpus = self._make_corpus(tmp_path)
        retriever = PolicyRetriever(corpus_path=corpus)
        assert len(retriever.snippets) > 0

    def test_retrieve_returns_list(self, tmp_path):
        corpus = self._make_corpus(tmp_path)
        retriever = PolicyRetriever(corpus_path=corpus)
        result = retriever.retrieve("credit score")
        assert isinstance(result, list)

    def test_retrieve_empty_query_returns_empty(self, tmp_path):
        corpus = self._make_corpus(tmp_path)
        retriever = PolicyRetriever(corpus_path=corpus)
        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []

    def test_retrieve_top_k_respected(self, tmp_path):
        corpus = self._make_corpus(tmp_path)
        retriever = PolicyRetriever(corpus_path=corpus)
        result = retriever.retrieve("credit income", top_k=1)
        assert len(result) <= 1

    def test_retrieve_returns_policy_snippets(self, tmp_path):
        corpus = self._make_corpus(tmp_path)
        retriever = PolicyRetriever(corpus_path=corpus)
        result = retriever.retrieve("credit score minimum", top_k=3)
        for snippet in result:
            assert isinstance(snippet, PolicySnippet)

    def test_missing_corpus_returns_no_snippets(self, tmp_path):
        retriever = PolicyRetriever(corpus_path=tmp_path / "nonexistent")
        assert retriever.snippets == []
        assert retriever.retrieve("anything") == []

    def test_snippets_property_returns_copy(self, tmp_path):
        corpus = self._make_corpus(tmp_path)
        retriever = PolicyRetriever(corpus_path=corpus)
        a = retriever.snippets
        b = retriever.snippets
        assert a == b
        # Ensure it returns a new list each time (defensive copy)
        a.clear()
        assert len(retriever.snippets) > 0

    def test_real_corpus_retrieval(self):
        """Smoke test against real corpus if it exists."""
        retriever = PolicyRetriever(corpus_path="docs/policy_corpus")
        # Should not raise regardless of whether corpus exists
        result = retriever.retrieve("credit score", top_k=3)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _chromadb_available
# ---------------------------------------------------------------------------


class TestChromadbAvailable:
    def test_returns_bool(self):
        result = _chromadb_available()
        assert isinstance(result, bool)

    def test_false_when_chromadb_not_importable(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "chromadb":
                raise ImportError("chromadb not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        assert _chromadb_available() is False


# ---------------------------------------------------------------------------
# enrich_narrative_with_policy
# ---------------------------------------------------------------------------


class TestEnrichNarrativeWithPolicy:
    def test_no_snippets_returns_original(self):
        narrative = "This application is approved."
        result = enrich_narrative_with_policy(narrative, [])
        assert result == narrative

    def test_snippets_appended(self):
        narrative = "Application denied."
        snippets = [PolicySnippet("Credit score below 650", "policy.md", "1.2")]
        result = enrich_narrative_with_policy(narrative, snippets)
        assert "Policy References" in result
        assert "Application denied." in result
        assert "1.2" in result

    def test_deduplicates_snippets(self):
        narrative = "Review required."
        s = PolicySnippet("Same content", "policy.md", "1.1")
        result = enrich_narrative_with_policy(narrative, [s, s, s])
        # Should only appear once in citations
        assert result.count("§1.1") == 1

    def test_multiple_unique_snippets(self):
        narrative = "Decision explanation."
        snippets = [
            PolicySnippet("Rule one", "policy.md", "1.1"),
            PolicySnippet("Rule two", "policy.md", "2.1"),
        ]
        result = enrich_narrative_with_policy(narrative, snippets)
        assert "§1.1" in result
        assert "§2.1" in result


# ---------------------------------------------------------------------------
# PolicySnippet.__str__
# ---------------------------------------------------------------------------


class TestPolicySnippetStr:
    def test_str_format(self):
        s = PolicySnippet("Minimum income requirement", "policy.md", "3.2")
        assert str(s) == "[policy.md §3.2] Minimum income requirement"

    def test_str_used_in_citation(self):
        s = PolicySnippet("Content here", "credit_policy.md", "1.5")
        result = enrich_narrative_with_policy("Narrative", [s])
        assert "[credit_policy.md §1.5] Content here" in result
