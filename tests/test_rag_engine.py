"""Verification tests for the air-gapped RAG engine.

Corpus-loading tests always run (pure filesystem/JSON logic, no model
needed). Vector-store tests are SKIPPED automatically if the local
embedding model hasn't been downloaded yet (./local_model/ missing),
since that requires the one-time internet-facing step in
src/download_model.py. Run that script first to enable them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gaad.firewall.router import RoutingDecision, RoutingVerdict
from gaad.firewall.metrics import AccountTopologyMetrics
from gaad.rag_engine.corpus_loader import CorpusError, load_corpus
from gaad.rag_engine.retrieval import (
    RetrievalRequestError,
    retrieve_regulatory_grounding,
)
from gaad.rag_engine.vector_store import (
    LocalRegulatoryVectorStore,
    VectorStoreError,
)

REAL_CORPUS_DIR = Path("data/rbi_circulars")
LOCAL_MODEL_DIR = Path("local_model")

requires_local_model = pytest.mark.skipif(
    not LOCAL_MODEL_DIR.is_dir(),
    reason=(
        "local embedding model not present at ./local_model/. Run "
        "'python src/download_model.py' first (requires internet access)."
    ),
)


def _fake_flagged_decision(account_id: str = "ACC-LAYER-000") -> RoutingDecision:
    metrics = AccountTopologyMetrics(
        account_id=account_id,
        local_clustering_coefficient=0.0,
        shared_device_account_count=2,
        max_multi_hop_velocity_hops=2,
        min_hop_interval_seconds=3300.0,
    )
    return RoutingDecision(
        account_id=account_id,
        verdict=RoutingVerdict.FLAGGED,
        metrics=metrics,
        triggered_reasons=(
            "shared_device_account_count=2 >= threshold=2",
            "max_multi_hop_velocity_hops=2 >= threshold=2",
        ),
    )


def _fake_safe_decision(account_id: str = "ACC-LEGIT-000") -> RoutingDecision:
    metrics = AccountTopologyMetrics(
        account_id=account_id,
        local_clustering_coefficient=0.0,
        shared_device_account_count=0,
        max_multi_hop_velocity_hops=1,
        min_hop_interval_seconds=None,
    )
    return RoutingDecision(
        account_id=account_id,
        verdict=RoutingVerdict.SAFE,
        metrics=metrics,
        triggered_reasons=(),
    )


# --- corpus_loader tests (no model required) ---


def test_bundled_corpus_loads_and_is_flagged_as_unverified() -> None:
    provisions, all_verified = load_corpus(REAL_CORPUS_DIR)
    assert len(provisions) >= 1
    assert all_verified is False  # bundled corpus is illustrative/mock


def test_missing_corpus_dir_raises_corpus_error(tmp_path: Path) -> None:
    with pytest.raises(CorpusError):
        load_corpus(tmp_path / "does_not_exist")


def test_empty_corpus_dir_raises_corpus_error(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_corpus"
    empty_dir.mkdir()
    with pytest.raises(CorpusError):
        load_corpus(empty_dir)


def test_malformed_json_raises_corpus_error(tmp_path: Path) -> None:
    bad_dir = tmp_path / "bad_corpus"
    bad_dir.mkdir()
    (bad_dir / "broken.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CorpusError):
        load_corpus(bad_dir)


def test_missing_required_field_raises_corpus_error(tmp_path: Path) -> None:
    bad_dir = tmp_path / "missing_field_corpus"
    bad_dir.mkdir()
    payload = {
        "verified_real_text": False,
        "provisions": [{"citation_id": "X-1", "title": "Missing text field"}],
    }
    (bad_dir / "provisions.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CorpusError):
        load_corpus(bad_dir)


# --- retrieval guard-clause test (no model required) ---


def test_retrieval_rejects_safe_verdict_before_touching_vector_store() -> None:
    decision = _fake_safe_decision()
    with pytest.raises(RetrievalRequestError):
        # Passing a sentinel instead of a real vector store: the guard
        # clause must reject SAFE verdicts before ever calling it.
        retrieve_regulatory_grounding(decision, vector_store=object())  # type: ignore[arg-type]


# --- vector store tests (require local_model/) ---


def test_vector_store_missing_model_dir_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(VectorStoreError, match="download_model.py"):
        LocalRegulatoryVectorStore(local_model_path=tmp_path / "nonexistent_model")


@requires_local_model
def test_vector_store_retrieves_layering_relevant_provision() -> None:
    store = LocalRegulatoryVectorStore()
    results = store.search(
        "accounts sharing a mobile device converge and funds move rapidly "
        "across multiple hops, layering pattern",
        top_k=2,
    )
    assert len(results) == 2
    top_keywords = " ".join(results[0].provision.keywords).lower()
    assert "device" in top_keywords or "layering" in top_keywords or "multi-hop" in top_keywords


@requires_local_model
def test_vector_store_rejects_empty_query() -> None:
    store = LocalRegulatoryVectorStore()
    with pytest.raises(VectorStoreError):
        store.search("   ", top_k=2)


@requires_local_model
def test_vector_store_rejects_non_positive_top_k() -> None:
    store = LocalRegulatoryVectorStore()
    with pytest.raises(VectorStoreError):
        store.search("mule ring layering", top_k=0)


@requires_local_model
def test_end_to_end_retrieval_for_flagged_decision() -> None:
    store = LocalRegulatoryVectorStore()
    decision = _fake_flagged_decision()
    result = retrieve_regulatory_grounding(decision, store, top_k=3)

    assert result.account_id == decision.account_id
    assert "ACC-LAYER-000" in result.query_text
    assert len(result.retrieved_provisions) == 3
    assert result.corpus_is_verified_real_text is False
    for retrieved in result.retrieved_provisions:
        assert retrieved.provision.citation_id
        assert -1.0 <= retrieved.similarity_score <= 1.0