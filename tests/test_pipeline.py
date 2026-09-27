"""Verification tests for Module 4: LangGraph pipeline + audit log.

Audit-log and pure compile_node tests always run. Full end-to-end
pipeline tests (which need the real embedding model) auto-skip until
./local_model/ exists -- run src/download_model.py first to enable them.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from gaad.data_layer.ledger_graph import build_mule_ring_layering_graph
from gaad.firewall.metrics import AccountTopologyMetrics
from gaad.firewall.router import (
    RoutingDecision,
    RoutingVerdict,
    route_accounts,
    route_single_account,
)
from gaad.pipeline.audit_log import AuditLogCompiler, AuditLogError
from gaad.pipeline.graph_pipeline import (
    MANDATORY_WARNING_KEY,
    compile_node,
    run_account_through_pipeline,
)
from gaad.rag_engine.corpus_loader import RegulatoryProvision
from gaad.rag_engine.retrieval import RegulatoryRetrievalResult
from gaad.rag_engine.vector_store import LocalRegulatoryVectorStore, RetrievedProvision

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
        triggered_reasons=("shared_device_account_count=2 >= threshold=2",),
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


# --- route_single_account: parity with route_accounts ---


def test_route_single_account_matches_route_accounts() -> None:
    g = build_mule_ring_layering_graph(seed=42)
    full = route_accounts(g)
    single = route_single_account(g, "ACC-LAYER-000")
    assert single == full["ACC-LAYER-000"]


# --- audit log: chaining and tamper detection ---


def test_audit_log_first_entry_has_genesis_previous_hash(tmp_path: Path) -> None:
    compiler = AuditLogCompiler(tmp_path / "audit.jsonl")
    entry = compiler.append_entry(
        account_id="A", verdict="SAFE", triggered_reasons=(), metrics={},
        regulatory_provisions=None, corpus_is_verified_real_text=None,
        system_audit_warning=None,
    )
    assert entry.previous_entry_hash == "GENESIS"
    assert entry.sequence_number == 1


def test_audit_log_chain_links_correctly_across_entries(tmp_path: Path) -> None:
    compiler = AuditLogCompiler(tmp_path / "audit.jsonl")
    e1 = compiler.append_entry(
        account_id="A", verdict="SAFE", triggered_reasons=(), metrics={},
        regulatory_provisions=None, corpus_is_verified_real_text=None,
        system_audit_warning=None,
    )
    e2 = compiler.append_entry(
        account_id="B", verdict="FLAGGED", triggered_reasons=("r",), metrics={},
        regulatory_provisions=None, corpus_is_verified_real_text=False,
        system_audit_warning="warn",
    )
    assert e2.previous_entry_hash == e1.entry_hash
    assert e2.sequence_number == 2
    assert compiler.verify_chain() is True


def test_audit_log_verify_chain_detects_tampering(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    compiler = AuditLogCompiler(log_path)
    compiler.append_entry(
        account_id="A", verdict="SAFE", triggered_reasons=(), metrics={},
        regulatory_provisions=None, corpus_is_verified_real_text=None,
        system_audit_warning=None,
    )
    compiler.append_entry(
        account_id="B", verdict="FLAGGED", triggered_reasons=("r",), metrics={},
        regulatory_provisions=None, corpus_is_verified_real_text=False,
        system_audit_warning="warn",
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1])
    tampered["verdict"] = "SAFE"  # tamper without recomputing the hash
    lines[1] = json.dumps(tampered)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(AuditLogError):
        compiler.verify_chain()


def test_audit_log_malformed_last_line_raises_on_append(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    log_path.write_text("{not valid json\n", encoding="utf-8")
    compiler = AuditLogCompiler(log_path)
    with pytest.raises(AuditLogError):
        compiler.append_entry(
            account_id="A", verdict="SAFE", triggered_reasons=(), metrics={},
            regulatory_provisions=None, corpus_is_verified_real_text=None,
            system_audit_warning=None,
        )


# --- compile_node: SYSTEM_AUDIT_WARNING enforcement (pure, no model needed) ---


def test_compile_node_injects_warning_for_unverified_corpus() -> None:
    decision = _fake_flagged_decision()
    provision = RegulatoryProvision(
        citation_id="X-1", title="T", text="body", keywords=("k",),
        source_file="f.json",
    )
    reg_result = RegulatoryRetrievalResult(
        account_id=decision.account_id,
        query_text="q",
        retrieved_provisions=(RetrievedProvision(provision=provision, similarity_score=0.9),),
        corpus_is_verified_real_text=False,
    )
    update = compile_node({"decision": decision, "regulatory_result": reg_result})
    record = update["audit_record"]

    assert MANDATORY_WARNING_KEY in record
    assert "ILLUSTRATIVE" in record[MANDATORY_WARNING_KEY]
    assert "MUST NOT be exported" in record[MANDATORY_WARNING_KEY]
    assert record["corpus_is_verified_real_text"] is False


def test_compile_node_omits_warning_for_safe_account_with_no_retrieval() -> None:
    decision = _fake_safe_decision()
    update = compile_node({"decision": decision, "regulatory_result": None})
    record = update["audit_record"]

    assert MANDATORY_WARNING_KEY not in record
    assert record["regulatory_provisions"] is None
    assert record["corpus_is_verified_real_text"] is None


def test_compile_node_would_omit_warning_if_corpus_were_verified() -> None:
    decision = _fake_flagged_decision()
    provision = RegulatoryProvision(
        citation_id="X-1", title="T", text="body", keywords=("k",),
        source_file="f.json",
    )
    reg_result = RegulatoryRetrievalResult(
        account_id=decision.account_id,
        query_text="q",
        retrieved_provisions=(RetrievedProvision(provision=provision, similarity_score=0.9),),
        corpus_is_verified_real_text=True,  # hypothetical future verified corpus
    )
    update = compile_node({"decision": decision, "regulatory_result": reg_result})
    assert MANDATORY_WARNING_KEY not in update["audit_record"]


# --- end-to-end pipeline (requires local_model/) ---


@requires_local_model
def test_end_to_end_pipeline_flagged_account_persists_with_warning(tmp_path: Path) -> None:
    g = build_mule_ring_layering_graph(seed=42)
    store = LocalRegulatoryVectorStore()
    compiler = AuditLogCompiler(tmp_path / "audit.jsonl")

    record = asyncio.run(
        run_account_through_pipeline(
            g, "ACC-LAYER-000", vector_store=store, audit_log_compiler=compiler
        )
    )

    assert record["verdict"] == "FLAGGED"
    assert MANDATORY_WARNING_KEY in record
    assert record["corpus_is_verified_real_text"] is False
    assert len(record["regulatory_provisions"]) == 3
    assert compiler.verify_chain() is True


@requires_local_model
def test_end_to_end_pipeline_safe_account_skips_retrieval(tmp_path: Path) -> None:
    g = build_mule_ring_layering_graph(seed=42)
    store = LocalRegulatoryVectorStore()
    compiler = AuditLogCompiler(tmp_path / "audit.jsonl")

    record = asyncio.run(
        run_account_through_pipeline(
            g, "ACC-LEGIT-000", vector_store=store, audit_log_compiler=compiler
        )
    )

    assert record["verdict"] == "SAFE"
    assert MANDATORY_WARNING_KEY not in record
    assert record["regulatory_provisions"] is None
    assert compiler.verify_chain() is True