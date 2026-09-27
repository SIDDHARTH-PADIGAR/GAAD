"""Asynchronous LangGraph orchestration: firewall -> RAG -> audit compiler.

NO generative LLM call exists anywhere in this module. "Agentic" here
means conditionally-branching orchestration (route -> maybe retrieve
-> compile -> persist), not autonomous text generation. This is what
makes the SYSTEM_AUDIT_WARNING guarantee structural rather than a
matter of prompting discipline: `compile_node` is plain, deterministic
Python, it runs unconditionally on every path through the graph
(SAFE and FLAGGED both pass through it before `persist`), and there is
no LLM step anywhere that could omit, reword, or "forget" to include it.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from gaad.firewall.router import RoutingVerdict, route_single_account
from gaad.pipeline.audit_log import AuditLogCompiler
from gaad.pipeline.state import PipelineState
from gaad.rag_engine.retrieval import retrieve_regulatory_grounding
from gaad.rag_engine.vector_store import LocalRegulatoryVectorStore

MANDATORY_WARNING_KEY = "SYSTEM_AUDIT_WARNING"

_UNVERIFIED_CORPUS_WARNING = (
    "SYSTEM AUDIT WARNING: The regulatory text used to ground this escalation "
    "is ILLUSTRATIVE / MOCK content and has NOT been verified against "
    "official RBI Master Directions or any production regulatory database. "
    "This record MUST NOT be exported to a regulator, used in a filed "
    "Suspicious Transaction Report, or otherwise treated as a "
    "compliance-grade citation until the underlying corpus in "
    "data/rbi_circulars/ has been replaced with legal-team-verified official "
    "text and this record has been re-audited."
)


def _build_system_audit_warning(corpus_is_verified_real_text: bool | None) -> str | None:
    """Single source of truth for warning injection. Pure and deterministic.

    Returns the mandatory warning string only when regulatory grounding
    was actually used (corpus_is_verified_real_text is not None) AND
    that corpus is not verified real text. Returns None for SAFE
    accounts (no retrieval occurred) or a future verified corpus.

    Called unconditionally by `compile_node` on every path through the
    graph -- no verdict, no upstream node, and no LLM can bypass it.
    """

    if corpus_is_verified_real_text is False:
        return _UNVERIFIED_CORPUS_WARNING
    return None


async def route_node(state: PipelineState) -> dict[str, Any]:
    """Runs the deterministic topological firewall for one account."""

    decision = await asyncio.to_thread(
        route_single_account,
        state["graph"],
        state["account_id"],
        thresholds=state.get("thresholds"),
    )
    return {"decision": decision}


def route_condition(state: PipelineState) -> Literal["retrieve", "compile"]:
    """Conditional edge: FLAGGED escalates to RAG; SAFE skips straight to compile."""

    if state["decision"].verdict == RoutingVerdict.FLAGGED:
        return "retrieve"
    return "compile"


def _make_retrieve_node(vector_store: LocalRegulatoryVectorStore):
    async def retrieve_node(state: PipelineState) -> dict[str, Any]:
        top_k = state.get("top_k", 3)
        result = await asyncio.to_thread(
            retrieve_regulatory_grounding,
            state["decision"],
            vector_store,
            top_k=top_k,
        )
        return {"regulatory_result": result}

    return retrieve_node


def compile_node(state: PipelineState) -> dict[str, Any]:
    """Deterministically compiles the final audit record for this account.

    Runs on BOTH the SAFE and FLAGGED paths -- this is the single choke
    point every account passes through before persistence, which is why
    the warning-injection rule enforced here cannot be bypassed.
    """

    decision = state["decision"]
    regulatory_result = state.get("regulatory_result")

    corpus_verified: bool | None = (
        regulatory_result.corpus_is_verified_real_text
        if regulatory_result is not None
        else None
    )
    warning = _build_system_audit_warning(corpus_verified)

    record: dict[str, Any] = {
        "account_id": decision.account_id,
        "verdict": decision.verdict.value,
        "triggered_reasons": list(decision.triggered_reasons),
        "metrics": asdict(decision.metrics),
        "regulatory_provisions": (
            [
                {
                    "citation_id": r.provision.citation_id,
                    "title": r.provision.title,
                    "similarity_score": r.similarity_score,
                    "source_file": r.provision.source_file,
                }
                for r in regulatory_result.retrieved_provisions
            ]
            if regulatory_result is not None
            else None
        ),
        "corpus_is_verified_real_text": corpus_verified,
    }

    if warning is not None:
        record[MANDATORY_WARNING_KEY] = warning

    return {"audit_record": record, "system_audit_warning": warning}


def _make_persist_node(audit_log_compiler: AuditLogCompiler):
    async def persist_node(state: PipelineState) -> dict[str, Any]:
        record = state["audit_record"]

        def _write() -> dict[str, Any]:
            entry = audit_log_compiler.append_entry(
                account_id=record["account_id"],
                verdict=record["verdict"],
                triggered_reasons=tuple(record["triggered_reasons"]),
                metrics=record["metrics"],
                regulatory_provisions=(
                    tuple(record["regulatory_provisions"])
                    if record["regulatory_provisions"] is not None
                    else None
                ),
                corpus_is_verified_real_text=record["corpus_is_verified_real_text"],
                system_audit_warning=record.get(MANDATORY_WARNING_KEY),
            )
            return entry.to_json_dict()

        persisted = await asyncio.to_thread(_write)
        return {"audit_record": persisted}

    return persist_node


def build_gaad_pipeline(
    *,
    vector_store: LocalRegulatoryVectorStore,
    audit_log_compiler: AuditLogCompiler,
):
    """Builds and compiles the async GAAD LangGraph pipeline.

    Graph shape:
        START -> route
              -> [conditional: FLAGGED -> retrieve, SAFE -> compile]
        retrieve -> compile -> persist -> END
        compile (reached directly on SAFE) -> persist -> END

    Args:
        vector_store: A constructed LocalRegulatoryVectorStore. Build
            this ONCE and reuse the returned compiled app across every
            account -- do not rebuild the vector store per account.
        audit_log_compiler: The hash-chained audit log sink.

    Returns:
        A compiled LangGraph app exposing `.ainvoke(state)`.
    """

    builder: StateGraph = StateGraph(PipelineState)

    builder.add_node("route", route_node)
    builder.add_node("retrieve", _make_retrieve_node(vector_store))
    builder.add_node("compile", compile_node)
    builder.add_node("persist", _make_persist_node(audit_log_compiler))

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        route_condition,
        {"retrieve": "retrieve", "compile": "compile"},
    )
    builder.add_edge("retrieve", "compile")
    builder.add_edge("compile", "persist")
    builder.add_edge("persist", END)

    return builder.compile()


async def run_account_through_pipeline(
    graph,
    account_id: str,
    *,
    vector_store: LocalRegulatoryVectorStore,
    audit_log_compiler: AuditLogCompiler,
    thresholds=None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Convenience one-shot wrapper for a single account.

    For processing many accounts, build the app ONCE via
    `build_gaad_pipeline` and reuse it across accounts instead of
    calling this repeatedly -- this wrapper rebuilds the (cheap) graph
    wiring each call but still requires an already-built `vector_store`.
    """

    app = build_gaad_pipeline(
        vector_store=vector_store, audit_log_compiler=audit_log_compiler
    )
    initial_state: PipelineState = {
        "graph": graph,
        "account_id": account_id,
        "thresholds": thresholds,
        "top_k": top_k,
    }
    final_state = await app.ainvoke(initial_state)
    return final_state["audit_record"]