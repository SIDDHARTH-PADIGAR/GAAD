"""Typed state schema threaded through the GAAD LangGraph pipeline."""

from __future__ import annotations

from typing import Any, TypedDict

import networkx as nx

from gaad.firewall.router import FirewallThresholds, RoutingDecision
from gaad.rag_engine.retrieval import RegulatoryRetrievalResult


class PipelineState(TypedDict, total=False):
    """State passed between every node of the GAAD pipeline graph.

    total=False because not every field is populated at every step:
    `regulatory_result` stays absent for SAFE accounts that never
    reach the retrieve node, and `audit_record` only exists after
    `compile` has run.

    Attributes:
        graph: The ledger graph being processed.
        account_id: The single account this run of the pipeline concerns.
        thresholds: Firewall thresholds; None means use FirewallThresholds() defaults.
        top_k: Max regulatory provisions to retrieve if FLAGGED.
        decision: Set by the `route` node.
        regulatory_result: Set by the `retrieve` node (FLAGGED path only).
        system_audit_warning: Set by the `compile` node; None if not applicable.
        audit_record: Set by the `compile` node, then overwritten with
            the persisted (hash-stamped) version by the `persist` node.
    """

    graph: nx.DiGraph
    account_id: str
    thresholds: FirewallThresholds | None
    top_k: int
    decision: RoutingDecision
    regulatory_result: RegulatoryRetrievalResult | None
    system_audit_warning: str | None
    audit_record: dict[str, Any]