"""Routing firewall: deterministic topological anomaly detection.

This package computes cheap graph metrics per account and routes
each account to either SAFE (stop, no further processing) or
FLAGGED (escalate to the air-gapped RAG engine).
"""

from gaad.firewall.metrics import AccountTopologyMetrics, compute_account_metrics
from gaad.firewall.router import (
    FirewallThresholds,
    RoutingDecision,
    RoutingVerdict,
    route_accounts,
    route_single_account,
)

__all__ = [
    "AccountTopologyMetrics",
    "compute_account_metrics",
    "FirewallThresholds",
    "RoutingDecision",
    "RoutingVerdict",
    "route_accounts",
    "route_single_account",
]