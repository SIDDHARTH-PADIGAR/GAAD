"""Deterministic routing: SAFE (stop) vs FLAGGED (escalate to RAG).

The routing rule is intentionally a explicit, auditable boolean
combination of thresholds -- never a learned/opaque score -- because
every FLAGGED verdict must be explainable to a human compliance
officer and defensible in an audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique

import networkx as nx

from gaad.firewall.metrics import AccountTopologyMetrics, compute_account_metrics


@unique
class RoutingVerdict(str, Enum):
    """Final routing outcome for an account."""

    SAFE = "SAFE"
    FLAGGED = "FLAGGED"


class RouterConfigError(ValueError):
    """Raised when firewall thresholds are configured invalidly."""


@dataclass(frozen=True, slots=True)
class FirewallThresholds:
    """Configurable thresholds defining what counts as suspicious.

    Attributes:
        min_clustering_coefficient: Local clustering coefficient must
            be >= this value to contribute to a FLAGGED verdict.
        min_shared_device_neighbors: Number of distinct device-sharing
            neighbors must be >= this value to contribute to a
            FLAGGED verdict (this is the "converges via shared mobile
            device identifier" signal).
        min_velocity_hops: Multi-hop velocity chain length must be
            >= this value to contribute to a FLAGGED verdict.
        max_hop_interval_seconds: If the minimum observed interval
            between consecutive hops is <= this value (funds moved
            fast), that also contributes to a FLAGGED verdict.
    """

    min_clustering_coefficient: float = 0.5
    min_shared_device_neighbors: int = 2
    min_velocity_hops: int = 2
    max_hop_interval_seconds: float = 3600.0  # 1 hour

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_clustering_coefficient <= 1.0):
            raise RouterConfigError(
                "min_clustering_coefficient must be in [0.0, 1.0]."
            )
        if self.min_shared_device_neighbors < 0:
            raise RouterConfigError("min_shared_device_neighbors must be >= 0.")
        if self.min_velocity_hops < 0:
            raise RouterConfigError("min_velocity_hops must be >= 0.")
        if self.max_hop_interval_seconds < 0:
            raise RouterConfigError("max_hop_interval_seconds must be >= 0.")


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """The full, explainable outcome of routing a single account.

    Attributes:
        account_id: The account this decision concerns.
        verdict: SAFE or FLAGGED.
        metrics: The raw topology metrics the decision was based on.
        triggered_reasons: Human-readable list of WHICH specific
            threshold(s) were breached. Empty for SAFE verdicts.
            This is what makes the verdict auditable: a compliance
            officer (or the audit log compiler in Module 4) can see
            exactly why an account was escalated.
    """

    account_id: str
    verdict: RoutingVerdict
    metrics: AccountTopologyMetrics
    triggered_reasons: tuple[str, ...]


def _decide(
    metrics: AccountTopologyMetrics, thresholds: FirewallThresholds
) -> RoutingDecision:
    reasons: list[str] = []

    device_signal = metrics.shared_device_account_count >= (
        thresholds.min_shared_device_neighbors
    )
    clustering_signal = (
        metrics.local_clustering_coefficient >= thresholds.min_clustering_coefficient
    )
    velocity_hops_signal = (
        metrics.max_multi_hop_velocity_hops >= thresholds.min_velocity_hops
    )
    velocity_speed_signal = (
        metrics.min_hop_interval_seconds is not None
        and metrics.min_hop_interval_seconds <= thresholds.max_hop_interval_seconds
    )

    if device_signal:
        reasons.append(
            f"shared_device_account_count={metrics.shared_device_account_count} "
            f">= threshold={thresholds.min_shared_device_neighbors}"
        )
    if clustering_signal:
        reasons.append(
            f"local_clustering_coefficient={metrics.local_clustering_coefficient:.3f} "
            f">= threshold={thresholds.min_clustering_coefficient:.3f}"
        )
    if velocity_hops_signal:
        reasons.append(
            f"max_multi_hop_velocity_hops={metrics.max_multi_hop_velocity_hops} "
            f">= threshold={thresholds.min_velocity_hops}"
        )
    if velocity_speed_signal:
        reasons.append(
            f"min_hop_interval_seconds={metrics.min_hop_interval_seconds:.1f} "
            f"<= threshold={thresholds.max_hop_interval_seconds:.1f}"
        )

    # Escalation rule: device convergence signal is required (it's the
    # strongest, most specific indicator of a mule ring per the brief),
    # AND at least one of {clustering, velocity-hops, velocity-speed}
    # must also fire. This mirrors real AML rule design: no single
    # generic signal (e.g. clustering alone) should be sufficient,
    # since legitimate small business networks can cluster tightly too.
    is_flagged = device_signal and (
        clustering_signal or velocity_hops_signal or velocity_speed_signal
    )

    verdict = RoutingVerdict.FLAGGED if is_flagged else RoutingVerdict.SAFE
    return RoutingDecision(
        account_id=metrics.account_id,
        verdict=verdict,
        metrics=metrics,
        triggered_reasons=tuple(reasons) if is_flagged else (),
    )


def route_accounts(
    graph: nx.DiGraph,
    *,
    thresholds: FirewallThresholds | None = None,
    velocity_window_minutes: int = 120,
    max_hops_checked: int = 6,
) -> dict[str, RoutingDecision]:
    """Routes every account in `graph` to SAFE or FLAGGED.

    For SAFE accounts, execution conceptually "ends immediately" --
    no downstream RAG/agentic call is warranted, and callers should
    skip such accounts entirely rather than passing them further down
    the pipeline. This function itself is 100% deterministic and
    side-effect free, so re-running it on the same graph and
    thresholds always yields identical decisions.

    Args:
        graph: Ledger graph to route every account in.
        thresholds: Firewall thresholds. Defaults to
            `FirewallThresholds()` if not provided.
        velocity_window_minutes: Passed through to metric computation.
        max_hops_checked: Passed through to metric computation.

    Returns:
        Mapping of account_id -> RoutingDecision for every node in
        `graph`.
    """

    active_thresholds = thresholds if thresholds is not None else FirewallThresholds()

    decisions: dict[str, RoutingDecision] = {}
    for account_id in graph.nodes:
        metrics = compute_account_metrics(
            graph,
            account_id,
            velocity_window_minutes=velocity_window_minutes,
            max_hops_checked=max_hops_checked,
        )
        decisions[account_id] = _decide(metrics, active_thresholds)

    return decisions