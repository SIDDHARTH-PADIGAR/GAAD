"""Topological Compliance evaluation: proves firewall metrics/verdicts
match independently recomputed graph ground truth.

Every recomputation here is written FROM SCRATCH against raw NetworkX
graph structure -- it does NOT call into gaad.firewall.metrics'
internal helper functions. This is deliberate: the point is to catch
regressions IN metrics.py itself, so the eval must not share code with
the thing it is checking.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from gaad.data_layer.ledger_graph import NODE_ATTR_ACCOUNT
from gaad.data_layer.models import AccountType
from gaad.firewall.router import RoutingDecision, RoutingVerdict


class TopologyEvalError(RuntimeError):
    """Raised when topological compliance evaluation cannot be performed."""


@dataclass(frozen=True, slots=True)
class MetricDiscrepancy:
    """One account where an independently recomputed metric disagreed
    with the firewall's reported value."""

    account_id: str
    metric_name: str
    firewall_value: float
    independent_value: float
    absolute_error: float


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Binary confusion matrix: LAYERING accounts are ground-truth-positive.

    This treats the planted typology label (AccountType.LAYERING) as
    ground truth for "should this account be escalated", independent of
    whatever metrics produced the verdict, and scores the firewall's
    actual SAFE/FLAGGED decisions against it.
    """

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def total(self) -> int:
        return (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return (self.true_positives / denom) if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return (self.true_positives / denom) if denom > 0 else 1.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        return (
            (self.true_positives + self.true_negatives) / self.total
            if self.total > 0
            else 1.0
        )


@dataclass(frozen=True, slots=True)
class TopologyReport:
    """Aggregate topological-compliance metrics for a routed graph.

    Attributes:
        accounts_checked: Total accounts evaluated.
        metric_discrepancies: Every case where a firewall-reported
            metric disagreed with independent recomputation. Empty
            list means zero degradation.
        max_absolute_error_by_metric: Worst-case error per metric name,
            for quick "is anything broken" inspection.
        confusion_matrix: Verdict-vs-planted-typology scoring.
    """

    accounts_checked: int
    metric_discrepancies: tuple[MetricDiscrepancy, ...]
    max_absolute_error_by_metric: dict[str, float]
    confusion_matrix: ConfusionMatrix

    @property
    def has_zero_degradation(self) -> bool:
        """True iff every independently recomputed metric matched exactly."""
        return len(self.metric_discrepancies) == 0


def _independent_clustering_coefficient(graph: nx.DiGraph, account_id: str) -> float:
    """Hand-written triangle-counting clustering coefficient.

    Deliberately reimplements the formula from first principles rather
    than calling nx.clustering, on an undirected adjacency view built
    by hand, so a bug in HOW metrics.py invokes NetworkX (wrong view,
    wrong node set, etc.) is caught rather than silently agreed with.
    """

    undirected_neighbors: set[str] = set(graph.predecessors(account_id)) | set(
        graph.successors(account_id)
    )
    undirected_neighbors.discard(account_id)

    k = len(undirected_neighbors)
    if k < 2:
        return 0.0

    triangle_count = 0
    neighbor_list = list(undirected_neighbors)
    for i in range(len(neighbor_list)):
        for j in range(i + 1, len(neighbor_list)):
            a, b = neighbor_list[i], neighbor_list[j]
            if graph.has_edge(a, b) or graph.has_edge(b, a):
                triangle_count += 1

    possible_triangles = k * (k - 1) / 2
    return triangle_count / possible_triangles if possible_triangles > 0 else 0.0


def _independent_shared_device_count(graph: nx.DiGraph, account_id: str) -> int:
    """Fresh brute-force scan of every OTHER node for device overlap.

    Rewritten independently from metrics.py's _shared_device_account_count
    to protect against a bug being introduced there without a
    corresponding independent check ever noticing.
    """

    own_devices = graph.nodes[account_id][NODE_ATTR_ACCOUNT].device_ids
    count = 0
    for other_id, other_data in graph.nodes(data=True):
        if other_id == account_id:
            continue
        if own_devices & other_data[NODE_ATTR_ACCOUNT].device_ids:
            count += 1
    return count


def evaluate_topological_compliance(
    graph: nx.DiGraph,
    decisions: dict[str, RoutingDecision],
) -> TopologyReport:
    """Recomputes ground truth independently and diffs against the firewall.

    Args:
        graph: The ledger graph the decisions were computed against.
        decisions: Output of gaad.firewall.router.route_accounts(graph).

    Returns:
        A TopologyReport. `has_zero_degradation` is True iff every
        recomputed metric matched exactly (absolute_error == 0.0 for
        all accounts and both metrics checked).

    Raises:
        TopologyEvalError: If `decisions` is empty, or references an
            account_id not present in `graph`.
    """

    if not decisions:
        raise TopologyEvalError(
            "evaluate_topological_compliance requires a non-empty decisions map."
        )

    discrepancies: list[MetricDiscrepancy] = []
    max_errors: dict[str, float] = {
        "local_clustering_coefficient": 0.0,
        "shared_device_account_count": 0.0,
    }

    tp = fp = tn = fn = 0

    for account_id, decision in decisions.items():
        if account_id not in graph:
            raise TopologyEvalError(
                f"Decision references account_id '{account_id}' not "
                "present in the supplied graph."
            )

        independent_clustering = _independent_clustering_coefficient(
            graph, account_id
        )
        clustering_error = abs(
            independent_clustering - decision.metrics.local_clustering_coefficient
        )
        if clustering_error > 1e-9:
            discrepancies.append(
                MetricDiscrepancy(
                    account_id=account_id,
                    metric_name="local_clustering_coefficient",
                    firewall_value=decision.metrics.local_clustering_coefficient,
                    independent_value=independent_clustering,
                    absolute_error=clustering_error,
                )
            )
        max_errors["local_clustering_coefficient"] = max(
            max_errors["local_clustering_coefficient"], clustering_error
        )

        independent_device_count = _independent_shared_device_count(graph, account_id)
        device_error = abs(
            independent_device_count - decision.metrics.shared_device_account_count
        )
        if device_error > 0:
            discrepancies.append(
                MetricDiscrepancy(
                    account_id=account_id,
                    metric_name="shared_device_account_count",
                    firewall_value=float(decision.metrics.shared_device_account_count),
                    independent_value=float(independent_device_count),
                    absolute_error=float(device_error),
                )
            )
        max_errors["shared_device_account_count"] = max(
            max_errors["shared_device_account_count"], float(device_error)
        )

        account_type = graph.nodes[account_id][NODE_ATTR_ACCOUNT].account_type
        is_ground_truth_positive = account_type == AccountType.LAYERING
        is_predicted_positive = decision.verdict == RoutingVerdict.FLAGGED

        if is_ground_truth_positive and is_predicted_positive:
            tp += 1
        elif is_ground_truth_positive and not is_predicted_positive:
            fn += 1
        elif not is_ground_truth_positive and is_predicted_positive:
            fp += 1
        else:
            tn += 1

    return TopologyReport(
        accounts_checked=len(decisions),
        metric_discrepancies=tuple(discrepancies),
        max_absolute_error_by_metric=max_errors,
        confusion_matrix=ConfusionMatrix(
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
        ),
    )