"""Deterministic topological metric computation for ledger accounts.

All functions here are pure: same graph in, same metrics out, every
time. No randomness, no I/O, no LLM calls. This is what makes the
firewall's decisions auditable and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from gaad.data_layer.ledger_graph import EDGE_ATTR_TRANSACTION, NODE_ATTR_ACCOUNT


class MetricsComputationError(RuntimeError):
    """Raised when topology metrics cannot be computed for a graph/account."""


@dataclass(frozen=True, slots=True)
class AccountTopologyMetrics:
    """Computed structural signals for a single account node.

    Attributes:
        account_id: The account these metrics describe.
        local_clustering_coefficient: NetworkX clustering coefficient
            computed on the undirected view of the graph, restricted
            to this node's neighborhood. Range [0.0, 1.0].
        shared_device_account_count: Number of DISTINCT other accounts
            ANYWHERE in the graph (not necessarily directly connected
            by a transaction edge) that share at least one device_id
            with this account. This models device-fingerprint linking
            as used in real AML entity resolution: a mule ring's
            member accounts are often never seen transacting with each
            other -- the shared device is the only link between them.
        max_multi_hop_velocity_hops: Length (in hops) of the longest
            simple path reachable from this account within
            `velocity_window_minutes`, walking only edges whose
            transaction timestamp is within the window of the
            previous edge on the path. Capped at `max_hops_checked`
            for performance on dense graphs.
        min_hop_interval_seconds: The smallest time gap, in seconds,
            between consecutive transactions on any outgoing path from
            this account. A very small value here means "money moved
            almost instantly across multiple hops" -- a classic
            layering signature. `None` if the account has no outgoing
            edges (nothing to measure velocity over).
    """

    account_id: str
    local_clustering_coefficient: float
    shared_device_account_count: int
    max_multi_hop_velocity_hops: int
    min_hop_interval_seconds: float | None


def _shared_device_account_count(graph: nx.DiGraph, account_id: str) -> int:
    """Counts distinct OTHER accounts anywhere in the graph sharing a device.

    Deliberately ignores transaction adjacency: this is an identity-
    resolution signal (accounts linked via a shared device fingerprint),
    not a money-flow signal. Members of a mule ring are frequently never
    directly connected by a transaction edge to each other -- the shared
    device is the only thing tying them together.
    """

    account = graph.nodes[account_id][NODE_ATTR_ACCOUNT]
    own_devices = account.device_ids

    shared_count = 0
    for other_id, other_data in graph.nodes(data=True):
        if other_id == account_id:
            continue
        other_account = other_data[NODE_ATTR_ACCOUNT]
        if own_devices & other_account.device_ids:
            shared_count += 1
    return shared_count


def _directional_chain(
    graph: nx.DiGraph,
    account_id: str,
    *,
    forward: bool,
    velocity_window_minutes: int,
    max_hops_checked: int,
) -> tuple[int, float | None]:
    """Walks time-consistent edges in ONE direction from `account_id`.

    forward=True walks out_edges (money leaving, moving forward in time).
    forward=False walks in_edges (money arriving, moving backward in time).

    Returns (max_hops_in_this_direction, min_interval_seconds_or_none).
    """

    window_seconds = velocity_window_minutes * 60
    best_hops = 0
    best_min_interval: float | None = None

    def dfs(node_id, depth, boundary_time, current_min_interval, visited) -> None:
        nonlocal best_hops, best_min_interval

        if depth > best_hops:
            best_hops = depth
            best_min_interval = current_min_interval
        elif depth == best_hops and current_min_interval is not None:
            if best_min_interval is None or current_min_interval < best_min_interval:
                best_min_interval = current_min_interval

        if depth >= max_hops_checked:
            return

        edges = (
            graph.out_edges(node_id, data=True)
            if forward
            else graph.in_edges(node_id, data=True)
        )
        for edge in edges:
            neighbor = edge[1] if forward else edge[0]
            edge_data = edge[2]
            if neighbor in visited:
                continue
            txn = edge_data[EDGE_ATTR_TRANSACTION]
            edge_time = txn.executed_at

            if boundary_time is not None:
                delta_seconds = (
                    (edge_time - boundary_time).total_seconds()
                    if forward
                    else (boundary_time - edge_time).total_seconds()
                )
                if delta_seconds < 0 or delta_seconds > window_seconds:
                    continue
                new_min = (
                    delta_seconds
                    if current_min_interval is None
                    else min(current_min_interval, delta_seconds)
                )
            else:
                new_min = current_min_interval

            dfs(neighbor, depth + 1, edge_time, new_min, visited | {neighbor})

    dfs(account_id, 0, None, None, frozenset({account_id}))
    return best_hops, best_min_interval


def _pivot_dwell_seconds(graph: nx.DiGraph, account_id: str) -> float | None:
    """Shortest non-negative gap between ANY incoming edge and ANY
    outgoing edge at `account_id` -- i.e. the fastest "funds in, funds
    straight back out" pairing observed at this node. A short dwell
    time is itself a layering signature, independent of chain length.
    Returns None if the account has no incoming or no outgoing edges.
    """

    in_times = [
        data[EDGE_ATTR_TRANSACTION].executed_at
        for _, _, data in graph.in_edges(account_id, data=True)
    ]
    out_times = [
        data[EDGE_ATTR_TRANSACTION].executed_at
        for _, _, data in graph.out_edges(account_id, data=True)
    ]
    if not in_times or not out_times:
        return None

    best: float | None = None
    for in_t in in_times:
        for out_t in out_times:
            delta = (out_t - in_t).total_seconds()
            if delta >= 0 and (best is None or delta < best):
                best = delta
    return best


def _max_multi_hop_velocity(
    graph: nx.DiGraph,
    account_id: str,
    *,
    velocity_window_minutes: int,
    max_hops_checked: int,
) -> tuple[int, float | None]:
    """Total time-consistent hop span THROUGH `account_id`, in either
    direction, plus the tightest interval observed (including dwell
    time at the pivot itself).

    A LAYERING account in a placement -> layering -> integration chain
    sits in the MIDDLE: its own forward-only view sees just 1 hop. This
    combines the backward chain (funds arriving) with the forward chain
    (funds leaving) so the full 2+ hop chain is attributed to every node
    it passes through, matching how a compliance officer would actually
    read the flow.
    """

    window_seconds = velocity_window_minutes * 60

    forward_hops, forward_min = _directional_chain(
        graph, account_id, forward=True,
        velocity_window_minutes=velocity_window_minutes,
        max_hops_checked=max_hops_checked,
    )
    backward_hops, backward_min = _directional_chain(
        graph, account_id, forward=False,
        velocity_window_minutes=velocity_window_minutes,
        max_hops_checked=max_hops_checked,
    )

    total_hops = forward_hops + backward_hops

    candidate_intervals = [m for m in (forward_min, backward_min) if m is not None]

    dwell = _pivot_dwell_seconds(graph, account_id)
    if dwell is not None and dwell <= window_seconds:
        candidate_intervals.append(dwell)

    min_interval = min(candidate_intervals) if candidate_intervals else None
    return total_hops, min_interval


def compute_account_metrics(
    graph: nx.DiGraph,
    account_id: str,
    *,
    velocity_window_minutes: int = 120,
    max_hops_checked: int = 6,
) -> AccountTopologyMetrics:
    """Computes all topological signals for a single account.

    Args:
        graph: The ledger graph (as built by `build_mule_ring_layering_graph`
            or equivalent). Must contain `account_id`.
        account_id: The node to compute metrics for.
        velocity_window_minutes: Max allowed gap between consecutive hops
            for them to count as part of the same "velocity chain".
        max_hops_checked: Safety cap on DFS depth to bound runtime on
            dense/large graphs.

    Returns:
        An `AccountTopologyMetrics` instance for `account_id`.

    Raises:
        MetricsComputationError: If `account_id` is not in `graph`, or if
            `velocity_window_minutes` / `max_hops_checked` are non-positive.
    """

    if account_id not in graph:
        raise MetricsComputationError(
            f"Account '{account_id}' not found in graph."
        )
    if velocity_window_minutes <= 0:
        raise MetricsComputationError("velocity_window_minutes must be > 0.")
    if max_hops_checked <= 0:
        raise MetricsComputationError("max_hops_checked must be > 0.")

    undirected_view = graph.to_undirected(as_view=True)
    clustering_coefficient: float = nx.clustering(undirected_view, account_id)

    shared_device_count = _shared_device_account_count(graph, account_id)

    max_hops, min_interval = _max_multi_hop_velocity(
        graph,
        account_id,
        velocity_window_minutes=velocity_window_minutes,
        max_hops_checked=max_hops_checked,
    )

    return AccountTopologyMetrics(
        account_id=account_id,
        local_clustering_coefficient=float(clustering_coefficient),
        shared_device_account_count=shared_device_count,
        max_multi_hop_velocity_hops=max_hops,
        min_hop_interval_seconds=min_interval,
    )