"""Verification tests for the Mule Ring Layering mock ledger graph."""

from __future__ import annotations

import pytest

from gaad.data_layer.ledger_graph import (
    LedgerGraphError,
    build_mule_ring_layering_graph,
    NODE_ATTR_ACCOUNT,
)
from gaad.data_layer.models import Account, AccountType


def test_graph_is_deterministic_for_fixed_seed() -> None:
    g1 = build_mule_ring_layering_graph(seed=7)
    g2 = build_mule_ring_layering_graph(seed=7)
    assert sorted(g1.nodes) == sorted(g2.nodes)
    assert sorted(g1.edges) == sorted(g2.edges)


def test_graph_node_and_edge_counts() -> None:
    g = build_mule_ring_layering_graph(
        num_placement=4,
        num_layering=3,
        num_integration=2,
        num_legitimate_noise=5,
        seed=42,
    )
    # 4 placement + 3 layering + 2 integration + 5 legit = 14 nodes
    assert g.number_of_nodes() == 14
    placement = [
        n for n, d in g.nodes(data=True)
        if d[NODE_ATTR_ACCOUNT].account_type == AccountType.PLACEMENT
    ]
    layering = [
        n for n, d in g.nodes(data=True)
        if d[NODE_ATTR_ACCOUNT].account_type == AccountType.LAYERING
    ]
    integration = [
        n for n, d in g.nodes(data=True)
        if d[NODE_ATTR_ACCOUNT].account_type == AccountType.INTEGRATION
    ]
    assert len(placement) == 4
    assert len(layering) == 3
    assert len(integration) == 2


def test_layering_accounts_share_single_mule_device() -> None:
    g = build_mule_ring_layering_graph(seed=42)
    layering_accounts: list[Account] = [
        d[NODE_ATTR_ACCOUNT]
        for _, d in g.nodes(data=True)
        if d[NODE_ATTR_ACCOUNT].account_type == AccountType.LAYERING
    ]
    device_sets = {acc.device_ids for acc in layering_accounts}
    assert len(device_sets) == 1, "All layering accounts must share one device."
    shared_devices = next(iter(device_sets))
    assert len(shared_devices) == 1


def test_placement_funds_reach_integration_within_two_hops() -> None:
    g = build_mule_ring_layering_graph(seed=42)
    placement_nodes = [
        n for n, d in g.nodes(data=True)
        if d[NODE_ATTR_ACCOUNT].account_type == AccountType.PLACEMENT
    ]
    integration_nodes = {
        n for n, d in g.nodes(data=True)
        if d[NODE_ATTR_ACCOUNT].account_type == AccountType.INTEGRATION
    }
    for p in placement_nodes:
        reachable_in_2 = set()
        for _, mid in g.out_edges(p):
            reachable_in_2.update(v for _, v in g.out_edges(mid))
        assert reachable_in_2 & integration_nodes, (
            f"Placement account {p} cannot reach any integration account "
            "within 2 hops."
        )


def test_no_isolated_nodes_and_no_self_loops() -> None:
    g = build_mule_ring_layering_graph(seed=42)
    assert list(g.nodes) == [n for n in g.nodes if g.degree(n) > 0]
    for n in g.nodes:
        assert not g.has_edge(n, n)


def test_invalid_ring_size_raises_ledger_graph_error() -> None:
    with pytest.raises(LedgerGraphError):
        build_mule_ring_layering_graph(num_placement=0)


def test_negative_noise_count_raises_ledger_graph_error() -> None:
    with pytest.raises(LedgerGraphError):
        build_mule_ring_layering_graph(num_legitimate_noise=-1)