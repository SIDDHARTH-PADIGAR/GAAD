"""Verification tests for the deterministic routing firewall."""

from __future__ import annotations

import pytest

from gaad.data_layer.ledger_graph import build_mule_ring_layering_graph
from gaad.data_layer.models import AccountType
from gaad.data_layer.ledger_graph import NODE_ATTR_ACCOUNT
from gaad.firewall.metrics import MetricsComputationError, compute_account_metrics
from gaad.firewall.router import (
    FirewallThresholds,
    RouterConfigError,
    RoutingVerdict,
    route_accounts,
)


@pytest.fixture(scope="module")
def mule_ring_graph():
    return build_mule_ring_layering_graph(
        num_placement=4,
        num_layering=3,
        num_integration=2,
        num_legitimate_noise=10,
        seed=42,
    )


def test_layering_accounts_are_flagged(mule_ring_graph) -> None:
    decisions = route_accounts(mule_ring_graph)
    layering_ids = [
        n
        for n, d in mule_ring_graph.nodes(data=True)
        if d[NODE_ATTR_ACCOUNT].account_type == AccountType.LAYERING
    ]
    for acc_id in layering_ids:
        decision = decisions[acc_id]
        assert decision.verdict == RoutingVerdict.FLAGGED, (
            f"Expected {acc_id} (LAYERING, shared mule device) to be FLAGGED, "
            f"got {decision.verdict} with metrics {decision.metrics}"
        )
        assert len(decision.triggered_reasons) > 0


def test_most_legitimate_noise_accounts_are_safe(mule_ring_graph) -> None:
    decisions = route_accounts(mule_ring_graph)
    legit_ids = [
        n
        for n, d in mule_ring_graph.nodes(data=True)
        if d[NODE_ATTR_ACCOUNT].account_type == AccountType.LEGITIMATE
    ]
    safe_count = sum(
        1 for acc_id in legit_ids if decisions[acc_id].verdict == RoutingVerdict.SAFE
    )
    # Legit accounts never share devices with anyone, so device_signal
    # (a hard requirement for FLAGGED) can never fire for them.
    assert safe_count == len(legit_ids)


def test_flagged_decisions_have_explainable_reasons(mule_ring_graph) -> None:
    decisions = route_accounts(mule_ring_graph)
    for decision in decisions.values():
        if decision.verdict == RoutingVerdict.FLAGGED:
            assert decision.triggered_reasons
            for reason in decision.triggered_reasons:
                assert "threshold" in reason
        else:
            assert decision.triggered_reasons == ()


def test_routing_is_deterministic(mule_ring_graph) -> None:
    d1 = route_accounts(mule_ring_graph)
    d2 = route_accounts(mule_ring_graph)
    assert {k: v.verdict for k, v in d1.items()} == {
        k: v.verdict for k, v in d2.items()
    }


def test_stricter_thresholds_flag_fewer_or_equal_accounts(mule_ring_graph) -> None:
    lenient = route_accounts(
        mule_ring_graph,
        thresholds=FirewallThresholds(
            min_shared_device_neighbors=1,
            min_clustering_coefficient=0.0,
        ),
    )
    strict = route_accounts(
        mule_ring_graph,
        thresholds=FirewallThresholds(
            min_shared_device_neighbors=10,
            min_clustering_coefficient=0.99,
            min_velocity_hops=10,
            max_hop_interval_seconds=0.0,
        ),
    )
    lenient_flagged = sum(
        1 for d in lenient.values() if d.verdict == RoutingVerdict.FLAGGED
    )
    strict_flagged = sum(
        1 for d in strict.values() if d.verdict == RoutingVerdict.FLAGGED
    )
    assert strict_flagged <= lenient_flagged


def test_invalid_thresholds_raise_router_config_error() -> None:
    with pytest.raises(RouterConfigError):
        FirewallThresholds(min_clustering_coefficient=1.5)
    with pytest.raises(RouterConfigError):
        FirewallThresholds(min_shared_device_neighbors=-1)


def test_unknown_account_raises_metrics_error(mule_ring_graph) -> None:
    with pytest.raises(MetricsComputationError):
        compute_account_metrics(mule_ring_graph, "ACC-DOES-NOT-EXIST")


def test_invalid_velocity_window_raises_metrics_error(mule_ring_graph) -> None:
    with pytest.raises(MetricsComputationError):
        compute_account_metrics(
            mule_ring_graph, "ACC-PLACE-000", velocity_window_minutes=0
        )