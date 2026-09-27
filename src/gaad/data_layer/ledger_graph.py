"""Synthetic ledger graph construction: Mule Ring Layering typology.

This module builds a deterministic, seedable NetworkX DiGraph that
models a realistic Indian-corporate-banking mule ring:

    PLACEMENT accounts (N)
        -> LAYERING accounts (M), all sharing ONE mobile device_id
            -> INTEGRATION accounts (K)

plus a configurable number of unrelated LEGITIMATE accounts as
background noise, so the topological firewall (Module 2) has both
signal and noise to discriminate between.

No network calls, no LLM calls. Pure, testable graph construction.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import networkx as nx

from gaad.data_layer.models import (
    Account,
    AccountType,
    ModelValidationError,
    Transaction,
)

# Node/edge attribute keys, centralized so downstream modules
# (firewall, audit compiler) never hardcode magic strings.
NODE_ATTR_ACCOUNT = "account"
EDGE_ATTR_TRANSACTION = "transaction"

_BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


class LedgerGraphError(RuntimeError):
    """Raised when the constructed ledger graph fails an invariant check."""


def _build_account(
    account_id: str,
    account_type: AccountType,
    holder_name: str,
    device_ids: frozenset[str],
    opened_at: datetime,
    kyc_risk_score: float,
) -> Account:
    try:
        return Account(
            account_id=account_id,
            account_type=account_type,
            holder_name=holder_name,
            device_ids=device_ids,
            opened_at=opened_at,
            kyc_risk_score=kyc_risk_score,
        )
    except ModelValidationError as exc:
        raise LedgerGraphError(
            f"Failed to construct account '{account_id}': {exc}"
        ) from exc


def build_mule_ring_layering_graph(
    *,
    num_placement: int = 4,
    num_layering: int = 3,
    num_integration: int = 2,
    num_legitimate_noise: int = 10,
    seed: int = 42,
) -> nx.DiGraph:
    """Builds a mock directed ledger graph containing one mule ring.

    Topology modeled:
        placement_i -> layering_j   (funds dispersed / fan-out)
        layering_j  -> integration_k (funds converge / fan-in)
    All `layering_j` accounts share a single mobile `device_id`,
    which is the structural signature the firewall looks for
    (high fan-in/fan-out combined with device convergence).

    A pool of `num_legitimate_noise` unrelated LEGITIMATE accounts
    with sparse, random legitimate-looking transfers is added so the
    graph isn't trivially "everything is suspicious."

    Args:
        num_placement: Number of initial placement (source) accounts.
        num_layering: Number of intermediate layering accounts sharing
            the shared mule device.
        num_integration: Number of final integration (exit) accounts.
        num_legitimate_noise: Number of unrelated background accounts.
        seed: RNG seed for full reproducibility across runs/tests.

    Returns:
        A `networkx.DiGraph` with `Account` objects on nodes (keyed
        under NODE_ATTR_ACCOUNT) and `Transaction` objects on edges
        (keyed under EDGE_ATTR_TRANSACTION).

    Raises:
        LedgerGraphError: If any structural parameter is invalid or
            any constructed Account/Transaction fails validation.
    """

    if num_placement < 1 or num_layering < 1 or num_integration < 1:
        raise LedgerGraphError(
            "num_placement, num_layering, and num_integration must each "
            "be >= 1 to form a mule ring."
        )
    if num_legitimate_noise < 0:
        raise LedgerGraphError("num_legitimate_noise must be >= 0.")

    rng = random.Random(seed)
    graph: nx.DiGraph = nx.DiGraph()

    # --- Shared mule device: the structural fingerprint of the ring ---
    shared_mule_device = "DEV-MULE-SHARED-0001"

    # --- Placement accounts ---
    placement_ids: list[str] = []
    for i in range(num_placement):
        acc_id = f"ACC-PLACE-{i:03d}"
        account = _build_account(
            account_id=acc_id,
            account_type=AccountType.PLACEMENT,
            holder_name=f"Placement Holder {i}",
            device_ids=frozenset({f"DEV-PLACE-{i:03d}"}),
            opened_at=_BASE_TIME - timedelta(days=rng.randint(30, 400)),
            kyc_risk_score=round(rng.uniform(0.2, 0.5), 3),
        )
        graph.add_node(acc_id, **{NODE_ATTR_ACCOUNT: account})
        placement_ids.append(acc_id)

    # --- Layering accounts: ALL share the same mule device ---
    layering_ids: list[str] = []
    for j in range(num_layering):
        acc_id = f"ACC-LAYER-{j:03d}"
        account = _build_account(
            account_id=acc_id,
            account_type=AccountType.LAYERING,
            holder_name=f"Layering Holder {j}",
            device_ids=frozenset({shared_mule_device}),
            opened_at=_BASE_TIME - timedelta(days=rng.randint(1, 20)),
            kyc_risk_score=round(rng.uniform(0.6, 0.9), 3),
        )
        graph.add_node(acc_id, **{NODE_ATTR_ACCOUNT: account})
        layering_ids.append(acc_id)

    # --- Integration accounts ---
    integration_ids: list[str] = []
    for k in range(num_integration):
        acc_id = f"ACC-INTEG-{k:03d}"
        account = _build_account(
            account_id=acc_id,
            account_type=AccountType.INTEGRATION,
            holder_name=f"Integration Holder {k}",
            device_ids=frozenset({f"DEV-INTEG-{k:03d}"}),
            opened_at=_BASE_TIME - timedelta(days=rng.randint(60, 500)),
            kyc_risk_score=round(rng.uniform(0.3, 0.6), 3),
        )
        graph.add_node(acc_id, **{NODE_ATTR_ACCOUNT: account})
        integration_ids.append(acc_id)

    # --- Fan-out edges: placement -> layering (rapid multi-hop dispersal) ---
    t_cursor = _BASE_TIME
    for p_id in placement_ids:
        for l_id in rng.sample(layering_ids, k=min(2, len(layering_ids))):
            t_cursor += timedelta(minutes=rng.randint(5, 45))
            txn = Transaction(
                source_account_id=p_id,
                dest_account_id=l_id,
                amount_inr=round(rng.uniform(45_000, 95_000), 2),
                executed_at=t_cursor,
                channel="IMPS",
            )
            graph.add_edge(p_id, l_id, **{EDGE_ATTR_TRANSACTION: txn})

    # --- Fan-in edges: layering -> integration (rapid convergence) ---
    for l_id in layering_ids:
        for i_id in integration_ids:
            t_cursor += timedelta(minutes=rng.randint(5, 30))
            txn = Transaction(
                source_account_id=l_id,
                dest_account_id=i_id,
                amount_inr=round(rng.uniform(60_000, 140_000), 2),
                executed_at=t_cursor,
                channel="IMPS",
            )
            graph.add_edge(l_id, i_id, **{EDGE_ATTR_TRANSACTION: txn})

    # --- Legitimate background noise: sparse, unrelated accounts ---
    legit_ids: list[str] = []
    for n in range(num_legitimate_noise):
        acc_id = f"ACC-LEGIT-{n:03d}"
        account = _build_account(
            account_id=acc_id,
            account_type=AccountType.LEGITIMATE,
            holder_name=f"Legitimate Holder {n}",
            device_ids=frozenset({f"DEV-LEGIT-{n:03d}"}),
            opened_at=_BASE_TIME - timedelta(days=rng.randint(100, 2000)),
            kyc_risk_score=round(rng.uniform(0.0, 0.3), 3),
        )
        graph.add_node(acc_id, **{NODE_ATTR_ACCOUNT: account})
        legit_ids.append(acc_id)

    # Guarantee baseline connectivity: chain every legit account into a
    # ring (n -> n+1 -> ... -> 0). This ensures every legit node has both
    # in-degree >= 1 and out-degree >= 1 BEFORE any randomness is applied,
    # so no node can end up isolated regardless of RNG outcome.
    if len(legit_ids) == 1:
        # A single legit account cannot form a ring without self-looping,
        # which violates our no-self-loop invariant. Attach it directly
        # to an integration account instead so it still has a real edge.
        acc_id = legit_ids[0]
        target = integration_ids[0]
        t_cursor += timedelta(hours=rng.randint(1, 72))
        txn = Transaction(
            source_account_id=acc_id,
            dest_account_id=target,
            amount_inr=round(rng.uniform(500, 20_000), 2),
            executed_at=t_cursor,
            channel=rng.choice(["UPI", "NEFT"]),
        )
        graph.add_edge(acc_id, target, **{EDGE_ATTR_TRANSACTION: txn})
    else:
        for n, acc_id in enumerate(legit_ids):
            other = legit_ids[(n + 1) % len(legit_ids)]
            t_cursor += timedelta(hours=rng.randint(1, 72))
            txn = Transaction(
                source_account_id=acc_id,
                dest_account_id=other,
                amount_inr=round(rng.uniform(500, 20_000), 2),
                executed_at=t_cursor,
                channel=rng.choice(["UPI", "NEFT"]),
            )
            graph.add_edge(acc_id, other, **{EDGE_ATTR_TRANSACTION: txn})

        # Layer extra random edges ON TOP of the guaranteed ring, so the
        # noise graph isn't a perfectly uniform cycle (more realistic and
        # gives the firewall genuine low-signal clustering to reject).
        for n, acc_id in enumerate(legit_ids):
            candidates = [x for x in legit_ids if x != acc_id]
            if candidates and rng.random() < 0.3:
                other = rng.choice(candidates)
                if not graph.has_edge(acc_id, other):
                    t_cursor += timedelta(hours=rng.randint(1, 72))
                    txn = Transaction(
                        source_account_id=acc_id,
                        dest_account_id=other,
                        amount_inr=round(rng.uniform(500, 20_000), 2),
                        executed_at=t_cursor,
                        channel=rng.choice(["UPI", "NEFT"]),
                    )
                    graph.add_edge(acc_id, other, **{EDGE_ATTR_TRANSACTION: txn})

    _validate_graph_invariants(graph)
    return graph


def _validate_graph_invariants(graph: nx.DiGraph) -> None:
    """Runs structural sanity checks that must hold for ANY ledger graph.

    Raises:
        LedgerGraphError: If any invariant is violated.
    """

    if graph.number_of_nodes() == 0:
        raise LedgerGraphError("Constructed ledger graph has zero nodes.")

    for node_id in graph.nodes:
        if graph.has_edge(node_id, node_id):
            raise LedgerGraphError(f"Self-loop detected on node '{node_id}'.")
        if NODE_ATTR_ACCOUNT not in graph.nodes[node_id]:
            raise LedgerGraphError(
                f"Node '{node_id}' is missing required '{NODE_ATTR_ACCOUNT}' "
                "attribute."
            )

    for u, v, data in graph.edges(data=True):
        if EDGE_ATTR_TRANSACTION not in data:
            raise LedgerGraphError(
                f"Edge ({u} -> {v}) is missing required "
                f"'{EDGE_ATTR_TRANSACTION}' attribute."
            )

    isolated = list(nx.isolates(graph))
    if isolated:
        raise LedgerGraphError(
            f"Ledger graph has {len(isolated)} isolated node(s): {isolated}"
        )