"""Module 6 (extension): Adversarial stress testing.

Two independent stress tests, run against UNMODIFIED core application
code:

1. STRUCTURAL NOISE TEST: an unseen graph where a real laundering
   chain routes through an innocent, high-degree "hub" account, and
   the ring avoids device reuse at the flagged hop specifically to
   defeat device-linkage detection.

2. MALFORMED INPUT FUZZER: deliberately broken inputs thrown at the
   model layer and a pipeline node, classified by whether they were
   caught by a DEFINED GAAD error, an UNDEFINED (raw builtin) error,
   or silently succeeded when they should not have.

This suite reports what it actually finds, including real gaps. It is
not written to demonstrate that everything already works.

Runs as both pytest (canary assertions -- if these start failing, the
underlying behavior changed and this file's findings/remediation notes
need re-review) and as a standalone script producing a structured JSON
findings log at audit_logs/adversarial_findings.json.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import pytest

from gaad.data_layer.ledger_graph import (
    EDGE_ATTR_TRANSACTION,
    LedgerGraphError,
    NODE_ATTR_ACCOUNT,
)
from gaad.data_layer.models import (
    Account,
    AccountType,
    ModelValidationError,
    Transaction,
)
from gaad.firewall.metrics import MetricsComputationError
from gaad.firewall.router import (
    RouterConfigError,
    RoutingVerdict,
    route_single_account,
)
from gaad.pipeline.graph_pipeline import route_node
from gaad.rag_engine.corpus_loader import CorpusError
from gaad.rag_engine.retrieval import RetrievalRequestError
from gaad.rag_engine.vector_store import VectorStoreError
from gaad.pipeline.audit_log import AuditLogError
from gaad.eval.grounding_eval import GroundingEvalError
from gaad.eval.topology_eval import TopologyEvalError

FINDINGS_LOG_PATH = Path("audit_logs/adversarial_findings.json")

# Every exception type this codebase has purpose-built as a structured,
# documented error boundary. Anything NOT in this tuple that escapes a
# fuzz case is, by definition, an undefined/raw error -- a gap.
DEFINED_GAAD_ERRORS: tuple[type[Exception], ...] = (
    ModelValidationError,
    LedgerGraphError,
    MetricsComputationError,
    RouterConfigError,
    CorpusError,
    VectorStoreError,
    RetrievalRequestError,
    AuditLogError,
    GroundingEvalError,
    TopologyEvalError,
)


# =============================================================================
# PART 1: STRUCTURAL NOISE TEST
# =============================================================================


def _build_structural_noise_graph() -> nx.DiGraph:
    """Builds an UNSEEN adversarial graph, never used elsewhere in this repo.

    Topology: a real placement -> layering -> integration laundering
    chain, but the middle hop is routed through an innocent, high-degree
    "hub" account (e.g. a payment gateway) that also serves 40 unrelated
    legitimate customers. The laundering ring deliberately uses a
    UNIQUE device at the layering hop -- never reused anywhere else --
    specifically to defeat device-linkage detection while preserving
    the fast multi-hop money-movement pattern.
    """

    base_time = datetime(2026, 6, 1, tzinfo=timezone.utc)
    g: nx.DiGraph = nx.DiGraph()

    def add_account(
        acc_id: str,
        acc_type: AccountType,
        device_id: str,
        *,
        opened_days_ago: int = 30,
        risk: float = 0.3,
    ) -> None:
        account = Account(
            account_id=acc_id,
            account_type=acc_type,
            holder_name=f"Holder {acc_id}",
            device_ids=frozenset({device_id}),
            opened_at=base_time - timedelta(days=opened_days_ago),
            kyc_risk_score=risk,
        )
        g.add_node(acc_id, **{NODE_ATTR_ACCOUNT: account})

    def add_txn(src: str, dst: str, amount: float, when: datetime) -> None:
        txn = Transaction(
            source_account_id=src,
            dest_account_id=dst,
            amount_inr=amount,
            executed_at=when,
            channel="IMPS",
        )
        g.add_edge(src, dst, **{EDGE_ATTR_TRANSACTION: txn})

    add_account("ACC-ADV-PLACE-000", AccountType.PLACEMENT, "DEV-ADV-PLACE-000")
    add_account(
        "ACC-ADV-HUB-000", AccountType.LEGITIMATE, "DEV-ADV-HUB-000", risk=0.1
    )
    # Deliberate device-rotation: unique, never-reused device at the
    # actual laundering hop.
    add_account(
        "ACC-ADV-LAYER-000",
        AccountType.LAYERING,
        "DEV-ADV-LAYER-UNIQUE-000",
        opened_days_ago=3,
        risk=0.7,
    )
    add_account("ACC-ADV-INTEG-000", AccountType.INTEGRATION, "DEV-ADV-INTEG-000")

    # 40 unrelated, legitimate customers of the hub -- genuine structural
    # noise, not a strawman: each has its own unique device and transacts
    # with the hub once, making it authentically high-degree/distributed.
    for i in range(40):
        cust_id = f"ACC-ADV-CUST-{i:03d}"
        add_account(cust_id, AccountType.LEGITIMATE, f"DEV-ADV-CUST-{i:03d}")
        add_txn(
            cust_id,
            "ACC-ADV-HUB-000",
            round(500 + i * 10, 2),
            base_time + timedelta(minutes=i),
        )

    # The actual laundering path, fast and multi-hop, routed through the hub.
    t1 = base_time + timedelta(hours=5)
    add_txn("ACC-ADV-PLACE-000", "ACC-ADV-HUB-000", 72_000.0, t1)
    t2 = t1 + timedelta(minutes=20)
    add_txn("ACC-ADV-HUB-000", "ACC-ADV-LAYER-000", 71_500.0, t2)
    t3 = t2 + timedelta(minutes=15)
    add_txn("ACC-ADV-LAYER-000", "ACC-ADV-INTEG-000", 71_000.0, t3)

    return g


@dataclass(frozen=True, slots=True)
class StructuralNoiseFinding:
    target_account_id: str
    firewall_verdict: str
    shared_device_account_count: int
    max_multi_hop_velocity_hops: int
    is_false_negative: bool
    attack_description: str
    root_cause: str
    remediation_notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_structural_noise_test() -> StructuralNoiseFinding:
    graph = _build_structural_noise_graph()
    target = "ACC-ADV-LAYER-000"
    decision = route_single_account(graph, target)

    is_false_negative = decision.verdict == RoutingVerdict.SAFE

    return StructuralNoiseFinding(
        target_account_id=target,
        firewall_verdict=decision.verdict.value,
        shared_device_account_count=decision.metrics.shared_device_account_count,
        max_multi_hop_velocity_hops=decision.metrics.max_multi_hop_velocity_hops,
        is_false_negative=is_false_negative,
        attack_description=(
            "Laundering chain (placement -> hub -> layering -> integration) "
            "routed through an innocent, high-degree hub account (40 "
            "unrelated legitimate customers). The layering account uses a "
            "unique, never-reused device specifically to avoid triggering "
            "device-linkage detection."
        ),
        root_cause=(
            "route_accounts()._decide() requires device_signal as a hard "
            "AND-gate before ANY other signal (clustering/velocity) can "
            "escalate an account. Velocity correctly detected the 2-hop "
            "chain (max_multi_hop_velocity_hops=2) but device_signal=False "
            "(shared_device_account_count=0 < threshold=2) suppresses "
            "escalation entirely, regardless of how strong the velocity "
            "signal is."
        ),
        remediation_notes=(
            "Recommend replacing the hard AND-gate with a weighted/scored "
            "combination (e.g. escalate if device_signal OR "
            "(velocity_hops_signal AND velocity_speed_signal), or a "
            "point-based threshold summing multiple weaker signals) so a "
            "sufficiently strong velocity+clustering pattern can escalate "
            "even without a corroborating device match. Also recommend "
            "extending device-linkage detection beyond immediate neighbors "
            "to a bounded N-hop neighborhood, so a cutout hub does not "
            "fully sever the device trail between ring members."
        ),
    )


# =============================================================================
# PART 2: MALFORMED INPUT FUZZER
# =============================================================================


@dataclass(frozen=True, slots=True)
class FuzzCaseResult:
    case_id: str
    description: str
    outcome: str  # CAUGHT_BY_DEFINED_ERROR | CAUGHT_BY_UNDEFINED_ERROR | SUCCEEDED_NO_ERROR
    exception_type: str | None
    exception_message: str | None
    expected_outcome: str
    is_known_gap: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _execute_fuzz_case(
    case_id: str, description: str, expected_outcome: str, fn: Callable[[], Any]
) -> FuzzCaseResult:
    try:
        fn()
    except DEFINED_GAAD_ERRORS as exc:
        outcome = "CAUGHT_BY_DEFINED_ERROR"
        exc_type, exc_msg = type(exc).__name__, str(exc)
    except Exception as exc:  # noqa: BLE001 -- intentionally broad, this IS the fuzzer
        outcome = "CAUGHT_BY_UNDEFINED_ERROR"
        exc_type, exc_msg = type(exc).__name__, str(exc)
    else:
        outcome = "SUCCEEDED_NO_ERROR"
        exc_type, exc_msg = None, None

    return FuzzCaseResult(
        case_id=case_id,
        description=description,
        outcome=outcome,
        exception_type=exc_type,
        exception_message=exc_msg,
        expected_outcome=expected_outcome,
        is_known_gap=(outcome != expected_outcome)
        if expected_outcome == "CAUGHT_BY_DEFINED_ERROR"
        else (outcome == expected_outcome),  # gap cases: matching = confirming the gap exists
    )


def _valid_aware_time() -> datetime:
    return datetime(2026, 6, 1, tzinfo=timezone.utc)


def _fuzz_negative_amount() -> None:
    Transaction(
        source_account_id="A",
        dest_account_id="B",
        amount_inr=-500.0,
        executed_at=_valid_aware_time(),
    )


def _fuzz_naive_timestamp() -> None:
    Transaction(
        source_account_id="A",
        dest_account_id="B",
        amount_inr=100.0,
        executed_at=datetime(2026, 6, 1),  # no tzinfo
    )


def _fuzz_self_loop() -> None:
    Transaction(
        source_account_id="A",
        dest_account_id="A",
        amount_inr=100.0,
        executed_at=_valid_aware_time(),
    )


def _fuzz_nan_kyc_score() -> None:
    Account(
        account_id="A",
        account_type=AccountType.LEGITIMATE,
        holder_name="H",
        device_ids=frozenset({"DEV-1"}),
        opened_at=_valid_aware_time(),
        kyc_risk_score=math.nan,
    )


def _fuzz_empty_string_device_id() -> None:
    # Account.__post_init__ checks `if not self.device_ids: raise ...`,
    # which validates the FROZENSET is non-empty -- it does NOT iterate
    # the set to check that individual device_id strings are non-empty.
    Account(
        account_id="A",
        account_type=AccountType.LEGITIMATE,
        holder_name="H",
        device_ids=frozenset({""}),
        opened_at=_valid_aware_time(),
        kyc_risk_score=0.5,
    )


def _fuzz_corrupted_graph_node() -> None:
    # Simulates corrupted pipeline state: a graph node present, but
    # missing its NODE_ATTR_ACCOUNT attribute entirely (e.g. a partial
    # write, a schema migration gap, or a malformed upstream feed).
    graph: nx.DiGraph = nx.DiGraph()
    graph.add_node("ACC-CORRUPT-001")  # no NODE_ATTR_ACCOUNT set

    async def _run() -> None:
        state = {"graph": graph, "account_id": "ACC-CORRUPT-001", "thresholds": None}
        await route_node(state)

    asyncio.run(_run())


def _run_fuzz_suite() -> list[FuzzCaseResult]:
    return [
        _execute_fuzz_case(
            "negative_transaction_amount",
            "Transaction constructed with amount_inr=-500.0",
            expected_outcome="CAUGHT_BY_DEFINED_ERROR",
            fn=_fuzz_negative_amount,
        ),
        _execute_fuzz_case(
            "naive_timezone_timestamp",
            "Transaction constructed with a timezone-naive executed_at",
            expected_outcome="CAUGHT_BY_DEFINED_ERROR",
            fn=_fuzz_naive_timestamp,
        ),
        _execute_fuzz_case(
            "self_loop_transaction",
            "Transaction constructed with source_account_id == dest_account_id",
            expected_outcome="CAUGHT_BY_DEFINED_ERROR",
            fn=_fuzz_self_loop,
        ),
        _execute_fuzz_case(
            "nan_kyc_risk_score",
            "Account constructed with kyc_risk_score=math.nan",
            expected_outcome="CAUGHT_BY_DEFINED_ERROR",
            fn=_fuzz_nan_kyc_score,
        ),
        _execute_fuzz_case(
            "empty_string_device_id",
            "Account constructed with device_ids=frozenset({''}) -- an "
            "empty-string device fingerprint inside an otherwise "
            "non-empty set",
            expected_outcome="CAUGHT_BY_DEFINED_ERROR",  # what SHOULD happen
            fn=_fuzz_empty_string_device_id,
        ),
        _execute_fuzz_case(
            "corrupted_graph_node_missing_account_attr",
            "route_node invoked on a graph node present but missing its "
            "NODE_ATTR_ACCOUNT attribute entirely",
            expected_outcome="CAUGHT_BY_DEFINED_ERROR",  # what SHOULD happen
            fn=_fuzz_corrupted_graph_node,
        ),
    ]


# =============================================================================
# STANDALONE REPORT
# =============================================================================


def _write_and_print_findings_log(
    structural: StructuralNoiseFinding, fuzz_results: list[FuzzCaseResult]
) -> None:
    known_gaps = [r for r in fuzz_results if r.is_known_gap]

    log_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "structural_noise_test": structural.to_dict(),
        "fuzz_cases": [r.to_dict() for r in fuzz_results],
        "summary": {
            "structural_false_negative_confirmed": structural.is_false_negative,
            "fuzz_cases_total": len(fuzz_results),
            "fuzz_cases_with_known_gaps": len(known_gaps),
            "fuzz_gap_case_ids": [r.case_id for r in known_gaps],
        },
    }

    FINDINGS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_LOG_PATH.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")

    print("=" * 70)
    print("GAAD MODULE 6 ADVERSARIAL SUITE -- STRUCTURED FAILURE LOG")
    print("=" * 70)
    print(json.dumps(log_payload, indent=2))
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"Structural noise -> False Negative confirmed: "
        f"{structural.is_false_negative}"
    )
    print(
        f"Fuzz cases: {len(fuzz_results)} total, "
        f"{len(known_gaps)} known gap(s): "
        f"{[r.case_id for r in known_gaps] or 'NONE'}"
    )
    print(f"Full log written to: {FINDINGS_LOG_PATH.resolve()}")
    print("=" * 70)


# =============================================================================
# PYTEST CANARIES
# =============================================================================


def test_structural_noise_currently_causes_a_false_negative() -> None:
    """CANARY: documents a CURRENT limitation.

    If this test starts failing (verdict becomes FLAGGED), the firewall
    rule changed -- go update the remediation notes in this file and in
    the Module 2 escalation rule's design rationale, don't just delete
    this test.
    """
    finding = _run_structural_noise_test()
    assert finding.is_false_negative is True
    assert finding.shared_device_account_count == 0
    # Chain is placement -> hub -> layering -> integration (4 nodes).
    # From the layering account: 2 hops backward (layering<-hub<-placement)
    # + 1 hop forward (layering->integration) = 3 total.
    assert finding.max_multi_hop_velocity_hops == 3  # velocity DID see it


@pytest.mark.parametrize(
    "case_id",
    [
        "negative_transaction_amount",
        "naive_timezone_timestamp",
        "self_loop_transaction",
        "nan_kyc_risk_score",
    ],
)
def test_fuzz_case_is_caught_by_a_defined_error(case_id: str) -> None:
    results = {r.case_id: r for r in _run_fuzz_suite()}
    result = results[case_id]
    assert result.outcome == "CAUGHT_BY_DEFINED_ERROR", (
        f"{case_id} was expected to be caught by a defined GAAD error, "
        f"got outcome={result.outcome} ({result.exception_type})"
    )


def test_fuzz_case_empty_device_id_is_a_known_gap() -> None:
    """CANARY: documents a CURRENT validation gap in Account.

    If this test starts failing (outcome != SUCCEEDED_NO_ERROR), Account
    now validates individual device_id strings -- great, but then update
    this test's expected_outcome and remove the "known gap" framing.
    """
    results = {r.case_id: r for r in _run_fuzz_suite()}
    result = results["empty_string_device_id"]
    assert result.outcome == "SUCCEEDED_NO_ERROR"
    assert result.is_known_gap is True


def test_fuzz_case_corrupted_graph_node_is_a_known_gap() -> None:
    """CANARY: documents that a corrupted graph node currently raises a
    raw KeyError from inside an async pipeline node, not a structured
    MetricsComputationError -- meaning it would propagate uncaught out
    of a real LangGraph .ainvoke() call today.
    """
    results = {r.case_id: r for r in _run_fuzz_suite()}
    result = results["corrupted_graph_node_missing_account_attr"]
    assert result.outcome == "CAUGHT_BY_UNDEFINED_ERROR"
    assert result.exception_type == "KeyError"
    assert result.is_known_gap is True


if __name__ == "__main__":
    structural_finding = _run_structural_noise_test()
    fuzz_findings = _run_fuzz_suite()
    _write_and_print_findings_log(structural_finding, fuzz_findings)
    sys.exit(0)