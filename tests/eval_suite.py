"""Module 5: Integrated Evaluation Suite.

Runs as both a pytest module (CI-friendly assertions) AND a standalone
script (`python tests/eval_suite.py`) that prints a human-readable
scorecard for manual/audit review.

Grounding tests require the local embedding model (./local_model/);
they auto-skip otherwise. Topology tests are pure graph math and
always run.

============================================================================
IMPORTANT SCOPING NOTE -- SYNTHETIC GROUND TRUTH VS. PRODUCTION VALIDATION
============================================================================
The confusion matrix (precision/recall/F1/accuracy) computed here treats
AccountType.LAYERING -- a label WE planted when we wrote
build_mule_ring_layering_graph() -- as ground truth for "should this
account have been flagged."

This is a legitimate check that the ORCHESTRATION PLUMBING works
end-to-end (firewall metrics -> thresholds -> verdict -> eval scoring
all agree with each other), but it is NOT a measurement of real-world
detection performance, and a perfect score here should NOT be read as
"the firewall is a good mule-ring detector." It largely can't be
otherwise: the same engineer wrote the graph generator, the threshold
rule, AND this eval, so all three agree by construction on the one
synthetic topology we've exercised. There is no learned model here to
overfit in the ML sense -- route_accounts() is a fixed, hand-written
rule with no training step -- but the SAME structural risk shows up as
circularity: a rule tuned against a benchmark it also defines will
look better than it is.

In a live bank deployment there is NO oracle label at account-open
time. AccountType simply does not exist as a field on a real ledger.
The production swap is:

    SYNTHETIC (this module)              PRODUCTION (future work)
    ------------------------              ------------------------
    AccountType.LAYERING             ->   Outcome of a HUMAN COMPLIANCE
    (planted at graph-generation           OFFICER'S REVIEW, recorded
    time, known immediately)               weeks/months later: did this
                                            account's escalation result
                                            in an STR actually being
                                            filed with the FIU?

    Ground truth is available         ->   Ground truth is available
    instantly, for every account           only in ARREARS, and only
                                            for accounts that were
                                            ALREADY escalated (no label
                                            exists for accounts the
                                            firewall never surfaced --
                                            a structural class-imbalance
                                            and censoring problem this
                                            synthetic eval has no
                                            reason to encounter)

    One fixed topology, generated      ->   A backtesting dataset built
    to be cleanly separable                 from real historical
                                            transaction/case data,
                                            almost certainly NOT cleanly
                                            separable by a simple
                                            threshold rule

    Precision/recall computed          ->   Precision/recall computed
    once, deterministically                 per time window, monitored
                                            for drift as typologies
                                            evolve and mules adapt to
                                            known detection rules

Swapping to that production scheme means replacing
`_ground_truth_label()` below with a lookup against a real STR
disposition table (e.g. a case-management system export keyed by
account_id), and treating any account with NO recorded disposition
as "unlabeled" rather than forcing it into SAFE/FLAGGED -- which in
turn means the confusion matrix formula itself would need to support
an "unknown" class, not just binary positive/negative. That is a
materially different evaluation design and is explicitly OUT OF SCOPE
for this module.
============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gaad.data_layer.ledger_graph import build_mule_ring_layering_graph
from gaad.eval.grounding_eval import evaluate_context_grounding
from gaad.eval.topology_eval import evaluate_topological_compliance
from gaad.firewall.router import RoutingVerdict, route_accounts
from gaad.rag_engine.retrieval import retrieve_regulatory_grounding
from gaad.rag_engine.vector_store import LocalRegulatoryVectorStore

LOCAL_MODEL_DIR = Path("local_model")
requires_local_model = pytest.mark.skipif(
    not LOCAL_MODEL_DIR.is_dir(),
    reason=(
        "local embedding model not present at ./local_model/. Run "
        "'python src/download_model.py' first (requires internet access)."
    ),
)


# --- Topological Compliance: pure graph math, always runs ---


def test_topology_zero_degradation_on_mule_ring_graph() -> None:
    g = build_mule_ring_layering_graph(seed=42)
    decisions = route_accounts(g)
    report = evaluate_topological_compliance(g, decisions)

    assert report.has_zero_degradation, (
        f"Independent recomputation disagreed with firewall metrics: "
        f"{report.metric_discrepancies}"
    )
    assert report.max_absolute_error_by_metric["local_clustering_coefficient"] == 0.0
    assert report.max_absolute_error_by_metric["shared_device_account_count"] == 0.0


def test_topology_confusion_matrix_perfect_on_mule_ring_graph() -> None:
    # NOTE: "perfect" here is a plumbing-integrity check against a
    # SYNTHETIC, planted label -- see the module-level scoping note
    # above before reading this as a real-world performance claim.
    g = build_mule_ring_layering_graph(seed=42)
    decisions = route_accounts(g)
    report = evaluate_topological_compliance(g, decisions)
    cm = report.confusion_matrix

    # This synthetic graph is constructed so LAYERING accounts are
    # exactly the ones that should be FLAGGED, and nothing else should be.
    assert cm.false_positives == 0
    assert cm.false_negatives == 0
    assert cm.precision == 1.0
    assert cm.recall == 1.0
    assert cm.f1_score == 1.0
    assert cm.accuracy == 1.0


def test_topology_eval_rejects_empty_decisions() -> None:
    g = build_mule_ring_layering_graph(seed=42)
    with pytest.raises(Exception):
        evaluate_topological_compliance(g, {})


# --- Context Grounding: requires the real embedding model ---


@requires_local_model
def test_grounding_zero_hallucinations_across_all_flagged_accounts() -> None:
    g = build_mule_ring_layering_graph(seed=42)
    decisions = route_accounts(g)
    store = LocalRegulatoryVectorStore()

    retrieval_results = [
        retrieve_regulatory_grounding(decision, store, top_k=3)
        for decision in decisions.values()
        if decision.verdict == RoutingVerdict.FLAGGED
    ]
    assert retrieval_results, "Expected at least one FLAGGED account to test against."

    report = evaluate_context_grounding(retrieval_results)

    # UNLIKE the confusion matrix above, this metric is NOT a synthetic
    # benchmark artifact. It holds regardless of which corpus or graph
    # is used, because it is a structural property of retrieval.py
    # (every returned RegulatoryProvision was itself loaded from disk
    # by corpus_loader.py -- there is no code path that can invent one).
    # A future production corpus swap should NOT change this result.
    assert report.is_fully_grounded, (
        f"Hallucinated citations detected: {report.hallucination_details}"
    )
    assert report.hallucinated_citations == 0
    assert report.grounding_precision == 1.0
    assert report.total_citations_checked > 0


def _print_scorecard() -> None:
    """Human-readable eval report, run via `python tests/eval_suite.py`."""

    print("=" * 70)
    print("GAAD MODULE 5 EVALUATION SCORECARD")
    print("=" * 70)
    print(
        "\nNOTE: Topological confusion-matrix ground truth is a SYNTHETIC, "
        "planted label (AccountType.LAYERING), used here to validate\n"
        "orchestration plumbing, not real-world detection performance. "
        "See the module docstring for the production backtesting design."
    )

    g = build_mule_ring_layering_graph(seed=42)
    decisions = route_accounts(g)

    topo_report = evaluate_topological_compliance(g, decisions)
    cm = topo_report.confusion_matrix

    print("\n--- Topological Compliance Accuracy ---")
    print(f"Accounts checked:        {topo_report.accounts_checked}")
    print(f"Zero degradation:        {topo_report.has_zero_degradation}")
    print(f"Max error (clustering):  {topo_report.max_absolute_error_by_metric['local_clustering_coefficient']}")
    print(f"Max error (device cnt):  {topo_report.max_absolute_error_by_metric['shared_device_account_count']}")
    print(f"Confusion matrix:        TP={cm.true_positives} FP={cm.false_positives} "
          f"TN={cm.true_negatives} FN={cm.false_negatives}")
    print(f"Precision / Recall / F1: {cm.precision:.3f} / {cm.recall:.3f} / {cm.f1_score:.3f}")
    print(f"Accuracy:                {cm.accuracy:.3f}")

    grounding_report = None
    if LOCAL_MODEL_DIR.is_dir():
        store = LocalRegulatoryVectorStore()
        retrieval_results = [
            retrieve_regulatory_grounding(decision, store, top_k=3)
            for decision in decisions.values()
            if decision.verdict == RoutingVerdict.FLAGGED
        ]
        grounding_report = evaluate_context_grounding(retrieval_results)

        print("\n--- Context Grounding ---")
        print(f"Citations checked:       {grounding_report.total_citations_checked}")
        print(f"Verified grounded:       {grounding_report.verified_grounded_citations}")
        print(f"Hallucinated:            {grounding_report.hallucinated_citations}")
        print(f"Grounding precision:     {grounding_report.grounding_precision:.3f}")
        print(f"Corpus coverage ratio:   {grounding_report.corpus_coverage_ratio:.3f} "
              f"({grounding_report.distinct_citation_ids_used}/{grounding_report.corpus_size})")
        print(f"Fully grounded:          {grounding_report.is_fully_grounded}")
    else:
        print("\n--- Context Grounding ---")
        print("SKIPPED: local_model/ not found. Run src/download_model.py first.")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if grounding_report is not None:
        print(
            f"Context Grounding Score:      "
            f"{grounding_report.grounding_precision * 100:.1f}% "
            f"({grounding_report.verified_grounded_citations}/"
            f"{grounding_report.total_citations_checked} citations verified "
            f"against on-disk corpus, {grounding_report.hallucinated_citations} hallucinated)"
        )
    else:
        print("Context Grounding Score:      N/A (local_model/ not present)")
    print(
        f"Topological Accuracy:         {cm.accuracy * 100:.1f}% "
        f"(vs. SYNTHETIC planted labels -- see scoping note above; "
        f"F1={cm.f1_score:.3f}, degradation={'NONE' if topo_report.has_zero_degradation else 'DETECTED'})"
    )
    print("=" * 70)


if __name__ == "__main__":
    _print_scorecard()
    sys.exit(0)