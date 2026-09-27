"""Module 5: Integrated Evaluation Suite.

Runs as both a pytest module (CI-friendly assertions) AND a standalone
script (`python tests/eval_suite.py`) that prints a human-readable
scorecard for manual/audit review.

Grounding tests require the local embedding model (./local_model/);
they auto-skip otherwise. Topology tests are pure graph math and
always run.
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

    if not LOCAL_MODEL_DIR.is_dir():
        print("\n--- Context Grounding ---")
        print("SKIPPED: local_model/ not found. Run src/download_model.py first.")
        return

    store = LocalRegulatoryVectorStore()
    retrieval_results = [
        retrieve_regulatory_grounding(decision, store, top_k=3)
        for decision in decisions.values()
        if decision.verdict == RoutingVerdict.FLAGGED
    ]
    ground_report = evaluate_context_grounding(retrieval_results)

    print("\n--- Context Grounding ---")
    print(f"Citations checked:       {ground_report.total_citations_checked}")
    print(f"Verified grounded:       {ground_report.verified_grounded_citations}")
    print(f"Hallucinated:            {ground_report.hallucinated_citations}")
    print(f"Grounding precision:     {ground_report.grounding_precision:.3f}")
    print(f"Corpus coverage ratio:   {ground_report.corpus_coverage_ratio:.3f} "
          f"({ground_report.distinct_citation_ids_used}/{ground_report.corpus_size})")
    print(f"Fully grounded:          {ground_report.is_fully_grounded}")
    print("=" * 70)


if __name__ == "__main__":
    _print_scorecard()
    sys.exit(0)