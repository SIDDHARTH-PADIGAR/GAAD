"""Integrated evaluation suite: mathematically scores GAAD's pipeline.

Two metric families:
    - grounding_eval: proves regulatory citations are never hallucinated
      relative to the on-disk corpus.
    - topology_eval: proves firewall metrics/verdicts match independently
      recomputed graph ground truth, without implementation degradation.

Every metric here is computed by exact, deterministic recomputation --
there is no sampling, no LLM-as-judge, and no fuzzy scoring, because
nothing upstream of this module is itself non-deterministic.
"""

from gaad.eval.grounding_eval import (
    GroundingEvalError,
    GroundingReport,
    HallucinationDetail,
    evaluate_context_grounding,
)
from gaad.eval.topology_eval import (
    ConfusionMatrix,
    TopologyEvalError,
    TopologyReport,
    evaluate_topological_compliance,
)

__all__ = [
    "GroundingEvalError",
    "GroundingReport",
    "HallucinationDetail",
    "evaluate_context_grounding",
    "ConfusionMatrix",
    "TopologyEvalError",
    "TopologyReport",
    "evaluate_topological_compliance",
]