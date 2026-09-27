"""Context Grounding evaluation: proves RAG output is never hallucinated.

The check is deliberately paranoid: it does NOT trust the in-memory
RegulatoryProvision objects the vector store already holds. It reloads
the corpus FRESH from disk, builds an independent lookup table, and
verifies every returned citation_id + text pair against that fresh
copy byte-for-byte. This catches both "the model invented a citation
that doesn't exist" (impossible by construction here, but proven, not
assumed) and "the in-memory object was mutated/stale relative to disk"
(a real class of bug this specifically guards against).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gaad.rag_engine.corpus_loader import load_corpus
from gaad.rag_engine.retrieval import RegulatoryRetrievalResult
from gaad.rag_engine.vector_store import DEFAULT_CORPUS_DIR


class GroundingEvalError(RuntimeError):
    """Raised when context-grounding evaluation cannot be performed."""


@dataclass(frozen=True, slots=True)
class HallucinationDetail:
    """Describes exactly one citation that failed independent verification.

    Attributes:
        account_id: The account whose retrieval result contained this citation.
        citation_id: The offending citation_id.
        reason: Why it failed ("citation_id_not_in_corpus" or "text_mismatch").
    """

    account_id: str
    citation_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class GroundingReport:
    """Aggregate context-grounding metrics across a batch of retrievals.

    Attributes:
        total_citations_checked: Total (account, citation) pairs examined.
        verified_grounded_citations: Count that matched the fresh
            on-disk corpus exactly (citation_id AND text).
        hallucinated_citations: Count that did not match. Should be 0.
        grounding_precision: verified / total. 1.0 means zero hallucination.
        hallucination_details: Full detail for every failure, empty if none.
        corpus_size: Number of provisions in the freshly-reloaded corpus,
            for context when interpreting coverage.
        distinct_citation_ids_used: How many distinct real citation_ids
            were actually retrieved across the batch.
        corpus_coverage_ratio: distinct_citation_ids_used / corpus_size.
            Not a correctness metric -- a diagnostic of whether the eval
            batch exercised a meaningful slice of the corpus.
    """

    total_citations_checked: int
    verified_grounded_citations: int
    hallucinated_citations: int
    grounding_precision: float
    hallucination_details: tuple[HallucinationDetail, ...]
    corpus_size: int
    distinct_citation_ids_used: int
    corpus_coverage_ratio: float

    @property
    def is_fully_grounded(self) -> bool:
        """True iff zero hallucinations were found."""
        return self.hallucinated_citations == 0


def evaluate_context_grounding(
    retrieval_results: list[RegulatoryRetrievalResult],
    *,
    corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
) -> GroundingReport:
    """Proves every citation in `retrieval_results` traces to the real corpus.

    Args:
        retrieval_results: RegulatoryRetrievalResult objects produced by
            Module 3's retrieve_regulatory_grounding, across any number
            of accounts.
        corpus_dir: Directory to independently reload the corpus from.
            Deliberately re-read from disk rather than trusting any
            already-constructed vector store.

    Returns:
        A GroundingReport with exact counts, not estimates.

    Raises:
        GroundingEvalError: If `retrieval_results` is empty, or if the
            corpus cannot be reloaded from `corpus_dir`.
    """

    if not retrieval_results:
        raise GroundingEvalError(
            "evaluate_context_grounding requires at least one "
            "RegulatoryRetrievalResult to evaluate."
        )

    try:
        fresh_provisions, _ = load_corpus(corpus_dir)
    except Exception as exc:  # corpus_loader raises CorpusError subclasses
        raise GroundingEvalError(
            f"Could not independently reload corpus from '{corpus_dir}' "
            f"for grounding verification: {exc}"
        ) from exc

    fresh_lookup: dict[str, str] = {p.citation_id: p.text for p in fresh_provisions}

    total = 0
    verified = 0
    details: list[HallucinationDetail] = []
    distinct_used: set[str] = set()

    for result in retrieval_results:
        for retrieved in result.retrieved_provisions:
            total += 1
            citation_id = retrieved.provision.citation_id
            reported_text = retrieved.provision.text

            if citation_id not in fresh_lookup:
                details.append(
                    HallucinationDetail(
                        account_id=result.account_id,
                        citation_id=citation_id,
                        reason="citation_id_not_in_corpus",
                    )
                )
                continue

            if fresh_lookup[citation_id] != reported_text:
                details.append(
                    HallucinationDetail(
                        account_id=result.account_id,
                        citation_id=citation_id,
                        reason="text_mismatch",
                    )
                )
                continue

            verified += 1
            distinct_used.add(citation_id)

    hallucinated = total - verified
    precision = (verified / total) if total > 0 else 1.0
    corpus_size = len(fresh_provisions)
    coverage = (len(distinct_used) / corpus_size) if corpus_size > 0 else 0.0

    return GroundingReport(
        total_citations_checked=total,
        verified_grounded_citations=verified,
        hallucinated_citations=hallucinated,
        grounding_precision=precision,
        hallucination_details=tuple(details),
        corpus_size=corpus_size,
        distinct_citation_ids_used=len(distinct_used),
        corpus_coverage_ratio=coverage,
    )