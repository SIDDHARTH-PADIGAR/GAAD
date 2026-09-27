"""Bridges firewall RoutingDecisions to regulatory retrieval queries.

Keeps query-construction logic (how a routing decision's
triggered_reasons become search text) separate from the vector store
itself, so the store stays a generic "text in, provisions out" engine
reusable outside the firewall context if needed later.
"""

from __future__ import annotations

from dataclasses import dataclass

from gaad.firewall.router import RoutingDecision, RoutingVerdict
from gaad.rag_engine.vector_store import LocalRegulatoryVectorStore, RetrievedProvision


class RetrievalRequestError(ValueError):
    """Raised when regulatory retrieval is requested for an invalid input."""


@dataclass(frozen=True, slots=True)
class RegulatoryRetrievalResult:
    """Grounding output for a single flagged account.

    Attributes:
        account_id: The account this retrieval concerns.
        query_text: The exact text sent to the vector store, kept for
            audit traceability (Module 4 will persist this verbatim).
        retrieved_provisions: Ranked regulatory passages relevant to
            the triggered reasons, most similar first.
        corpus_is_verified_real_text: Propagated from the vector store
            so downstream consumers (and the audit log) can tell
            whether these citations came from verified official text
            or the illustrative/mock corpus.
    """

    account_id: str
    query_text: str
    retrieved_provisions: tuple[RetrievedProvision, ...]
    corpus_is_verified_real_text: bool


def _build_query_text(decision: RoutingDecision) -> str:
    """Turns a FLAGGED decision's structural signals into search text."""

    reason_text = "; ".join(decision.triggered_reasons)
    return (
        f"Account {decision.account_id} flagged for suspicious money "
        f"laundering typology. Structural signals observed: {reason_text}."
    )


def retrieve_regulatory_grounding(
    decision: RoutingDecision,
    vector_store: LocalRegulatoryVectorStore,
    *,
    top_k: int = 3,
) -> RegulatoryRetrievalResult:
    """Retrieves grounding regulatory provisions for a FLAGGED decision.

    Args:
        decision: A RoutingDecision from Module 2. Must be FLAGGED --
            retrieval is only meaningful (and, per the firewall's
            "stop immediately" design, only invoked) for escalated
            accounts.
        vector_store: A constructed LocalRegulatoryVectorStore.
        top_k: Max number of provisions to retrieve.

    Raises:
        RetrievalRequestError: If `decision.verdict` is not FLAGGED.
            This check runs BEFORE touching `vector_store`, so it is
            safe to call even before a vector store has been
            constructed, as a pure guard-clause check.
    """

    if decision.verdict != RoutingVerdict.FLAGGED:
        raise RetrievalRequestError(
            f"Regulatory retrieval requested for account "
            f"'{decision.account_id}' with verdict "
            f"{decision.verdict.value}. Retrieval is only valid for "
            "FLAGGED accounts -- SAFE accounts must stop at the firewall."
        )

    query_text = _build_query_text(decision)
    results = vector_store.search(query_text, top_k=top_k)

    return RegulatoryRetrievalResult(
        account_id=decision.account_id,
        query_text=query_text,
        retrieved_provisions=tuple(results),
        corpus_is_verified_real_text=vector_store.corpus_is_verified_real_text,
    )