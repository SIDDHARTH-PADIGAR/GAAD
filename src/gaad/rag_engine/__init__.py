"""Air-gapped RAG engine: local regulatory retrieval for flagged accounts.

Nothing in this package makes a network call. The embedding model is
loaded exclusively from a local directory (see vector_store.py); the
regulatory text is loaded exclusively from local JSON files (see
corpus_loader.py).
"""

from gaad.rag_engine.corpus_loader import (
    CorpusError,
    RegulatoryProvision,
    load_corpus,
)
from gaad.rag_engine.vector_store import (
    LocalRegulatoryVectorStore,
    RetrievedProvision,
    VectorStoreError,
)
from gaad.rag_engine.retrieval import (
    RegulatoryRetrievalResult,
    RetrievalRequestError,
    retrieve_regulatory_grounding,
)

__all__ = [
    "CorpusError",
    "RegulatoryProvision",
    "load_corpus",
    "LocalRegulatoryVectorStore",
    "RetrievedProvision",
    "VectorStoreError",
    "RegulatoryRetrievalResult",
    "RetrievalRequestError",
    "retrieve_regulatory_grounding",
]