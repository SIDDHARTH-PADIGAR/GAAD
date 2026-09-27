"""Local, air-gapped vector store over the regulatory corpus.

Loads a sentence-transformers model EXCLUSIVELY from a local directory
(never downloads at runtime) and performs pure in-memory cosine
similarity search. No network calls are made anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gaad.rag_engine.corpus_loader import RegulatoryProvision, load_corpus


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot be built or queried."""


DEFAULT_LOCAL_MODEL_PATH = Path("local_model")
DEFAULT_CORPUS_DIR = Path("data/rbi_circulars")


@dataclass(frozen=True, slots=True)
class RetrievedProvision:
    """A single search result: a provision plus its similarity score."""

    provision: RegulatoryProvision
    similarity_score: float


class LocalRegulatoryVectorStore:
    """Embeds the regulatory corpus once and serves local similarity search.

    Air-gap guarantee: the sentence-transformers model is loaded ONLY
    from `local_model_path` on disk. If that path is missing, this
    raises immediately with instructions to run
    `src/download_model.py` on a non-air-gapped machine first -- it
    never attempts to fetch weights from the internet.
    """

    def __init__(
        self,
        *,
        local_model_path: Path | str = DEFAULT_LOCAL_MODEL_PATH,
        corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
    ) -> None:
        model_path = Path(local_model_path)
        if not model_path.is_dir():
            raise VectorStoreError(
                f"Local embedding model not found at '{model_path}'. "
                "This engine never downloads weights at runtime. Run "
                "'python src/download_model.py' on a machine with internet "
                f"access, then copy the resulting '{model_path}' directory "
                "onto this air-gapped machine before retrying."
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise VectorStoreError(
                "sentence-transformers is not installed. Run: "
                "pip install sentence-transformers"
            ) from exc

        try:
            self._model = SentenceTransformer(str(model_path))
        except Exception as exc:  # model dir exists but is corrupt/incomplete
            raise VectorStoreError(
                f"Failed to load embedding model from '{model_path}': {exc}"
            ) from exc

        self.provisions, self.corpus_is_verified_real_text = load_corpus(corpus_dir)

        corpus_texts = [
            f"{p.title}. {p.text} Keywords: {', '.join(p.keywords)}"
            for p in self.provisions
        ]
        embeddings = self._model.encode(
            corpus_texts, convert_to_numpy=True, normalize_embeddings=True
        )
        self._embeddings: np.ndarray = np.asarray(embeddings, dtype=np.float32)

    def search(self, query: str, *, top_k: int = 3) -> list[RetrievedProvision]:
        """Returns the top_k most similar provisions to `query`.

        Args:
            query: Free text to search for (e.g. built from a
                RoutingDecision's triggered_reasons).
            top_k: Maximum number of results to return.

        Raises:
            VectorStoreError: If `query` is empty/whitespace or
                `top_k` is not positive.
        """

        if not query or not query.strip():
            raise VectorStoreError("search query must be non-empty.")
        if top_k <= 0:
            raise VectorStoreError("top_k must be > 0.")

        query_embedding = np.asarray(
            self._model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            ),
            dtype=np.float32,
        )[0]

        # Embeddings are pre-normalized, so dot product == cosine similarity.
        scores = self._embeddings @ query_embedding

        k = min(top_k, len(self.provisions))
        top_indices = np.argsort(-scores)[:k]

        return [
            RetrievedProvision(
                provision=self.provisions[i],
                similarity_score=float(scores[i]),
            )
            for i in top_indices
        ]