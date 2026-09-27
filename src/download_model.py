"""Standalone, INTERNET-FACING setup script.

Run this ONCE on a machine that has internet access. It downloads the
sentence-transformers model and saves it to a local folder. Copy that
folder onto the air-gapped machine at the same relative path; nothing
else in this repository ever downloads model weights at runtime.

Usage:
    python src/download_model.py
    python src/download_model.py --model all-MiniLM-L6-v2 --output local_model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def download_model(model_name: str, output_dir: Path) -> None:
    """Downloads `model_name` and saves it to `output_dir`.

    Raises:
        SystemExit: If sentence-transformers is not installed.
    """

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "ERROR: sentence-transformers is not installed.\n"
            "Run: pip install sentence-transformers",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Downloading '{model_name}' (requires internet access)...")
    model = SentenceTransformer(model_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    print(f"Saved model weights to '{output_dir.resolve()}'.")
    print(
        "Copy this directory onto the air-gapped machine at the same "
        "relative path before running the main pipeline."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-time, internet-facing download of the local embedding "
            "model for GAAD's air-gapped RAG engine."
        )
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name to download (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--output",
        default="local_model",
        help="Local directory to save the model weights to (default: ./local_model).",
    )
    args = parser.parse_args()
    download_model(args.model, Path(args.output))


if __name__ == "__main__":
    main()