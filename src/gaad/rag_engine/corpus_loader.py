"""Regulatory corpus loading: reads RBI-related provisions from disk.

This module is the ONLY place that knows how the regulatory text is
stored on disk. Nothing in vector_store.py or retrieval.py hardcodes
any clause number, section title, or legal text -- everything comes
from files under a data directory, so a bank's legal team can replace
that directory's contents with verified official text without
touching a single line of application code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class CorpusError(RuntimeError):
    """Raised when the regulatory corpus directory is missing, empty, or malformed."""


@dataclass(frozen=True, slots=True)
class RegulatoryProvision:
    """One retrievable unit of regulatory text.

    Attributes:
        citation_id: Stable identifier for this provision, e.g.
            "RBI/MD-KYC/ILLUSTRATIVE-4(2)(a)". NOT guaranteed to be a
            real RBI citation -- see the corpus-level
            `verified_real_text` flag returned by `load_corpus`.
        title: Short human-readable heading.
        text: The regulatory text body.
        keywords: Free-text tags used to aid retrieval interpretability.
        source_file: Which file on disk this provision was loaded from,
            for audit traceability back to the source document.
    """

    citation_id: str
    title: str
    text: str
    keywords: tuple[str, ...]
    source_file: str


def load_corpus(data_dir: Path | str) -> tuple[list[RegulatoryProvision], bool]:
    """Loads every provision from every *.json file under `data_dir`.

    Each JSON file must contain:
        {
          "verified_real_text": bool,
          "provisions": [
             {"citation_id": str, "title": str, "text": str, "keywords": [str, ...]},
             ...
          ]
        }

    Args:
        data_dir: Directory to scan for *.json regulatory files.

    Returns:
        A tuple of (all provisions found, whether ALL source files
        claim verified_real_text=True). If even one file is marked
        False (illustrative/mock), the second element is False, so
        callers can surface an explicit "not verified real regulatory
        text" signal all the way through to the audit log.

    Raises:
        CorpusError: If `data_dir` does not exist, contains no *.json
            files, or any file is malformed / missing required fields.
    """

    path = Path(data_dir)
    if not path.is_dir():
        raise CorpusError(
            f"Regulatory corpus directory '{path}' does not exist. "
            "Create it and add at least one *.json provisions file."
        )

    json_files = sorted(path.glob("*.json"))
    if not json_files:
        raise CorpusError(
            f"No *.json files found in regulatory corpus directory '{path}'."
        )

    provisions: list[RegulatoryProvision] = []
    all_verified = True

    for json_file in json_files:
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CorpusError(f"Malformed JSON in '{json_file}': {exc}") from exc

        if not isinstance(raw, dict) or "provisions" not in raw:
            raise CorpusError(
                f"'{json_file}' must be a JSON object with a 'provisions' key."
            )

        verified = bool(raw.get("verified_real_text", False))
        if not verified:
            all_verified = False

        entries = raw["provisions"]
        if not isinstance(entries, list):
            raise CorpusError(f"'{json_file}' 'provisions' must be a JSON array.")

        for idx, entry in enumerate(entries):
            missing = [
                field
                for field in ("citation_id", "title", "text")
                if field not in entry
            ]
            if missing:
                raise CorpusError(
                    f"'{json_file}' provisions[{idx}] missing required "
                    f"field(s): {missing}"
                )
            provisions.append(
                RegulatoryProvision(
                    citation_id=entry["citation_id"],
                    title=entry["title"],
                    text=entry["text"],
                    keywords=tuple(entry.get("keywords", [])),
                    source_file=json_file.name,
                )
            )

    if not provisions:
        raise CorpusError(
            f"Regulatory corpus directory '{path}' contains *.json files "
            "but zero provisions were parsed from them."
        )

    return provisions, all_verified