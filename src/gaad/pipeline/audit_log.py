"""Hash-chained, append-only audit log compiler.

Every entry's hash is computed over its own canonical content PLUS
the previous entry's hash. This means any modification to, or removal
of, a historical entry changes the hash chain from that point forward
in a way `verify_chain()` will detect. This is the "unalterable
processing trace log" required by the project brief.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS_HASH = "GENESIS"
DEFAULT_AUDIT_LOG_PATH = Path("audit_logs/gaad_audit_log.jsonl")


class AuditLogError(RuntimeError):
    """Raised when the audit log cannot be written to, or its chain is broken."""


@dataclass(frozen=True, slots=True)
class AuditLogEntry:
    """One persisted, hash-chained record in the audit log.

    Attributes:
        sequence_number: 1-indexed position of this entry in the chain.
        account_id: The account this entry concerns.
        verdict: "SAFE" or "FLAGGED".
        triggered_reasons: Firewall reasons, empty tuple for SAFE.
        metrics: The account's raw AccountTopologyMetrics, as a dict.
        regulatory_provisions: Retrieved citations (FLAGGED only), or
            None for SAFE accounts that never queried the RAG engine.
        corpus_is_verified_real_text: None if no retrieval occurred;
            otherwise whether the RAG corpus used was verified official
            text (False for the current illustrative/mock corpus).
        system_audit_warning: The mandatory warning string if the
            corpus was unverified, else None.
        created_at: ISO-8601 UTC timestamp.
        previous_entry_hash: The prior entry's entry_hash, or GENESIS_HASH.
        entry_hash: SHA-256 hex digest binding this entry to the chain.
    """

    sequence_number: int
    account_id: str
    verdict: str
    triggered_reasons: tuple[str, ...]
    metrics: dict[str, Any]
    regulatory_provisions: tuple[dict[str, Any], ...] | None
    corpus_is_verified_real_text: bool | None
    system_audit_warning: str | None
    created_at: str
    previous_entry_hash: str
    entry_hash: str

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["triggered_reasons"] = list(self.triggered_reasons)
        payload["regulatory_provisions"] = (
            list(self.regulatory_provisions)
            if self.regulatory_provisions is not None
            else None
        )
        # The mandatory warning must survive persistence under the SAME
        # top-level key compile_node used (MANDATORY_WARNING_KEY =
        # "SYSTEM_AUDIT_WARNING"), not the dataclass field's lowercase
        # name. Without this, persist_node's overwrite of audit_record
        # with this dict silently renames/drops the mandatory key.
        payload.pop("system_audit_warning", None)
        if self.system_audit_warning is not None:
            payload["SYSTEM_AUDIT_WARNING"] = self.system_audit_warning
        return payload


def _canonical_payload_for_hash(
    *,
    sequence_number: int,
    account_id: str,
    verdict: str,
    triggered_reasons: tuple[str, ...],
    metrics: dict[str, Any],
    regulatory_provisions: tuple[dict[str, Any], ...] | None,
    corpus_is_verified_real_text: bool | None,
    system_audit_warning: str | None,
    created_at: str,
    previous_entry_hash: str,
) -> str:
    """Deterministic JSON serialization used as the hash input.

    sort_keys=True guarantees the same logical entry always produces
    the same byte string, which is required for verify_chain() to be
    independently re-computable.
    """

    payload = {
        "sequence_number": sequence_number,
        "account_id": account_id,
        "verdict": verdict,
        "triggered_reasons": list(triggered_reasons),
        "metrics": metrics,
        "regulatory_provisions": (
            list(regulatory_provisions) if regulatory_provisions is not None else None
        ),
        "corpus_is_verified_real_text": corpus_is_verified_real_text,
        "system_audit_warning": system_audit_warning,
        "created_at": created_at,
        "previous_entry_hash": previous_entry_hash,
    }
    return json.dumps(payload, sort_keys=True, default=str)


class AuditLogCompiler:
    """Append-only, hash-chained audit log backed by a JSONL file on disk."""

    def __init__(self, log_path: Path | str = DEFAULT_AUDIT_LOG_PATH) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_last_hash(self) -> tuple[str, int]:
        """Returns (previous_entry_hash, next_sequence_number)."""

        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return GENESIS_HASH, 1

        last_line = ""
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_line = stripped

        if not last_line:
            return GENESIS_HASH, 1

        try:
            last_entry = json.loads(last_line)
        except json.JSONDecodeError as exc:
            raise AuditLogError(
                f"Audit log '{self.log_path}' has a malformed final line: {exc}"
            ) from exc

        return last_entry["entry_hash"], last_entry["sequence_number"] + 1

    def append_entry(
        self,
        *,
        account_id: str,
        verdict: str,
        triggered_reasons: tuple[str, ...],
        metrics: dict[str, Any],
        regulatory_provisions: tuple[dict[str, Any], ...] | None,
        corpus_is_verified_real_text: bool | None,
        system_audit_warning: str | None,
    ) -> AuditLogEntry:
        """Appends one new, hash-chained entry to the log file.

        Raises:
            AuditLogError: If the existing log's final line is malformed.
        """

        previous_hash, sequence_number = self._read_last_hash()
        created_at = datetime.now(timezone.utc).isoformat()

        canonical = _canonical_payload_for_hash(
            sequence_number=sequence_number,
            account_id=account_id,
            verdict=verdict,
            triggered_reasons=triggered_reasons,
            metrics=metrics,
            regulatory_provisions=regulatory_provisions,
            corpus_is_verified_real_text=corpus_is_verified_real_text,
            system_audit_warning=system_audit_warning,
            created_at=created_at,
            previous_entry_hash=previous_hash,
        )
        entry_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        entry = AuditLogEntry(
            sequence_number=sequence_number,
            account_id=account_id,
            verdict=verdict,
            triggered_reasons=triggered_reasons,
            metrics=metrics,
            regulatory_provisions=regulatory_provisions,
            corpus_is_verified_real_text=corpus_is_verified_real_text,
            system_audit_warning=system_audit_warning,
            created_at=created_at,
            previous_entry_hash=previous_hash,
            entry_hash=entry_hash,
        )

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_json_dict(), sort_keys=True))
            f.write("\n")

        return entry

    def verify_chain(self) -> bool:
        """Re-walks the entire log and recomputes every hash from scratch.

        Returns:
            True if the chain is fully intact.

        Raises:
            AuditLogError: If tampering or corruption is detected,
                naming the first bad sequence_number found.
        """

        if not self.log_path.exists():
            return True

        expected_previous = GENESIS_HASH
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                entry = json.loads(stripped)

                if entry["previous_entry_hash"] != expected_previous:
                    raise AuditLogError(
                        f"Audit chain broken at sequence_number="
                        f"{entry['sequence_number']}: previous_entry_hash does "
                        "not match the prior entry's actual hash."
                    )

                recomputed = hashlib.sha256(
                    _canonical_payload_for_hash(
                        sequence_number=entry["sequence_number"],
                        account_id=entry["account_id"],
                        verdict=entry["verdict"],
                        triggered_reasons=tuple(entry["triggered_reasons"]),
                        metrics=entry["metrics"],
                        regulatory_provisions=(
                            tuple(entry["regulatory_provisions"])
                            if entry["regulatory_provisions"] is not None
                            else None
                        ),
                        corpus_is_verified_real_text=entry["corpus_is_verified_real_text"],
                                                # The persisted JSON stores this under
                        # MANDATORY_WARNING_KEY ("SYSTEM_AUDIT_WARNING")
                        # when present, and omits the key entirely when
                        # not -- matching to_json_dict()'s serialization.
                        # Reading the old lowercase key here would
                        # KeyError against every entry written since
                        # that rename.
                        system_audit_warning=entry.get("SYSTEM_AUDIT_WARNING"),
                        created_at=entry["created_at"],
                        previous_entry_hash=entry["previous_entry_hash"],
                    ).encode("utf-8")
                ).hexdigest()

                if recomputed != entry["entry_hash"]:
                    raise AuditLogError(
                        f"Audit chain broken at sequence_number="
                        f"{entry['sequence_number']}: stored entry_hash does not "
                        "match recomputed hash. Entry may have been tampered with."
                    )

                expected_previous = entry["entry_hash"]

        return True