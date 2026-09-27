"""Typed domain models for the GAAD ledger data layer.

These models are intentionally framework-agnostic (plain dataclasses,
no ORM/NetworkX coupling) so they can be reused by the firewall, RAG,
and audit-compiler modules without pulling in graph-construction code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, unique


@unique
class AccountType(str, Enum):
    """Role an account plays within a laundering typology.

    PLACEMENT: initial deposit point for illicit funds.
    LAYERING: intermediate hop used to obscure the money trail.
    INTEGRATION: final exit point where funds are reintroduced
        into the legitimate economy.
    LEGITIMATE: a normal, unrelated account used as background noise.
    """

    PLACEMENT = "PLACEMENT"
    LAYERING = "LAYERING"
    INTEGRATION = "INTEGRATION"
    LEGITIMATE = "LEGITIMATE"


class ModelValidationError(ValueError):
    """Raised when a domain model is constructed with invalid data."""


@dataclass(frozen=True, slots=True)
class Device:
    """A mobile/device fingerprint associated with account access.

    Attributes:
        device_id: Stable hardware/software fingerprint (e.g. IMEI hash).
        first_seen: UTC timestamp the device was first observed.
    """

    device_id: str
    first_seen: datetime

    def __post_init__(self) -> None:
        if not self.device_id or not self.device_id.strip():
            raise ModelValidationError("Device.device_id must be non-empty.")
        if self.first_seen.tzinfo is None:
            raise ModelValidationError(
                f"Device '{self.device_id}' first_seen must be timezone-aware."
            )


@dataclass(frozen=True, slots=True)
class Account:
    """A bank account node in the ledger network.

    Attributes:
        account_id: Unique account identifier (e.g. masked account number).
        account_type: Role this account plays (see AccountType).
        holder_name: Display name of the account holder (mock data only).
        device_ids: Devices historically used to access this account.
        opened_at: UTC timestamp the account was opened.
        kyc_risk_score: Bank-assigned KYC risk score in [0.0, 1.0].
    """

    account_id: str
    account_type: AccountType
    holder_name: str
    device_ids: frozenset[str]
    opened_at: datetime
    kyc_risk_score: float

    def __post_init__(self) -> None:
        if not self.account_id or not self.account_id.strip():
            raise ModelValidationError("Account.account_id must be non-empty.")
        if not self.holder_name or not self.holder_name.strip():
            raise ModelValidationError("Account.holder_name must be non-empty.")
        if self.opened_at.tzinfo is None:
            raise ModelValidationError(
                f"Account '{self.account_id}' opened_at must be timezone-aware."
            )
        if not (0.0 <= self.kyc_risk_score <= 1.0):
            raise ModelValidationError(
                f"Account '{self.account_id}' kyc_risk_score must be in "
                f"[0.0, 1.0], got {self.kyc_risk_score}."
            )
        if not self.device_ids:
            raise ModelValidationError(
                f"Account '{self.account_id}' must have at least one device_id."
            )
        blank_device_ids = [d for d in self.device_ids if not d or not d.strip()]
        if blank_device_ids:
            raise ModelValidationError(
                f"Account '{self.account_id}' device_ids contains "
                f"{len(blank_device_ids)} blank/empty-string device "
                "identifier(s). Every device_id must be a non-empty, "
                "non-whitespace string."
            )


@dataclass(frozen=True, slots=True)
class Transaction:
    """A directed funds-transfer edge between two accounts.

    Attributes:
        source_account_id: Account the funds leave.
        dest_account_id: Account the funds arrive at.
        amount_inr: Transfer amount in INR. Must be strictly positive.
        executed_at: UTC timestamp the transfer executed.
        channel: Rail used, e.g. "IMPS", "NEFT", "UPI".
    """

    source_account_id: str
    dest_account_id: str
    amount_inr: float
    executed_at: datetime
    channel: str = "IMPS"

    def __post_init__(self) -> None:
        if self.source_account_id == self.dest_account_id:
            raise ModelValidationError(
                f"Transaction cannot self-loop on account "
                f"'{self.source_account_id}'."
            )
        if self.amount_inr <= 0:
            raise ModelValidationError(
                f"Transaction amount_inr must be > 0, got {self.amount_inr}."
            )
        if self.executed_at.tzinfo is None:
            raise ModelValidationError(
                "Transaction.executed_at must be timezone-aware."
            )
        if not self.channel.strip():
            raise ModelValidationError("Transaction.channel must be non-empty.")


def utc_now() -> datetime:
    """Returns the current UTC time, timezone-aware.

    Centralized so every module in GAAD sources time the same way,
    which matters later for the hash-chained audit log's ordering
    guarantees.
    """

    return datetime.now(timezone.utc)