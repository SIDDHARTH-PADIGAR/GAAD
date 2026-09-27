"""Asynchronous LangGraph orchestration and hash-chained audit logging.

This package contains NO generative LLM calls. It orchestrates the
deterministic firewall (Module 2) and the local RAG engine (Module 3)
as a conditionally-branching state machine, then persists the result
to an append-only, tamper-evident audit log.
"""

from gaad.pipeline.audit_log import (
    AuditLogCompiler,
    AuditLogEntry,
    AuditLogError,
    DEFAULT_AUDIT_LOG_PATH,
)
from gaad.pipeline.graph_pipeline import (
    MANDATORY_WARNING_KEY,
    build_gaad_pipeline,
    run_account_through_pipeline,
)
from gaad.pipeline.state import PipelineState

__all__ = [
    "AuditLogCompiler",
    "AuditLogEntry",
    "AuditLogError",
    "DEFAULT_AUDIT_LOG_PATH",
    "MANDATORY_WARNING_KEY",
    "build_gaad_pipeline",
    "run_account_through_pipeline",
    "PipelineState",
]