# GAAD — Graph Anomaly Agentic Dossier Pipeline

Air-gapped, audit-ready backend that bridges graph-based AML anomaly
detection with an agentic RAG layer for RBI-regulation-grounded
Suspicious Transaction Report (STR) drafting.

## Setup
See SHELL COMMANDS in module logs. Local dev only — no external
network calls are made by any module in this repository.

## Modules
- `data_layer`: synthetic ledger graph construction (Module 1)
- `firewall`: topological anomaly routing (Module 2 — pending)
- `rag_engine`: air-gapped regulatory retrieval (Module 3 — pending)
- `audit_compiler`: hash-chained audit log (Module 4 — pending)