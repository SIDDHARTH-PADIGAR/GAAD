# GAAD: Graph Anomaly Agentic Dossier Pipeline

## System Nomenclature & High-Level Purpose

GAAD is a deterministic middleware layer that bridges NetworkX graph engines with LangGraph state machines. It sits between a bank's transaction-graph scoring system and its compliance officers, and it does one job: turn raw network topology features into localized, audit-ready compliance narratives, entirely inside an air-gapped environment.

The problem this solves is specific. Graph intelligence engines at Tier-1 Indian banks score accounts for laundering risk using structural features (clustering coefficients, device convergence, multi-hop velocity). Those scores are not readable by a human compliance officer, and they carry no legal grounding. Someone still has to trace the network path by hand, look up the relevant RBI KYC/AML provisions, and write a citation-backed Suspicious Transaction Report. GAAD automates that translation step deterministically, with a persistent, tamper-evident audit trail, without a single call leaving the bank's network.

There is no generative LLM anywhere in this pipeline. "Agentic" here refers to conditional, multi-step orchestration (route, retrieve, compile, persist), not autonomous text generation. That distinction is load-bearing: every decision GAAD makes is reproducible, explainable, and independently re-verifiable against the same graph and the same corpus, every time.

## Core Operational Flow

GAAD runs as a four-stage pipeline per account.

**1. Topological Feature Extraction**
A NetworkX `DiGraph` models the ledger network: accounts as nodes, transactions as directed edges. For each account, GAAD computes local clustering coefficient, shared-device account count (identity resolution across the full node set, not just transaction-adjacent neighbors), and bidirectional multi-hop transfer velocity (hop count and minimum time interval, walked forward and backward through the account so a middle-of-chain node gets full credit for the chain it sits inside).

**2. Routing Firewall**
A deterministic Python rule evaluates the extracted metrics against configurable thresholds. If the account does not exhibit device convergence combined with at least one of clustering, velocity-hop, or velocity-speed signals, the pipeline terminates immediately with a `SAFE` verdict. No retrieval call, no token spend, no latency cost. Only `FLAGGED` accounts proceed to stage three. This is the cost-control gate: most accounts in a real ledger are clean, and the firewall is built to reject them cheaply and explainably, with every triggered threshold recorded on the decision object.

**3. Air-Gapped Retrieval Engine**
For `FLAGGED` accounts only, a local vector store built on `sentence-transformers` (`all-MiniLM-L6-v2`) performs cosine similarity search against a regulatory corpus, entirely in-process, with zero network calls. The embedding model is loaded exclusively from a local directory; there is no runtime download path. Retrieval returns ranked regulatory provisions with similarity scores, grounding the escalation in specific citations rather than a bare anomaly score.

**4. Structured Log Compiler**
Every account, `SAFE` or `FLAGGED`, passes through a single compile node before persistence. That node deterministically builds a strict JSON payload: account ID, verdict, triggered reasons, raw topology metrics, retrieved provisions (if any), and a corpus-verification flag. The record is then appended to a SHA-256 hash-chained log, where each entry's hash covers its own content plus the previous entry's hash. Any modification to a historical entry breaks the chain from that point forward, and `verify_chain()` detects it on demand.

## Automated Verification & Scoring

Testing philosophy here is exact recomputation, not sampled or estimated scoring. Because routing and retrieval are both pure functions of the graph and the corpus, every metric below is checked by independently rebuilding ground truth from scratch and diffing against what the pipeline actually produced, using code that shares no implementation with the thing it is checking.

Context grounding is verified by reloading the regulatory corpus fresh from disk (bypassing any in-memory vector store) and confirming every retrieved citation ID and text body matches a real corpus entry byte-for-byte. Topological accuracy is verified by hand-written, independent reimplementations of clustering coefficient (triangle-counting from first principles) and shared-device count (brute-force node scan), diffed against the firewall's reported values.

Verified scorecard output from `tests/eval_suite.py`:

```
======================================================================
GAAD MODULE 5 EVALUATION SCORECARD
======================================================================

NOTE: Topological confusion-matrix ground truth is a SYNTHETIC, planted label (AccountType.LAYERING), used here to validate
orchestration plumbing, not real-world detection performance. See the module docstring for the production backtesting design.

--- Topological Compliance Accuracy ---
Accounts checked:        19
Zero degradation:        True
Max error (clustering):  0.0
Max error (device cnt):  0.0
Confusion matrix:        TP=3 FP=0 TN=16 FN=0
Precision / Recall / F1: 1.000 / 1.000 / 1.000
Accuracy:                1.000

--- Context Grounding ---
Citations checked:       9
Verified grounded:       9
Hallucinated:            0
Grounding precision:     1.000
Corpus coverage ratio:   0.600 (3/5)
Fully grounded:          True

======================================================================
SUMMARY
======================================================================
Context Grounding Score:      100.0% (9/9 citations verified against on-disk corpus, 0 hallucinated)
Topological Accuracy:         100.0% (vs. SYNTHETIC planted labels -- see scoping note above; F1=1.000, degradation=NONE)
======================================================================
```

Two of these numbers mean different things and should not be read the same way. Context grounding precision (100%) is a structural guarantee: retrieval can only ever return objects it loaded from the on-disk corpus, so zero hallucination is a property of the code, not a benchmark result. Topological accuracy (100%) is scored against a synthetic, planted label on a graph built specifically to be separable by this rule, which makes it a plumbing-integrity check, not a real-world detection performance claim. Production validation requires backtesting against historically filed STR outcomes, where ground truth exists only in arrears and only for accounts that were already escalated.

## Adversarial Stress-Testing & Fuzzer Analysis

Self-consistency checks against a benchmark the same engineer designed prove the wiring works. They do not prove the detection logic is sound against an adversary who knows the rule. `tests/adversarial_suite.py` exists to find real failure modes, not confirm the absence of any.

It found one. A laundering chain was constructed routing through an innocent, high-degree hub account (a payment-gateway stand-in with 40 unrelated legitimate customers), with the layering hop using a unique, never-reused device specifically to defeat device-linkage detection. Result: a confirmed false negative. The velocity signal correctly detected the full 3-hop chain through the hub, but the firewall's escalation rule requires device convergence as a hard AND-gate before any other signal can trigger an escalation. Zero shared-device matches suppressed the flag entirely, regardless of how strong the velocity signal was.

Structural noise test result, from `tests/adversarial_suite.py`:

```json
{
  "target_account_id": "ACC-ADV-LAYER-000",
  "firewall_verdict": "SAFE",
  "shared_device_account_count": 0,
  "max_multi_hop_velocity_hops": 3,
  "is_false_negative": true,
  "attack_description": "Laundering chain (placement -> hub -> layering -> integration) routed through an innocent, high-degree hub account (40 unrelated legitimate customers). The layering account uses a unique, never-reused device specifically to avoid triggering device-linkage detection.",
  "root_cause": "route_accounts()._decide() requires device_signal as a hard AND-gate before ANY other signal (clustering/velocity) can escalate an account. Velocity correctly detected the 3-hop chain (max_multi_hop_velocity_hops=3) but device_signal=False (shared_device_account_count=0 < threshold=2) suppresses escalation entirely, regardless of how strong the velocity signal is.",
  "remediation_notes": "Recommend replacing the hard AND-gate with a weighted/scored combination (e.g. escalate if device_signal OR (velocity_hops_signal AND velocity_speed_signal), or a point-based threshold summing multiple weaker signals) so a sufficiently strong velocity+clustering pattern can escalate even without a corroborating device match. Also recommend extending device-linkage detection beyond immediate neighbors to a bounded N-hop neighborhood, so a cutout hub does not fully sever the device trail between ring members."
}
```

This is an open architectural finding, not yet patched. It is documented here deliberately, because a compliance-facing system that hides its own known blind spots is worse than one that states them plainly.

Separately, the fuzzer threw six deliberately malformed inputs directly at the model and pipeline layers: negative transaction amounts, timezone-naive timestamps, self-looping transactions, NaN KYC risk scores, empty-string device identifiers, and a graph node missing its account attribute entirely. The first four were caught cleanly by structured `ModelValidationError`s from day one. The remaining two were confirmed gaps on first run and have since been patched: `Account` now rejects blank or whitespace-only device ID strings at construction time, and `compute_account_metrics` now catches missing node attributes and re-raises them as a structured `MetricsComputationError` naming the corrupted node, rather than letting a raw `KeyError` propagate out of the state machine.

Fuzzer scorecard from `tests/adversarial_suite.py`, post-patch:

```json
"fuzz_cases": [
    {
      "case_id": "negative_transaction_amount",
      "outcome": "CAUGHT_BY_DEFINED_ERROR",
      "exception_type": "ModelValidationError",
      "exception_message": "Transaction amount_inr must be > 0, got -500.0.",
      "is_known_gap": false
    },
    {
      "case_id": "naive_timezone_timestamp",
      "outcome": "CAUGHT_BY_DEFINED_ERROR",
      "exception_type": "ModelValidationError",
      "exception_message": "Transaction.executed_at must be timezone-aware.",
      "is_known_gap": false
    },
    {
      "case_id": "self_loop_transaction",
      "outcome": "CAUGHT_BY_DEFINED_ERROR",
      "exception_type": "ModelValidationError",
      "exception_message": "Transaction cannot self-loop on account 'A'.",
      "is_known_gap": false
    },
    {
      "case_id": "nan_kyc_risk_score",
      "outcome": "CAUGHT_BY_DEFINED_ERROR",
      "exception_type": "ModelValidationError",
      "exception_message": "Account 'A' kyc_risk_score must be in [0.0, 1.0], got nan.",
      "is_known_gap": false
    },
    {
      "case_id": "empty_string_device_id",
      "outcome": "CAUGHT_BY_DEFINED_ERROR",
      "exception_type": "ModelValidationError",
      "exception_message": "Account 'A' device_ids contains 1 blank/empty-string device identifier(s). Every device_id must be a non-empty, non-whitespace string.",
      "is_known_gap": false
    },
    {
      "case_id": "corrupted_graph_node_missing_account_attr",
      "outcome": "CAUGHT_BY_DEFINED_ERROR",
      "exception_type": "MetricsComputationError",
      "exception_message": "Account 'ACC-CORRUPT-001' (or a node it is compared against) is missing required node data: 'account'. This indicates a corrupted or partially-written graph node -- every node must carry a 'account' attribute before metrics can be computed.",
      "is_known_gap": false
    }
  ],
  "summary": {
    "fuzz_cases_total": 6,
    "fuzz_cases_with_known_gaps": 0,
    "fuzz_gap_case_ids": []
  }
```

Six out of six malformed inputs now terminate in a structured, typed GAAD error instead of a raw exception, before ever reaching the audit log.

## On-Premise Deployment & Data Decoupling Protocol

Deployment is split into two phases with a hard boundary between them.

**Phase one: internet-facing setup, off the air-gapped box.** `src/download_model.py` is the only component in this repository that ever touches the network. Run it once, on a machine with internet access, to pull the `all-MiniLM-L6-v2` weights and save them to `./local_model/`. Copy that directory onto the target machine over your standard secure transfer channel (USB, internal file transfer, whatever the bank's air-gap policy mandates).

**Phase two: runtime, fully offline.** `LocalRegulatoryVectorStore` loads the embedding model exclusively from `./local_model/` on disk. There is no fallback path, no lazy download, no telemetry call. If that directory is missing, the constructor raises immediately with the exact remediation command, rather than attempting any network access. This is the enforced boundary: nothing downstream of phase one can reach the internet, by construction, not by convention.

Regulatory text is fully decoupled from application logic. Every provision GAAD cites lives in JSON files under `data/rbi_circulars/`, dynamically loaded at startup by `corpus_loader.py`. No clause number, section title, or legal text is hardcoded anywhere in the retrieval, routing, or pipeline code. A bank's legal team replaces the contents of that directory with verified, sourced official text and nothing else in the codebase needs to change, no redeploy of application logic, no dependency risk.

The corpus currently shipped is illustrative, not verified. Every provisions file carries a `verified_real_text` boolean, defaulted to `false`, and that flag is propagated end to end through retrieval into the audit log itself. Any record built from an unverified corpus carries a top-level `SYSTEM_AUDIT_WARNING` key in its persisted JSON, injected deterministically by the compile stage of the state machine before that record is ever written to disk. That stage runs unconditionally on every account, `SAFE` or `FLAGGED`, so the warning cannot be bypassed by any upstream node. Do not export any record carrying that key to a regulator until the corpus has been replaced with counsel-verified official text and the record has been re-audited.
