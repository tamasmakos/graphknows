---
name: agent-owasp-compliance
description: |
  Check any AI agent codebase against the OWASP Agentic Security Initiative (ASI) Top 10 risks.
  Use this skill when:
  - Evaluating an agent system's security posture before production deployment
  - Running a compliance check against OWASP ASI 2026 standards
  - Mapping existing security controls to the 10 agentic risks
  - Generating a compliance report for security review or audit
  - Comparing agent framework security features against the standard
  - Any request like "is my agent OWASP compliant?", "check ASI compliance", or "agentic security audit"
---

# Agent OWASP ASI Compliance Check

Evaluate AI agent systems against the OWASP Agentic Security Initiative (ASI) Top 10 — the industry standard for agent security posture.

## Overview

The OWASP ASI Top 10 defines the critical security risks specific to autonomous AI agents — not LLMs, not chatbots, but agents that call tools, access systems, and act on behalf of users. This skill checks whether your agent implementation addresses each risk.

```
Codebase → Scan for each ASI control:
  ASI-01: Prompt Injection Protection
  ASI-02: Tool Use Governance
  ASI-03: Agency Boundaries
  ASI-04: Escalation Controls
  ASI-05: Trust Boundary Enforcement
  ASI-06: Logging & Audit
  ASI-07: Identity Management
  ASI-08: Policy Integrity
  ASI-09: Supply Chain Verification
  ASI-10: Behavioral Monitoring
→ Generate Compliance Report (X/10 covered)
```

## The 10 Risks

| Risk | Name | What to Look For |
|------|------|-----------------|
| ASI-01 | Prompt Injection | Input validation before tool calls, not just LLM output filtering |
| ASI-02 | Insecure Tool Use | Tool allowlists, argument validation, no raw shell execution |
| ASI-03 | Excessive Agency | Capability boundaries, scope limits, principle of least privilege |
| ASI-04 | Unauthorized Escalation | Privilege checks before sensitive operations, no self-promotion |
| ASI-05 | Trust Boundary Violation | Trust verification between agents, signed credentials, no blind trust |
| ASI-06 | Insufficient Logging | Structured audit trail for all tool calls, tamper-evident logs |
| ASI-07 | Insecure Identity | Cryptographic agent identity, not just string names |
| ASI-08 | Policy Bypass | Deterministic policy enforcement, no LLM-based permission checks |
| ASI-09 | Supply Chain Integrity | Signed plugins/tools, integrity verification, dependency auditing |
| ASI-10 | Behavioral Anomaly | Drift detection, circuit breakers, kill switch capability |

---

## Check ASI-01: Prompt Injection Protection

Look for input validation that runs **before** tool execution, not after LLM generation.

**What passing looks like:**
```python
# GOOD: Validate before tool execution
result = policy_engine.evaluate(user_input)
if result.action == "deny":
    return "Request blocked by policy"
tool_result = await execute_tool(validated_input)
```

**What failing looks like:**
```python
# BAD: User input goes directly to tool
tool_result = await execute_tool(user_input)  # No validation
```

---

## Check ASI-02: Insecure Tool Use

Verify tools have allowlists, argument validation, and no unrestricted execution.

**Passing example:**
```python
ALLOWED_TOOLS = {"search", "read_file", "create_ticket"}

def execute_tool(name: str, args: dict):
    if name not in ALLOWED_TOOLS:
        raise PermissionError(f"Tool '{name}' not in allowlist")
    return tools[name](**validated_args)
```

---

## Check ASI-03: Excessive Agency

Verify agent capabilities are bounded — not open-ended.

**Failing:** Agent has access to all tools by default.
**Passing:** Agent capabilities defined as a fixed allowlist, unknown tools denied.

---

## Check ASI-04: Unauthorized Escalation

Verify agents cannot promote their own privileges.

**Failing:** Agent can modify its own configuration or permissions.
**Passing:** Privilege changes require out-of-band approval (e.g., Ring 0 requires SRE attestation).

---

## Check ASI-05: Trust Boundary Violation

In multi-agent systems, verify that agents verify each other's identity before accepting instructions.

**Passing example:**
```python
def accept_task(sender_id: str, task: dict):
    trust = trust_registry.get_trust(sender_id)
    if not trust.meets_threshold(0.7):
        raise PermissionError(f"Agent {sender_id} trust too low: {trust.current()}")
    if not verify_signature(task, sender_id):
        raise SecurityError("Task signature verification failed")
    return process_task(task)
```

---

## Check ASI-06: Insufficient Logging

Verify all agent actions produce structured, tamper-evident audit entries.

**Failing:** Agent actions logged via `print()` or not logged at all.
**Passing:** Structured JSONL audit trail with chain hashes, exported to secure storage.

---

## Check ASI-07: Insecure Identity

**Failing indicators:**
- Agent identified by `agent_name = "my-agent"` (string only)
- No authentication between agents
- Shared credentials across agents

**Passing indicators:**
- DID-based identity (`did:web:`, `did:key:`)
- Ed25519 or similar cryptographic signing
- Per-agent credentials with rotation

---

## Check ASI-08: Policy Bypass

**Failing:** Agent decides its own permissions via prompt ("Am I allowed to...?").
**Passing:** PolicyEvaluator.evaluate() returns allow/deny deterministically, no LLM involved.

---

## Check ASI-09: Supply Chain Integrity

**What to search for:**
- `INTEGRITY.json` or manifest files with SHA-256 hashes
- Signature verification on plugin installation
- Dependency pinning (no `@latest` or unbounded ranges)
- SBOM generation

---

## Check ASI-10: Behavioral Anomaly

**Failing:** No mechanism to stop a misbehaving agent automatically.
**Passing:** Circuit breaker trips after N failures, trust decays without activity, kill switch available.

---

## Compliance Report Format

```markdown
# OWASP ASI Compliance Report
Generated: [date]
Project: [project name]

## Summary: X/10 Controls Covered

| Risk | Status | Finding |
|------|--------|---------|
| ASI-01 Prompt Injection | PASS | PolicyEngine validates input before tool calls |
| ASI-02 Insecure Tool Use | PASS | Tool allowlist enforced |
| ASI-03 Excessive Agency | PASS | Execution rings limit capabilities |
| ASI-04 Unauthorized Escalation | PASS | Ring promotion requires attestation |
| ASI-05 Trust Boundary | FAIL | No identity verification between agents |
| ASI-06 Insufficient Logging | PASS | AuditChain with SHA-256 chain hashes |
| ASI-07 Insecure Identity | FAIL | Agents use string names, no crypto identity |
| ASI-08 Policy Bypass | PASS | Deterministic PolicyEvaluator, no LLM in path |
| ASI-09 Supply Chain | FAIL | No integrity manifests or plugin signing |
| ASI-10 Behavioral Anomaly | PASS | Circuit breakers and trust decay active |

## Critical Gaps
- ASI-05: Add agent identity verification using DIDs or signed tokens
- ASI-07: Replace string agent names with cryptographic identity
- ASI-09: Generate INTEGRITY.json manifests for all plugins
```

---

## Quick Assessment Questions

1. Does user input pass through validation before reaching any tool? (ASI-01)
2. Is there an explicit list of what tools the agent can call? (ASI-02)
3. Can the agent do anything, or are its capabilities bounded? (ASI-03)
4. Can the agent promote its own privileges? (ASI-04)
5. Do agents verify each other's identity before accepting tasks? (ASI-05)
6. Is every tool call logged with enough detail to replay it? (ASI-06)
7. Does each agent have a unique cryptographic identity? (ASI-07)
8. Is policy enforcement deterministic (not LLM-based)? (ASI-08)
9. Are plugins/tools integrity-verified before use? (ASI-09)
10. Is there a circuit breaker or kill switch? (ASI-10)

If you answer "no" to any of these, that's a gap to address.
