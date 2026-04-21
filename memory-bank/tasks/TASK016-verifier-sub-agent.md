# TASK016 — Verifier Sub-Agent

**Status:** Pending  
**Added:** 2026-04-21  
**Updated:** 2026-04-21  
**Phase:** C — Agent Intelligence (requires TASK014 complete)

---

## Original Request
After the agent synthesizes an answer, there is no check that each claim is actually supported by the retrieved chunks. Add a lightweight verifier LLM call that scores each sentence/claim as "supported" or "unsupported", triggers one retry for unsupported claims (re-retrieve + re-synthesize that fragment), and exposes the verification result in the API response and SSE stream.

---

## Thought Process
The verifier is a post-synthesis LLM call, not a separate agent. It receives:
- The synthesized answer broken into claims
- The supporting chunks (top-k from TASK015)

And returns per-claim: `{"claim": "...", "verdict": "supported|unsupported|unknown", "evidence_chunk_id": "..."}`.

On finding unsupported claims:
1. Run a targeted re-retrieval for those specific claims (call `search_chunks` with the unsupported sentence as query)
2. Re-synthesize just those fragments
3. Run verifier again (single pass — no infinite loop)
4. Merge into final answer

The retry loop is bounded (max 1 retry per verifier run) to prevent infinite oscillation. An `GRAPHRAG_VERIFIER_ENABLED` env flag makes it easy to toggle off for speed.

**Important**: Do not use the verifier to rewrite factual statements — only to flag them. The agent's answer is returned as-is if unsupported claims can't be fixed in one retry.

---

## Implementation Plan

1. **New `services/graphrag/src/agent/verifier.py`**:
   ```python
   class VerifierResult(BaseModel):
       claims: list[ClaimVerdict]
       verified_answer: str | None  # only set if retry changed the answer
   
   class ClaimVerdict(BaseModel):
       claim: str
       verdict: Literal["supported", "unsupported", "unknown"]
       evidence_chunk_id: str | None
   
   async def verify_answer(
       answer: str,
       chunks: list[RetrievedChunk],
       llm: LLM,
   ) -> VerifierResult:
       ...
   ```

2. **Verifier prompt** (`prompts/verifier.txt`):
   - Input: answer + chunks
   - Output: JSON array of `{claim, verdict, evidence_chunk_id}`
   - Use structured output (`with_structured_output(list[ClaimVerdict])`)

3. **Wire into `workflow.py`** (after `_synthesize()`, before returning):
   ```python
   if settings.verifier_enabled:
       verification = await verify_answer(answer, top_chunks, llm)
       if unsupported := [c for c in verification.claims if c.verdict == "unsupported"]:
           # one retry: re-retrieve + re-synthesize unsupported fragment
           ...
   ```

4. **Emit SSE `reasoning` events during verification**:
   ```python
   yield {"type": "reasoning", "content": f"Verifying {len(claims)} claims..."}
   yield {"type": "reasoning", "content": f"⚠ {n} claims unsupported — retrying..."}
   ```

5. **Add `verification: VerificationResult | None` to `AgentResponse`** — populated after verifier runs.

6. **`GRAPHRAG_VERIFIER_ENABLED` env flag** (default `true`):
   - When `false`, skips verifier entirely (faster, useful for dev).
   - Document in `.env.example`.

7. **Unit tests**:
   - `test_verifier_returns_supported_for_grounded_claim(mock_llm)`
   - `test_verifier_triggers_retry_on_unsupported(mock_llm, mock_neo4j)`
   - `test_verifier_disabled_when_flag_false()`

---

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 16.1 | Create agent/verifier.py (VerifierResult, verify_answer) | Not Started | 2026-04-21 | |
| 16.2 | Write prompts/verifier.txt | Not Started | 2026-04-21 | |
| 16.3 | Wire verifier into workflow.py post-synthesis | Not Started | 2026-04-21 | |
| 16.4 | Implement single-pass retry for unsupported claims | Not Started | 2026-04-21 | |
| 16.5 | Emit reasoning SSE events during verification | Not Started | 2026-04-21 | |
| 16.6 | Add verification field to AgentResponse | Not Started | 2026-04-21 | |
| 16.7 | Add GRAPHRAG_VERIFIER_ENABLED env flag | Not Started | 2026-04-21 | |
| 16.8 | Unit tests | Not Started | 2026-04-21 | |

## Progress Log
### 2026-04-21
- Task created. No verifier exists. Answer quality entirely unvalidated.
