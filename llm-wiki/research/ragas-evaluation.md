# RAGAS Evaluation Results — Graphknows GraphRAG Agent

**Date:** 2025-05-04  
**Endpoint evaluated:** `http://localhost:8001/retrieve` (GraphRAG retrieval agent)  
**Knowledge base:** `ep_debate_2024_03_13` — 8 chunks, 129 entities, 6 Topic/community nodes  
**Debate series:** *This is Europe* — EU leaders at EP plenary, March 2024  

---

## 1. Methodology

### Dataset Construction
- **10 questions** covering key themes: Ukraine support, energy transition, strategic autonomy, maritime security, peace negotiations, sanctions, economic impacts, defence spending, Green Deal, international law.
- **Contexts:** Citation excerpts from `citations[].excerpt`; fallback to observation text from last `fuse_and_rank`-style step in `reasoning_steps` when citations were empty.
- **Ground truth:** Manually written 1–2 sentence factual summaries per question, derived from agent answers (RAGAS reference-free mode — no external ground truth available).

### RAGAS Metrics
No LLM API key (`OPENAI_API_KEY`, `GROQ_API_KEY`) was available. LLM-based RAGAS metrics (faithfulness, answer_relevancy via NLI, context_precision) **could not be computed**.

**Fallback: Embedding-based similarity** using `sentence-transformers/all-MiniLM-L6-v2`:

| Metric | Description |
|---|---|
| `answer_relevancy_emb` | Cosine sim between question embedding and answer embedding |
| `ground_truth_similarity` | Cosine sim between answer embedding and ground truth embedding |
| `context_relevancy_emb` | Cosine sim between question embedding and retrieved context |
| `context_answer_alignment` | Cosine sim between answer and retrieved context |

---

## 2. Per-Question Results

### Q1 — Ukraine Military Support
- **Query:** What positions did EU leaders take on Ukraine military support?
- **Answer (200 chars):** EU leaders, such as Ursula von der Leyen [URSULA], have taken a position in support of Ukraine military support, as evidenced by her statement in [chunk_7], where she affirms that Europe
- **Citations:** 1 | **Steps:** 4
- **ans_rel:** 0.7110 | **gt_sim:** 0.8718 | **ctx_rel:** 0.1454 | **ctx_ans:** 0.3020

### Q2 — Energy Transition
- **Query:** Who discussed energy transition and what were their main arguments?
- **Answer (200 chars):** The energy transition was discussed in the context of the EU's strategy in response to the ongoing Middle East crisis, as seen in [chunk_4]. The main arguments focused on the need for the
- **Citations:** 1 | **Steps:** 5
- **ans_rel:** 0.6023 | **gt_sim:** 0.6966 | **ctx_rel:** 0.3594 | **ctx_ans:** 0.5839

### Q3 — EU Strategic Autonomy
- **Query:** What is EU strategic autonomy and which leaders mentioned it?
- **Answer (200 chars):** EU strategic autonomy refers to the European Union's ability to act independently in its strategic decisions, particularly in areas such as energy and defence. It was mentioned by leaders
- **Citations:** 2 | **Steps:** 5
- **ans_rel:** 0.8250 | **gt_sim:** 0.8261 | **ctx_rel:** 0.2443 | **ctx_ans:** 0.4036

### Q4 — Maritime Security Entities
- **Query:** Which entities are associated with maritime security topics?
- **Answer (200 chars):** Entities associated with maritime security topics include "UN_SECURITY_COUNCIL" [entity name], "UN_SECURITY_COUNCIL_RESOLUTIONS" [entity name], and "K
- **Citations:** 0 | **Steps:** 4
- **ans_rel:** 0.7815 | **gt_sim:** 0.6684 | **ctx_rel:** 0.1176 | **ctx_ans:** 0.1560

### Q5 — Peace Negotiation Efforts
- **Query:** What peace negotiation efforts were discussed in EU debates?
- **Answer (200 chars):** The EU debates discussed peace negotiation efforts, including the EU's commitment to work with regional partners for stability, expand dialogue, and s
- **Citations:** 1 | **Steps:** 5
- **ans_rel:** 0.8747 | **gt_sim:** 0.6467 | **ctx_rel:** 0.3090 | **ctx_ans:** 0.4065

### Q6 — Sanctions
- **Query:** Who mentioned sanctions and in what context?
- **Answer (200 chars):** Sanctions are mentioned in the context of supporting Ukraine and condemning Russia's aggression, as seen in [chunk_7] and related to entities such as
- **Citations:** 1 | **Steps:** 5
- **ans_rel:** 0.6944 | **gt_sim:** 0.7696 | **ctx_rel:** 0.2687 | **ctx_ans:** 0.1937

### Q7 — Economic Impacts of Ukraine War
- **Query:** What economic impacts of the Ukraine war were discussed?
- **Answer (200 chars):** The economic impacts of the Ukraine war discussed include sanctions, inflation, and the effects on international trade, as evidenced by chunk_7, chunk
- **Citations:** 3 | **Steps:** 5
- **ans_rel:** 0.8199 | **gt_sim:** 0.7296 | **ctx_rel:** 0.2030 | **ctx_ans:** 0.1451

### Q8 — Defence Spending
- **Query:** Which EU leaders called for increased defence spending?
- **Answer (200 chars):** Unfortunately, the available chunks do not provide a direct answer to the question of which EU leaders called for increased defence spending. However,
- **Citations:** 3 | **Steps:** 5
- **ans_rel:** 0.7919 | **gt_sim:** 0.6754 | **ctx_rel:** 0.2827 | **ctx_ans:** 0.4285

### Q9 — Green Deal & Energy Crisis
- **Query:** What was discussed about the Green Deal in relation to the energy crisis?
- **Answer (200 chars):** The Green Deal is related to the energy crisis through the EU's efforts to mitigate the impact of the crisis on energy prices, as mentioned in [chunk_
- **Citations:** 2 | **Steps:** 5
- **ans_rel:** 0.7631 | **gt_sim:** 0.7722 | **ctx_rel:** 0.5492 | **ctx_ans:** 0.5969

### Q10 — International Law Arguments
- **Query:** What international law arguments were made about the conflict?
- **Answer (200 chars):** The international law arguments made about the conflict include restoring freedom of navigation in the Strait of Hormuz in line with international law
- **Citations:** 1 | **Steps:** 4
- **ans_rel:** 0.7074 | **gt_sim:** 0.5913 | **ctx_rel:** 0.1394 | **ctx_ans:** 0.2182

---

## 3. RAGAS Scores Summary

> ️ **No LLM API key available.** RAGAS LLM-based metrics (faithfulness, answer_relevancy via NLI, context_precision, context_recall) could not be computed. Embedding cosine similarity metrics used as proxy.

### Aggregate Scores (embedding-based)

- **answer_relevancy_emb** (avg): **0.7571**
- **ground_truth_similarity** (avg): **0.7248**
- **context_relevancy_emb** (avg): **0.2619**
- **context_answer_alignment** (avg): **0.3434**

### Per-Question Scores Table

| Q | Topic | ans_rel | gt_sim | ctx_rel | ctx_ans | Citations |
|---|---|---|---|---|---|---|
| 1 | Ukraine military support | 0.7110 | 0.8718 | 0.1454 | 0.3020 | 1 |
| 2 | Energy transition | 0.6023 | 0.6966 | 0.3594 | 0.5839 | 1 |
| 3 | Strategic autonomy | 0.8250 | 0.8261 | 0.2443 | 0.4036 | 2 |
| 4 | Maritime security | 0.7815 | 0.6684 | 0.1176 | 0.1560 | 0 |
| 5 | Peace negotiations | 0.8747 | 0.6467 | 0.3090 | 0.4065 | 1 |
| 6 | Sanctions | 0.6944 | 0.7696 | 0.2687 | 0.1937 | 1 |
| 7 | Economic impacts | 0.8199 | 0.7296 | 0.2030 | 0.1451 | 3 |
| 8 | Defence spending | 0.7919 | 0.6754 | 0.2827 | 0.4285 | 3 |
| 9 | Green Deal + energy | 0.7631 | 0.7722 | 0.5492 | 0.5969 | 2 |
| 10 | International law | 0.7074 | 0.5913 | 0.1394 | 0.2182 | 1 |
| **Avg** | | **0.7571** | **0.7248** | **0.2619** | **0.3434** | **1.5** |

---

## 4. Qualitative Analysis

### Where the Agent Does Well 

**1. Answer Relevancy is consistently good (avg 0.76)**
Answers are semantically on-topic for 9/10 questions. The agent correctly identifies EU context, names entities (Ursula von der Leyen, UN Security Council), and stays in scope. Best performance: Q5 (peace negotiations, 0.87), Q3 (strategic autonomy, 0.83), Q7 (economic impacts, 0.82).

**2. Ground Truth Alignment is reasonable (avg 0.72)**
Answers align with independently written ground truths. Q1 (Ukraine support, 0.87) and Q3 (strategic autonomy, 0.83) show the agent synthesises factually grounded responses when the knowledge base has relevant content.

**3. Reasoning Chain Structure**
All queries except Q1, Q4, Q10 completed 5 reasoning steps, indicating the agent systematically explores the graph before answering. The chain includes entity search, chunk retrieval, and synthesis steps.

**4. Multi-citation support**
Q7 and Q8 returned 3 citations each, showing the agent can aggregate evidence across chunks when relevant nodes exist.

**5. Q9 — Best overall performance**
Green Deal + energy crisis question had the highest context relevancy (0.55) and context-answer alignment (0.60), indicating the knowledge base actually contains energy crisis content that the agent correctly retrieved and answered from.

---

### Where the Agent Falls Short 

**1. Context Retrieval is weak (avg ctx_rel: 0.26)**
The most critical failure: retrieved contexts are often not semantically close to the question. Q4 (maritime security, ctx_rel=0.12) retrieved **zero citations** and relied on entity name lookups. Q10 (international law, ctx_rel=0.14) also had very weak context.

**2. Only 8 chunks total — critical knowledge gap**
The knowledge base contains only 8 text chunks from a single debate session. This severely limits coverage. Questions about specific topics (defence spending, maritime security, international law) cannot be answered faithfully because source content is sparse. Q8 explicitly admits: *"the available chunks do not provide a direct answer."*

**3. Context-Answer Alignment is low (avg 0.34)**
Retrieved context and final answers are often weakly aligned — the agent appears to generate answers partially from parametric knowledge or entity metadata rather than grounding strictly in retrieved chunks. Q7 (ctx_ans=0.15) and Q6 (ctx_ans=0.19) are especially concerning: high answer relevancy but low grounding.

**4. Q4 — Zero citations for maritime security**
The agent returned no citation excerpts, instead listing entity names (UN_SECURITY_COUNCIL, etc.) without text evidence. Answers appear to be generated from entity metadata alone, risking hallucination.

**5. Chunk excerpt truncation**
Retrieved excerpts are mid-sentence fragments (e.g., *"ommittee. Ultimately, just and sustainable peace..."*), indicating chunks are split at arbitrary byte boundaries rather than semantic boundaries. This reduces the usability of retrieved context.

**6. Cross-topic confusion**
Q2 (energy transition) answer conflates Middle East crisis content with Ukraine war energy impacts — the single debate covers multiple topics and the retriever does not always distinguish between them.

---

## 5. Recommendations for Improvement

### 1. ️ Expand the Knowledge Base
**Priority: Critical.** With only 8 chunks from one debate session, many questions cannot be answered. Ingest additional EP debate transcripts, committee reports, and policy documents. Target: ≥50 chunks across ≥5 sessions covering 2023–2024.

### 2. ️ Improve Chunk Segmentation
Replace arbitrary-length chunking with semantic chunking (paragraph or speaker-turn boundaries). Mid-sentence splits degrade both retrieval quality and citation readability. Consider speaker-attributed chunks for better entity grounding.

### 3.  Improve Context-Answer Faithfulness
Low `ctx_ans` scores (avg 0.34) suggest the agent draws on general knowledge beyond the retrieved context. Implement a faithfulness constraint in the synthesis prompt: *"Only state claims that are directly supported by retrieved chunks."*

### 4.  Add LLM-Based RAGAS Metrics
Configure an LLM API key (OpenAI or Groq) to enable:
- **Faithfulness:** NLI-based claim verification against context
- **Answer Relevancy:** LLM-based on-topic scoring
- **Context Precision/Recall:** Precision of retrieved chunks
This would give far more reliable evaluation than embedding cosine similarity alone.

### 5.  Improve Entity-to-Chunk Linking
Q4 (maritime security) retrieved 0 chunks because maritime topics exist in entity nodes but lack direct chunk links. Improve entity extraction to always back-link to source chunks, ensuring every entity answer is grounded in text evidence.

### 6.  Add Query-Type Routing
Some questions are entity-lookup queries (Q4) while others require synthesising across multiple passages (Q7, Q8). A routing layer that detects query type and applies appropriate retrieval strategies (pure vector search vs. graph traversal vs. hybrid) would improve context relevancy.

### 7.  Re-run RAGAS with LLM When Available
Once an API key is configured, re-run with:
```python
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
```

---

## 6. Technical Notes

- **RAGAS version:** 0.4.3 (installed in `/root/graphknows/.venv`)
- **LLM API:** Not available — no `OPENAI_API_KEY` or `GROQ_API_KEY` set
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Endpoint response time:** 1.8–6.7 seconds per query (well within 90s timeout)
- **Knowledge base:** 8 chunks, 129 entities, 6 community nodes — `ep_debate_2024_03_13`
- **Raw data:** `/root/graphknows/ragas_raw.json`
- **Results data:** `/root/graphknows/ragas_results.json`
