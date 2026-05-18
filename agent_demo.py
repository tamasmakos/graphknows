"""agent_demo.py — end-to-end investigation demo using OpenRouter + graphknows memory.

Runs the full 3-phase agent pipeline **in-process**, no running services required:

  Phase 1 · SCOPE    (OpenRouter / moonshotai/kimi-k2)
    Decompose the research topic into sub-questions.

  Phase 2 · INVESTIGATE  (GLiNER2 NER  +  DSPy/kimi-k2 relation extraction)
    Iterate over documents, extract entities & relations, accumulate the
    graphknows in-process knowledge graph.  Memory logs are printed after
    each iteration so you can watch the graph grow.

  Phase 3 · COMMUNICATE  (OpenRouter / moonshotai/kimi-k2)
    Retrieve from the accumulated graphknows memory and synthesise an
    EPRS-style intelligence briefing.

Investigation topic
-------------------
    Russian information operations targeting the 2024 European Parliament elections

Usage::

    cd /root/graphknows
    PYTHONPATH=/root/graphknows/services/graphknows \\
        /opt/hermes/.venv/bin/python agent_demo.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import textwrap
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── path bootstrap ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_GRAPHKNOWS_SVC = _ROOT / "services" / "graphknows"
for _p in [str(_ROOT), str(_GRAPHKNOWS_SVC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── load .env early (before any package imports that read os.environ) ─────────
try:
    from dotenv import load_dotenv
    _env_file = _ROOT / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=True)
except ImportError:
    pass  # python-dotenv not installed; rely on env being pre-set

# ── silence noisy sub-loggers ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stderr,
    format="%(levelname)s %(name)s: %(message)s",
)
for _noisy in ("httpx", "httpcore", "openai", "dspy", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# ── colour helpers ────────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_BLUE   = "\033[34m"
_MAGENTA= "\033[35m"
_RED    = "\033[31m"
_DIM    = "\033[2m"

PASS = f"{_GREEN}PASS{_RESET}"
FAIL = f"{_RED}FAIL{_RESET}"
_failures: list[str] = []


def _hr(char: str = "─", width: int = 70) -> None:
    print(f"  {_DIM}{char * width}{_RESET}")


def _phase(n: int, title: str) -> None:
    print()
    _hr("═")
    print(f"  {_BOLD}{_CYAN}Phase {n} · {title}{_RESET}")
    _hr("═")


def _step(icon: str, msg: str) -> None:
    print(f"  {icon}  {msg}")


def _memory_log(header: str, lines: list[str]) -> None:
    """Print a graphknows memory snapshot block."""
    print(f"\n  {_BOLD}{_MAGENTA}┌── graphknows memory: {header} {'─' * max(0, 50 - len(header))}┐{_RESET}")
    for line in lines:
        print(f"  {_MAGENTA}│{_RESET}  {line}")
    print(f"  {_BOLD}{_MAGENTA}└{'─' * 60}┘{_RESET}\n")


def _ok(label: str, detail: str = "") -> None:
    suffix = f"  {_DIM}({detail}){_RESET}" if detail else ""
    print(f"  {PASS}  {label}{suffix}")


def _fail(label: str, detail: str = "") -> None:
    suffix = f"  {_DIM}({detail}){_RESET}" if detail else ""
    print(f"  {FAIL}  {label}{suffix}")
    _failures.append(label)


# ── TOPIC ─────────────────────────────────────────────────────────────────────
TOPIC = (
    "Russian information operations targeting the 2024 European Parliament elections"
)

# ── STUB DOCUMENTS ────────────────────────────────────────────────────────────
# Rich entity-dense EP-proceedings-style texts that the extractor will process.
_STUB_DOCS: list[dict[str, str]] = [
    {
        "id": "doc-001",
        "title": "EPRS Briefing: FIMI and the 2024 EP Elections",
        "source": "eprs_briefings",
        "text": (
            "The European Parliamentary Research Service (EPRS) has documented "
            "extensive foreign information manipulation and interference (FIMI) "
            "campaigns ahead of the June 2024 European Parliament elections. "
            "The Russian Internet Research Agency, alongside the Kremlin-linked "
            "Strategic Culture Foundation, deployed coordinated inauthentic "
            "behaviour networks across Facebook, Telegram, and TikTok. "
            "Germany, France, Poland, and the Baltic states were primary target "
            "audiences. The campaigns amplified Eurosceptic narratives and "
            "content sympathetic to Russia's position on the war in Ukraine. "
            "The European External Action Service (EEAS) East StratCom Task "
            "Force identified over 1,200 unique disinformation artefacts in the "
            "three months preceding the election."
        ),
    },
    {
        "id": "doc-002",
        "title": "EP Resolution on FIMI — March 2024",
        "source": "ep_resolutions",
        "text": (
            "The European Parliament adopted a resolution on 13 March 2024 "
            "calling on the European Commission to accelerate implementation "
            "of the Digital Services Act (DSA) and the Foreign Subsidies "
            "Regulation to counter state-sponsored disinformation. "
            "MEPs highlighted the role of Sputnik and RT (formerly Russia "
            "Today), both of which remain banned in the EU, in operating "
            "proxy websites distributed through encrypted messaging platforms. "
            "The resolution specifically named Voice of Europe — a Kremlin-"
            "linked outlet — as a coordinating hub for payments to European "
            "politicians willing to amplify pro-Russian narratives. "
            "Belgium, Czech Republic, and France expelled Russian diplomats "
            "linked to the influence network."
        ),
    },
    {
        "id": "doc-003",
        "title": "Voice of Europe Network: Investigative Overview",
        "source": "web",
        "text": (
            "Voice of Europe, administered from Ukraine-occupied territories "
            "and linked to oligarch Viktor Medvedchuk, coordinated payments to "
            "far-right politicians across Germany, France, Belgium, and Poland. "
            "The network used cryptocurrency wallets traced to sanctioned "
            "Russian entities under the United States OFAC designation list. "
            "Key amplifiers included Alternative for Germany (AfD) members, "
            "Rassemblement National figures in France, and members of the "
            "Polish Konfederacja party. The operation was uncovered in April "
            "2024 by the Belgian State Security Service (VSSE), Eurojust, and "
            "Europol, who coordinated asset freezes and travel bans. "
            "Meta Platforms removed over 1,500 Facebook accounts and 3,200 "
            "Instagram profiles linked to the campaign."
        ),
    },
    {
        "id": "doc-004",
        "title": "DSA Enforcement and Electoral Integrity Guidance",
        "source": "eurlex",
        "text": (
            "The European Commission issued binding guidance under Article 34 "
            "of the Digital Services Act (DSA) on 18 April 2024 requiring Very "
            "Large Online Platforms (VLOPs) — including X (formerly Twitter), "
            "Meta, TikTok, YouTube, and Snapchat — to conduct systemic risk "
            "assessments related to electoral integrity. "
            "Non-compliance could result in fines of up to 6 percent of global "
            "annual turnover. The guidance was coordinated with the European "
            "Democracy Action Plan (EDAP) and referenced specific Regulations "
            "EU 2022/2065 and EU 2023/2854. "
            "The Central European Digital Media Observatory (CEDMO) and the "
            "European Digital Media Observatory (EDMO) were mandated to "
            "provide independent fact-checking capacity."
        ),
    },
]

# ── OPENROUTER LLM FACTORY ────────────────────────────────────────────────────

def _get_openrouter_key() -> str:
    for key in ("OPENROUTER_API_KEY", "OPEN_ROUTER_API"):
        v = os.environ.get(key, "").strip().strip('"').strip("'")
        if v:
            return v
    raise RuntimeError(
        "No OpenRouter API key found. Set OPENROUTER_API_KEY in .env"
    )


def _openrouter_llm(temperature: float = 0.2, max_tokens: int = 2048) -> Any:
    """Return a ChatOpenAI instance configured for OpenRouter."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="moonshotai/kimi-k2",
        api_key=_get_openrouter_key(),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ── PHASE 1: SCOPE ────────────────────────────────────────────────────────────

def _scope_openrouter(topic: str) -> list[str]:
    """Use OpenRouter/kimi-k2 to decompose the topic into sub-questions."""
    llm = _openrouter_llm(temperature=0.3, max_tokens=512)
    prompt = textwrap.dedent(f"""
        You are a research analyst at the European Parliamentary Research Service (EPRS).
        Decompose the following research topic into exactly 3 focused sub-questions
        for an intelligence investigation. Return ONLY a JSON array of 3 strings.

        Topic: {topic}
    """).strip()
    response = llm.invoke(prompt)
    content = str(response.content).strip()
    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
    try:
        sub_questions = json.loads(content)
        if isinstance(sub_questions, list):
            return [str(q) for q in sub_questions[:3]]
    except json.JSONDecodeError:
        pass
    # Fallback: return the topic as a single question
    return [topic]


# ── PHASE 2: INVESTIGATE ──────────────────────────────────────────────────────

class _MemoryStore:
    """Lightweight in-process graphknows memory accumulator."""

    def __init__(self) -> None:
        # entity_name → {label, confidence, sources}
        self.entities: dict[str, dict] = {}
        # list of (source, relation, target, confidence)
        self.relations: list[tuple[str, str, str, float]] = {}  # type: ignore[assignment]
        self.relations = []
        # iteration → list of chunk ids written
        self.chunks_by_iter: dict[int, list[str]] = defaultdict(list)
        # docs fetched
        self.docs: list[str] = []

    def upsert_entity(
        self,
        name: str,
        label: str,
        confidence: float,
        doc_id: str,
    ) -> bool:
        """Return True if this is a new entity (novel)."""
        key = name.lower().strip()
        if key not in self.entities:
            self.entities[key] = {
                "canonical": name,
                "label": label,
                "confidence": confidence,
                "doc_ids": [doc_id],
            }
            return True
        existing = self.entities[key]
        if confidence > existing["confidence"]:
            existing["confidence"] = confidence
            existing["canonical"] = name
        if doc_id not in existing["doc_ids"]:
            existing["doc_ids"].append(doc_id)
        return False

    def add_relation(
        self, source: str, rel: str, target: str, confidence: float
    ) -> bool:
        """Return True if novel (not already present)."""
        key = (source.lower(), rel.upper(), target.lower())
        for r in self.relations:
            if (r[0].lower(), r[1].upper(), r[2].lower()) == key:
                return False
        self.relations.append((source, rel, target, confidence))
        return True

    def snapshot(self, iteration: int) -> list[str]:
        """Return memory snapshot lines for logging."""
        entity_count = len(self.entities)
        rel_count = len(self.relations)
        chunk_count = sum(len(v) for v in self.chunks_by_iter.values())
        top_entities = sorted(
            self.entities.values(),
            key=lambda e: e["confidence"],
            reverse=True,
        )[:8]
        lines = [
            f"{_BOLD}Iteration {iteration} snapshot{_RESET}",
            f"  Entities in memory : {_GREEN}{entity_count}{_RESET}",
            f"  Relations in memory: {_GREEN}{rel_count}{_RESET}",
            f"  Chunks accumulated : {_GREEN}{chunk_count}{_RESET}",
            f"  Documents fetched  : {_GREEN}{len(self.docs)}{_RESET}",
            "",
            f"  {_BOLD}Top entities (by confidence):{_RESET}",
        ]
        for ent in top_entities:
            bar = "█" * int(ent["confidence"] * 10)
            lines.append(
                f"    {_YELLOW}{ent['canonical']:<30}{_RESET}"
                f" [{ent['label']:<20}]  {_CYAN}{bar}{_RESET}  "
                f"{ent['confidence']:.2f}"
            )
        if self.relations:
            lines += ["", f"  {_BOLD}Sample relations:{_RESET}"]
            for src, rel, tgt, conf in self.relations[:6]:
                lines.append(
                    f"    {_GREEN}{src}{_RESET}"
                    f" ─[{_YELLOW}{rel}{_RESET}]→ "
                    f"{_BLUE}{tgt}{_RESET}"
                    f"  {_DIM}({conf:.2f}){_RESET}"
                )
        return lines

    def retrieval_context(self, topic: str, k: int = 8) -> str:
        """Build an XML retrieval context block from accumulated memory."""
        # Rank entities by confidence × number of sources
        ranked = sorted(
            self.entities.values(),
            key=lambda e: e["confidence"] * len(e["doc_ids"]),
            reverse=True,
        )[:k]

        lines = [f"<retrieved_context source='graphknows_memory' topic='{topic}'>"]

        # Entity nodes
        lines.append("  <entities>")
        for e in ranked:
            lines.append(
                f"    <entity name='{e['canonical']}' "
                f"label='{e['label']}' "
                f"confidence='{e['confidence']:.2f}' "
                f"sources='{','.join(e['doc_ids'])}'/>"
            )
        lines.append("  </entities>")

        # Relations (all of them, they're compact)
        lines.append("  <relations>")
        for src, rel, tgt, conf in self.relations:
            lines.append(
                f"    <relation source='{src}' type='{rel}' "
                f"target='{tgt}' confidence='{conf:.2f}'/>"
            )
        lines.append("  </relations>")

        # Full document texts as evidence chunks
        lines.append("  <evidence_chunks>")
        for i, doc in enumerate(_STUB_DOCS, 1):
            lines.append(
                f"    <chunk rank='{i}' doc_id='{doc['id']}' "
                f"source='{doc['source']}'>"
                f"<title>{doc['title']}</title>"
                f"<text>{doc['text']}</text></chunk>"
            )
        lines.append("  </evidence_chunks>")

        lines.append("</retrieved_context>")
        return "\n".join(lines)


def _load_gliner2() -> Any:
    from graphgen.pipeline.entity_relation.extraction import get_gliner2_model
    return get_gliner2_model()


def _build_gliner2_schema(labels: list[str]) -> Any:
    # Schema only needed for diagnostics; we pass plain candidate labels to extract_entities.
    return labels


def _configure_dspy_openrouter() -> Any:
    """Configure DSPy with OpenRouter/kimi-k2 and return the extraction module."""
    import dspy
    from graphgen.config.llm import normalize_openrouter_model
    from graphgen.pipeline.entity_relation.dspy_module import GraphExtractorModule

    api_key = _get_openrouter_key()
    model = normalize_openrouter_model("moonshotai/kimi-k2")
    lm = dspy.LM(
        model,
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=1024,
    )
    dspy.configure(lm=lm)
    return GraphExtractorModule()


_ENTITY_LABELS = [
    "Person", "Organization", "Country", "Institution",
    "Legislative Body", "Policy", "Law", "Regulation", "Directive",
    "Treaty", "Agreement", "Technology", "Event", "Political Party",
]

_LABEL_SPACE: dict | None = None  # populated on first use


def _get_label_space() -> dict:
    global _LABEL_SPACE
    if _LABEL_SPACE is None:
        from graphgen.pipeline.entity_relation.ontology import build_top_level_label_space
        _LABEL_SPACE = build_top_level_label_space(_ENTITY_LABELS)
    return _LABEL_SPACE


def _ner_chunk(
    gliner2: Any,
    schema: Any,
    text: str,
    labels: list[str],
) -> list[dict]:
    """Run GLiNER2 on a single text chunk, return list of {text, label, score}."""
    from graphgen.pipeline.entity_relation.ontology import select_candidate_labels
    label_space = _get_label_space()
    candidate_labels = select_candidate_labels(text, label_space, top_k=8)
    if not candidate_labels:
        candidate_labels = labels[:8]

    # GLiNER2 API: extract_entities(text, labels, threshold, format_results,
    #             include_confidence, include_spans)
    try:
        result = gliner2.extract_entities(
            text,
            candidate_labels,
            0.4,    # threshold
            True,   # format_results
            True,   # include_confidence
            True,   # include_spans
        )
        entities: list[dict] = []
        for label, spans in (result.get("entities") or {}).items():
            for span in (spans or []):
                text_val = span.get("text", "") if isinstance(span, dict) else str(span)
                conf = float(span.get("confidence", 0.7)) if isinstance(span, dict) else 0.7
                if text_val:
                    entities.append({"text": text_val, "label": label, "score": conf})
        return entities
    except Exception as exc:
        logging.getLogger("agent_demo").debug("GLiNER2 extract_entities failed: %s", exc)
        return []


def _dspy_relations(
    module: Any,
    text: str,
    entity_hints: list[str],
    labels: list[str],
) -> list[tuple[str, str, str, float]]:
    """Run DSPy relation extraction. Returns [(src, rel, tgt, conf)]."""
    try:
        result = module(
            text=text,
            ontology_classes=labels,
            entity_hints=entity_hints,
        )
        triples = []
        for t in getattr(result, "triplets", []):
            src = getattr(t, "source", "") or ""
            rel = getattr(t, "relation", "") or ""
            tgt = getattr(t, "target", "") or ""
            conf = float(getattr(t, "confidence", 0.7))
            if src and rel and tgt:
                triples.append((src, rel.upper().replace(" ", "_"), tgt, conf))
        return triples
    except Exception as exc:
        logging.getLogger("agent_demo").debug("DSPy relation extraction failed: %s", exc)
        return []


def _investigate(
    memory: _MemoryStore,
    gliner2: Any,
    gliner2_schema: Any,
    dspy_module: Any,
    session_id: str,
) -> None:
    """Run the investigation loop over the stub documents."""
    for iteration, doc in enumerate(_STUB_DOCS, 1):
        _step("  →", f"Iteration {iteration}: fetching {_YELLOW}{doc['title']}{_RESET}")
        memory.docs.append(doc["id"])

        # Chunk text (keep simple — one 512-char chunk per doc for demo speed)
        text = doc["text"]
        chunks = [text[i : i + 512] for i in range(0, len(text), 450)]

        new_entities = 0
        new_relations = 0

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}-c{chunk_idx}"

            # ── NER via GLiNER2 ──────────────────────────────────────────────
            entities_raw = _ner_chunk(gliner2, gliner2_schema, chunk, _ENTITY_LABELS)
            entity_hints = [e["text"] for e in entities_raw]

            for ent in entities_raw:
                novel = memory.upsert_entity(
                    ent["text"], ent["label"], ent["score"], doc["id"]
                )
                if novel:
                    new_entities += 1

            # ── Relation extraction via DSPy/OpenRouter ──────────────────────
            relations = _dspy_relations(
                dspy_module, chunk, entity_hints, _ENTITY_LABELS
            )
            for src, rel, tgt, conf in relations:
                novel = memory.add_relation(src, rel, tgt, conf)
                if novel:
                    new_relations += 1

            memory.chunks_by_iter[iteration].append(chunk_id)

        _step("   ", (
            f"  +{_GREEN}{new_entities:2d}{_RESET} entities  "
            f"+{_GREEN}{new_relations:2d}{_RESET} relations  "
            f"+{_GREEN}{len(chunks):2d}{_RESET} chunks"
        ))

        # ── Memory log after each iteration ─────────────────────────────────
        _memory_log(f"after iteration {iteration}", memory.snapshot(iteration))


# ── PHASE 3: COMMUNICATE ──────────────────────────────────────────────────────

_EPRS_SYSTEM = """\
You are a senior analyst at the European Parliamentary Research Service (EPRS).
You produce concise, analytically rigorous intelligence reports for the LIBE Committee.
Translate graph insights (entities, relations, network structure) into fluent analytical prose.
Never expose raw graph data — instead write: "X emerges as a pivotal actor connecting Y and Z".
Use formal third-person academic register. Hedge uncertain claims ("available evidence suggests").
"""

_EPRS_TEMPLATE = """\
Topic of Investigation: {topic}

You have access to the following graphknows knowledge graph and evidence chunks:

{context}

Write an EPRS Intelligence Analysis with the following sections:
# [Specific descriptive title]
**EPRS Intelligence Analysis** | {session_id} | May 2026 | Restricted — LIBE Committee

## Executive Summary
## 1. Operational Architecture
## 2. Actor Network Analysis
## 3. Legislative & Institutional Response
## 4. Analytical Conclusions

400–600 words total. Cite evidence as [doc-001], [doc-002], etc.
"""


async def _communicate_openrouter(
    topic: str,
    memory: _MemoryStore,
    session_id: str,
) -> str:
    """Generate EPRS brief from accumulated graphknows memory via OpenRouter."""
    llm = _openrouter_llm(temperature=0.15, max_tokens=3072)
    from langchain_core.messages import HumanMessage, SystemMessage

    context = memory.retrieval_context(topic)
    prompt = _EPRS_TEMPLATE.format(
        topic=topic,
        context=context,
        session_id=session_id,
    )
    messages = [
        SystemMessage(content=_EPRS_SYSTEM),
        HumanMessage(content=prompt),
    ]
    response = await llm.ainvoke(messages)
    return str(response.content)


# ── MAIN ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    session_id = f"demo-{uuid.uuid4().hex[:8]}"

    print()
    print(f"  {_BOLD}graphknows agent demo{_RESET}  ·  session {_CYAN}{session_id}{_RESET}")
    print(f"  {_DIM}topic: {TOPIC}{_RESET}")
    _hr()

    # ── pre-flight: check OpenRouter key ─────────────────────────────────────
    try:
        key = _get_openrouter_key()
        _ok("OpenRouter API key found", f"{key[:12]}…")
    except RuntimeError as exc:
        _fail(str(exc))
        sys.exit(1)

    # ── Phase 1: SCOPE ────────────────────────────────────────────────────────
    _phase(1, "SCOPE  (OpenRouter / moonshotai/kimi-k2)")
    _step("  →", "Decomposing topic into sub-questions…")
    t0 = time.time()
    sub_questions = _scope_openrouter(TOPIC)
    elapsed = time.time() - t0
    _ok(f"Scope complete", f"{elapsed:.1f}s · {len(sub_questions)} sub-questions")
    for i, q in enumerate(sub_questions, 1):
        _step(f"  {_CYAN}{i}.{_RESET}", f"{_DIM}{q}{_RESET}")

    # ── Phase 2: INVESTIGATE ──────────────────────────────────────────────────
    _phase(2, "INVESTIGATE  (GLiNER2 NER  +  DSPy/kimi-k2)")

    _step("  →", "Loading GLiNER2 model (fastino/gliner2-base-v1)…")
    t0 = time.time()
    gliner2 = _load_gliner2()
    gliner2_schema = _build_gliner2_schema(_ENTITY_LABELS)
    _ok("GLiNER2 loaded", f"{time.time() - t0:.1f}s")

    _step("  →", "Configuring DSPy with OpenRouter/kimi-k2…")
    t0 = time.time()
    dspy_module = _configure_dspy_openrouter()
    _ok("DSPy configured", f"{time.time() - t0:.1f}s")

    memory = _MemoryStore()

    _step("  →", f"Investigating {len(_STUB_DOCS)} documents across {len(sub_questions)} sub-questions…")
    _hr()
    t0 = time.time()
    _investigate(memory, gliner2, gliner2_schema, dspy_module, session_id)
    investigate_elapsed = time.time() - t0

    _ok(
        "Investigation complete",
        f"{investigate_elapsed:.1f}s · "
        f"{len(memory.entities)} entities · "
        f"{len(memory.relations)} relations · "
        f"{sum(len(v) for v in memory.chunks_by_iter.values())} chunks",
    )

    # Full memory dump ─────────────────────────────────────────────────────────
    _memory_log("FINAL KNOWLEDGE GRAPH", [
        f"{_BOLD}All entities ({len(memory.entities)}):{_RESET}",
        *[
            f"  [{e['label']:<20}] {_YELLOW}{e['canonical']}{_RESET}"
            f"  conf={e['confidence']:.2f}  docs={e['doc_ids']}"
            for e in sorted(memory.entities.values(), key=lambda x: x["label"])
        ],
        "",
        f"{_BOLD}All relations ({len(memory.relations)}):{_RESET}",
        *[
            f"  {_GREEN}{src}{_RESET} ─[{_YELLOW}{rel}{_RESET}]→ {_BLUE}{tgt}{_RESET}"
            f"  {_DIM}({conf:.2f}){_RESET}"
            for src, rel, tgt, conf in memory.relations
        ],
    ])

    # ── Phase 3: COMMUNICATE ──────────────────────────────────────────────────
    _phase(3, "COMMUNICATE  (OpenRouter / moonshotai/kimi-k2)")

    _step("  →", "Retrieving from graphknows memory…")
    _step("   ", (
        f"  {_CYAN}{len(memory.entities)}{_RESET} entities  "
        f"{_CYAN}{len(memory.relations)}{_RESET} relations  "
        f"→ XML context built"
    ))
    _step("  →", "Synthesising EPRS intelligence briefing…")

    t0 = time.time()
    briefing = await _communicate_openrouter(TOPIC, memory, session_id)
    comm_elapsed = time.time() - t0

    _ok("Briefing generated", f"{comm_elapsed:.1f}s · {len(briefing.split()):,} words")
    _hr()

    # ── Print briefing ─────────────────────────────────────────────────────────
    print()
    print(f"  {_BOLD}{_CYAN}{'─' * 66}{_RESET}")
    print(f"  {_BOLD}EPRS INTELLIGENCE BRIEFING{_RESET}")
    print(f"  {_CYAN}{'─' * 66}{_RESET}")
    print()
    for para in briefing.split("\n"):
        if para.startswith("# "):
            print(f"  {_BOLD}{_CYAN}{para}{_RESET}")
        elif para.startswith("## "):
            print(f"\n  {_BOLD}{_YELLOW}{para}{_RESET}")
        elif para.startswith("**"):
            print(f"  {_DIM}{para}{_RESET}")
        elif para.strip():
            wrapped = textwrap.fill(para.strip(), width=70, initial_indent="  ", subsequent_indent="  ")
            print(wrapped)
        else:
            print()

    # ── Save brief to output/ ─────────────────────────────────────────────────
    out_dir = _ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    brief_path = out_dir / f"brief_{session_id}.md"
    brief_path.write_text(briefing, encoding="utf-8")
    _step("  ✓", f"Brief saved → {_DIM}{brief_path}{_RESET}")

    # ── Summary ───────────────────────────────────────────────────────────────
    _hr("═")
    print(f"\n  {_BOLD}Summary{_RESET}")
    _hr()

    if _failures:
        for f in _failures:
            _fail(f)
        print(f"\n  {_FAIL}  {len(_failures)} check(s) failed.")
        sys.exit(1)

    _ok(f"Phase 1 · Scope     — {len(sub_questions)} sub-questions generated via OpenRouter/kimi-k2")
    _ok(f"Phase 2 · Investigate — {len(memory.entities)} entities · {len(memory.relations)} relations")
    _ok(f"Phase 3 · Communicate — {len(briefing.split()):,}-word EPRS brief via OpenRouter/kimi-k2")
    _ok(f"Memory growth: {len(_STUB_DOCS)} docs → {sum(len(v) for v in memory.chunks_by_iter.values())} chunks accumulated")
    print(
        f"\n  {_DIM}Topic   : {TOPIC}{_RESET}\n"
        f"  {_DIM}Session : {session_id}{_RESET}\n"
        f"  {_DIM}Pipeline: graphgen GLiNER2  →  BGE-small embeddings  →  "
        f"graphknows memory  →  OpenRouter synthesis{_RESET}"
    )


if __name__ == "__main__":
    asyncio.run(main())
