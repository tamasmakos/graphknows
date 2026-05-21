"""Dialog writeback: session turns -> Neo4j Dialog/Utterance/Participant nodes."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from graphknows.document_ingestor import ChunkData, DocumentIngestor

log = logging.getLogger(__name__)


@dataclass
class TurnData:
    """A single conversation turn."""

    turn_index: int
    role: str
    participant_name: str
    text: str
    timestamp: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class DialogResult:
    dialog_id: str
    session_id: str
    utterance_count: int
    mention_count: int
    errors: list[str] = field(default_factory=list)


class DialogWriteBack:
    """Write session dialog turns to Neo4j as Dialog/Utterance/Entity paths."""

    def __init__(
        self,
        neo4j_driver: Any,
        embedding_model: Any | None = None,
        ner_ontology: list[str] | None = None,
    ) -> None:
        self._driver = neo4j_driver
        self._embed = embedding_model
        self._ontology = ner_ontology or [
            "PERSON",
            "ORGANIZATION",
            "LOCATION",
            "CONCEPT",
            "POLICY",
            "LAW",
            "INSTITUTION",
            "TECHNOLOGY",
        ]

    def write(self, session_id: str, turns: list[TurnData]) -> DialogResult:
        """Write dialog turns to Neo4j. Idempotent via MERGE on dialog_id + utterance_id."""
        dialog_id = hashlib.sha256(session_id.encode()).hexdigest()[:12]
        utterance_count = 0
        mention_count = 0
        errors: list[str] = []
        now = datetime.now(timezone.utc).isoformat()

        try:
            _ner = DocumentIngestor(neo4j_driver=None)
        except Exception as exc:
            log.warning("NER backend unavailable — skipping entity linking: %s", exc)
            _ner = None

        with self._driver.session() as s:
            s.run(
                "MERGE (dia:Dialog {id: $dialog_id}) "
                "SET dia.session_id = $session_id, dia.started_at = $started_at",
                dialog_id=dialog_id,
                session_id=session_id,
                started_at=now,
            )
            s.run(
                "MATCH (seg:Segment {session_id: $session_id}) "
                "MATCH (dia:Dialog {id: $dialog_id}) "
                "MERGE (seg)-[:HAS_DIALOG]->(dia)",
                session_id=session_id,
                dialog_id=dialog_id,
            )

            for turn in turns:
                utterance_id = hashlib.sha256(
                    f"{session_id}:{turn.turn_index}:{turn.role}".encode()
                ).hexdigest()[:16]
                participant_id = hashlib.sha256(
                    f"{session_id}:{turn.role}:{turn.participant_name}".encode()
                ).hexdigest()[:12]

                try:
                    s.run(
                        "MATCH (dia:Dialog {id: $dialog_id}) "
                        "MERGE (p:Participant {id: $participant_id}) "
                        "SET p.role = $role, p.name = $participant_name "
                        "MERGE (dia)-[:PARTICIPANT]->(p) "
                        "MERGE (u:Utterance {id: $utterance_id}) "
                        "SET u.text = $text, u.turn_index = $turn_index, "
                        "    u.timestamp = $timestamp, u.embedding = $embedding "
                        "MERGE (p)-[:SAID]->(u)",
                        dialog_id=dialog_id,
                        participant_id=participant_id,
                        role=turn.role,
                        participant_name=turn.participant_name,
                        utterance_id=utterance_id,
                        text=turn.text,
                        turn_index=turn.turn_index,
                        timestamp=turn.timestamp,
                        embedding=turn.embedding or None,
                    )
                    utterance_count += 1
                except Exception as exc:
                    msg = f"utterance {turn.turn_index}/{turn.role}: {exc}"
                    errors.append(msg)
                    log.warning("Dialog writeback skipped turn: %s", msg)
                    continue

                if _ner is None:
                    continue

                try:
                    chunk = ChunkData(
                        id=utterance_id,
                        text=turn.text,
                        char_offset=0,
                        token_count=len(turn.text.split()),
                    )
                    mentions = _ner._extract_entities(chunk, self._ontology)  # noqa: SLF001
                    for mention in mentions:
                        try:
                            s.run(
                                "MATCH (e:Entity {name: $entity_name}) "
                                "MATCH (u:Utterance {id: $utterance_id}) "
                                "MERGE (u)-[:MENTIONS]->(e)",
                                entity_name=mention.name,
                                utterance_id=utterance_id,
                            )
                            mention_count += 1
                        except Exception as exc:
                            msg = f"mention {mention.name}: {exc}"
                            errors.append(msg)
                            log.debug("Skipped mention: %s", msg)
                except Exception as exc:
                    msg = f"NER failed for turn {turn.turn_index}: {exc}"
                    errors.append(msg)
                    log.debug(msg)

        return DialogResult(
            dialog_id=dialog_id,
            session_id=session_id,
            utterance_count=utterance_count,
            mention_count=mention_count,
            errors=errors,
        )
