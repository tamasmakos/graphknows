"""Host-agnostic facade over the graphknows memory backend.

Composes MemoryIngestor, DialogWriter, SessionFlusher, and recall
functions behind a stable generic contract. Host-specific parsing
and normalization stay outside this class.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from graphknows.memory.adapter_types import (
    AdapterCapabilities,
    ArtifactRecord,
    ParticipantRecord,
    PromoteOptions,
    RecallRequest,
    TurnRecord,
)
from graphknows.memory.dialog import DialogWriter, Participant, ToolCall, Turn
from graphknows.memory.flush import FlushOptions, SessionFlusher
from graphknows.memory.ingestor import MemoryIngestor

# recall_short_term is imported lazily inside recall() to avoid a circular
# import: recall_stm → memory.ingestor → memory.__init__ → adapter → recall_stm


@dataclass
class GraphknowsAdapter:
    """Host-agnostic facade over the graphknows memory backend.

    Composes MemoryIngestor, DialogWriter, SessionFlusher, and recall
    functions behind a stable generic contract. Host-specific parsing
    and normalization stay outside this class.

    Args:
        mg_store: Memgraph store instance.
        pg_store: PgVector store instance.
        neo_store: Neo4j store instance.
        extractor: Entity extractor (GLiNER2EntityExtractor or compatible).
        embed_fn: Optional embedding function (text -> list[float]).
    """

    mg_store: Any
    pg_store: Any
    neo_store: Any
    extractor: Any
    embed_fn: Any | None = None

    _active_session_id: str | None = field(default=None, repr=False)
    _active_dialog_id: str | None = field(default=None, repr=False)
    _ingestor: MemoryIngestor | None = field(default=None, repr=False)
    _dialog_writer: DialogWriter | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize composed ingestor and dialog writer."""
        self._ingestor = MemoryIngestor(
            self.mg_store,
            self.pg_store,
            self.extractor,
            self.embed_fn,
        )
        self._dialog_writer = DialogWriter(
            self.mg_store,
            self.pg_store,
            self.extractor,
            self.embed_fn,
        )

    def open_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record that a session has started.

        Calls any setup needed on stores. No-op if stores don't require it.
        Stores ``session_id`` internally as ``self._active_session_id``.

        Args:
            session_id: Unique session identifier.
            metadata: Optional session metadata.
        """
        self._active_session_id = session_id
        if metadata and hasattr(self.mg_store, "write_session_metadata"):
            from graphknows.memory.ingestor import _run

            _run(self.mg_store.write_session_metadata(session_id, metadata))

    def start_dialog(
        self,
        session_id: str,
        dialog_id: str,
        participants: list[ParticipantRecord],
    ) -> None:
        """Initialize a dialog and register participants.

        Args:
            session_id: Parent session namespace.
            dialog_id: Unique dialog identifier.
            participants: List of participants in the dialog.
        """
        self._active_dialog_id = dialog_id
        mapped_participants = [
            Participant(
                id=p.participant_id,
                role=p.role,
                display_name=p.display_name or "",
            )
            for p in participants
        ]
        self._dialog_writer.start_dialog(
            session_id,
            dialog_id,
            mapped_participants,
        )

    def write_turn(self, turn: TurnRecord) -> None:
        """Write a single conversation turn into the interaction graph.

        Args:
            turn: Turn data including text, role, and tool calls.
        """
        dialog_id = self._active_dialog_id
        mapped_tool_calls = [
            ToolCall(
                name=tc.tool_name,
                result_summary=tc.output_summary,
            )
            for tc in turn.tool_calls
        ]
        mapped_turn = Turn(
            id=turn.turn_id,
            index=0,
            role=turn.role,
            text=turn.content,
            timestamp=time.time(),
            tool_calls=mapped_tool_calls,
        )
        self._dialog_writer.write_turn(
            turn.session_id,
            dialog_id,
            mapped_turn,
        )

    def ingest_artifact(
        self,
        artifact: ArtifactRecord,
        session_id: str,
    ) -> None:
        """Ingest an artifact into short-term memory.

        Args:
            artifact: Artifact record to ingest.
            session_id: Session namespace.
        """
        metadata = {
            **artifact.metadata,
            "memory_class": artifact.memory_class.value,
            "artifact_id": artifact.artifact_id,
        }
        self._ingestor.ingest_text(
            artifact.content,
            session_id=session_id,
            title=artifact.path,
            source_hint=artifact.mime_type,
            metadata=metadata,
        )

    def recall(self, request: RecallRequest) -> list[dict[str, Any]]:
        """Recall memories from short-term memory.

        Args:
            request: Recall query parameters.

        Returns:
            List of recall hits as dictionaries.
        """
        from graphknows.retrieval.recall_stm import recall_short_term  # noqa: PLC0415

        hits = recall_short_term(
            self.mg_store,
            self.pg_store,
            query=request.query,
            session_id=request.session_id,
            top_k=request.top_k,
            embed_fn=self.embed_fn,
        )
        hit_dicts = [asdict(hit) for hit in hits]
        if not request.include_raw:
            hit_dicts = [
                hit
                for hit in hit_dicts
                if hit.get("provenance", {}).get("memory_class") != "raw_evidence"
            ]
        return hit_dicts

    async def promote(self, options: PromoteOptions) -> dict[str, Any]:
        """Promote short-term memory to long-term memory.

        Args:
            options: Promotion configuration.

        Returns:
            Summary dictionary with promotion results.
        """
        flusher = SessionFlusher(
            self.mg_store,
            self.pg_store,
            self.neo_store,
        )
        flush_options = FlushOptions(
            summarize_topics=options.run_topics,
            run_gds=options.run_gds,
        )
        result = await flusher.flush(
            options.session_id,
            options=flush_options,
        )
        summary = {
            "session_id": options.session_id,
            "run_gds": options.run_gds,
            "run_topics": options.run_topics,
        }
        if hasattr(result, "nodes_written"):
            summary["nodes_written"] = result.nodes_written
        if hasattr(result, "edges_written"):
            summary["edges_written"] = result.edges_written
        if hasattr(result, "chunks_promoted"):
            summary["chunks_promoted"] = result.chunks_promoted
        if hasattr(result, "turns_promoted"):
            summary["turns_promoted"] = result.turns_promoted
        if hasattr(result, "topics_created"):
            summary["topics_created"] = result.topics_created
        if hasattr(result, "merged_entities"):
            summary["merged_entities"] = result.merged_entities
        if hasattr(result, "elapsed_s"):
            summary["elapsed_s"] = result.elapsed_s
        if hasattr(result, "errors"):
            summary["errors"] = result.errors
        return summary

    def capabilities(self) -> AdapterCapabilities:
        """Return the adapter capability flags.

        Returns:
            AdapterCapabilities indicating supported features.
        """
        return AdapterCapabilities(
            supports_stm=True,
            supports_ltm=True,
            supports_dialog_graph=True,
            supports_artifact_ingest=True,
            supports_recall=True,
            supports_promote=True,
        )
