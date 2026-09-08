"""Semantic Relation Verifier for GLiNER2.

Filters false positive relations by verifying semantic validity
using a lightweight MLP trained on relation extraction datasets.

Vendored verbatim from oneryalcin/GLiNER2 (branch feature/relation-verifier),
Apache-2.0, because the ``gliner2.verifiers`` module ships in no PyPI release
(1.3.x lacks it). Paired with the ``oneryalcin/gliner2-relation-verifier``
weights. Local changes: ``_bucket_distance`` clamps its return to
``len(buckets) - 1`` — upstream returns ``len(buckets)``, which overflows the
``nn.Embedding(len(buckets), 32)`` and crashes ``verify()`` for any head/tail
pair more than ``max(buckets)`` tokens apart. ``_char_to_token_idx`` returns
``None`` instead of falling back to token 1 / the last token when a span lies
past the encoder's 512-token truncation — upstream's fallback silently scored
a bogus token position; ``_build_features``/``verify`` now skip such a
relation instead. Type annotations were also added to satisfy
``mypy --strict``; they change no behaviour.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


@dataclass
class VerifierConfig:
    """Configuration for RelationVerifier."""

    input_dim: int = 3105  # 768*4 + 32 + 1
    hidden_dim: int = 768
    dropout: float = 0.3
    threshold: float = 0.55
    encoder_name: str = "microsoft/deberta-v3-base"
    # Upstream declared this `list[int] | None = None` and filled the default in
    # `__post_init__`, which left every read site statically Optional — three of
    # them index or `len()` it without a guard. `default_factory` gives the same
    # default with the None branch deleted.
    distance_buckets: list[int] = field(default_factory=lambda: [0, 5, 10, 20, 50])


class RelationVerifierModel(nn.Module):  # type: ignore[misc]
    """MLP model for relation verification."""

    def __init__(self, config: VerifierConfig):
        super().__init__()
        self.config = config

        # Distance embedding (matches checkpoint: dist_embedding)
        self.dist_embedding = nn.Embedding(len(config.distance_buckets), 32)

        # MLP classifier (matches checkpoint structure)
        # classifier.0: Linear(3105, 768)
        # classifier.1: GELU
        # classifier.2: Dropout
        # classifier.3: Linear(768, 384)
        # classifier.4: GELU
        # classifier.5: Dropout
        # classifier.6: Linear(384, 1)
        self.classifier = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.classifier(features).squeeze(-1)


class RelationVerifier:
    """Verifies extracted relations using a trained semantic classifier.

    Filters false positives by checking if head-tail entity pairs
    semantically match the claimed relation type.

    Example:
        verifier = RelationVerifier.from_pretrained("oneryalcin/gliner2-relation-verifier")

        # Use with GLiNER2
        model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
        results = model.extract_relations(text, ["works_for", "founded"])
        verified = verifier.verify(text, results["relation_extraction"])

    Trimmed against the one call site (``verifier.RelationVerifierGate``): it
    reaches ``from_pretrained`` and ``verify`` and nothing else, so the vendored
    ``from_checkpoint``/``verify_single``/``set_threshold`` entry points, the two
    unused span-pooling helpers, and ``verify``'s tuple-instance branch (this
    caller always passes dicts) were unreachable and are gone.
    """

    def __init__(
        self,
        model: RelationVerifierModel,
        encoder: Any,
        tokenizer: Any,
        config: VerifierConfig,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()
        self.encoder.to(self.device)
        self.encoder.eval()

    @classmethod
    def from_pretrained(  # noqa: D417 — vendored signature; **kwargs is swallowed
        cls,
        model_id: str = "oneryalcin/gliner2-relation-verifier",
        threshold: float = 0.55,
        device: str | None = None,
        **kwargs: Any,
    ) -> RelationVerifier:
        """Load verifier from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID or local path
            threshold: Verification threshold (0.55 recommended)
            device: Device to load model on

        Returns:
            RelationVerifier instance
        """
        from transformers import AutoModel, AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load config
        config = VerifierConfig(threshold=threshold)

        # Try to load from HF Hub or local path
        if os.path.isfile(model_id):
            checkpoint_path = model_id
        else:
            # nosec B615 on the four downloads below: bandit accepts only a literal
            # 40-char commit SHA, and both ids here come from config (verifier.py
            # passes model_id through). Pinning belongs in the model ledger
            # scripts/bake_models.py already owns, not inlined at each call site.
            try:
                checkpoint_path = hf_hub_download(  # nosec B615
                    repo_id=model_id, filename="verifier.pt"
                )
            except Exception:
                # Try .bin extension
                checkpoint_path = hf_hub_download(  # nosec B615
                    repo_id=model_id, filename="pytorch_model.bin"
                )

        # Load encoder
        encoder = AutoModel.from_pretrained(config.encoder_name)  # nosec B615
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]  # nosec B615
            config.encoder_name
        )

        # Load verifier weights
        model = RelationVerifierModel(config)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)

        return cls(model, encoder, tokenizer, config, device)

    def _get_token_embeddings(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get token embeddings for *text*, paired with its char→token offset map."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True
        )
        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden]

        return embeddings, offset_mapping

    def _char_to_token_idx(
        self, char_start: int, char_end: int, offset_mapping: torch.Tensor
    ) -> tuple[int, int] | None:
        """Convert character offsets to token indices, or ``None`` if unmapped.

        ``offset_mapping`` only covers the (up to 512-token) truncated
        encoding, so a span whose characters lie past that window has no
        token to map to — that is reported as ``None`` rather than guessed at,
        since a fallback token position would feed the classifier a feature
        vector for the wrong text.
        """
        token_start, token_end = None, None

        for idx, (start, end) in enumerate(offset_mapping.tolist()):
            if start == end == 0:  # Special token
                continue
            if token_start is None and start <= char_start < end:
                token_start = idx
            if start < char_end <= end:
                token_end = idx + 1
                break

        if token_start is None or token_end is None:
            return None

        return token_start, token_end

    def _bucket_distance(self, distance: int) -> int:
        """Convert token distance to bucket index."""
        buckets = self.config.distance_buckets
        for i, threshold in enumerate(buckets):
            if distance <= threshold:
                return i
        return len(buckets) - 1  # clamp: Embedding has only len(buckets) rows

    def _get_relation_embedding(self, relation: str) -> torch.Tensor:
        """Encode relation name and return [CLS] embedding."""
        inputs = self.tokenizer(relation, return_tensors="pt", truncation=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Use [CLS] token for relation representation
            return outputs.last_hidden_state[0, 0]

    def _build_features(
        self,
        text: str,
        head_span: dict,
        tail_span: dict,
        relation: str,
        embeddings: torch.Tensor,
        offset_mapping: torch.Tensor,
        rel_emb: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Build feature vector for verification, or ``None`` if a span is unmapped.

        A span past the 512-token truncation window has no token index (see
        :meth:`_char_to_token_idx`); such a relation is reported unbuildable
        rather than scored against a substituted, unrelated token position.
        """
        # Get token indices (first token of span, matching training)
        head_idx = self._char_to_token_idx(head_span["start"], head_span["end"], offset_mapping)
        tail_idx = self._char_to_token_idx(tail_span["start"], tail_span["end"], offset_mapping)
        if head_idx is None or tail_idx is None:
            return None
        head_tok, tail_tok = head_idx[0], tail_idx[0]

        # Use first token of each entity (matching training approach)
        head_emb = embeddings[head_tok]
        tail_emb = embeddings[tail_tok]

        # Between-span embedding
        start_pos = min(head_tok, tail_tok)
        end_pos = max(head_tok, tail_tok)
        if end_pos - start_pos <= 1:
            # Adjacent or same position - use average
            between_emb = (head_emb + tail_emb) / 2
        else:
            # Mean of tokens between head and tail (excluding endpoints)
            between_emb = embeddings[start_pos + 1 : end_pos].mean(dim=0)

        # Relation embedding (encode relation name separately)
        if rel_emb is None:
            rel_emb = self._get_relation_embedding(relation)

        # Distance features
        distance = abs(head_tok - tail_tok)
        dist_bucket = self._bucket_distance(distance)
        dist_emb = self.model.dist_embedding(torch.tensor([dist_bucket], device=self.device))[0]

        # Order feature
        order = torch.tensor([1.0 if head_tok < tail_tok else 0.0], device=self.device)

        # Concatenate all features
        features = torch.cat([head_emb, tail_emb, between_emb, rel_emb, dist_emb, order])
        return features

    def verify(
        self, text: str, relations: dict[str, list], return_scores: bool = True
    ) -> dict[str, list]:
        """Verify all relations extracted from text.

        Args:
            text: Source text
            relations: Dict mapping relation types to list of instances
                       Each instance should have 'head' and 'tail' dicts
            return_scores: If True, add 'verifier_score' to each instance

        Returns:
            Filtered relations dict with only verified instances
        """
        if not relations:
            return {}

        # Get embeddings once for efficiency
        embeddings, offset_mapping = self._get_token_embeddings(text)

        # Cache relation embeddings
        rel_embeddings = {}

        verified: dict[str, list] = {}
        for rel_type, instances in relations.items():
            verified[rel_type] = []

            # Get or compute relation embedding
            if rel_type not in rel_embeddings:
                rel_embeddings[rel_type] = self._get_relation_embedding(rel_type)
            rel_emb = rel_embeddings[rel_type]

            for instance in instances:
                head = instance.get("head", {})
                tail = instance.get("tail", {})

                # Skip if missing position info
                if "start" not in head or "start" not in tail:
                    continue

                # Build features and verify
                features = self._build_features(
                    text, head, tail, rel_type, embeddings, offset_mapping, rel_emb=rel_emb
                )
                # Skip a relation whose span lies past the truncation window
                # rather than scoring a substituted, unrelated token position.
                if features is None:
                    continue

                with torch.no_grad():
                    logit = self.model(features.unsqueeze(0))
                    score = torch.sigmoid(logit).item()

                if score >= self.config.threshold:
                    if return_scores:
                        instance = {**instance, "verifier_score": score}
                    verified[rel_type].append(instance)

        return verified
