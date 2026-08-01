"""Unit tests for the frame-definition embedding index (_frame_index).

Uses a tiny in-process matrix so no embedder/FrameNet is touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from graphknows.channels import _frame_index as fi


@pytest.fixture(autouse=True)
def _reset_index():
    fi._names, fi._matrix = None, None
    yield
    fi._names, fi._matrix = None, None


def _tiny_index() -> tuple[list[str], np.ndarray]:
    # 3 unit vectors in 2-D: East, North, and the 45-degree diagonal.
    names = ["East", "North", "NorthEast"]
    mat = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    return names, mat


def test_match_frames_ranks_by_cosine():
    fi._names, fi._matrix = _tiny_index()  # inject index; skip build
    out = fi.match_frames([2.0, 0.0], top_k=2)  # unnormalized input
    assert [n for n, _ in out] == ["East", "NorthEast"]
    assert out[0][1] == pytest.approx(1.0, abs=1e-5)
    assert out[1][1] == pytest.approx(0.7071, abs=1e-3)


def test_match_frames_min_sim_filters():
    fi._names, fi._matrix = _tiny_index()
    out = fi.match_frames([1.0, 0.0], top_k=3, min_sim=0.5)
    assert [n for n, _ in out] == ["East", "NorthEast"]  # North (cos 0.0) dropped


def test_match_frames_default_floor_drops_low_cosine():
    fi._names, fi._matrix = _tiny_index()
    # Default floor (DEFAULT_MIN_SIM=0.20) drops North (cos 0.0) without an explicit arg.
    out = fi.match_frames([1.0, 0.0], top_k=3)
    assert [n for n, _ in out] == ["East", "NorthEast"]


def test_frame_embedding_returns_matrix_row():
    fi._names, fi._matrix = _tiny_index()
    assert fi.frame_embedding("North") == pytest.approx([0.0, 1.0])
    assert fi.frame_embedding("Nonexistent") == []


def test_frame_summary_reads_definition(monkeypatch):
    monkeypatch.setattr("graphknows.channels._framenet._frame_def", lambda name: f"def of {name}")
    assert fi.frame_summary("Motion") == "def of Motion"


def test_zero_vector_skips_build_and_returns_empty(monkeypatch):
    def _boom(force: bool = False):
        raise AssertionError("index built for a degenerate vector")

    monkeypatch.setattr(fi, "_ensure", _boom)
    assert fi.match_frames([0.0, 0.0]) == []


def test_cache_round_trip(monkeypatch, tmp_path):
    names, mat = _tiny_index()
    monkeypatch.setenv("GRAPHKNOWS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(fi, "_cache_key", lambda: "test_frames")
    build_calls = {"n": 0}

    def _fake_build() -> tuple[list[str], np.ndarray]:
        build_calls["n"] += 1
        return names, mat

    monkeypatch.setattr(fi, "_build_matrix", _fake_build)

    # First _ensure builds + writes the disk cache.
    fi._ensure()
    assert build_calls["n"] == 1
    assert (tmp_path / "test_frames.npy").exists()
    assert (tmp_path / "test_frames.json").exists()

    # Drop the in-process index; second _ensure loads from disk (no rebuild).
    fi._names, fi._matrix = None, None
    n2, m2 = fi._ensure()
    assert build_calls["n"] == 1  # not rebuilt
    assert n2 == names
    np.testing.assert_allclose(m2, mat, rtol=1e-6)

    # Matches computed off the disk-loaded matrix are identical.
    assert fi.match_frames([1.0, 0.0], top_k=1)[0][0] == "East"
