from services.graphgen.patch import runner


class _FakeDriver:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_runner_dry_run_returns_non_empty_summary(monkeypatch):
    monkeypatch.setattr(
        runner.GraphDatabase,
        "driver",
        lambda *_args, **_kwargs: _FakeDriver(),
    )

    monkeypatch.setitem(
        runner.PHASE_RUNNERS,
        "structural",
        lambda _driver, dry_run, batch_size: {
            "phase": "structural",
            "before": 2,
            "after": 0,
            "updated": 2,
            "updated_by_type": {"DOC": 1, "CHUNK": 1},
            "dry_run": dry_run,
            "batch_size": batch_size,
        },
    )
    monkeypatch.setitem(
        runner.PHASE_RUNNERS,
        "legacy",
        lambda _driver, dry_run, batch_size: {
            "phase": "legacy",
            "before": 1,
            "after": 0,
            "updated": 1,
            "updated_by_class": {"PERSON": 1},
            "dry_run": dry_run,
            "batch_size": batch_size,
        },
    )

    summary = runner.run_patch(phases=["structural", "legacy"], dry_run=True, batch_size=25)

    assert summary["dry_run"] is True
    assert summary["batch_size"] == 25
    assert summary["phases"]
    assert len(summary["phases"]) == 2
    assert summary["totals"]["updated"] == 3
