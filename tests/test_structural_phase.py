from services.graphgen.patch.phases.structural import STRUCTURAL_MAPPINGS, run_structural_phase


def test_structural_mappings_have_expected_node_types():
    assert set(STRUCTURAL_MAPPINGS.keys()) == {"CHUNK", "SEGMENT", "DOC", "TOPIC", "SUBTOPIC"}


def test_structural_phase_dry_run_reports_pending_updates(monkeypatch):
    pending_by_type = {
        "CHUNK": 2,
        "SEGMENT": 1,
        "DOC": 1,
        "TOPIC": 0,
        "SUBTOPIC": 0,
    }

    def fake_collect_count(_driver, node_type, _ontology_class):
        return pending_by_type[node_type]

    monkeypatch.setattr("services.graphgen.patch.phases.structural._collect_count", fake_collect_count)

    result = run_structural_phase(driver=object(), dry_run=True, batch_size=100)

    assert result["phase"] == "structural"
    assert result["dry_run"] is True
    assert result["before"] > 0
    assert result["updated"] == result["before"]
    assert result["after"] == 0
