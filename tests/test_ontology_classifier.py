from services.graphgen.patch.ontology_classifier import classify_by_name


def test_party_name_maps_to_political_party():
    assert classify_by_name("EUROPEAN_PARTY") == "POLITICAL_PARTY"


def test_minister_and_commissioner_map_to_person():
    assert classify_by_name("FINANCE_MINISTER") == "PERSON"
    assert classify_by_name("EU_COMMISSIONER_FOR_TRADE") == "PERSON"


def test_date_like_maps_to_event():
    assert classify_by_name("2024_01_01") == "EVENT"


def test_policy_tokens_map_to_policy():
    assert classify_by_name("DIGITAL_SERVICES_DIRECTIVE") == "POLICY"
    assert classify_by_name("CLIMATE_REGULATION") == "POLICY"
    assert classify_by_name("SECURITY_TREATY") == "POLICY"


def test_generic_name_defaults_to_concept():
    assert classify_by_name("abstract_idea") == "CONCEPT"
