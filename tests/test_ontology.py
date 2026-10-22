"""Tests for the facility ontology."""

from __future__ import annotations

import pytest

from iai.ontology import Ontology, OntologyError, load_ontology


def test_load_reference_ontology_validates(ontology):
    report = ontology.validate()
    assert report.ok, report.errors
    assert ontology.summary()["nodes"] > 40


def test_find_site_and_equipment(ontology):
    chiller = ontology.resolve("Chiller 1")
    assert chiller.type == "equipment"
    assert ontology.site_of(chiller.id) == "site.dc-lon-01"
    sensors = ontology.sensors_for(chiller.id)
    assert any(sensor.id.endswith("condenser_pressure_kpa") for sensor in sensors)


def test_unknown_alias_raises(ontology):
    with pytest.raises(OntologyError):
        ontology.resolve("Chiller 99")


def test_find_in_text_prefers_longest_match(ontology):
    node = ontology.find_in_text("Check Chiller 1 on site")
    assert node is not None
    assert node.id == "plant.chiller-1"


def test_shortest_path_and_subgraph(ontology):
    path = ontology.shortest_path("plant.chiller-1", "plant.cooling-tower-1")
    assert path is not None
    assert path[0] == "plant.chiller-1"
    nodes = ontology.subgraph("plant.chiller-1", hops=1)
    assert any(node.id == "plant.cooling-tower-1" for node in nodes)


def test_validation_catches_duplicate_and_dangling():
    payload = {
        "version": "1.0.0",
        "nodes": [
            {"id": "site.a", "type": "site", "name": "A", "attributes": {}},
            {"id": "site.a", "type": "site", "name": "A2", "attributes": {}},
        ],
        "edges": [{"src": "site.a", "dst": "missing", "relation": "part_of"}],
    }
    from iai.ontology import Ontology

    with pytest.raises(OntologyError):
        Ontology.from_dict(payload)
    # dangling edge validation with unique nodes
    payload2 = {
        "version": "1.0.0",
        "nodes": [
            {"id": "site.a", "type": "site", "name": "A", "attributes": {}},
            {"id": "site.b", "type": "site", "name": "B", "attributes": {}},
        ],
        "edges": [{"src": "site.a", "dst": "missing", "relation": "part_of"}],
    }
    ontology2 = Ontology.from_dict(payload2)
    report2 = ontology2.validate()
    assert not report2.ok
    assert any("references" in error or "unknown" in error for error in report2.errors)


def test_derived_cycle_detection():
    payload = {
        "version": "1.0.0",
        "nodes": [
            {"id": "site.a", "type": "site", "name": "A", "attributes": {}},
            {
                "id": "derived.a",
                "type": "derived_tag",
                "name": "A",
                "site_id": "site.a",
                "attributes": {"unit": "x", "expression": "b + 1", "inputs": {"b": "derived.b"}},
            },
            {
                "id": "derived.b",
                "type": "derived_tag",
                "name": "B",
                "site_id": "site.a",
                "attributes": {"unit": "x", "expression": "a + 1", "inputs": {"a": "derived.a"}},
            },
        ],
        "edges": [],
    }
    ontology = Ontology.from_dict(payload)
    report = ontology.validate()
    assert not report.ok
    assert any("cycle" in error for error in report.errors)


def test_setpoint_validation(ontology):
    node = ontology.resolve("Chiller 1 chilled water supply setpoint")
    assert node.type == "setpoint"
    assert node.attributes["min"] < node.attributes["max"]


@pytest.fixture
def ontology():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return load_ontology(root / "data" / "ontology.json")
