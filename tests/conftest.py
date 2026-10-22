"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from iai.app import Application
from iai.ontology import load_ontology


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def ontology(data_dir):
    return load_ontology(data_dir / "ontology.json")


@pytest.fixture(scope="session")
def app(data_dir):
    return Application.load(data_dir)


@pytest.fixture
def fresh_app(tmp_path, data_dir):
    import shutil

    target = tmp_path / "data"
    shutil.copytree(data_dir, target)
    return Application.load(target)
