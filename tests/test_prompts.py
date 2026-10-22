"""Tests for versioned prompts."""

from __future__ import annotations

import pytest

from iai.prompts import PromptError, PromptLibrary, PromptTemplate


def test_prompt_render_strict():
    template = PromptTemplate(
        id="demo",
        version="1.0.0",
        template="Hello {{ name }}, site {{ site }}.",
        variables=("name", "site"),
    )
    assert template.render(name="Op", site="lon") == "Hello Op, site lon.\n"
    with pytest.raises(PromptError):
        template.render(name="Op")
    with pytest.raises(PromptError):
        template.render(name="Op", site="lon", extra="x")


def test_prompt_checksum_stable():
    template = PromptTemplate(id="demo", version="1.0.0", template="Hi {{ name }}.", variables=("name",))
    assert len(template.checksum) == 64


def test_prompt_library_load(data_dir):
    library = PromptLibrary.load_dir(data_dir / "prompts")
    assert "system" in library.ids()
    rendered = library.render("system", role="operator", site_id="site.dc-lon-01", tool_policy="read", untrusted_context="none")
    assert "operator" in rendered


def test_prompt_library_missing_variable(data_dir):
    library = PromptLibrary.load_dir(data_dir / "prompts")
    with pytest.raises(PromptError):
        library.render("system", role="operator", site_id="x", tool_policy="read")


def test_prompt_front_matter_validation(tmp_path):
    bad = tmp_path / "bad.md"
    bad.write_text("# no front matter")
    with pytest.raises(PromptError):
        PromptLibrary.load_dir(tmp_path)
