"""A profile saved over its file keeps what the file says beyond its settings.

A profile is a YAML file people read and annotate, and the paper's five open with a header saying which
task each is and where its values came from. `profiles.save` is how the web's editor, `tandem planners use`
and `tandem executors use` write one; it used to write a fresh dump, so the first edit in the web wiped
every comment. Now only what changed is rewritten -- and when that cannot be done exactly, the plain dump
is still what is written, never a setting other than as validated.
"""

from __future__ import annotations

import difflib

import pytest
from fastapi.testclient import TestClient
from helpers import FakeFactory, isolate_registry

from tandem.cli import planners as planners_cli
from tandem.core import profiles
from tandem.core import settings as settings_mod
from tandem.planners import registry
from tandem.server.app import create_app

NAME = "store-bread-in-closed-box"


@pytest.fixture
def paper(machine_rig, monkeypatch):
    isolate_registry(monkeypatch)
    profiles.seed_builtins()
    cfg = settings_mod.load()
    cfg.active_profile = NAME
    settings_mod.save(cfg)
    return profiles.path_of(NAME)


def _changed_lines(before: str, after: str) -> list[str]:
    return [
        line
        for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]


def test_a_save_that_changes_nothing_leaves_the_file_as_it_was(paper):
    before = paper.read_text()
    profiles.save(profiles.load(NAME))
    assert paper.read_text() == before


def test_an_edit_in_the_web_rewrites_that_line_and_keeps_the_header_and_comments(paper):
    before = paper.read_text()
    client = TestClient(create_app())
    body = client.get(f"/api/profiles/{NAME}").json()["profile"]
    body["task"]["target_episodes"] = 50
    body["planner"]["options"]["tamp"]["num_particles"] = 1024
    assert client.put(f"/api/profiles/{NAME}", json=body).status_code == 200

    after = paper.read_text()
    assert _changed_lines(before, after) == [
        "-  target_episodes: 20",
        "+  target_episodes: 50",
        "-      num_particles: 512",
        "+      num_particles: 1024",
    ]
    assert "Store Bread in Closed Box  —  task 5 of the TANDEM paper" in after
    loaded = profiles.load(NAME)
    assert loaded.task.target_episodes == 50 and loaded.planner.options["tamp"]["num_particles"] == 1024


def test_a_setting_taken_out_goes_and_one_put_in_is_added(paper):
    profile = profiles.load(NAME)
    del profile.planner.options["tamp"]["placement_flatness_tol"]
    profile.planner.options["tamp"]["grasp_threshold"] = 0.02
    profile.task.goal = "the bread is in the box"
    profiles.save(profile)

    text = paper.read_text()
    assert "placement_flatness_tol" not in text and "grasp_threshold: 0.02" in text
    assert "goal: the bread is in the box" in text and "# this task's own" in text
    assert profiles.load(NAME) == profile


def test_a_planner_switch_and_back_keeps_the_comments(paper):
    registry.register_backend("pure", FakeFactory("pure"))
    planners_cli.use_planner("pure", profile_name=NAME)
    assert "task 5 of the TANDEM paper" in paper.read_text()
    planners_cli.use_planner("tiptop", profile_name=NAME)
    after = paper.read_text()
    assert "task 5 of the TANDEM paper" in after and "# Phase planning" in after
    paper_options = profiles.load_file(profiles.builtin_path(NAME), name=NAME).planner.options
    assert profiles.load(NAME).planner.options == paper_options
    # The tamp block left the file and came back from the stash, a plain dump: its settings are the
    # paper's again, but the group comments inside it went with it.
    assert "# DATAFARM" not in after


def test_a_file_that_is_not_a_mapping_is_replaced_by_the_plain_dump(paper):
    profile = profiles.load(NAME)
    paper.write_text("- not\n- a profile\n")
    profiles.save(profile)
    assert profiles.load(NAME) == profile
    assert not paper.read_text().startswith("#"), "nothing to keep, so the plain dump"


def test_a_file_that_is_not_yaml_is_replaced_by_the_plain_dump(paper):
    profile = profiles.load(NAME)
    paper.write_text("task: [unclosed\n")
    profiles.save(profile)
    assert profiles.load(NAME) == profile
