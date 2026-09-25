"""A rejected `tandem profile edit` restores the profile, and keeps what the person wrote beside it.

Restored over, the edit was gone, and had to be made again from nothing.
"""

from __future__ import annotations

from typer.testing import CliRunner

from tandem.cli.app import app
from tandem.core import profiles


def test_a_rejected_edit_is_kept_beside_the_restored_profile(isolated_env, machine_rig, monkeypatch, tmp_path):
    profiles.create("cups", prompt="put the cup on the plate")
    before = profiles.path_of("cups").read_text()
    breaker = tmp_path / "break.sh"
    breaker.write_text("#!/bin/sh\nprintf 'version: 3\\ntask:\\n  bogus: 1\\n' > \"$1\"\n")
    breaker.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(breaker))

    result = CliRunner().invoke(app, ["profile", "edit", "cups"])
    assert result.exit_code == 1
    assert profiles.path_of("cups").read_text() == before
    kept = profiles.path_of("cups").with_name("cups.yml.rejected")
    assert kept.read_text() == "version: 3\ntask:\n  bogus: 1\n"
    assert str(kept) in result.exception.message
    assert profiles.list_names() == ["cups"], "the kept text is not a profile"
