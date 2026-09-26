"""Listing the ZED cameras for `tandem init`: under whichever interpreter has pyzed, and a role for each."""

from __future__ import annotations

from tandem.core import rig as rig_mod
from tandem.core import zed


def test_the_listing_suggests_a_mini_for_the_wrist_and_runs_under_an_interpreter_with_pyzed(tmp_path):
    cams = [zed.ZedCamera("1", "ZED 2i"), zed.ZedCamera("2", "ZED X Mini"), zed.ZedCamera("3", "ZED 2")]
    assert zed.suggest(cams, rig_mod.ROLES) == {"hand": "2", "external": "1", "external_2": "3"}
    assert zed.suggest(cams[:1], rig_mod.ROLES) == {"hand": "1"}, "with no Mini, the first camera is offered"

    # A stand-in interpreter that prints what `get_device_list` would, and one that has no pyzed.
    no_pyzed = tmp_path / "no-pyzed"
    no_pyzed.write_text("#!/bin/sh\necho \"ModuleNotFoundError: No module named 'pyzed'\" >&2\nexit 1\n")
    with_pyzed = tmp_path / "with-pyzed"
    with_pyzed.write_text(
        "#!/bin/sh\necho 'ZED SDK banner'\n"
        "echo '[{\"serial\": \"14846828\", \"model\": \"ZED-M\", \"state\": \"AVAILABLE\"}]'\n"
    )
    for script in (no_pyzed, with_pyzed):
        script.chmod(0o755)
    assert zed.detect([no_pyzed]) is None
    assert zed.detect([no_pyzed, with_pyzed]) == [zed.ZedCamera("14846828", "ZED-M", "AVAILABLE")]
