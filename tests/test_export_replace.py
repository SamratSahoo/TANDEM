"""A dataset at the export's destination is replaced only by a finished one, and only if tandem built it.

`tandem export lerobot` deleted whatever was at ``<out>/<repo>`` before it had read a single episode. A
build that then found nothing it could export, or failed part-way, left the user with neither the old
dataset nor a new one. A destination one level off (`--out ~`, or a repo named like a directory of their
own) cost a directory that had never been a dataset at all.

These run the real writer on real (tiny) videos. What is under test is the files on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import write_trajectory

from tandem.core import trajectories
from tandem.core.errors import TandemError

pytest.importorskip("av")
pytest.importorskip("pyarrow")

REPO = "me/data"
N = 12


def _clip(path: Path, frames: int) -> None:
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=15)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(frames):
            image = np.full((48, 64, 3), (i * 17) % 255, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _episode(profile, timestamp: str) -> Path:
    directory = write_trajectory(profile, timestamp, n_frames=N, with_plan=False)
    _clip(directory / "external_cam.mp4", N)
    _clip(directory / "hand_cam.mp4", N)
    return directory


def _build(profile, tmp_path, **kwargs) -> dict:
    from tandem.export import build

    return build.build_dataset(profile, repo_id=REPO, out_root=tmp_path / "out", **kwargs)


def _dataset(tmp_path) -> Path:
    return tmp_path / "out" / REPO


def _beside(tmp_path) -> list[str]:
    """Everything in the directory the dataset lives in: a staging or replaced directory left behind shows."""
    return sorted(p.name for p in _dataset(tmp_path).parent.iterdir())


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _episodes_in(root: Path) -> int:
    return json.loads((root / "meta" / "info.json").read_text())["total_episodes"]


def _unexportable(directory: Path) -> None:
    """Rewrite a trajectory's state with a continuous gripper, which the export refuses to write."""
    path = directory / trajectories.STATE_FILE
    with np.load(path) as store:
        arrays = {key: store[key] for key in store.files}
    arrays["cmd_gripper"] = np.linspace(0.0, 1.0, len(arrays["cmd_gripper"]), dtype=np.float32)
    np.savez(path, **arrays)


# --- a rebuild -----------------------------------------------------------------------------------------


def test_a_rebuild_replaces_the_dataset_it_built(profile, tmp_path):
    from tandem.export import build

    _episode(profile, "2026-01-01_00-00-01")
    first = _build(profile, tmp_path)
    assert first["written"] == 1 and _episodes_in(_dataset(tmp_path)) == 1
    marker = json.loads((_dataset(tmp_path) / build.MARKER).read_text())
    assert (marker["written_by"], marker["repo_id"], marker["episodes"]) == ("tandem", REPO, 1)

    _episode(profile, "2026-01-01_00-00-02")
    second = _build(profile, tmp_path)
    assert second["written"] == 2 and _episodes_in(_dataset(tmp_path)) == 2
    assert _beside(tmp_path) == ["data"], "a staging or replaced directory was left beside the dataset"


# --- nothing is lost to a build that does not finish -----------------------------------------------------


def test_a_build_with_nothing_to_export_leaves_the_last_dataset_as_it_was(profile, tmp_path):
    good = _episode(profile, "2026-01-01_00-00-01")
    _build(profile, tmp_path)
    before = _files(_dataset(tmp_path))

    _unexportable(good)
    with pytest.raises(TandemError, match="can be exported: 2026-01-01_00-00-01: cmd_gripper is not binary") as caught:
        _build(profile, tmp_path)
    assert "is as it was" in caught.value.hint

    assert _files(_dataset(tmp_path)) == before, "the last dataset was deleted for a build that wrote nothing"
    assert _beside(tmp_path) == ["data"]


def test_a_build_whose_every_episode_fails_to_decode_leaves_the_last_dataset_as_it_was(profile, tmp_path):
    good = _episode(profile, "2026-01-01_00-00-01")
    _build(profile, tmp_path)
    before = _files(_dataset(tmp_path))

    (good / "hand_cam.mp4").unlink()
    with pytest.raises(TandemError, match="could be written: 2026-01-01_00-00-01: missing exterior_1 and/or wrist"):
        _build(profile, tmp_path)
    assert _files(_dataset(tmp_path)) == before
    assert _beside(tmp_path) == ["data"]


def test_a_build_that_fails_part_way_leaves_the_last_dataset_as_it_was(profile, tmp_path, monkeypatch):
    from tandem.export import build

    _episode(profile, "2026-01-01_00-00-01")
    _build(profile, tmp_path)
    before = _files(_dataset(tmp_path))
    _episode(profile, "2026-01-01_00-00-02")

    def finalize(self):
        raise RuntimeError("the disk filled up")

    monkeypatch.setattr(build.V3DatasetWriter, "finalize", finalize)
    with pytest.raises(RuntimeError, match="the disk filled up"):
        _build(profile, tmp_path)
    assert _files(_dataset(tmp_path)) == before
    assert _beside(tmp_path) == ["data"], "the half-built dataset was left behind"


def test_the_swap_puts_the_old_dataset_back_when_the_new_one_cannot_be_moved_in(tmp_path, monkeypatch):
    """The one moment there could be neither: between moving the old one aside and the new one in."""
    import os

    from tandem.export import build

    dataset, staging = tmp_path / "data", tmp_path / ".data.building-x"
    dataset.mkdir()
    (dataset / "old").write_text("the last dataset")
    staging.mkdir()
    (staging / "new").write_text("the new one")
    rename = os.rename

    def refuse_the_new_one(src, dst):
        if Path(src) == staging:
            raise OSError("no")
        rename(src, dst)

    monkeypatch.setattr(build.os, "rename", refuse_the_new_one)
    with pytest.raises(OSError):
        build._swap_in(staging, dataset)
    assert (dataset / "old").read_text() == "the last dataset"
    assert sorted(p.name for p in tmp_path.iterdir()) == [".data.building-x", "data"]


def _interrupt_after(monkeypatch, which: str):
    """os.rename that does its job and then raises KeyboardInterrupt, for the rename whose source is ``which``.

    Python acts on a pending Ctrl-C as a call returns, so this is where one really lands.
    """
    import os

    from tandem.export import build

    rename = os.rename

    def rename_then_interrupt(src, dst):
        rename(src, dst)
        if Path(src).name.startswith(which):
            raise KeyboardInterrupt

    monkeypatch.setattr(build.os, "rename", rename_then_interrupt)


def test_a_ctrl_c_as_the_old_dataset_is_moved_aside_puts_it_back(profile, tmp_path, monkeypatch):
    """The first rename was outside the rollback: the new dataset was deleted as a failed build, and the
    old one was left at a hidden name, with nothing at the destination and nothing said."""
    _episode(profile, "2026-01-01_00-00-01")
    _build(profile, tmp_path)
    before = _files(_dataset(tmp_path))
    _episode(profile, "2026-01-01_00-00-02")

    _interrupt_after(monkeypatch, "data")
    with pytest.raises(KeyboardInterrupt):
        _build(profile, tmp_path)
    assert _files(_dataset(tmp_path)) == before
    assert _beside(tmp_path) == ["data"]


def test_a_ctrl_c_as_the_new_dataset_is_moved_in_leaves_it_there(profile, tmp_path, monkeypatch):
    """The rollback then tried to move the old one back onto the new one, and the Ctrl-C came out as
    `OSError: Directory not empty`, with the export said to have failed and the old one left hidden."""
    _episode(profile, "2026-01-01_00-00-01")
    _build(profile, tmp_path)
    _episode(profile, "2026-01-01_00-00-02")

    _interrupt_after(monkeypatch, ".data.building-")
    with pytest.raises(KeyboardInterrupt):
        _build(profile, tmp_path)
    assert _episodes_in(_dataset(tmp_path)) == 2, "the finished dataset is the one in place"
    assert _beside(tmp_path) == ["data"], "the replaced dataset was left behind"


def test_what_an_export_that_did_not_finish_left_is_named_and_kept(profile, tmp_path, caplog):
    """A build killed outright leaves gigabytes under a name `ls` does not show; nothing ever said so."""
    _episode(profile, "2026-01-01_00-00-01")
    out = _dataset(tmp_path).parent
    out.mkdir(parents=True)
    (out / ".data.building-dead0000").mkdir()
    (out / ".data.replaced-dead0000").mkdir()
    (out / ".data.replaced-dead0000" / "meta").mkdir()
    (out / ".other.building-dead0000").mkdir()

    with caplog.at_level("WARNING", logger="tandem.export"):
        _build(profile, tmp_path)
    said = caplog.text
    assert ".data.building-dead0000 is what an export that did not finish" in said
    assert ".data.replaced-dead0000 is the dataset an export was replacing when it stopped" in said
    assert "last complete one" in said
    assert ".other." not in said, "another dataset's leftovers are its own export's to mention"
    assert _beside(tmp_path) == [
        ".data.building-dead0000",
        ".data.replaced-dead0000",
        ".other.building-dead0000",
        "data",
    ], "only this export's own staging directory is its to remove"


def test_the_dataset_is_as_readable_as_any_directory_the_exporter_makes(profile, tmp_path):
    """Staged with `tempfile.mkdtemp`, the root came out 0700 whatever the umask: a shared lab machine or a
    training container running as another user got "permission denied" on it."""
    import os
    import stat

    _episode(profile, "2026-01-01_00-00-01")
    previous = os.umask(0o002)
    try:
        _build(profile, tmp_path)
    finally:
        os.umask(previous)
    root = _dataset(tmp_path)
    assert stat.S_IMODE(root.stat().st_mode) == 0o775
    assert stat.S_IMODE(root.stat().st_mode) == stat.S_IMODE((root / "meta").stat().st_mode)


# --- a directory tandem did not build ------------------------------------------------------------------


def test_a_directory_tandem_did_not_build_is_not_replaced_without_force(profile, tmp_path):
    from tandem.export import build

    _episode(profile, "2026-01-01_00-00-01")
    theirs = _dataset(tmp_path)
    theirs.mkdir(parents=True)
    (theirs / "thesis.tex").write_text("four years")

    with pytest.raises(TandemError, match="not a dataset tandem built") as caught:
        _build(profile, tmp_path)
    assert "--force" in caught.value.hint
    assert _files(theirs) == {"thesis.tex": b"four years"} and _beside(tmp_path) == ["data"]

    replaced = _build(profile, tmp_path, force=True)
    assert replaced["written"] == 1 and (theirs / build.MARKER).is_file()
    assert not (theirs / "thesis.tex").exists() and _beside(tmp_path) == ["data"]


def test_an_empty_directory_is_no_loss_and_is_built_into(profile, tmp_path):
    _episode(profile, "2026-01-01_00-00-01")
    _dataset(tmp_path).mkdir(parents=True)
    assert _build(profile, tmp_path)["written"] == 1


def test_the_cli_refuses_and_then_forces(profile, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from tandem.cli import export as export_cli
    from tandem.cli.app import app

    # Its handlers would outlive this test on the process-wide logger.
    monkeypatch.setattr(export_cli, "_configure_logging", lambda: None)
    _episode(profile, "2026-01-01_00-00-01")
    (_dataset(tmp_path) / "keep").mkdir(parents=True)
    args = ["export", "lerobot", profile.name, "--repo", REPO, "--out", str(tmp_path / "out")]

    refused = CliRunner().invoke(app, args)
    assert refused.exit_code != 0 and "--force" in refused.exception.hint
    assert (_dataset(tmp_path) / "keep").is_dir()

    forced = CliRunner().invoke(app, [*args, "--force"])
    assert forced.exit_code == 0, forced.output
    assert not (_dataset(tmp_path) / "keep").exists()


def test_the_marker_is_not_pushed(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from tandem.export import build

    calls: list[dict] = []

    class Api:
        def __init__(self, token=None):
            pass

        def create_repo(self, *args, **kwargs):
            pass

        def upload_folder(self, **kwargs):
            calls.append(kwargs)

        def create_tag(self, *args, **kwargs):
            pass

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))
    build._upload(tmp_path, REPO, private=True, token=None)
    assert calls and build.MARKER in calls[0]["ignore_patterns"]
