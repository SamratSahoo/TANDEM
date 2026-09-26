"""What a trial leaves on disk, and what may be done to it afterwards -- the review's findings, pinned.

* The merge never loses a leg: a status it cannot file under is refused before anything moves, and a
  failure once the legs are parked moves each one back before the scratch directory is removed.
* A trial the method settled (excluded, aborted, failed part-way) stays out of the dataset whatever
  directory it is later moved to: relabel refuses it without ``--force``, the web route likewise,
  ``traj merge --status success`` refuses it, and the export skips it. Every relabel rewrites the
  record, so ``hitl.json`` and the directory it sits in never disagree.
* The recording contract names clips from ``CAMERA_FILES`` only, and a planner can say where its own
  plan file is.
* The teleop driver's docstrings point at tandem's own pieces.

The video tools are stood in for (``fake_video``): what is under test is where the legs are.
"""

from __future__ import annotations

import errno
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tandem.core import merge as merge_mod
from tandem.core import trajectories
from tandem.core.errors import TandemError
from tandem.core.merge import MergeError

FPS = 15
N = 20
TRAJECTORY = "abcabcabcabcabca"


# --------------------------------------------------------------------------- legs on disk


def _leg(profile, name: str, *, source: str, t0: float, status: str = "eval", phase: dict | None = None) -> Path:
    """One leg in the on-disk format: state, a _meta.json and a (placeholder) clip."""
    directory = profile.status_dir(status) / name
    directory.mkdir(parents=True)
    frame_time = t0 + np.arange(N, dtype=np.float64) / FPS
    joints = np.tile(np.arange(7, dtype=np.float32), (N, 1))
    np.savez(
        directory / trajectories.STATE_FILE,
        joint_position=joints,
        gripper_position=np.zeros(N, np.float32),
        cmd_joint_position=joints,
        cmd_joint_velocity=np.zeros_like(joints),
        cmd_gripper=np.zeros(N, np.float32),
        frame_time=frame_time,
    )
    meta = {
        "instruction": "put the toy in the box",
        "fps": FPS,
        "n_frames": N,
        "trajectory_id": TRAJECTORY,
        "segment_source": source,
        "cameras": {"exterior_image_1_left": "external_cam.mp4"},
        "record_start": float(frame_time[0]),
        "record_stop": float(frame_time[0]) + N / FPS,
        **(phase or {}),
    }
    (directory / trajectories.META_FILE).write_text(json.dumps(meta))
    (directory / "external_cam.mp4").write_bytes(f"clip of {name}".encode())
    return directory


def _hand_off(profile, *, perception: bool = True) -> list[Path]:
    """A robot leg (the primary, with a perception/ dump and a plan) and a person's leg after it."""
    robot = _leg(profile, "2026-05-05_00-00-00", source="tamp", t0=5000.0)
    if perception:
        (robot / "perception").mkdir()
        (robot / "perception" / "scene.json").write_text("{}")
        (robot / "tiptop_plan.json").write_text("{}")
    person = _leg(profile, "2026-05-05_00-05-00", source="teleop", t0=5300.0)
    return [robot, person]


def _snapshot(directories: list[Path]) -> dict[str, bytes]:
    """Every file of every leg, by path, so "intact" means byte for byte."""
    out: dict[str, bytes] = {}
    for directory in directories:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                out[str(path)] = path.read_bytes()
    return out


@pytest.fixture
def fake_video(monkeypatch):
    """Stand in for ffprobe/ffmpeg: every clip holds as many frames as its leg has states."""

    def leg_video_frames(leg, cameras, tools_dir):
        with np.load(leg["dir"] / trajectories.STATE_FILE) as store:
            n = len(store["frame_time"])
        return n, {cam: n for cam in cameras}

    def concat_videos(legs, camera, leg_frames, dest, scratch, tools_dir):
        dest.write_bytes(b"joined")
        return sum(leg_frames)

    monkeypatch.setattr(merge_mod, "_leg_video_frames", leg_video_frames)
    monkeypatch.setattr(merge_mod, "_concat_videos", concat_videos)


def _scratch(profile) -> Path:
    return profile.trajectories_dir() / f".merge-{TRAJECTORY}"


# --------------------------------------------------------------------------- the merge keeps every leg


def test_a_status_the_merge_cannot_file_under_is_refused_before_a_leg_moves(profile, fake_video):
    """`tandem traj merge <id> --status Success` used to park both legs in the scratch directory, then
    fail to resolve the status, and delete the scratch directory -- and both legs with it."""
    legs = _hand_off(profile)
    before = _snapshot(legs)

    with pytest.raises(MergeError, match="Unknown status 'Success'.*'success'"):
        merge_mod.merge(profile, TRAJECTORY, status="Success")

    assert _snapshot(legs) == before
    assert not _scratch(profile).exists()


@pytest.mark.parametrize("fault", ["copytree", "copy2", "move-second-leg", "move-into-place"])
def test_a_merge_that_fails_part_way_puts_every_leg_back(profile, fake_video, monkeypatch, fault):
    """A full disk while surfacing the primary's perception dump, or a rename that fails: the legs end
    up exactly where they were, and a merge afterwards succeeds. They used to be rmtree'd with the
    scratch directory they had been parked in, while the log said "the legs are intact"."""
    legs = _hand_off(profile)
    before = _snapshot(legs)
    name = "move" if fault.startswith("move") else fault
    real = getattr(shutil, name)

    def disk_full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    def faulty(src, dst, *args, **kwargs):
        if fault in ("copytree", "copy2"):
            disk_full()
        if fault == "move-second-leg" and Path(src) == legs[1]:
            disk_full()
        if fault == "move-into-place" and Path(src).name.startswith(".merge-"):
            disk_full()
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(merge_mod.shutil, name, faulty)
    with pytest.raises(OSError):
        merge_mod.merge(profile, TRAJECTORY, status="success")

    assert _snapshot(legs) == before, "a leg was lost or changed by a merge that failed"
    assert not _scratch(profile).exists()
    assert len(merge_mod.find_legs(profile, TRAJECTORY)) == 2

    monkeypatch.setattr(merge_mod.shutil, name, real)
    result = merge_mod.merge(profile, TRAJECTORY, status="success")
    assert result["merged"] and result["n_legs"] == 2


def test_legs_a_failed_merge_left_parked_are_never_deleted_by_the_next(profile, fake_video):
    """A merge that could not move its legs back left them in the scratch directory. Every merge used to
    start by removing that directory."""
    legs = _hand_off(profile)
    stranded = _scratch(profile) / "segments" / "00_tamp_2026-05-05_00-00-00"
    stranded.parent.mkdir(parents=True)
    shutil.move(str(legs[0]), str(stranded))

    with pytest.raises(MergeError, match="did not finish"):
        merge_mod.merge(profile, TRAJECTORY, status="success")
    assert (stranded / trajectories.STATE_FILE).is_file()


def test_a_destination_already_taken_is_refused_before_a_leg_moves(profile, fake_video):
    legs = _hand_off(profile)
    before = _snapshot(legs)
    (profile.status_dir("success") / legs[0].name).mkdir(parents=True)

    with pytest.raises(MergeError, match="already exists"):
        merge_mod.merge(profile, TRAJECTORY, status="success")
    assert _snapshot(legs) == before


def test_traj_merge_with_a_mistyped_status_exits_non_zero_and_leaves_the_legs(profile, fake_video):
    from typer.testing import CliRunner

    from tandem.cli.app import app

    legs = _hand_off(profile)
    before = _snapshot(legs)
    result = CliRunner().invoke(app, ["traj", "merge", TRAJECTORY, "--status", "Success", "-p", profile.name])

    assert result.exit_code != 0
    assert isinstance(result.exception, TandemError) and "'success'" in result.exception.message
    assert "untouched" in result.exception.hint
    assert _snapshot(legs) == before


# --------------------------------------------------------------------------- a trial the method settled


EXCLUDED = {
    "outcome": "excluded",
    "excluded": True,
    "failure_stage": "verification",
    "filed_under": "failure",
    "outcome_reason": "the human phase 'open the box' did not verify",
    "verifications": [{"atom": "IsOpen(white_box)", "satisfied": False, "phase": 1}],
}


def _filed(profile, status: str, name: str, record: dict | None) -> Path:
    directory = _leg(profile, name, source="tamp", t0=9000.0, status=status)
    if record is not None:
        (directory / trajectories.HITL_FILE).write_text(json.dumps(record))
    return directory


def _record(directory: Path) -> dict:
    return json.loads((directory / trajectories.HITL_FILE).read_text())


def test_a_listing_says_a_trial_was_excluded(profile):
    directory = _filed(profile, "failure", "2026-02-01_10-00-02", EXCLUDED)
    shown = trajectories.read(directory).to_dict()
    assert (shown["excluded"], shown["outcome"], shown["failure_stage"], shown["settled"]) == (
        True,
        "excluded",
        "verification",
        "excluded",
    )
    plain = trajectories.read(_filed(profile, "failure", "2026-02-01_10-00-03", None)).to_dict()
    assert (plain["excluded"], plain["outcome"], plain["settled"]) == (False, None, None)


@pytest.mark.parametrize(
    "record",
    [
        EXCLUDED,
        {"outcome": "aborted", "failure_stage": None, "filed_under": "failure", "outcome_reason": "preempted"},
        {"outcome": "failure", "failure_stage": "tamp_execution", "filed_under": "failure"},
    ],
    ids=["excluded", "aborted", "tamp-execution"],
)
def test_relabelling_a_settled_trial_as_a_success_needs_force(profile, record):
    directory = _filed(profile, "failure", "2026-02-01_10-00-02", record)
    traj = trajectories.read(directory)

    with pytest.raises(TandemError, match=f"ended {traj.settled}") as refused:
        trajectories.relabel(profile, traj, "success")
    assert "--force" in refused.value.hint
    assert directory.is_dir(), "a refused relabel moved the trial"

    moved = trajectories.relabel(profile, traj, "success", force=True)
    assert moved.status == "success"
    written = _record(moved.path)
    assert written["filed_under"] == "success"
    assert written["outcome"] == "success" and written["excluded"] is False
    assert written["overruled"]["outcome"] == record["outcome"]
    assert written["overruled"]["by"] == "relabel"
    assert moved.settled is None, "an overruled trial is a demonstration now, on purpose"


def test_every_relabel_rewrites_where_the_record_says_the_trial_is(profile):
    directory = _filed(
        profile,
        "success",
        "2026-02-01_10-00-02",
        {"outcome": "success", "failure_stage": None, "excluded": False, "filed_under": "success"},
    )
    moved = trajectories.relabel(profile, trajectories.read(directory), "failure")
    assert (_record(moved.path)["outcome"], _record(moved.path)["filed_under"]) == ("failure", "failure")
    back = trajectories.relabel(profile, moved, "eval")
    assert _record(back.path)["filed_under"] is None


def test_the_relabel_route_refuses_a_settled_trial_without_force(profile):
    from fastapi.testclient import TestClient

    from tandem.server.app import create_app

    _filed(profile, "failure", "2026-02-01_10-00-02", EXCLUDED)
    client = TestClient(create_app())
    url = f"/api/trajectories/{profile.name}/2026-02-01_10-00-02/relabel"

    refused = client.post(url, json={"status": "success"})
    assert refused.status_code == 400 and "excluded" in refused.json()["error"]
    assert client.post(url, json={"status": "success", "force": True}).status_code == 200


def test_traj_relabel_takes_force(profile):
    from typer.testing import CliRunner

    from tandem.cli.app import app

    _filed(profile, "failure", "2026-02-01_10-00-02", EXCLUDED)
    runner = CliRunner()
    args = ["traj", "relabel", "2026-02-01_10-00-02", "success", "-p", profile.name]
    refused = runner.invoke(app, args)
    assert refused.exit_code != 0 and "ended excluded" in refused.exception.message
    forced = runner.invoke(app, [*args, "--force"])
    assert forced.exit_code == 0, forced.output
    assert "overruled" in forced.output


def test_a_merge_filed_as_a_success_refuses_an_excluded_lineage(profile, fake_video):
    legs = _hand_off(profile)
    (legs[0] / trajectories.HITL_FILE).write_text(json.dumps(EXCLUDED))
    before = _snapshot(legs)

    with pytest.raises(MergeError, match="ended excluded"):
        merge_mod.merge(profile, TRAJECTORY, status="success")
    assert _snapshot(legs) == before


@pytest.fixture
def export_writes(monkeypatch):
    """build_dataset with its writer and decoder stood in for: which directories it would export."""
    pytest.importorskip("av")
    pytest.importorskip("pyarrow")
    from tandem.export import build

    written: list[str] = []

    class Writer:
        def __init__(self, *args, **kwargs):
            pass

        def finalize(self):
            pass

    monkeypatch.setattr(build, "V3DatasetWriter", Writer)
    monkeypatch.setattr(
        build, "_add_episode", lambda writer, directory, task: written.append(directory.name) or {"frames": N}
    )
    return build, written


def test_the_export_skips_a_settled_trial_wherever_it_is_filed(profile, tmp_path, export_writes):
    """success/ is only where somebody put the trial; the record is what the method decided."""
    build, written = export_writes
    _filed(profile, "success", "2026-02-01_10-00-01", {"outcome": "success", "filed_under": "success"})
    _filed(profile, "success", "2026-02-01_10-00-02", EXCLUDED)
    _filed(profile, "success", "2026-02-01_10-00-03", {"outcome": "aborted", "filed_under": "success"})
    _filed(profile, "success", "2026-02-01_10-00-04", None)  # phase planning off: exported as ever
    overruled = _filed(profile, "failure", "2026-02-01_10-00-05", EXCLUDED)
    trajectories.relabel(profile, trajectories.read(overruled), "success", force=True)

    summary = build.build_dataset(profile, repo_id="me/data", out_root=tmp_path / "out")

    assert written == ["2026-02-01_10-00-01", "2026-02-01_10-00-04", "2026-02-01_10-00-05"]
    skipped = dict(summary["skipped"])
    assert set(skipped) == {"2026-02-01_10-00-02", "2026-02-01_10-00-03"}
    assert "excluded" in skipped["2026-02-01_10-00-02"]
    assert summary["considered"] == 5


def test_the_export_says_so_when_every_success_is_held_back(profile, tmp_path, export_writes):
    build, _ = export_writes
    _filed(profile, "success", "2026-02-01_10-00-02", EXCLUDED)
    with pytest.raises(TandemError, match="1 more are filed there"):
        build.build_dataset(profile, repo_id="me/data", out_root=tmp_path / "out")


# --------------------------------------------------------------------------- the recording contract


def test_a_clip_under_a_name_the_merge_does_not_join_is_not_a_complete_leg(profile):
    """It passed `is_complete` and the kit, and then could not be merged, listed or exported."""
    from tandem.planners.base import ExecuteResult, LegSpec
    from tandem.planners.testing import ConformanceError, check_leg

    directory = _leg(profile, "2026-03-03_00-00-00", source="tamp", t0=1.0)
    meta = json.loads((directory / trajectories.META_FILE).read_text())
    meta["cameras"] = {"exterior_image_1_left": "front.mp4"}
    (directory / trajectories.META_FILE).write_text(json.dumps(meta))
    (directory / "front.mp4").write_bytes(b"clip")

    assert not trajectories.is_complete(directory)
    leg = LegSpec(trajectory_id=TRAJECTORY, instruction="put the toy in the box")
    result = ExecuteResult(ok=True, n_frames=N, rollout_dir=str(directory))
    with pytest.raises(ConformanceError, match="front.mp4 is not one of external_cam.mp4"):
        check_leg(result, leg, directory)

    meta["cameras"] = {"exterior_image_1_left": "external_cam.mp4"}
    (directory / trajectories.META_FILE).write_text(json.dumps(meta))
    assert trajectories.is_complete(directory)
    check_leg(result, leg, directory)


def test_a_planner_names_its_own_plan_file(profile):
    directory = _leg(profile, "2026-03-03_00-00-00", source="tamp", t0=1.0)
    meta = json.loads((directory / trajectories.META_FILE).read_text())
    assert not trajectories.read(directory).has_plan

    meta["plan_file"] = "plan.json"
    (directory / trajectories.META_FILE).write_text(json.dumps(meta))
    (directory / "plan.json").write_text(json.dumps({"drops": 3}))
    traj = trajectories.read(directory)
    assert traj.has_plan and trajectories.plan(traj) == {"drops": 3}

    # A name off disk that is not a bare file name is ignored, not followed.
    meta["plan_file"] = "../plan.json"
    (directory / trajectories.META_FILE).write_text(json.dumps(meta))
    assert not trajectories.read(directory).has_plan


def test_a_merged_hand_off_names_its_primary_legs_plan_file(profile, fake_video):
    legs = _hand_off(profile, perception=False)
    meta = json.loads((legs[0] / trajectories.META_FILE).read_text())
    meta["plan_file"] = "plan.json"
    (legs[0] / trajectories.META_FILE).write_text(json.dumps(meta))
    (legs[0] / "plan.json").write_text("{}")

    result = merge_mod.merge(profile, TRAJECTORY, status="success")
    assert trajectories.read(Path(result["dir"])).has_plan


# --------------------------------------------------------------------------- the teleop driver's words


def test_the_teleop_driver_points_at_tandems_own_pieces():
    """Carried over from the monorepo, it cited files and a server tandem does not have."""
    import tandem

    source = (Path(tandem.__file__).parent / "teleop" / "driver.py").read_text()
    for stale in ("ARCHITECTURE.md", "Node server", "collect/", "build_lerobot", "merge_trajectory.py"):
        assert stale not in source, f"driver.py still mentions {stale!r}"
    package = Path(tandem.__file__).parent
    for named in ("teleop/child.py", "executors/teleop.py", "export/build.py", "teleop/raw_episode.py"):
        assert named in source and (package / named).is_file(), named
    assert "https://prpl-group.com/tandem/docs/adding-a-human-executor/#the-teleop-executor" in source
