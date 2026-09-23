"""The data layer a phase-planned hand-off relies on.

* ``action_joint_velocity`` is an OPTIONAL state array: TipTop 1c6daf3 records it on TAMP legs,
  teleop legs never do, and the merge must join such legs instead of refusing the array.
* Every leg says which phase φ_k it records, and the merged episode keeps saying so per segment.
* The teleop driver's recording window covers all n of its frames, not n-1.
* A leg is complete by the recording contract, not by carrying TipTop's plan file.

The video concatenation is faked (``fake_video``): what is under test is the state arrays and the
metadata, and a real ffmpeg would only make these tests depend on the machine.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tandem import teleop as teleop_pkg
from tandem.core import merge as merge_mod
from tandem.core import trajectories
from tandem.core.merge import DROID_JV_GAIN, MergeError

FPS = 15
N = 40


def _camera_indices(frame_time, n_cam: int, record_start: float, record_stop: float) -> np.ndarray:
    """The standalone export's state-frame -> camera-frame map (export/build.py), restated so these
    tests do not need the `export` extra (av, pyarrow). test_the_window_map_is_the_exporters pins
    the two together wherever that extra is installed."""
    idx = np.round((np.asarray(frame_time) - record_start) * (n_cam / (record_stop - record_start)))
    return np.clip(idx, 0, n_cam - 1).astype(int)


# --------------------------------------------------------------------------- fixtures


def _leg_state(n: int, provenance: str) -> dict:
    """The state arrays of one leg, in one of the provenances a hand-off mixes.

    Ported from the monorepo's collect/_selftest.py:

    * ``captured`` - a TAMP leg from TipTop 1c6daf3, which wrote action_joint_velocity itself. It
                     must pass through VERBATIM, not be recomputed from its operands.
    * ``teleop``   - a live-teleop leg: its cmd_joint_velocity is the IK command, which already
                     IS the DROID action. Ground truth; passes through with no note.
    * ``legacy``   - a TAMP leg from before that capture, carrying the planner's feedforward
                     rad/s. The one that must be RECOMPUTED.
    * ``stub``     - cmd_joint_position/velocity are a zero placeholder: underivable.
    """
    jp = np.tile(np.arange(7, dtype=np.float32), (n, 1))
    if provenance == "stub":
        zeros = np.zeros((n, 7), np.float32)
        return {"joint_position": jp, "cmd_joint_position": zeros, "cmd_joint_velocity": zeros}
    # A commanded target the arm lags behind by a varying amount, so 5 * (cmd_jp - jp) varies.
    lag = (0.01 * (1 + np.arange(n, dtype=np.float32)))[:, None] * np.ones((1, 7), np.float32)
    cmd_jp = (jp + lag).astype(np.float32)
    droid_jv = (DROID_JV_GAIN * (cmd_jp - jp)).astype(np.float32)
    if provenance == "teleop":
        return {"joint_position": jp, "cmd_joint_position": cmd_jp, "cmd_joint_velocity": droid_jv}
    # The plan's feedforward rad/s: a different quantity, deliberately anti-correlated with the
    # identity so nothing could mistake one for the other.
    plan_jv = (-0.5 * droid_jv + 0.2).astype(np.float32)
    out = {"joint_position": jp, "cmd_joint_position": cmd_jp, "cmd_joint_velocity": plan_jv}
    if provenance == "captured":
        out["action_joint_velocity"] = droid_jv
    return out


def _write_leg(
    profile,
    timestamp: str,
    *,
    trajectory_id: str,
    source: str,
    provenance: str,
    t0: float,
    n: int = N,
    phase: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """One leg in the on-disk format: state, a _meta.json, and a (placeholder) camera clip."""
    directory = profile.status_dir("eval") / timestamp
    directory.mkdir(parents=True)
    frame_time = t0 + np.arange(n, dtype=np.float64) / FPS
    np.savez(
        directory / "robot_state.npz",
        gripper_position=np.zeros(n, np.float32),
        cmd_gripper=np.zeros(n, np.float32),
        frame_time=frame_time,
        **_leg_state(n, provenance),
        **(extra or {}),
    )
    meta = {
        "instruction": "put the toy in the box",
        "fps": FPS,
        "n_frames": n,
        "config_id": f"{source}/x",
        "timestamp": timestamp,
        "source": source,
        "cameras": {"exterior_image_1_left": "external_cam.mp4"},
        "record_start": float(frame_time[0]),
        "record_stop": float(frame_time[0]) + n / FPS,
        "trajectory_id": trajectory_id,
        "segment_source": source,
        **(phase or {}),
    }
    if provenance == "captured":
        meta["action_convention"] = "droid_joint_velocity"
    (directory / "_meta.json").write_text(json.dumps(meta))
    (directory / "external_cam.mp4").write_bytes(b"not really a video")
    return directory


def _hand_off(profile, trajectory_id: str, *legs: tuple) -> list[Path]:
    """Write one trajectory's legs, five minutes apart on the wall clock and in their names.

    Each leg is ``(source, provenance)`` or ``(source, provenance, {"phase": …, "extra": …})``.
    """
    return [
        _write_leg(
            profile,
            f"2026-05-05_00-{5 * i:02d}-00",
            trajectory_id=trajectory_id,
            source=source,
            provenance=provenance,
            t0=5000.0 + 300.0 * i,
            **(options[0] if options else {}),
        )
        for i, (source, provenance, *options) in enumerate(legs)
    ]


@pytest.fixture
def fake_video(monkeypatch):
    """Stand in for ffprobe/ffmpeg: every clip holds as many frames as its leg has states."""

    def leg_video_frames(leg, cameras, runtime_dir):
        with np.load(leg["dir"] / trajectories.STATE_FILE) as store:
            n = len(store["frame_time"])
        return n, {cam: n for cam in cameras}

    def concat_videos(legs, camera, leg_frames, dest, scratch, runtime_dir):
        dest.write_bytes(b"joined")
        return sum(leg_frames)

    monkeypatch.setattr(merge_mod, "_leg_video_frames", leg_video_frames)
    monkeypatch.setattr(merge_mod, "_concat_videos", concat_videos)


def _merged(result: dict) -> tuple[dict, dict]:
    directory = Path(result["dir"])
    with np.load(directory / trajectories.STATE_FILE) as store:
        arrays = {key: store[key] for key in store.files}
    return arrays, json.loads((directory / trajectories.META_FILE).read_text())


# --------------------------------------------------------------------------- the optional array


def test_a_captured_tamp_leg_merges_with_a_teleop_leg_that_has_no_action(profile, fake_video):
    """The hand-off TipTop 1c6daf3 produces: its TAMP leg carries action_joint_velocity, the
    teleop leg cannot. Refusing the array blocked the bump; dropping it would lose what the
    capture wrote. The teleop leg's IK command IS the DROID action, so it fills its own frames."""
    _hand_off(profile, "mix2", ("tamp", "captured"), ("teleop", "teleop"))

    result = merge_mod.merge(profile, "mix2")
    assert result["merged"] is True and result["n_legs"] == 2
    arrays, meta = _merged(result)

    action = arrays["action_joint_velocity"]
    assert action.shape == (2 * N, 7)
    assert action.dtype == np.float32
    assert np.array_equal(action[:N], _leg_state(N, "captured")["action_joint_velocity"])
    assert np.array_equal(action[N:], _leg_state(N, "teleop")["cmd_joint_velocity"])
    # The stored command is untouched: the export (still on cmd_joint_velocity) sees what it saw.
    assert np.array_equal(arrays["cmd_joint_velocity"][:N], _leg_state(N, "captured")["cmd_joint_velocity"])
    assert meta["action_convention"] == "droid_joint_velocity"
    assert meta["action_notes"] == {}
    assert result["action_notes"] == {}


def test_a_mixed_provenance_hand_off_resolves_every_leg_in_its_own_provenance(profile, fake_video):
    """Ported from the monorepo's _selftest [6]: three legs, three different right answers, and
    once they are concatenated no single answer would do."""
    _hand_off(profile, "mix3", ("tamp", "captured"), ("teleop", "teleop"), ("tamp", "legacy"))

    result = merge_mod.merge(profile, "mix3")
    arrays, meta = _merged(result)
    action = arrays["action_joint_velocity"]
    assert action.shape == (3 * N, 7)

    assert np.allclose(action[:N], _leg_state(N, "captured")["action_joint_velocity"]), "kept VERBATIM"
    assert np.allclose(action[N : 2 * N], _leg_state(N, "teleop")["cmd_joint_velocity"]), "ground truth"
    legacy = _leg_state(N, "legacy")
    identity = DROID_JV_GAIN * (legacy["cmd_joint_position"] - legacy["joint_position"])
    assert np.allclose(action[2 * N :], identity), "a pre-capture TAMP leg is RECOMPUTED"
    assert not np.allclose(action[2 * N :], legacy["cmd_joint_velocity"]), "NOT the plan's feedforward rad/s"

    # Only the recomputed leg is reported; the two that needed nothing are silent.
    assert list(meta["action_notes"]) == ["2026-05-05_00-10-00"]
    assert "RECOMPUTED" in meta["action_notes"]["2026-05-05_00-10-00"]
    assert meta["action_convention"] == "droid_joint_velocity"


def test_legs_that_never_recorded_the_action_merge_exactly_as_before(profile, fake_video):
    """Nothing is invented for a trajectory whose legs never carried the array: merging legacy
    legs gives the arrays it always gave, and no convention is claimed for an array that is not
    there."""
    _hand_off(profile, "old", ("tamp", "legacy"), ("teleop", "teleop"))

    arrays, meta = _merged(merge_mod.merge(profile, "old"))
    assert set(arrays) == {*merge_mod.STATE_KEYS, "video_time"}
    assert "action_convention" not in meta
    assert "action_notes" not in meta


def test_an_inherited_convention_is_not_left_claiming_an_array_that_is_absent(profile, fake_video):
    """The merged _meta.json starts as a copy of the primary leg's. A primary that (wrongly)
    declares a convention without the array must not pass that claim on."""
    primary, _ = _hand_off(profile, "claim", ("tamp", "legacy"), ("teleop", "teleop"))
    meta = json.loads((primary / "_meta.json").read_text())
    meta["action_convention"] = "droid_joint_velocity"
    (primary / "_meta.json").write_text(json.dumps(meta))

    _, merged_meta = _merged(merge_mod.merge(profile, "claim"))
    assert "action_convention" not in merged_meta


def test_an_action_array_of_the_wrong_length_is_refused(profile):
    short = {"extra": {"action_joint_velocity": np.zeros((N - 1, 7), np.float32)}}
    _hand_off(profile, "short", ("tamp", "legacy", short), ("teleop", "teleop"))
    legs = merge_mod.find_legs(profile, "short")
    with pytest.raises(MergeError, match="action_joint_velocity has 39 rows"):
        merge_mod._concat_state(legs, [N, N], FPS)


def test_widening_the_allow_list_did_not_open_it_to_everything(profile):
    """Ported from the monorepo's _selftest [7]: an unknown array is still a schema change."""
    unknown = {"extra": {"some_new_array": np.zeros(N, np.float32)}}
    _hand_off(profile, "odd", ("tamp", "captured"), ("teleop", "teleop", unknown))
    legs = merge_mod.find_legs(profile, "odd")
    with pytest.raises(MergeError, match="some_new_array"):
        merge_mod._concat_state(legs, [N, N], FPS)


@pytest.mark.parametrize(
    ("provenance", "source", "passes_through", "note"),
    [
        ("teleop", "teleop", True, None),
        ("legacy", "policy", True, None),
        ("legacy", "tamp", False, "RECOMPUTED"),
        ("stub", "tamp", True, "constant"),
    ],
)
def test_the_derivation_follows_the_declared_source(provenance, source, passes_through, note):
    arrays = _leg_state(N, provenance)
    value, why = merge_mod._derive_action_joint_velocity(arrays, source)
    assert np.array_equal(value, arrays["cmd_joint_velocity"]) is passes_through
    if note is None:
        assert why is None
    else:
        assert note in why


def test_a_tamp_leg_whose_target_is_a_copy_of_the_measurement_is_not_recomputed():
    """5 * (q - q) is zero: plausible-looking and wrong. Pass the stored array through, loudly."""
    arrays = _leg_state(N, "legacy")
    # The arm moves (so this is not the constant-placeholder case) and the "target" follows it.
    arrays["joint_position"] = arrays["joint_position"] + np.linspace(0, 1, N, dtype=np.float32)[:, None]
    arrays["cmd_joint_position"] = arrays["joint_position"].copy()
    value, why = merge_mod._derive_action_joint_velocity(arrays, "tamp")
    assert np.array_equal(value, arrays["cmd_joint_velocity"])
    assert "copy of the measured joint_position" in why


def test_an_arm_that_is_not_a_7_joint_fr3_is_passed_through_under_a_note():
    """DROID_JV_GAIN is the FR3's; rewriting another embodiment's action with it would be silent
    corruption under a note claiming it is correct."""
    n = 10
    arrays = {
        "joint_position": np.zeros((n, 12), np.float32),
        "cmd_joint_position": np.ones((n, 12), np.float32),
        "cmd_joint_velocity": np.full((n, 12), 0.3, np.float32),
    }
    value, why = merge_mod._derive_action_joint_velocity(arrays, "tamp")
    assert np.array_equal(value, arrays["cmd_joint_velocity"])
    assert "not a 7-joint arm" in why


# --------------------------------------------------------------------------- phases in segments[]


def _phase(index: int, total: int, description: str | None = None) -> dict:
    phase = {"phase_index": index, "n_phases": total}
    if description is not None:
        phase["phase_description"] = description
    return {"phase": phase}


def test_segments_say_which_phase_each_leg_recorded(profile, fake_video):
    """τ = ((τ_1, φ_1), …): each stretch of the merged episode maps back to its phase. The
    primary leg's own phase must not be left at the top, labelling the whole trajectory."""
    _hand_off(
        profile,
        "phased",
        ("tamp", "captured", _phase(0, 3, "put the bread on the plate")),
        ("teleop", "teleop", _phase(1, 3, "open the box")),
        ("tamp", "captured", _phase(2, 3, "put the bread in the box")),
    )

    result = merge_mod.merge(profile, "phased")
    _, meta = _merged(result)
    assert [s["phase_index"] for s in meta["segments"]] == [0, 1, 2]
    assert [s["phase_description"] for s in meta["segments"]] == [
        "put the bread on the plate",
        "open the box",
        "put the bread in the box",
    ]
    assert [s["source"] for s in meta["segments"]] == ["tamp", "teleop", "tamp"]
    assert meta["segments"][1]["config_id"] == "teleop/x"
    assert result["segments"] == meta["segments"]
    # One leg's phase is not the trajectory's; the plan's length is.
    assert "phase_index" not in meta
    assert "phase_description" not in meta
    assert meta["n_phases"] == 3


def test_legs_without_phases_get_no_invented_ones(profile, fake_video):
    """A leg from before phase planning has no phase; numbering it by position would pair frames
    with the wrong subgoal the moment one leg was skipped."""
    _hand_off(profile, "partial", ("tamp", "legacy", _phase(0, 2)), ("teleop", "teleop"))

    _, meta = _merged(merge_mod.merge(profile, "partial"))
    assert meta["segments"][0]["phase_index"] == 0
    assert "phase_index" not in meta["segments"][1]
    assert "n_phases" not in meta["segments"][1]
    assert meta["n_phases"] == 2


def test_legs_that_disagree_on_the_plan_length_leave_it_to_the_segments(profile, fake_video):
    _hand_off(profile, "disagree", ("tamp", "legacy", _phase(0, 2)), ("teleop", "teleop", _phase(1, 3)))
    _, meta = _merged(merge_mod.merge(profile, "disagree"))
    assert "n_phases" not in meta
    assert [s["n_phases"] for s in meta["segments"]] == [2, 3]


# --------------------------------------------------------------------------- the teleop driver


@pytest.fixture
def driver(monkeypatch):
    """The teleop driver, imported the way the DROID interpreter runs it: as a script beside its
    helpers. Its module-level sys.path edit is rolled back afterwards."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("tandem_teleop_driver_under_test", teleop_pkg.driver_path())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Clock:
    """time.time/time.sleep for the record loop: every sleep advances exactly as asked, so the
    loop runs at precisely CONTROL_HZ and frame_time is known in advance."""

    def __init__(self, start: float):
        self.now = start

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _Env:
    def __init__(self):
        self.q = np.zeros(7)

    def get_observation(self):
        image = np.zeros((4, 6, 4), np.uint8)
        return {
            "image": {"EXT_left": image, "HAND_left": image},
            "robot_state": {"joint_positions": self.q.copy(), "gripper_position": 0.0},
        }

    def step(self, action):
        self.q = self.q + 0.02
        return {"joint_velocity": np.full(7, 0.1), "joint_position": self.q + 0.02}


class _Policy:
    def forward(self, obs):
        return np.zeros(7), 0.0


def _record(driver, monkeypatch, parent: Path, n_frames: int, **flags) -> Path:
    """One teleop leg recorded by the real record loop, with the robot, cameras and clock faked."""
    commands = iter([None] * n_frames + [{"cmd": "end"}])
    monkeypatch.setattr(driver, "_poll_line", lambda: next(commands))
    monkeypatch.setattr(driver, "time", _Clock(1_800_000_000.0))
    monkeypatch.setattr(driver, "_write_video", lambda frames, path: Path(path).write_bytes(b"clip"))

    ep_dir = parent / "2026-05-05_00-05-00"
    ep_dir.mkdir()
    args = driver.Args(external_camera_id="EXT", hand_camera_id="HAND", trajectory_id="abc", **flags)
    n, quit_session = driver.record_episode(_Env(), _Policy(), ep_dir, args, events="")
    assert n == n_frames and quit_session is False
    return ep_dir


def test_the_driver_stamps_the_phase_it_was_told(driver, monkeypatch, tmp_path):
    phase = {"phase_index": 1, "n_phases": 3, "phase_description": "open the box"}
    ep_dir = _record(driver, monkeypatch, tmp_path, 12, **phase)
    meta = json.loads((ep_dir / "_meta.json").read_text())
    assert meta["phase_index"] == 1
    assert meta["n_phases"] == 3
    assert meta["phase_description"] == "open the box"
    assert meta["segment_source"] == "teleop"
    assert meta["trajectory_id"] == "abc"


def test_phase_zero_is_a_phase(driver, monkeypatch, tmp_path):
    ep_dir = _record(driver, monkeypatch, tmp_path, 5, phase_index=0)
    meta = json.loads((ep_dir / "_meta.json").read_text())
    assert meta["phase_index"] == 0
    assert "n_phases" not in meta and "phase_description" not in meta


def test_without_phase_flags_the_meta_is_what_it_always_was(driver, monkeypatch, tmp_path):
    ep_dir = _record(driver, monkeypatch, tmp_path, 5)
    meta = json.loads((ep_dir / "_meta.json").read_text())
    assert not set(meta) & {"phase_index", "n_phases", "phase_description"}
    assert driver.Args().phase_index is None


def test_the_recording_window_covers_every_frame_not_all_but_one(driver, monkeypatch, tmp_path):
    """The driver writes one image per state frame, so state frame i IS camera frame i. Every
    consumer maps a frame through the window, and a window ending AT the last frame's timestamp
    spans n-1 periods: from mid-leg on, each state frame was paired with the NEXT image."""
    n = 30
    ep_dir = _record(driver, monkeypatch, tmp_path, n)
    meta = json.loads((ep_dir / "_meta.json").read_text())
    with np.load(ep_dir / "robot_state.npz") as store:
        frame_time = store["frame_time"]

    assert meta["record_start"] == pytest.approx(frame_time[0])
    assert meta["record_stop"] == pytest.approx(frame_time[-1] + 1.0 / FPS)
    # The standalone export's map is now the identity...
    idx = _camera_indices(frame_time, n, meta["record_start"], meta["record_stop"])
    assert np.array_equal(idx, np.arange(n))
    # ...where the old window put the second half of the leg one image late.
    old = _camera_indices(frame_time, n, float(frame_time[0]), float(frame_time[-1]))
    assert np.count_nonzero(old != np.arange(n)) >= n // 2 - 1


def test_a_merged_teleop_leg_lands_on_its_own_images(driver, monkeypatch, profile):
    """The same fix, seen through the merge's video_time: frame i of a teleop leg sits at
    (frames of earlier legs + i) / fps in the joined clip, exactly."""
    n = 30
    _hand_off(profile, "abc", ("tamp", "legacy"))
    teleop_leg = _record(driver, monkeypatch, profile.status_dir("eval"), n)

    legs = merge_mod.find_legs(profile, "abc")
    assert [leg["dir"] for leg in legs][1] == teleop_leg
    state = merge_mod._concat_state(legs, [N, n], FPS)
    video_time = state["arrays"]["video_time"]
    assert np.allclose(video_time[N:] * FPS, N + np.arange(n))


def test_the_window_map_is_the_exporters():
    build = pytest.importorskip("tandem.export.build", reason="needs the `export` extra (av, pyarrow)")
    rng = np.random.default_rng(3)
    frame_time = 1000.0 + np.sort(rng.uniform(0, 5, size=70))
    for start, stop in [(999.5, 1005.5), (1000.0, 1004.0), (float(frame_time[0]), float(frame_time[-1]))]:
        ours = _camera_indices(frame_time, 80, start, stop)
        assert np.array_equal(build._camera_indices(frame_time, 80, start, stop), ours)


def test_the_window_of_a_leg_that_ran_slow_still_ends_on_its_last_frame(driver):
    """Nominal 1/15 s would under-reach a loop that really ran at 12 Hz; one MEAN period past
    the last frame is right at whatever rate the loop ran."""
    frame_time = 1000.0 + np.arange(24) / 12.0
    stop = driver.recording_window_stop(frame_time)
    idx = _camera_indices(frame_time, len(frame_time), float(frame_time[0]), stop)
    assert np.array_equal(idx, np.arange(len(frame_time)))


# --------------------------------------------------------------------------- completeness


def _leg_on_disk(parent: Path, *, meta: dict | None, clips=("external_cam.mp4",), plan=False, state=True):
    directory = parent / "leg"
    directory.mkdir()
    if state:
        np.savez(directory / "robot_state.npz", frame_time=np.arange(3, dtype=np.float64))
    if meta is not None:
        (directory / "_meta.json").write_text(json.dumps(meta))
    for clip in clips:
        (directory / clip).write_bytes(b"clip")
    if plan:
        (directory / "tiptop_plan.json").write_text("{}")
    return directory


def test_a_leg_from_a_planner_that_is_not_tiptop_is_complete_without_tiptops_plan(tmp_path):
    meta = {"segment_source": "tamp", "cameras": {"exterior_image_1_left": "external_cam.mp4"}}
    directory = _leg_on_disk(tmp_path, meta=meta)
    assert trajectories.is_complete(directory)
    assert trajectories.read(directory).complete


def test_a_teleop_leg_is_complete(tmp_path):
    meta = {"segment_source": "teleop", "cameras": {"a": "external_cam.mp4", "b": "hand_cam.mp4"}}
    directory = _leg_on_disk(tmp_path, meta=meta, clips=("external_cam.mp4", "hand_cam.mp4"))
    assert trajectories.read(directory).complete


def test_a_leg_missing_a_clip_its_meta_names_is_incomplete(tmp_path):
    meta = {"cameras": {"a": "external_cam.mp4", "b": "hand_cam.mp4"}}
    directory = _leg_on_disk(tmp_path, meta=meta, plan=True)
    assert not trajectories.is_complete(directory)


def test_a_leg_without_state_or_without_any_video_is_incomplete(tmp_path):
    directory = _leg_on_disk(tmp_path, meta={"cameras": {}}, state=False)
    assert not trajectories.is_complete(directory)
    (directory / "robot_state.npz").write_bytes(b"")
    (directory / "external_cam.mp4").unlink()
    assert not trajectories.is_complete(directory), "no clip named and none on disk"


def test_old_data_without_a_meta_file_is_still_complete_on_its_plan(tmp_path):
    """Written before _meta.json existed: what made it complete then still does."""
    directory = _leg_on_disk(tmp_path, meta=None, plan=True)
    assert trajectories.is_complete(directory)
    (directory / "tiptop_plan.json").unlink()
    assert not trajectories.is_complete(directory), "no _meta.json and no plan: nothing says what it is"


def test_a_camera_map_cannot_point_outside_the_leg(tmp_path):
    directory = _leg_on_disk(tmp_path, meta={"cameras": {"a": "../elsewhere.mp4"}})
    (tmp_path / "elsewhere.mp4").write_bytes(b"clip")
    # The escaping name is ignored, so the leg is judged by the clips actually beside it.
    assert trajectories.is_complete(directory)
    (directory / "external_cam.mp4").unlink()
    assert not trajectories.is_complete(directory)


def test_a_merged_hand_off_is_complete(profile, fake_video):
    _hand_off(profile, "whole", ("tamp", "captured"), ("teleop", "teleop"))
    result = merge_mod.merge(profile, "whole")
    traj = trajectories.read(Path(result["dir"]))
    assert traj.complete
    assert not traj.has_plan
