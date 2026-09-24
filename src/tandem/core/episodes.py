"""Where a trial's legs live on disk, and how they become one episode.

A trial is recorded as several legs: one for each robot phase, from the planner, and one for each
hand-off, from the teleop driver. Each leg has its own directory under
``<profile>/trajectories/eval/``, and every leg carries the trajectory id the session minted. This
module handles the disk side of that. It allocates a leg's directory, takes back the directories
nothing was recorded into, and, once the operator has labeled the trial, files the trial under
success/ or failure/. A trial the method EXCLUDED (a human phase that never verified) is filed under
failure/ with no label, and its record says ``excluded: true``. Either way the legs are then merged
into one episode, tau = ((tau_1, phi_1), .., (tau_N, phi_N)) in the paper's terms (Sec. IV-E), and
the phase record is written beside it.

Nothing here holds a session. Each function is given the profile it files into and somewhere to log.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tandem.core.profiles import Profile
    from tandem.planning.plan import PhasePlan


class LegDirs:
    """Where one session allocates its legs, and where it moves the ones nothing was recorded into."""

    def __init__(self, profile: Profile, session_dir: Path, *, log: Callable[[str], None]) -> None:
        self.profile = profile
        self.session_dir = Path(session_dir)
        self._log = log

    def new(self) -> Path:
        """A fresh directory for one leg, where a trajectory actually lives.

        Under ``<profile>/trajectories/eval/``, not the session's scratch directory: that is where
        the teleop driver writes its legs, where ``merge.find_legs`` looks for them, and where
        ``tandem traj list`` reads. A leg written anywhere else is invisible to all three — it
        would never be merged, never be labeled, and never appear in the dataset.

        Named with a wall-clock stamp like every other trajectory, with a suffix only if two legs
        start inside the same second, which second-resolution names otherwise silently collide on.
        """
        import datetime

        from tandem.core.profiles import STATUSES

        root = self.profile.status_dir("eval")
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        def taken(candidate: str) -> bool:
            # Across EVERY status, not just eval: a labeled leg moves to success/ or failure/ and
            # frees its name here, and a directory name IS a trajectory's id — two episodes sharing
            # one is a collision that surfaces much later, in a listing or a merge that overwrites.
            return any((self.profile.status_dir(status) / candidate).exists() for status in STATUSES)

        name, suffix = stamp, 1
        while taken(name):
            suffix += 1
            name = f"{stamp}-{suffix}"
        directory = root / name
        directory.mkdir(parents=True)
        return directory

    def retire(self, leg_dir: Path) -> None:
        """Take a leg directory back out of the dataset if nothing was recorded into it.

        Every pass allocates one under ``eval/`` because perception writes into it before anyone
        knows whether a recording will follow — and for a human phase, or a phase the planner could
        not plan, none ever does. Left there, each is listed by ``tandem traj list`` and the web UI
        as a zero-frame episode, so a session's worth of them buries the real ones.

        Moved rather than deleted: the perception dump is the first thing worth looking at when a
        phase went wrong, and the session directory is where post-mortem material lives.
        """
        import shutil

        try:
            if (leg_dir / "_meta.json").is_file():
                return  # something was recorded here; it is a real leg
            keep = self.session_dir / "perception" / leg_dir.name
            keep.parent.mkdir(parents=True, exist_ok=True)
            if keep.exists():
                shutil.rmtree(keep, ignore_errors=True)
            shutil.move(str(leg_dir), str(keep))
        except OSError as exc:
            self._log(f"could not tidy away the unused leg directory {leg_dir.name}: {exc}")


def merge_trajectory(
    profile: Profile,
    trajectory_id: str,
    status: str | None,
    plan: PhasePlan | None = None,
    *,
    tools_dir: Path | None,
    vlm_dir: Path | None,
    log: Callable[[str], None],
    emit: Callable[[dict], None],
    reason: str | None = None,
) -> None:
    """Join a task's legs into one trajectory, in the background.

    Every trial with legs on disk comes through here, an excluded one included: excluded means kept
    out of the dataset, not deleted, and its legs are what a reader auditing the exclusion needs.
    ``reason`` is why the phase loop ended the trial, when it did (``TrialOutcome.reason``).

    Failure here must never take down a session: the legs are untouched on disk (the merge
    never partially writes) and `tandem traj merge` can retry once the cause is fixed.
    """
    from tandem.core import merge as merge_mod

    episode_dir = promote_primary_leg(profile, trajectory_id, status, log=log)
    try:
        result = merge_mod.merge(profile, trajectory_id, status=status, tools_dir=tools_dir)
    except Exception as exc:
        log(f"could not merge trajectory {trajectory_id}: {exc}")
        log(f"the legs are intact; retry with: tandem traj merge {trajectory_id}")
        write_phase_record(plan, episode_dir, status=status, vlm_dir=vlm_dir, log=log, reason=reason)
        return
    # `dir` is absent when the merge declined for want of state data, so fall back to the leg
    # the label was filed against — the record has to land somewhere a reader will open.
    # Not `Path(...) or episode_dir`: Path("") is PosixPath("."), which is truthy, so an
    # absent `dir` would file the record in the working directory.
    merged_dir = Path(result["dir"]) if result.get("dir") else episode_dir
    write_phase_record(plan, merged_dir, status=status, vlm_dir=vlm_dir, log=log, reason=reason)
    if result.get("merged"):
        log(
            f"merged {result['n_legs']} legs into one trajectory "
            f"({result['n_frames']} frames) at {result['dir']}"
        )
        emit({"type": "merged", **result})


def promote_primary_leg(
    profile: Profile, trajectory_id: str, status: str | None, *, log: Callable[[str], None]
) -> Path | None:
    """Move the leg the label refers to out of ``eval/`` and into ``success/`` or ``failure/``.

    Every leg is recorded into ``eval/`` — it is a staging bucket, and a teleop leg stays there
    unlabeled because the verdict belongs to the whole task and is given once. The label has to
    land somewhere, though, and for a task that produced only ONE leg the merge will decline to
    do anything ("single leg"), so nothing else would ever move it.

    The leg chosen is the one ``merge`` will treat as primary, so a merge that follows relocates
    the joined trajectory rather than leaving a labeled leg and an unlabeled merge behind.
    """
    import shutil

    from tandem.core import merge as merge_mod

    if not status:
        return None
    try:
        legs = merge_mod.find_legs(profile, trajectory_id)
    except Exception:
        return None
    if not legs:
        return None
    primary = next((leg for leg in legs if leg["source"] == "tamp"), legs[0])
    if primary["status"] == status:
        return primary["dir"]
    destination = profile.status_dir(status) / primary["dir"].name
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(primary["dir"]), str(destination))
    except OSError as exc:
        log(f"could not file the episode under {status}: {exc}")
        return primary["dir"]
    return destination


def trial_outcome(outcome: str | None, status: str | None) -> dict:
    """How the trial ended, as ``hitl.json`` states it: the loop's word, then the operator's label.

    ``outcome`` is what the phase loop recorded (``PhasePlan.set_outcome``), None for a trial that
    ran to the end. ``status`` is where the episode was filed, ``success`` or ``failure``: the
    operator's label, or ``failure`` for a trial filed without one.

    An ``excluded`` or ``aborted`` trial keeps that outcome whatever it was filed under. Excluded is
    the whole point of the record -- such a trial is filed under failure/ WITHOUT a label prompt, and
    reading the filing as its outcome would put it back among the ordinary failures a dataset keeps.
    Any other trial takes the label when there is one, since the label is the verdict the operator
    was asked for (with ``on_verification_failure: label`` it overrules the check on purpose), and
    otherwise the loop's own word.
    """
    final = outcome if outcome in ("excluded", "aborted") or not status else status
    return {"outcome": final, "excluded": final == "excluded", "filed_under": status}


def write_phase_record(
    plan: PhasePlan | None,
    directory: Path | None,
    *,
    status: str | None = None,
    vlm_dir: Path | None,
    log: Callable[[str], None],
    reason: str | None = None,
) -> None:
    """Drop the plan, the invented predicates and every verdict beside the finished episode.

    Written AFTER the merge and into the merged directory, not into a leg. The merge surfaces
    only the first planner leg's extra files, and that copy is the earliest snapshot — a low
    phase index and no verifications at all — so a record written per leg reads as though the
    task barely started.

    ``status`` is where the episode was filed. It settles the record's ``outcome`` for a trial the
    loop did not end itself (``trial_outcome``); ``failure_stage`` is always the loop's, the stage
    at which it ended the attempt, whatever the label said afterwards. ``reason`` is the loop's own
    account of why it ended the attempt, written as ``outcome_reason`` when there is one: the
    failure stage says where a trial stopped, and this says what was wrong there.
    """
    if plan is None or directory is None or not Path(directory).is_dir():
        return
    directory = Path(directory)
    record = plan.to_json()
    record.update(trial_outcome(record.get("outcome"), status))
    if reason:
        record["outcome_reason"] = reason
    try:
        (directory / "hitl.json").write_text(json.dumps(record, indent=2, default=str))
    except OSError as exc:
        log(f"could not write the phase record: {exc}")

    # The images and replies belong with the record they explain. Copied rather than moved, so
    # a failure here cannot destroy the only copy of the trail.
    source = vlm_dir
    if source is not None and Path(source).is_dir():
        import shutil

        try:
            shutil.copytree(source, directory / "vlm", dirs_exist_ok=True)
        except OSError as exc:
            log(f"could not file the model audit trail: {exc}")
