"""`tandem traj` — inspect collected trajectories."""

from __future__ import annotations

import json
import shutil
import sys

import typer

from tandem.cli import theme
from tandem.core import profiles, series, trajectories
from tandem.core import settings as settings_mod
from tandem.core.errors import TandemError

app = typer.Typer(no_args_is_help=True, help="Inspect collected trajectories.")

_STATUS_STYLE = {"success": "ok", "failure": "err", "eval": "warn"}

# Every command below reads a profile only to find its trajectories, so each loads it read-only
# (``require_installed=False``): a profile collected with a planner or executor this machine does not
# have is still browsed, relabeled, merged and copied here -- a laptop is where that is done.


def is_human_segment(segment: dict) -> bool:
    """Whether a merged trajectory's segment is a human phase's leg, whoever carried it out.

    Any source other than the planner's ("tamp") is a human executor's -- teleop, or a policy standing in
    for the person (``executors.SEGMENT_SOURCES``). Counting only "teleop" showed a policy executor's
    legs as the planner's, and its hand-offs as none.
    """
    return str(segment.get("source") or "tamp") != "tamp"


def segment_label(source: str) -> str:
    """How a segment's source is shown: TAMP, human, or human with the executor kind (policy)."""
    if source in ("tamp", "", None):
        return "[accent]TAMP[/accent]"
    if source == "teleop":
        return "[violet]human[/violet]"
    return f"[violet]human ({source})[/violet]"


@app.command("list", help="List a profile's trajectories, newest first.")
def list_trajectories(
    profile_name: str = typer.Argument(None, help="Profile name (default: the active one)."),
    status: str = typer.Option(None, "--status", "-s", help="Only eval, success or failure."),
    limit: int = typer.Option(30, "--limit", "-n", help="How many to show (0 for all)."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    profile = profiles.load(profile_name, require_installed=False)
    if status and status not in profiles.STATUSES:
        raise TandemError(f"Unknown status {status!r}.", hint=f"One of: {', '.join(profiles.STATUSES)}")

    items = trajectories.list_all(profile, status=status)
    shown = items[:limit] if limit else items

    if as_json:
        typer.echo(json.dumps([t.to_dict() for t in shown], indent=2))
        return

    if not items:
        theme.info(f"No trajectories in profile {profile.name!r} yet.")
        theme.next_steps([(f"tandem collect {profile.name}", "run a collection session")])
        return

    table = theme.table("timestamp", "", "frames", "length", "cams", "task", "")
    for traj in shown:
        style = _STATUS_STYLE.get(traj.status, "faint")
        chip = f"[{style}]{traj.status}[/{style}]"
        flags = []
        if traj.segments and len(traj.segments) > 1:
            human = sum(1 for s in traj.segments if is_human_segment(s))
            flags.append(f"[violet]hand-off ×{human}[/violet]")
        if not traj.complete:
            flags.append("[faint]incomplete[/faint]")
        if traj.settled:
            # The chip says where it is filed; this says what the method decided it is.
            stage = f" at {traj.failure_stage}" if traj.failure_stage else ""
            flags.append(f"[warn]{traj.settled}{stage}[/warn]")
        table.add_row(
            traj.id,
            chip,
            str(traj.n_frames) if traj.n_frames else "[faint]—[/faint]",
            f"{traj.duration_s:.1f}s" if traj.duration_s else "[faint]—[/faint]",
            str(len(traj.cameras)),
            _truncate(traj.instruction, 38),
            "  ".join(flags),
        )
    theme.console().print(table)

    counts = trajectories.counts(profile)
    theme.info(
        f"{counts['success']} success  ·  {counts['failure']} failure  ·  {counts['eval']} unlabeled",
        f"{len(items)} total in {profile.name}",
    )
    if limit and len(items) > limit:
        theme.info(f"showing {limit} of {len(items)}", "use --limit 0 for all")


@app.command("show", help="Show one trajectory in detail.")
def show(
    traj_id: str = typer.Argument(..., help="Timestamp id, or a unique prefix."),
    profile_name: str = typer.Option(None, "--profile", "-p", help="Profile name."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    profile = profiles.load(profile_name, require_installed=False)
    traj = trajectories.find(profile, traj_id)

    if as_json:
        payload = traj.to_dict()
        payload["summary"] = series.summary(traj)
        payload["meta"] = traj.meta
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    style = _STATUS_STYLE.get(traj.status, "faint")
    theme.blank()
    theme.heading(traj.id, f"[{style}]{traj.status}[/{style}]")
    theme.kv(
        [
            ("task", traj.instruction),
            ("frames", f"{traj.n_frames} at {traj.fps} Hz  ({traj.duration_s:.1f}s)"),
            ("cameras", [trajectories.CAMERA_LABELS.get(c, c) for c in traj.cameras]),
            ("plan", "recorded" if traj.has_plan else None),
            ("outcome", _outcome_line(traj)),
            ("path", traj.path),
        ]
    )

    if traj.segments and len(traj.segments) > 1:
        theme.blank()
        theme.heading("hand-off legs", f"{len(traj.segments)} segments merged into one trajectory")
        seg_table = theme.table("#", "source", "timestamp", "frames", "video")
        for i, seg in enumerate(traj.segments):
            label = segment_label(str(seg.get("source") or "tamp"))
            start, stop = seg.get("video_start"), seg.get("video_stop")
            span = f"{start:.1f}–{stop:.1f}s" if isinstance(start, (int, float)) and isinstance(stop, (int, float)) else "—"
            seg_table.add_row(str(i + 1), label, str(seg.get("timestamp", "")), str(seg.get("n_frames", "")), span)
        theme.console().print(seg_table)

    stats = series.summary(traj)
    if stats:
        theme.blank()
        theme.heading("motion")
        rows: list[tuple[str, object]] = []
        if "joint_travel_rad" in stats:
            rows.append(("joint travel", f"{stats['joint_travel_rad']:.2f} rad total"))
        if "gripper_events" in stats:
            rows.append(("gripper events", stats["gripper_events"]))
        if "peak_cmd_velocity" in stats:
            rows.append(("peak commanded vel", f"{stats['peak_cmd_velocity']:.2f}"))
        if "nonidle_kept" in stats:
            kept = stats["nonidle_kept"]
            pct = (kept / traj.n_frames * 100) if traj.n_frames else 0
            rows.append(("non-idle frames", f"{kept} of {traj.n_frames}  ({pct:.0f}% survive training's filter)"))
        if stats.get("frac_clipped"):
            rows.append(("velocity at rail", f"{stats['frac_clipped']:.1%} of commanded elements clipped to ±1"))
        theme.kv(rows)

    steps = []
    recorder, _ = recorded_by(traj, profile)
    if can_replay(recorder):
        steps.append((f"tandem traj open {traj.id}", f"replay it in {recorder}'s own viewer"))
    steps.append(("tandem ui", "videos and charts in the browser"))
    theme.next_steps(steps)


def recorded_by(traj, profile) -> tuple[str, str]:
    """(the planner that recorded ``traj``, how that is known).

    Not simply the profile's planner: `tandem planners use` switches a profile and leaves its earlier
    trajectories where they are, and a leg handed to another planner's viewer is either refused or,
    worse, read as a rollout it is not. In order: the planner a leg says recorded it (``planner`` in
    _meta.json), the one the trial's phase record names (hitl.json), the recorder a leg names as its
    ``source`` when that is a planner this machine knows, and only then the profile's.
    """
    from tandem.planners import registry

    known = set(registry.available())
    meta = traj.meta or {}
    if isinstance(meta.get("planner"), str) and meta["planner"]:
        return meta["planner"], "the trajectory's _meta.json"
    try:
        stated = trajectories.read_hitl(traj.path).get("planner")
    except Exception:
        stated = None
    if isinstance(stated, str) and stated:
        return stated, "its hitl.json"
    source = meta.get("source")
    if isinstance(source, str) and source in known:
        return source, "the recorder named in its _meta.json"
    return profile.planner.backend, f"profile {profile.name!r}"


def can_replay(planner: str) -> bool:
    """Whether ``planner`` has a viewer of its own for `tandem traj open`. A planner written with the SDK
    always HAS a replay method -- the base one, which refuses -- so that one does not count."""
    from tandem.planners import registry

    try:
        factory = registry.factory(planner)
    except TandemError:
        return False
    hook = getattr(factory, "replay", None)
    if not callable(hook):
        return False
    from tandem.planners.sdk import Planner

    if isinstance(factory, type) and issubclass(factory, Planner):
        return getattr(factory.replay, "__func__", None) is not getattr(Planner.replay, "__func__", None)
    return True


@app.command("relabel", help="Move a trajectory between success, failure and eval.")
def relabel(
    traj_id: str = typer.Argument(..., help="Timestamp id, or a unique prefix."),
    status: str = typer.Argument(..., help="success | failure | eval"),
    profile_name: str = typer.Option(None, "--profile", "-p", help="Profile name."),
    force: bool = typer.Option(
        False,
        "--force",
        help="File a trial the method excluded (or ended part-way) as a success anyway. Its hitl.json "
        "says it was overruled.",
    ),
) -> None:
    profile = profiles.load(profile_name, require_installed=False)
    traj = trajectories.find(profile, traj_id)
    old = traj.status
    updated = trajectories.relabel(profile, traj, status, force=force)
    theme.ok(f"{updated.id}: {old} → {status}", str(updated.path))
    if force and traj.settled and status == "success":
        theme.warn(f"overruled: the method had settled it as {traj.settled}", "recorded in its hitl.json")


@app.command("rm", help="Delete a trajectory from disk.")
def remove(
    traj_id: str = typer.Argument(..., help="Timestamp id, or a unique prefix."),
    profile_name: str = typer.Option(None, "--profile", "-p", help="Profile name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    profile = profiles.load(profile_name, require_installed=False)
    traj = trajectories.find(profile, traj_id)
    size_mb = trajectories.read(traj.path, with_size=True).size_bytes / 1e6
    theme.warn(f"{traj.id} ({traj.status}, {size_mb:.0f} MB) will be deleted. There is no undo.")
    if not yes and not typer.confirm("Delete it?", default=False):
        raise typer.Abort()
    trajectories.delete(profile, traj)
    theme.ok(f"Deleted {traj.id}")


@app.command("open", help="Replay a trajectory in its planner's own viewer.")
def open_(
    traj_id: str = typer.Argument(..., help="Timestamp id, or a unique prefix."),
    profile_name: str = typer.Option(None, "--profile", "-p", help="Profile name."),
) -> None:
    """The viewer is the planner's -- the one that RECORDED the trajectory, which is not always the one the
    profile plans with now (`recorded_by`). TiPToP's replays the saved plan in Rerun, inside its runtime,
    since it needs cuRobo and cuTAMP to load the robot model and the TAMP scene."""
    from tandem.planners import registry

    profile = profiles.load(profile_name, require_installed=False)
    traj = trajectories.find(profile, traj_id)
    cfg = settings_mod.load()
    recorder, known_from = recorded_by(traj, profile)
    if recorder != profile.planner.backend:
        theme.info(f"Recorded by {recorder} ({known_from}); profile {profile.name!r} plans with "
                   f"{profile.planner.backend} now")
    theme.busy(f"Opening {traj.id} with {recorder}", str(traj.path))
    registry.replay(recorder, traj.path, settings=cfg)


@app.command("path", help="Print a trajectory's directory.")
def path_(
    traj_id: str = typer.Argument(..., help="Timestamp id, or a unique prefix."),
    profile_name: str = typer.Option(None, "--profile", "-p", help="Profile name."),
) -> None:
    profile = profiles.load(profile_name, require_installed=False)
    typer.echo(str(trajectories.find(profile, traj_id).path))


@app.command("merge", help="Join the legs of a teleop hand-off into one trajectory.")
def merge(
    trajectory_id: str = typer.Argument(
        None, help="Lineage id. Omit to merge every trajectory that has unmerged legs."
    ),
    profile_name: str = typer.Option(None, "--profile", "-p", help="Profile name."),
    status: str = typer.Option(None, "--status", help="Status for the merged trajectory."),
) -> None:
    """Normally automatic — a session merges as soon as you label the trajectory. Use this to
    retry after a failure; the merge never partially writes, so the legs are always intact."""
    from tandem.core import merge as merge_mod
    from tandem.planners import registry

    profile = profiles.load(profile_name, require_installed=False)
    cfg = settings_mod.load()
    targets = [trajectory_id] if trajectory_id else merge_mod.pending_trajectory_ids(profile)

    if not targets:
        theme.info("Nothing to merge — no trajectory has more than one unmerged leg.")
        return

    for target in targets:
        try:
            result = merge_mod.merge(
                profile, target, status=status, tools_dir=registry.tools_dir(profile.planner.backend, cfg)
            )
        except merge_mod.MergeError as exc:
            raise TandemError(
                str(exc), hint="The legs are untouched on disk; fix the cause and re-run."
            ) from exc
        if result.get("merged"):
            theme.ok(f"Merged {result['n_legs']} legs · {result['n_frames']} frames", result["dir"])
            for name in result.get("legs_skipped") or []:
                theme.warn(f"leg {name} captured no state and is not in the trajectory")
            for name in result.get("proportional_fallback_legs") or []:
                theme.warn(f"leg {name} had no recording window; its frames were aligned proportionally")
            for camera in result.get("cameras_dropped") or []:
                theme.warn(f"{camera} is missing from at least one leg and was dropped")
        else:
            theme.info(f"{target}: {result.get('reason')}")


@app.command("copy", help="Copy a trajectory into another profile.")
def copy(
    traj_id: str = typer.Argument(..., help="Timestamp id, or a unique prefix."),
    dest_profile: str = typer.Argument(..., help="Destination profile."),
    profile_name: str = typer.Option(None, "--profile", "-p", help="Source profile."),
) -> None:
    source = profiles.load(profile_name, require_installed=False)
    dest = profiles.load(dest_profile, require_installed=False)
    traj = trajectories.find(source, traj_id)
    target = dest.status_dir(traj.status) / traj.id
    if target.exists():
        raise TandemError(f"{target} already exists.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(traj.path, target)
    theme.ok(f"Copied {traj.id} to {dest.name}", str(target))


def _outcome_line(traj) -> str | None:
    """How the phase record says the trial ended, when there is one, with what the method settled."""
    if not traj.outcome and not traj.settled:
        return None
    line = traj.outcome or "not labeled"
    if traj.failure_stage:
        line += f" at {traj.failure_stage}"
    if traj.settled:
        line += "  (settled by the method: not a demonstration)"
    return line


def _truncate(text: str, width: int) -> str:
    text = text or ""
    return text if len(text) <= width else text[: width - 1] + "…"


if __name__ == "__main__":
    sys.exit(app())
