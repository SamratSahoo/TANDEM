"""`tandem doctor` — is everything tandem needs present and working?"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from tandem.cli import theme
from tandem.core import probe, profiles, render
from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError, TandemError

GROUP_TITLES = {
    "core": "environment",
    "runtime": "runtime",
    "gpu": "gpu",
    "credentials": "credentials",
    "hardware": "hardware",
    "profile": "profile",
}


def collect_checks(*, profile_name: str | None = None, probe_hardware: bool = True) -> list[probe.Check]:
    cfg = settings_mod.load()
    checks: list[probe.Check] = [probe.check_python(), probe.check_platform()]

    runtime_check, runtime_ready, runtime_root, active = _runtime_check(profile_name, cfg)
    checks.append(probe.check_disk(runtime_root or cfg.resolved_runtime_dir()))
    checks.append(probe.check_pixi())
    checks.append(probe.check_ffmpeg())
    checks.append(runtime_check)
    checks.extend(_other_planner_checks(active, cfg))

    checks.append(probe.check_nvidia_driver())
    checks.append(probe.check_cuda_runtime())
    checks.append(probe.check_nvcc())

    # A missing key blocks collection, but on a machine that cannot collect anyway it is only
    # worth a note — a visualization-only install should not report a failure it cannot act on.
    gemini = probe.check_gemini_key()
    if gemini.state == probe.FAIL and not runtime_ready:
        gemini.state = probe.WARN
        gemini.detail = "not set (only needed to collect)"
    checks.append(gemini)

    # Profile-specific checks: the settings that decide whether a session can even start.
    try:
        profile = profiles.load(profile_name)
    except ProfileError as exc:
        checks.append(
            probe.Check("profile", probe.FAIL, exc.message.split("\n")[0], exc.hint or "", group="profile")
        )
        return checks

    checks.append(probe.Check("profile", probe.OK, f"{profile.name} · {profile.robot.type}", group="profile"))

    configured = profile.cameras.configured()
    missing = profiles.missing_calibration(profile)
    if not configured:
        checks.append(
            probe.Check(
                "cameras",
                probe.SKIP,
                "none configured — this profile can be browsed but not collected into",
                group="profile",
            )
        )
    elif missing:
        checks.append(
            probe.Check(
                "camera calibration",
                probe.FAIL,
                f"no extrinsics for {', '.join(missing)}",
                f"Extrinsics are keyed by serial. Add them to {profile.calibration_file()}.",
                group="profile",
            )
        )
    else:
        checks.append(
            probe.Check(
                "camera calibration",
                probe.OK,
                f"{len(configured)} camera(s) calibrated",
                group="profile",
            )
        )

    warnings = render.check_assets(profile, runtime_dir=cfg.resolved_runtime_dir())
    # missing-calibration is already its own row above; do not say it twice.
    warnings = [w for w in warnings if not w.startswith("no camera extrinsics")]
    for warning in warnings:
        checks.append(probe.Check("tamp settings", probe.WARN, warning, group="profile"))
    if not warnings:
        n = len(profile.tamp)
        checks.append(
            probe.Check(
                "tamp settings",
                probe.OK,
                f"{n} override(s)" if n else "stock settings",
                group="profile",
            )
        )

    checks.append(_phase_planning_check(profile))
    executor_check = _human_executor_check(profile)
    if executor_check is not None:
        checks.append(executor_check)

    if probe_hardware:
        checks.append(probe.check_zed_sdk())
        checks.append(probe.check_robot(profile.robot.host, profile.robot.port))
        checks.append(probe.check_robot_state_port(profile.robot.host, profile.robot.state_port))
        checks.append(probe.check_m2t2(profile.perception.m2t2.url))

    return checks


def _runtime_check(profile_name: str | None, cfg) -> tuple[probe.Check, bool, Path | None, str | None]:
    """The runtime of the planner the profile uses: (the check, whether it is ready, where it is, the
    planner's name -- None when there is no telling which planner that is)."""
    from tandem.cli.runtime import planner_runtime
    from tandem.planners.runtime import RecipeRuntime

    try:
        planner, runtime = planner_runtime(profile_name=profile_name, settings=cfg)
    except TandemError as exc:
        # The profile row below says what is wrong with the profile; this one only says that without
        # it, there is no telling whose runtime to look at.
        return (
            probe.Check("gpu runtime", probe.WARN, "unknown: " + exc.message.split("\n")[0], group="runtime"),
            False,
            None,
            None,
        )
    if runtime is None:
        return (
            probe.Check("gpu runtime", probe.OK, f"{planner} is pure Python", group="runtime"),
            True,
            None,
            planner,
        )

    status = runtime.status()
    root = Path(status.path) if status.path else None
    if status.installed:
        detail = f"{planner} ready at {status.path}"
        built = runtime.inspect().built_at if isinstance(runtime, RecipeRuntime) else None
        if built:
            detail += f"  · built {built}"
        return probe.Check("gpu runtime", probe.OK, detail, group="runtime"), True, root, planner
    return (
        probe.Check(
            "gpu runtime",
            probe.WARN,
            f"{planner}: " + "; ".join(status.problems or ["not built"]),
            f"Run `tandem planners install {planner}` (`tandem init` does too). Not needed to visualize "
            "trajectories.",
            group="runtime",
        ),
        False,
        root,
        planner,
    )


def _other_planner_checks(active: str | None, cfg) -> list[probe.Check]:
    """One row per planner the profile does not use: whether this machine has it, and whether it loads.

    None of these stops a session, so none of them fails: a planner that is simply not installed is
    a note. Two are warnings, because each is something the machine holds that is wrong -- a runtime
    built from commits its planner no longer pins (25 GB that a session with it would refuse), and an
    installed plugin that will not load (which ``tandem planners list`` explains).
    """
    from tandem.cli import planners as planners_cli
    from tandem.planners import registry

    entries = registry.catalog()
    working = {entry.name for entry in entries if entry.ok}
    checks = []
    for entry in entries:
        if entry.name == active and (entry.ok or entry.name not in working):
            continue  # the row above
        label = f"planner {entry.name}"
        if not entry.ok:
            checks.append(
                probe.Check(
                    label,
                    probe.WARN,
                    entry.error or "it will not load",
                    "Reinstall or uninstall the package that provides it; `tandem planners list` shows the rest.",
                    group="runtime",
                )
            )
            continue
        state = planners_cli.runtime_state(entry.name, entry.info, cfg)
        status = state["status"]
        if status == planners_cli.INSTALLED:
            checks.append(probe.Check(label, probe.OK, "installed · not used by this profile", group="runtime"))
        elif status == planners_cli.OUTDATED:
            checks.append(
                probe.Check(
                    label,
                    probe.WARN,
                    state["detail"],
                    f"`tandem planners install {entry.name}` updates it; `tandem planners remove {entry.name}` "
                    "frees the disk.",
                    group="runtime",
                )
            )
        elif status == planners_cli.BROKEN:
            checks.append(probe.Check(label, probe.WARN, state["detail"], group="runtime"))
        else:
            what = "pure Python" if status == planners_cli.NO_RUNTIME else "not installed"
            checks.append(probe.Check(label, probe.SKIP, f"{what} · not used by this profile", group="runtime"))
    return checks


def _phase_planning_check(profile) -> probe.Check:
    """Whether a session with phase planning on would get as far as proposing a plan.

    Answerable without the runtime now that the phase planner is tandem's own — which is the point.
    Before, "will this work?" needed a warm cuRobo, an open camera and an arm, so the answer arrived
    with an operator already standing next to one.
    """
    from tandem.core import secrets
    from tandem.planners import registry

    backend = profile.planner.backend
    if not profile.hitl.enabled:
        return probe.Check(
            "phase planning",
            probe.SKIP,
            f"off · the {backend} planner gets the whole instruction as one goal",
            "Set hitl.enabled to let a model split it into robot and human steps.",
            group="profile",
        )

    try:
        caps = registry.capabilities(backend)
    except TandemError as exc:
        return probe.Check("phase planning", probe.FAIL, exc.message, exc.hint or "", group="profile")

    if not secrets.gemini_api_key():
        return probe.Check(
            "phase planning",
            probe.FAIL,
            "on, but no Gemini API key is set — nothing can propose a plan",
            "Run `tandem config set-gemini-key`.",
            group="profile",
        )

    # Through render, so this names the file a session would actually use rather than one relative
    # to wherever doctor was run from.
    cache = render.resolve_cache_path(profile)
    if cache:
        parent = Path(cache).parent
        if not parent.is_dir():
            return probe.Check(
                "phase planning",
                probe.WARN,
                f"the proposal cache directory does not exist: {parent}",
                "Create it, or clear hitl.cache_path.",
                group="profile",
            )

    detail = (
        f"{profile.hitl.proposal_model} splits the task · {profile.hitl.vlm_model} checks it · "
        f"goals in {backend}'s {', '.join(caps.goal_predicates)}"
    )
    hint = ""
    if profile.hitl.on_robot_phase_failure == "teleop":
        hint = "A phase the planner cannot plan is offered to you as teleop."
    return probe.Check("phase planning", probe.OK, detail, hint, group="profile")


def _human_executor_check(profile) -> probe.Check | None:
    """Whether the human steps of a phase-planned session can be carried out here, and recorded.

    Worth its own row because the answer changes what a session can do, not only whether it starts.
    While recording, a human step is completed only by a leg of ``hitl.human_executor``: "I did it"
    alone is refused unless ``hitl.allow_unrecorded_human_phase`` is set. So an executor this machine
    is not set up for leaves every human step with one answer -- give up -- and the operator would
    otherwise learn that at the first human phase, with the robot part-way through the task.

    None when phase planning is off: there are no human phases to carry out.
    """
    from tandem.executors import base as executors

    if not profile.hitl.enabled:
        return None
    name = profile.hitl.human_executor
    try:
        info = executors.info(name)
    except TandemError as exc:
        # The session refuses to start on exactly this (Session.start describes the executor first).
        return probe.Check("human executor", probe.FAIL, exc.message, exc.hint or "", group="profile")
    if info.ready:
        return probe.Check("human executor", probe.OK, f"{name} · {info.display_name}", group="profile")
    detail = f"{name} needs setup: {'; '.join(info.unmet)}"
    if profile.hitl.allow_unrecorded_human_phase:
        hint = (
            "Until then a human step is accepted when you say it is done, unrecorded "
            "(hitl.allow_unrecorded_human_phase). `tandem executors list` shows what each executor needs."
        )
    else:
        hint = (
            "While recording, a human step is completed only through it, so until then the only answer at "
            "one is to give up. `tandem executors list` shows what each executor needs; "
            "hitl.allow_unrecorded_human_phase: true accepts a step done by hand, unrecorded."
        )
    return probe.Check("human executor", probe.WARN, detail, hint, group="profile")


def doctor(
    profile_name: str = typer.Option(None, "--profile", "-p", help="Check this profile instead of the active one."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
    skip_hardware: bool = typer.Option(
        False, "--no-hardware", help="Skip the robot / camera / grasp-server probes."
    ),
) -> None:
    """Every check says what it found and, when something is wrong, what to do about it."""
    checks = collect_checks(profile_name=profile_name, probe_hardware=not skip_hardware)

    if as_json:
        payload = {
            "checks": [c.to_dict() for c in checks],
            "summary": _summary(checks),
        }
        typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(0 if not _summary(checks)["fail"] else 1)

    theme.blank()
    for group in ("core", "runtime", "gpu", "credentials", "profile", "hardware"):
        rows = [c for c in checks if c.group == group]
        if not rows:
            continue
        theme.heading(GROUP_TITLES.get(group, group))
        table = theme.table("", "", "", box_style="none")
        table.columns[0].width = 3
        table.columns[1].style = "default"
        table.columns[1].width = 22
        table.columns[2].style = "faint"
        for check in rows:
            table.add_row(theme.status_glyph(check.state), check.name, check.detail)
        theme.console().print(table)
        theme.blank()

    problems = [c for c in checks if c.state in (probe.FAIL, probe.WARN) and c.hint]
    if problems:
        theme.rule("what to do")
        for check in problems:
            glyph = theme.FAIL if check.state == probe.FAIL else theme.WARN
            style = "err" if check.state == probe.FAIL else "warn"
            theme.console().print(f"  [{style}]{glyph}[/{style}] [bold]{check.name}[/bold]")
            theme.console().print(f"    [faint]{check.hint}[/faint]")
        theme.blank()

    counts = _summary(checks)
    runtime_ready = any(c.name == "gpu runtime" and c.state == probe.OK for c in checks)

    if counts["fail"]:
        theme.fail(f"{counts['fail']} check(s) failed", f"{counts['warn']} warning(s)")
        raise typer.Exit(1)
    if counts["warn"]:
        note = (
            "collection may still work — see above"
            if runtime_ready
            else "visualization is ready; collection needs the items above"
        )
        theme.warn(f"{counts['warn']} warning(s)", note)
    else:
        theme.ok("Everything checks out")


def _summary(checks: list[probe.Check]) -> dict[str, int]:
    out = {"ok": 0, "warn": 0, "fail": 0, "skip": 0}
    for check in checks:
        out[check.state] = out.get(check.state, 0) + 1
    return out
