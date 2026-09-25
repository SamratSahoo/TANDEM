"""`tandem doctor` — is everything tandem needs present and working?

What tandem needs is checked here: Python, ffmpeg, the runtime of the planner the profile uses (and
the disk and pixi, when that runtime is one to build), this machine's rig and its cameras, the profile,
phase planning and its human executor. What that PLANNER needs -- a GPU, a key for its perception, a robot shim, a grasp
server -- only the planner knows, so it is asked (its factory's ``doctor_checks``, through the
registry), and a profile that plans with another planner is shown that planner's rows instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from tandem.cli import theme
from tandem.core import probe, profiles, secrets
from tandem.core import rig as rig_mod
from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError, TandemError, one_line

GROUP_TITLES = {
    "core": "environment",
    "runtime": "runtime",
    "gpu": "gpu",
    "credentials": "credentials",
    "rig": "rig",
    "hardware": "hardware",
    "profile": "profile",
}

#: The order the groups are shown in.
GROUPS = ("core", "runtime", "gpu", "credentials", "rig", "profile", "hardware")

#: The row naming the runtime of the planner the profile uses, whichever planner that is.
RUNTIME_ROW = "planner runtime"


def collect_checks(*, profile_name: str | None = None, probe_hardware: bool = True) -> list[probe.Check]:
    from tandem.planners import registry

    cfg = settings_mod.load()
    checks: list[probe.Check] = [probe.check_python(), probe.check_platform()]

    runtime_check, runtime_ready, runtime_root, active = _runtime_check(profile_name, cfg)
    builds, solves = _what_the_runtime_needs(active, cfg)
    if builds:
        checks.append(probe.check_disk(runtime_root or cfg.resolved_runtime_dir()))
    if solves:
        checks.append(probe.check_pixi())
    else:
        checks.append(
            probe.Check(
                "pixi", probe.SKIP, f"not needed · {active}'s runtime is no pixi environment", group="runtime"
            )
        )
    checks.append(probe.check_ffmpeg())
    checks.append(runtime_check)
    checks.extend(_optional_step_checks(active, cfg))
    checks.extend(_other_planner_checks(active, cfg))
    # This machine's rig: tandem's own robot and cameras. What a planner needs of it is the planner's rows.
    checks.extend(rig_checks())

    # Profile-specific checks: the settings that decide whether a session can even start.
    try:
        profile = profiles.load(profile_name)
    except ProfileError as exc:
        checks.append(_gemini_check(None, runtime_ready))
        checks.append(
            probe.Check("profile", probe.FAIL, one_line(exc.message), exc.hint or "", group="profile")
        )
        return checks

    backend = profile.planner.backend
    checks.append(_gemini_check(profile, runtime_ready))
    checks.append(probe.Check("profile", probe.OK, f"{profile.name} · plans with {backend}", group="profile"))
    checks.append(_phase_planning_check(profile))
    executor_check = _human_executor_check(profile)
    if executor_check is not None:
        checks.append(executor_check)

    # What only the planner knows it needs. Never raises: a planner whose own checks break is one row.
    checks.extend(registry.doctor_checks(backend, profile, settings=cfg, probe_hardware=probe_hardware))
    return checks


def _gemini_check(profile, runtime_ready: bool) -> probe.Check:
    """The Gemini key, as far as tandem itself is concerned: phase planning calls Gemini, and nothing
    else of tandem's does. A planner that calls it too (TiPToP's perception) says so in its own row."""
    check = probe.check_gemini_key()
    if check.state == probe.OK:
        return check
    if profile is None:
        check.state, check.detail = probe.WARN, "not set (phase planning needs it)"
    elif not profile.hitl.enabled:
        check.state = probe.SKIP
        check.detail = "not set · phase planning is off, so tandem itself does not need it"
    elif not runtime_ready:
        # A missing key blocks collection, but on a machine that cannot collect anyway it is only
        # worth a note — a visualization-only install should not report a failure it cannot act on.
        check.state, check.detail = probe.WARN, "not set (only needed to collect)"
    else:
        check.detail = "not set, and phase planning is on"
    return check


def rig_checks() -> list[probe.Check]:
    """rig.yml, and whether it has cameras to record from.

    The teleop legs record from the cameras whichever planner runs, so they are tandem's to check; whether
    a planner can also localise from them (its calibration) is the planner's.
    """
    try:
        rig = rig_mod.load()
    except TandemError as exc:
        return [probe.Check("rig", probe.FAIL, one_line(exc.message), exc.hint or "`tandem rig edit`", group="rig")]
    if not rig_mod.exists():
        rows = [
            probe.Check(
                "rig",
                probe.WARN,
                "not set up",
                "`tandem init`, or `tandem rig set robot.host ADDRESS` and `tandem rig set "
                "cameras.external.serial SERIAL`.",
                group="rig",
            )
        ]
    else:
        rows = [probe.Check("rig", probe.OK, f"{rig.file()} · {rig.summary()}", group="rig")]
    rows.append(_cameras_check(rig))
    return rows


def _cameras_check(rig) -> probe.Check:
    configured = rig.cameras.configured()
    if not configured:
        return probe.Check(
            "cameras",
            probe.SKIP,
            "none configured: this machine can browse trajectories but not collect",
            "`tandem rig set cameras.external.serial SERIAL` (and cameras.hand.serial) to collect here.",
            group="rig",
        )
    missing = rig.cameras.perception_missing()
    if missing:
        return probe.Check(
            "cameras",
            probe.FAIL,
            missing,
            f"`tandem rig set cameras.{rig.cameras.perception}.serial SERIAL`, or `tandem rig set "
            "cameras.perception ROLE` to read another camera. A session will not start without it.",
            group="rig",
        )
    detail = ", ".join(f"{role} {cam.serial}" for role, cam in configured.items())
    return probe.Check("cameras", probe.OK, f"{detail} · perception reads {rig.cameras.perception}", group="rig")


def _what_the_runtime_needs(active: str | None, cfg) -> tuple[bool, bool]:
    """Whether the planner in use has a runtime to build, so the disk it takes matters, and whether
    building it solves a pixi environment, so pixi matters. A pure-Python planner needs neither, and a
    missing pixi is no failure on a machine that will never build anything. Both True when there is no
    telling which planner that is: the rows then show, as they always have.
    """
    if active is None:
        return True, True
    from tandem.cli import runtime as runtime_cli
    from tandem.planners import registry

    try:
        rt = registry.runtime(active, cfg)
    except Exception:
        return True, True
    if rt is None:
        return False, False
    return True, runtime_cli.needs_pixi(rt)


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
            probe.Check(RUNTIME_ROW, probe.WARN, "unknown: " + one_line(exc.message), group="runtime"),
            False,
            None,
            None,
        )
    if runtime is None:
        return (
            probe.Check(RUNTIME_ROW, probe.OK, f"{planner} is pure Python", group="runtime"),
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
        return probe.Check(RUNTIME_ROW, probe.OK, detail, group="runtime"), True, root, planner
    return (
        probe.Check(
            RUNTIME_ROW,
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


def _optional_step_checks(active: str | None, cfg) -> list[probe.Check]:
    """One row per optional part of the planner's runtime (TiPToP's: the ZED Python API): there, or what does
    not work without it and how to get it. A warning at worst: the runtime works without it, and the part
    of it that needs one (a camera, say) has its own row where it is checked."""
    if active is None:
        return []
    from tandem.planners import registry
    from tandem.planners.runtime import RecipeRuntime

    try:
        rt = registry.runtime(active, cfg)
    except Exception:  # the runtime row says what is wrong with it
        return []
    if not isinstance(rt, RecipeRuntime):
        return []
    st = rt.inspect()
    if not st.exists:
        return []  # nothing is built yet, and the runtime row says to build it
    done = dict(st.steps)
    rows = []
    for step in (step for step in rt.recipe.steps if step.optional):
        name = step.title.lower()
        if done.get(step.name):
            rows.append(probe.Check(name, probe.OK, step.done, group="runtime"))
        elif step.unmet():
            rows.append(probe.Check(name, probe.WARN, step.todo, step.missing, group="runtime"))
        else:
            rows.append(
                probe.Check(
                    name, probe.WARN, step.todo, f"`tandem planners install {active}` installs it.", group="runtime"
                )
            )
    return rows


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
            checks.append(
                probe.Check(label, probe.OK, "installed · not used by this profile", group="runtime")
            )
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
            checks.append(
                probe.Check(label, probe.SKIP, f"{what} · not used by this profile", group="runtime")
            )
    return checks


def _phase_planning_check(profile) -> probe.Check:
    """Whether a session with phase planning on would get as far as proposing a plan.

    Answerable without the runtime now that the phase planner is tandem's own — which is the point.
    Before, "will this work?" needed a warm cuRobo, an open camera and an arm, so the answer arrived
    with an operator already standing next to one.
    """
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

    frames = _frame_checks_unanswerable(backend, profile.hitl)
    if frames:
        checks = " and ".join(f"hitl.{key}" for key in frames)
        return probe.Check(
            "phase planning",
            probe.FAIL,
            f"on, but the {backend} planner cannot capture a camera frame, and {checks} "
            f"{'needs' if len(frames) == 1 else 'need'} one — a session will refuse to start",
            f"Implement capture_frame(camera=...) in the planner, or set {checks} to false -- knowing that "
            "no human step will then be verified.",
            group="profile",
        )

    # Through the profile, so this names the file a session would actually use rather than one relative
    # to wherever doctor was run from.
    cache = profiles.resolve_cache_path(profile)
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


def _frame_checks_unanswerable(backend: str, hitl) -> list[str]:
    """The checks ``hitl`` turns on that need a frame the planner ``backend`` can never capture.

    The session refuses such a planner once it is warm (``Session._require_frames``); this says so
    before anyone is standing at the robot. Only what the class alone settles: a planner written with
    the SDK that never overrode ``capture_frame``. A sidecar's verbs are known only once it runs, and a
    hand-written factory's only once it builds a backend, so both are left to the session.
    """
    from tandem.core.session import FRAME_CHECKS
    from tandem.planners import registry
    from tandem.planners.sdk import Planner
    from tandem.planners.sidecar import SidecarPlanner

    needs = [key for key in FRAME_CHECKS if getattr(hitl, key, False)]
    if not needs:
        return []
    try:
        factory = registry.factory(backend)
    except TandemError:
        return []
    if not (isinstance(factory, type) and issubclass(factory, Planner)):
        return []
    own = getattr(factory, "capture_frame", None)
    if issubclass(factory, SidecarPlanner) and own is SidecarPlanner.capture_frame:
        return []
    return needs if own is Planner.capture_frame else []


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
    profile_name: str = typer.Option(
        None, "--profile", "-p", help="Check this profile instead of the active one."
    ),
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
    for group in GROUPS:
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
    runtime_ready = any(c.name == RUNTIME_ROW and c.state == probe.OK for c in checks)

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
