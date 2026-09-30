"""`tandem init` — the onboarding wizard.

It sets the machine up -- the planner and its runtime, the Gemini key, the rig (the robot's address,
the arm, the cameras), teleop -- and adds the paper's five tasks as profiles. Everything a profile holds
is a task; everything asked about this machine goes to the rig, which every profile shares.

Idempotent and resumable: every step checks its own postcondition first and says
"already done" rather than redoing work. Interrupting it and re-running is safe — cuRobo's
build fingerprint means even the expensive kernel compile is skipped when nothing changed.
Without a terminal (or with --yes) it asks nothing: --robot-host, --robot-type and --camera say
what it would have asked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import typer

from tandem.cli import runtime as runtime_cli
from tandem.cli import theme
from tandem.core import layout, paths, probe, profiles, secrets
from tandem.core import rig as rig_mod
from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError, TandemError, one_line
from tandem.planners import registry


def init(
    viz_only: bool = typer.Option(
        False,
        "--viz-only",
        help="Set up for browsing and visualizing trajectories only — no GPU runtime, no robot.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Accept every default; ask nothing."),
    repair: bool = typer.Option(
        False,
        "--repair",
        help="Redo steps that are already done: rebuild the planner's runtime, and ask again for the key, "
        "the robot and cameras, and the teleop settings. No profile is ever touched by it.",
    ),
    planner: str = typer.Option(
        None,
        "--planner",
        help="The planner to set this machine up for, and the one new profiles get (see `tandem planners "
        "list`). Default: the machine's default (tiptop).",
    ),
    profile_name: str = typer.Option(
        None, "--profile", help="Make this profile the active one (default: keep it, else cover-bread-rolls)."
    ),
    robot_host: str = typer.Option(
        None, "--robot-host", help="The robot computer's address (the NUC), such as 172.16.0.2."
    ),
    robot_type: str = typer.Option(None, "--robot-type", help="The arm, such as fr3_robotiq or panda_robotiq."),
    cameras: list[str] = typer.Option(
        None,
        "--camera",
        help="A camera's serial by role: hand=SERIAL, external=SERIAL or external_2=SERIAL. Repeatable.",
    ),
) -> None:
    interactive = theme.is_tty() and not yes
    # Checked before anything is asked or built: a typo in a flag is said now, not after a twenty-minute build.
    rig_flags = _rig_flags(robot_host=robot_host, robot_type=robot_type, cameras=cameras or [])
    theme.banner("setup")

    cfg = settings_mod.load()

    # ---- 1. what kind of install -------------------------------------------
    if not viz_only and interactive:
        theme.heading("What is this machine for?")
        theme.console().print(
            "  [accent]1[/accent]  workstation  [faint]— collect trajectories on a real robot, and visualize them[/faint]\n"
            "  [accent]2[/accent]  laptop       [faint]— browse and visualize trajectories collected elsewhere[/faint]"
        )
        choice = typer.prompt("  Choose", default="1").strip()
        viz_only = choice.startswith("2")
        theme.blank()

    mode = "visualization only" if viz_only else "workstation"
    theme.info(f"Setting up for {mode}.")
    theme.blank()

    # ---- 2. preflight -------------------------------------------------------
    theme.rule("checking this machine")
    checks = _preflight(viz_only=viz_only)
    _render_checks(checks)
    if not viz_only:  # a laptop joins no hand-off's legs, and apt's ffmpeg is minutes of install
        checks = _ensure_ffmpeg(checks, interactive=interactive)
    if not viz_only:
        _stop_on_blocking(checks, interactive=interactive, unapplied=rig_flags)
    theme.blank()

    # ---- 3. where data lives ------------------------------------------------
    theme.rule("where to keep trajectories")
    data_root = cfg.resolved_data_root()
    if interactive:
        answer = typer.prompt("  Data root", default=str(data_root)).strip()
        data_root = Path(answer).expanduser().resolve()
    cfg.data_root = str(data_root)
    paths.ensure_dir(data_root)
    settings_mod.save(cfg)
    theme.ok("Data root", str(data_root))
    theme.blank()

    # ---- 3b. profiles in the layout before version 3 ------------------------
    # Before anything reads the profiles: until they are moved they are not loaded at all, and the rig
    # this machine needs is set up from them when it has none yet.
    if layout.pending():
        from tandem.cli import profile as profile_cli

        theme.rule("profiles in the old layout")
        report = profile_cli.run_migration()
        if report is not None and report.aborted is not None:
            # The rig comes from these profiles, and every later step reads the rig: nothing is set up over it.
            raise report.aborted
        if report is not None and report.failed:
            theme.warn(
                f"{len(report.failed)} profile(s) could not be moved, or only partly",
                "`tandem profile migrate` again once each is fixed",
            )
        theme.blank()

    # After the data root is settled and old profiles are moved, before twenty minutes go into a build.
    if profile_name is not None:
        _check_profile_to_activate(profile_name)

    # ---- 4. the rig: this machine's robot and cameras -----------------------
    # Before the planner's checks and its build: it needs no GPU, and --robot-host and --camera given to an
    # init that a missing driver then stops are written all the same, not dropped without a word.
    asked_rig = False
    if not viz_only:
        theme.rule("robot")
        asked_rig = _setup_rig(rig_flags, interactive=interactive, repair=repair)
        theme.blank()
    elif rig_flags:
        theme.warn(
            "--robot-host, --robot-type and --camera were not applied",
            "a visualization-only machine has no robot; `tandem rig set` sets them on one that does",
        )

    # ---- 5. the planner -----------------------------------------------------
    # Chosen before anything is built: the runtime step builds THIS planner's runtime. A laptop builds
    # nothing; a planner named on its command line is still made the one its new profiles get.
    if not viz_only:
        theme.rule("planner")
        planner = _choose_planner(planner, interactive=interactive)
        _planner_preflight(planner, interactive=interactive, repair=repair)
        theme.blank()
    if planner is not None:
        from tandem.cli import planners as planners_cli

        # The planner this machine is for, so the one its new profiles get. No existing profile is
        # switched: that is `tandem planners use`, and a profile's planner is its own.
        planners_cli.set_default_planner(planner)

    # ---- 6. the planner's runtime -------------------------------------------
    if not viz_only:
        theme.rule("planner runtime")
        _build_runtime(planner, interactive=interactive, repair=repair)
        theme.blank()

    # ---- 6a. the perception servers -----------------------------------------
    # The servers the planner calls (TiPToP: M2T2 and FoundationStereo, every rollout); a session starts them.
    if not viz_only and planner is not None and registry.services(planner):
        theme.rule("perception servers")
        _build_servers(planner, interactive=interactive, repair=repair)
        theme.blank()

    # ---- 6b. the cameras ------------------------------------------------------
    # After the runtime: its ZED step is what gives this machine the ZED Python API that lists the cameras.
    if not viz_only:
        theme.rule("cameras")
        _setup_cameras(planner, interactive=interactive, ask=asked_rig)
        theme.blank()

    # ---- 7. the Gemini key --------------------------------------------------
    theme.rule("gemini api key")
    source = secrets.gemini_key_source()
    if source != "none" and not repair:
        theme.ok(f"Already set  {secrets.mask(secrets.gemini_api_key())}", f"from {source}")
    elif viz_only:
        theme.info("Not needed for visualization — skipping.")
    else:
        theme.info("Phase planning asks Gemini to split each task into steps and to check each human one,")
        theme.info("and a planner may call it too (TiPToP's perception does, every rollout).")
        theme.info("Get a key at https://aistudio.google.com/apikey")
        if interactive:
            key = typer.prompt("  Gemini API key (blank to skip)", default="", hide_input=True).strip()
            if key:
                path = secrets.set_gemini_api_key(key)
                theme.ok(f"Stored  {secrets.mask(key)}", str(path))
            else:
                theme.warn("Skipped", "set it later with `tandem config set-gemini-key`")
        else:
            theme.warn("No key set", "run `tandem config set-gemini-key`")
    theme.blank()

    # ---- 8. the paper's five tasks, and the active profile ------------------
    theme.rule("profiles")
    active = _setup_profiles(profile_name)
    if planner is not None and not viz_only:
        _say_when_the_planners_differ(active, planner)
    theme.blank()

    # ---- 9. teleop hand-off (optional) --------------------------------------
    if not viz_only:
        theme.rule("teleop hand-off  (optional)")
        _setup_teleop(interactive=interactive, repair=repair)
        theme.blank()

    # ---- 10. done -----------------------------------------------------------
    _summary(viz_only=viz_only, profile_name=active, planner=planner)


# --------------------------------------------------------------------------- steps


def _choose_planner(requested: str | None, *, interactive: bool) -> str:
    """Which planner this machine is set up for: the catalog shown, one chosen, and checked.

    ``--planner`` wins; otherwise the machine's default, which at a terminal with more than one planner
    to choose from is offered as the answer to a question. Whichever it is becomes the default, so a
    re-run sets the same planner up again and new profiles plan with it. The choice is checked against
    the registry here, before twenty minutes go into a runtime.
    """
    from tandem.cli import planners as planners_cli

    payload = planners_cli.catalog_payload()
    planners_cli.render_catalog(payload["planners"])
    for row in payload["planners"]:
        if row["status"] == planners_cli.BROKEN:
            theme.fail(f"{row['name']}: {row['detail']}")

    usable = [row["name"] for row in payload["planners"] if row["ok"]]
    if requested:
        choice = requested
    elif interactive and len(usable) > 1:
        default = payload["default_planner"] if payload["default_planner"] in usable else usable[0]
        choice = typer.prompt("  Planner", default=default).strip()
    else:
        choice = payload["default_planner"]

    info = registry.info(choice)  # unknown or broken: the registry's own error, with the nearest name
    theme.ok(f"Planner: {info.title}", "new profiles plan with it")
    return choice


def _build_runtime(planner: str, *, interactive: bool, repair: bool) -> None:
    """Build ``planner``'s runtime, with the optional steps its recipe can take on this machine.

    The consent flow for pixi and the build itself are `tandem planners install`'s (``cli/runtime``),
    so the wizard and the command cannot drift apart. `init` is "accept every default" when it cannot
    ask, so without a terminal it installs pixi rather than failing. A runtime that is built but has an
    optional step it can take now -- the ZED Python API, once the ZED SDK is installed -- is built again,
    which does that step and skips everything already done.
    """
    planner, runtime = runtime_cli.planner_runtime(planner=planner)
    title = registry.info(planner).title
    if runtime is None:
        theme.ok(f"{title} is pure Python", "there is no runtime to build")
        return
    status = runtime.status()
    pending = runtime_cli.optional_steps_to_run(runtime)

    if status.installed and not repair and not pending:
        theme.ok("Runtime is already built", str(status.path))
        runtime_cli.say_notes(status)
        return

    if runtime_cli.needs_pixi(runtime):
        runtime_cli.ensure_pixi(title, ask=interactive, allowed=True)

    if status.installed and not repair:
        theme.ok("Runtime is already built", f"{status.path}; now installing {', '.join(pending)}")
    else:
        for note in getattr(getattr(runtime, "recipe", None), "notes", ()) or ():
            theme.info(note)
    if interactive and not typer.confirm("  Build it now?", default=True):
        theme.warn("Skipped", f"run `tandem planners install {planner}` when you are ready")
    else:
        runtime_cli.run_build(runtime, force=repair, planner=planner)


def _build_servers(planner: str, *, interactive: bool, repair: bool) -> None:
    """The helper servers the planner runs (TiPToP: M2T2 and FoundationStereo), built as runtimes beside its own.

    Those whose URL is another machine's are that machine's to build. A failed build is said and init goes on:
    a session reports a server it cannot reach.
    """
    from tandem.cli import servers as servers_cli

    services = servers_cli.planner_services(planner)
    local = [service for service in services if service.local()]
    if not local:
        theme.ok("Its servers are on another machine", ", ".join(service.url() for service in services))
        return
    theme.info("A collection session starts them when it needs them, and stops them when it ends.")
    try:
        if not servers_cli.install_servers(planner, yes=not interactive, repair=repair):
            theme.warn("Skipped", "`tandem servers install` builds them when you are ready")
    except TandemError as exc:
        theme.warn(f"A server did not build: {exc.message}", exc.hint or "")
        theme.info("`tandem servers install` tries again; it skips what is done.")


def _preflight(*, viz_only: bool) -> list[probe.Check]:
    """What tandem itself needs of the machine. What the planner needs is asked once it is chosen.

    Disk space and pixi are not asked here. They are what a planner's RUNTIME needs, and before the
    planner is chosen there is no telling whether it has one: a pure-Python planner needs neither, and
    stopping on a missing pixi before init has had the chance to install it stopped `init --yes` on
    every fresh machine.
    """
    checks = [probe.check_python(), probe.check_platform()]
    checks.append(probe.check_ffmpeg())
    return checks


def _ensure_ffmpeg(checks: list[probe.Check], *, interactive: bool) -> list[probe.Check]:
    """Install ffmpeg when the preflight found none: ``checks`` with its check made again after. A workstation's
    only: joining the legs of a teleop hand-off is collection's, and a laptop is told ffmpeg is missing.

    With Homebrew on a Mac, apt-get on Linux (through sudo unless this is root). At a terminal it is asked
    first and sudo may ask for a password; without one it is "accept every default", as pixi is, but sudo
    is never left waiting on a password nobody can type (``sudo -n``). A failed install is said and init
    goes on: ffmpeg is needed only to join the legs of a teleop hand-off.
    """
    at = next((i for i, check in enumerate(checks) if check.name == "ffmpeg"), None)
    if at is None or checks[at].state == probe.OK:
        return checks
    command = _ffmpeg_install_command(interactive=interactive)
    if command is None:
        theme.warn("ffmpeg is not installed, and there is no brew or apt-get here to install it", checks[at].hint)
        return checks
    theme.blank()
    theme.info("ffmpeg joins the legs of a teleop hand-off into one trajectory.")
    theme.info(f"It installs with `{' '.join(command)}`.")
    if interactive and not typer.confirm("  Install ffmpeg now?", default=True):
        theme.warn("Skipped", f"`{' '.join(command)}` installs it when you are ready")
        return checks
    theme.busy("Installing ffmpeg", "this can take a few minutes")
    output = _install_ffmpeg(command, interactive=interactive)
    found = probe.check_ffmpeg()
    if found.state == probe.OK:
        theme.ok("ffmpeg installed", found.detail)
    else:
        theme.warn("ffmpeg did not install", f"run `{' '.join(command)}` yourself, then `tandem doctor`")
        for line in output[-10:]:
            theme.console().print(f"    {line}", style="faint", markup=False, highlight=False)
    return [*checks[:at], found, *checks[at + 1:]]


def _ffmpeg_install_command(*, interactive: bool) -> list[str] | None:
    """The package manager's command that installs ffmpeg on this machine, or None when there is none."""
    import os
    import shutil
    import sys

    if sys.platform == "darwin":
        return ["brew", "install", "ffmpeg"] if shutil.which("brew") else None
    if not shutil.which("apt-get"):
        return None
    # Through env, not the environment: sudo drops DEBIAN_FRONTEND, and apt's configure step must not stop
    # on a question (tzdata's time zone, on a fresh machine).
    command = ["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y", "--no-install-recommends", "ffmpeg"]
    if os.geteuid() == 0:
        return command
    if not shutil.which("sudo"):
        return None
    return ["sudo", *([] if interactive else ["-n"]), *command]


def _install_ffmpeg(command: list[str], *, interactive: bool) -> list[str]:
    """Run ``command``, and return what it printed. Its outcome is judged by ffmpeg being there after.

    Without a terminal nothing can answer it, so it reads nothing and is stopped after twenty minutes
    rather than holding init up for good.
    """
    import subprocess

    try:
        proc = subprocess.run(
            command,
            stdin=None if interactive else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
            timeout=None if interactive else 20 * 60,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.output or ""
        out = out.decode("utf-8", "backslashreplace") if isinstance(out, bytes) else out
        return [*out.splitlines(), "stopped after 20 minutes"]
    except OSError as exc:
        return [str(exc)]
    return proc.stdout.splitlines()


def _runtime_checks(planner: str, *, repair: bool) -> list[probe.Check]:
    """Disk and pixi, for the chosen planner's runtime only, and only when it is about to be built.

    A missing pixi is not a blocking problem here: the runtime step installs it (with consent at a
    terminal, and under --yes without asking, as "accept every default" promises).
    """
    cfg = settings_mod.load()
    try:
        rt = registry.runtime(planner, cfg)
    except Exception:  # the runtime step reports a planner whose runtime cannot be located
        return []
    if rt is None:
        return [probe.Check("runtime", probe.SKIP, f"not needed · {planner} is pure Python", group="runtime")]
    status = rt.status()
    if status.installed and not repair:
        return []
    checks = [probe.check_disk(Path(status.path) if status.path else cfg.resolved_runtime_dir())]
    if runtime_cli.needs_pixi(rt):
        pixi = probe.check_pixi()
        if pixi.state == probe.FAIL:
            pixi = probe.Check("pixi", probe.WARN, "not found · init installs it, into ~/.pixi", group="runtime")
        checks.append(pixi)
    return checks


def _planner_preflight(planner: str, *, interactive: bool, repair: bool = False) -> None:
    """What the chosen planner needs of this machine -- disk and pixi for its runtime, a GPU, a camera SDK
    -- before its runtime is built.

    The runtime's needs as `tandem doctor` asks them, then the planner's own doctor checks, with no
    profile yet: they are the only ones that know. Stops on a failure the same way the machine checks
    do, so twenty minutes are not spent building a runtime that cannot run here.
    """
    checks = _runtime_checks(planner, repair=repair)
    checks += registry.doctor_checks(planner, None, settings=settings_mod.load(), probe_hardware=True)
    if not checks:
        return
    theme.blank()
    theme.info(f"what {registry.info(planner).title} needs of this machine")
    _render_checks(checks)
    _stop_on_blocking(checks, interactive=interactive)


def _stop_on_blocking(
    checks: list[probe.Check], *, interactive: bool, unapplied: dict[str, Any] | None = None
) -> None:
    """Stop on a failed check: asked at a terminal, refused without one. ``unapplied``: the rig flags this
    stop leaves unwritten, which the refusal says rather than drop them without a word."""
    blocking = [c for c in checks if c.state == probe.FAIL]
    if not blocking:
        return
    theme.blank()
    for check in blocking:
        theme.fail(check.name, check.detail)
        if check.hint:
            theme.console().print(f"    [faint]{check.hint}[/faint]")
    theme.blank()
    if interactive:
        if not typer.confirm("  Continue anyway?", default=False):
            raise typer.Abort()
    else:
        runtime_only = all(c.group == "runtime" for c in blocking)
        raise TandemError(
            "Preflight found blocking problems: " + ", ".join(c.name for c in blocking),
            hint="Fix them and re-run `tandem init`."
            + ("" if runtime_only else " On a laptop, --viz-only sets up for browsing only.")
            + (
                " --robot-host, --robot-type and --camera were not applied: give them again then, or "
                "`tandem rig set KEY VALUE` now."
                if unapplied
                else ""
            ),
        )


def _render_checks(checks: list[probe.Check]) -> None:
    table = theme.table("", "", "", box_style="none")
    table.columns[0].width = 3
    table.columns[1].width = 22
    table.columns[2].style = "faint"
    for check in checks:
        table.add_row(theme.status_glyph(check.state), check.name, check.detail)
    theme.console().print(table)


# --------------------------------------------------------------------------- the rig


def _rig_flags(*, robot_host: str | None, robot_type: str | None, cameras: list[str]) -> dict[str, Any]:
    """``--robot-host``, ``--robot-type`` and ``--camera ROLE=SERIAL`` as rig changes, each checked as typed."""
    import difflib

    changes: dict[str, Any] = {}
    if robot_host is not None:
        changes["robot.host"] = _checked_rig_value("robot.host", robot_host)
    if robot_type is not None:
        changes["robot.type"] = _checked_rig_value("robot.type", robot_type)
    for given in cameras:
        role, sep, serial = given.partition("=")
        role, serial = role.strip(), serial.strip()
        if not sep or not serial:
            raise TandemError(
                f"--camera {given!r} is not ROLE=SERIAL.",
                hint="For example --camera hand=14846828 --camera external=32439448.",
            )
        if role not in rig_mod.ROLES:
            close = difflib.get_close_matches(role, rig_mod.ROLES, n=1, cutoff=0.5)
            raise TandemError(
                f"--camera {given!r}: {role!r} is not a camera role"
                + (f" (did you mean {close[0]!r}?)." if close else "."),
                hint=f"The roles are {', '.join(rig_mod.ROLES)}.",
            )
        changes[f"cameras.{role}.serial"] = _checked_rig_value(f"cameras.{role}.serial", serial)
    return changes


def _checked_rig_value(key: str, value: str) -> str:
    """``value`` for the rig's ``key``, or a TandemError saying what is wrong with it -- before it is written."""
    from pydantic import ValidationError

    field = key.rsplit(".", 1)[-1]
    model = rig_mod.RobotSpec if key.startswith("robot.") else rig_mod.CameraSpec
    try:
        model.model_validate({field: value} if model is rig_mod.RobotSpec else {"serial": value})
    except ValidationError as exc:
        problem = "; ".join(str(err["msg"]).removeprefix("Value error, ") for err in exc.errors())
        raise TandemError(f"{key}: {problem}") from None
    return value.strip()


def _setup_rig(flags: dict[str, Any], *, interactive: bool, repair: bool) -> bool:
    """This machine's rig: the robot's address and arm. Returns whether the rig's questions were asked.

    Asked at a terminal when there is no rig yet (and again with --repair), each question defaulting to
    what the rig says now; the flags (--camera ones included) are applied either way, and are the defaults
    of the questions. One write, validated whole. A rig already there is otherwise left as it is and shown:
    `tandem rig set` changes one setting at any time. The cameras are asked in a later step
    (``_setup_cameras``), once the runtime can list them.

    A planner's machine settings are not written: its defaults stay its own (so a later tandem's better
    default reaches this machine), and `tandem rig show` lists each with its default beside the ones set.
    """
    layout.refuse_rig_change()  # old profiles that could not be moved still hold this machine's rig
    existed = rig_mod.exists()
    rig = rig_mod.load()  # a rig.yml that does not validate stops here, naming the line and `tandem rig edit`
    asked = interactive and (not existed or repair)
    changes = dict(flags)
    if asked:
        changes.update(_ask_rig(rig, flags))
    changes = {key: value for key, value in changes.items() if _differs(rig, key, value)}
    if changes or not existed:
        rig = rig_mod.update(changes)
    rig_mod.ensure_calibration_file(rig)  # a planner's calibration script writes into it

    theme.ok(
        f"Rig: {rig.summary()}",
        str(rig.file()) if asked or not existed else f"{rig.file()} · `tandem rig set KEY VALUE` changes a setting",
    )
    return asked


def _ask_rig(rig: rig_mod.Rig, flags: dict[str, Any]) -> dict[str, Any]:
    """The robot's questions, each defaulting to what a flag gave, else to what the rig says now."""
    changes: dict[str, Any] = {}
    changes["robot.host"] = _ask("Robot address (the NUC)", "robot.host", flags.get("robot.host", rig.robot.host))
    changes["robot.type"] = _ask("Arm type", "robot.type", flags.get("robot.type", rig.robot.type))
    return changes


#: The camera questions' labels, by role.
_CAMERA_LABELS = {
    "hand": "Wrist camera serial",
    "external": "External camera serial",
    "external_2": "Second external camera serial",
}


def _setup_cameras(planner: str | None, *, interactive: bool, ask: bool) -> None:
    """The cameras by role, offered from the ZED cameras the SDK lists, each to confirm or overwrite.

    Asked at a terminal when the robot's questions were (a new rig, or --repair), or when the rig has no
    cameras yet. The ZED listing runs under the planner's runtime, which has the ZED Python API once the
    ZED SDK is installed; without it, each serial is typed. --camera flags were applied with the robot, so
    they are the defaults here.
    """
    from tandem.cli import rig as rig_cli
    from tandem.core import zed

    rig = rig_mod.load()
    configured = rig.cameras.configured()
    if interactive and (ask or not configured):
        found = zed.detect(zed.interpreters(planner, settings_mod.load()))
        _say_found(found)
        changes = _ask_cameras(rig, found)
        changes = {key: value for key, value in changes.items() if _differs(rig, key, value)}
        if changes:
            rig = rig_mod.update(changes)
    elif not configured:
        found = zed.detect(zed.interpreters(planner, settings_mod.load()))
        if found:
            theme.info(
                "ZED cameras connected: " + ", ".join(f"{cam.serial} ({cam.model})" for cam in found),
                "`tandem rig set cameras.ROLE.serial SERIAL`, or `tandem init` at a terminal",
            )

    configured = rig.cameras.configured()
    if configured:
        missing = rig.missing_calibration()
        theme.ok(
            "Cameras: " + ", ".join(f"{role} {cam.serial}" for role, cam in configured.items()),
            f"{len(configured) - len(missing)} of {len(configured)} with extrinsics in {rig.calibration_file()}",
        )
    else:
        theme.warn(
            "No cameras yet: this machine cannot collect until it has them",
            "`tandem rig set cameras.hand.serial SERIAL` and `tandem rig set cameras.external.serial SERIAL`",
        )
    # What the planner says will stop it collecting on this rig -- extrinsics it reads that are missing, an arm
    # it does not drive -- in its own words, with its own fix.
    rig_cli.warn_planner_rig_checks()


def _say_found(found: list | None) -> None:
    if found is None:
        theme.info(
            "Could not list the ZED cameras: the ZED Python API is not installed yet (install the ZED SDK, "
            "then `tandem planners install tiptop`)",
            "type each serial: it is on the camera's label, and ZED Explorer shows it",
        )
    elif not found:
        theme.warn("The ZED SDK sees no camera", "check each is plugged in; type the serials to set them anyway")
    else:
        theme.info(f"The ZED SDK sees {len(found)} camera(s):")
        for cam in found:
            note = "" if cam.available else " · in use by another process"
            theme.info(f"  {cam.serial}  {cam.model}{note}")


def _ask_cameras(rig: rig_mod.Rig, found: list | None) -> dict[str, Any]:
    """Each role's serial, defaulting to the rig's, else to what the listing suggests; then perception's camera."""
    from tandem.core import zed

    seen = {cam.serial for cam in found or ()}
    suggested = zed.suggest(found or [], rig_mod.ROLES)
    changes: dict[str, Any] = {}
    taken: dict[str, str] = {}
    for role in rig_mod.ROLES:
        cam = getattr(rig.cameras, role)
        default = cam.serial if cam is not None else suggested.get(role, NONE)
        while True:
            serial = _ask(
                f"{_CAMERA_LABELS[role]} ('{NONE}' if there is none)", f"cameras.{role}.serial", default, blank=True
            )
            if serial != NONE and serial in taken:
                theme.warn(f"{serial} is already the {taken[serial]} camera; a camera fills one role")
                continue
            break
        if serial == NONE:
            changes[f"cameras.{role}"] = None
            continue
        taken[serial] = role
        changes[f"cameras.{role}.serial"] = serial
        if found is not None and serial not in seen:
            theme.warn(f"{serial} is not connected right now", "kept; plug it in before collecting")
    perception = rig.cameras.perception
    while True:
        answer = typer.prompt("  Which camera does perception read? [external/hand]", default=perception).strip()
        if answer in ("external", "hand"):
            break
        theme.warn(f"{answer!r} is neither external nor hand")
    changes["cameras.perception"] = answer
    return changes


#: What a camera question takes for "this machine has no such camera".
NONE = "none"


def _ask(label: str, key: str, default: str, *, blank: bool = False) -> str:
    """One rig question, asked again until the answer is a value the rig takes."""
    while True:
        answer = typer.prompt(f"  {label}", default=default).strip()
        if blank and answer.lower() in (NONE, ""):
            return NONE
        try:
            return _checked_rig_value(key, answer)
        except TandemError as exc:
            theme.warn(exc.message)


def _differs(rig: rig_mod.Rig, key: str, value: Any) -> bool:
    """Whether setting ``key`` to ``value`` changes the rig: an unchanged answer leaves the file alone."""
    node: Any = rig.model_dump(mode="python")
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return value is not None
        node = node[part]
    return node != value


# --------------------------------------------------------------------------- profiles


def _setup_profiles(requested: str | None) -> str:
    """Add the paper's five tasks that are missing, and settle the active profile. Returns its name.

    `--profile` if given (it must exist), else the active profile when it exists, else the first of the
    paper's. None of them is ever overwritten: once copied, a profile is its owner's.
    """
    written = profiles.seed_builtins()
    if written:
        theme.ok(
            f"Added the paper's {'five tasks' if len(written) == len(profiles.BUILTIN) else ', '.join(written)}",
            str(profiles.profiles_root()),
        )
    elif not profiles.held_back():
        theme.ok("The paper's five tasks are here", str(profiles.profiles_root()))
    for name in profiles.held_back():
        theme.warn(
            f"{name} was not added: a profile of that name is still in the old layout",
            "`tandem profile migrate` moves it; `tandem init` again then adds the rest",
        )

    cfg = settings_mod.load()
    if requested is not None:
        _check_profile_to_activate(requested)
        active = requested
    elif profiles.exists(cfg.active_profile):
        active = cfg.active_profile
    else:
        active = profiles.BUILTIN[0]
    if cfg.active_profile != active:
        cfg.active_profile = active
        settings_mod.save(cfg)
    theme.ok(f"Active profile: {active}", "`tandem profile use NAME` switches")
    return active


def _check_profile_to_activate(name: str) -> None:
    """--profile names a profile that exists, or one of the paper's (which the profiles step adds)."""
    if profiles.exists(name) or name in profiles.BUILTIN:
        return
    import difflib

    known = sorted({*profiles.list_names(), *profiles.BUILTIN})
    close = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
    raise ProfileError(
        f"--profile {name!r}: there is no such profile to make active.",
        hint=(f"Did you mean {close[0]!r}? " if close else "")
        + f'`tandem profile create {name} --prompt "..."` makes it; `tandem profile list` shows the rest.',
    )


def _say_when_the_planners_differ(active: str, planner: str) -> None:
    """A machine set up for one planner, whose active profile plans with another, is told how to line them up."""
    try:
        backend = profiles.load(active, require_installed=False).planner.backend
    except TandemError:
        return
    if backend != planner:
        theme.info(
            f"Profile {active!r} plans with {backend}, and this machine is set up for {planner}",
            f'`tandem profile create NAME --prompt "..." --use` makes a profile that plans with {planner}; '
            f"`tandem planners use {planner}` switches this one",
        )


def _setup_teleop(*, interactive: bool, repair: bool) -> None:
    from tandem.cli import executors as executors_cli
    from tandem.executors import teleop as teleop_executor

    cfg = settings_mod.load()
    if cfg.teleop.enabled and not repair and not teleop_executor.unmet_requirements(cfg):
        theme.ok("Teleop hand-off is configured", cfg.teleop.droid_dir or f"VR, {cfg.teleop.controller} controller")
        return

    theme.info('"Switch to teleop" lends the arm to a human mid-task and takes it back,')
    theme.info("without homing — the plan resumes from wherever they left it.")
    theme.info("tandem builds the teleop driver's environment (DROID's workstation side) for it.")

    if not interactive:
        theme.info("Skipped (non-interactive). Set it up with `tandem executors install teleop`.")
        return
    if not typer.confirm("  Set it up now?", default=False):
        theme.info("Skipped — collection works without it; the hand-off button stays disabled.")
        return

    controller = typer.prompt(
        "  Which VR controller drives the arm (right or left)",
        default=cfg.teleop.controller or "right",
        type=click.Choice(["right", "left"]),
        show_choices=False,
    )
    if controller != cfg.teleop.controller:
        cfg.teleop.controller = controller
        settings_mod.save(cfg)
    try:
        # The person just said yes, so this asks nothing more (pixi was offered with the planner's runtime).
        executors_cli.install_teleop(yes=True)
    except TandemError as exc:
        # Teleop is optional: a failed build is said, with how to retry, and the rest of init goes on.
        theme.warn(f"The teleop runtime did not build: {exc.message}", exc.hint or "")
        theme.info("Collection works without it. `tandem executors install teleop` tries again.")


def _summary(*, viz_only: bool, profile_name: str, planner: str | None) -> None:
    cfg = settings_mod.load()
    theme.rule("ready")
    try:
        planner, runtime = runtime_cli.planner_runtime(planner=planner, profile_name=profile_name)
        where = runtime.status().path if runtime is not None else "none needed (pure Python)"
    except TandemError as exc:
        planner, where = "unknown", one_line(exc.message)
    if rig_mod.exists():
        rig_row = f"{paths.rig_file()}"
    else:
        rig_row = "none (visualization only)" if viz_only else "not written"
    theme.kv(
        [
            ("profile", profile_name),
            ("planner", planner),
            ("data root", cfg.resolved_data_root()),
            ("rig", rig_row),
            ("runtime", "not installed (visualization only)" if viz_only else where),
            ("config", paths.config_file()),
        ]
    )
    steps = [("tandem doctor", "confirm everything is wired up")]
    if not viz_only:
        steps.append(("tandem collect", f"collect with {profile_name}"))
        steps.append(('tandem profile create NAME --prompt "..."', "a task of your own, with the paper's settings"))
    steps.append(("tandem ui", "browse and visualize trajectories in the browser"))
    theme.next_steps(steps)
    theme.blank()
