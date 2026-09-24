"""`tandem init` — the onboarding wizard.

Idempotent and resumable: every step checks its own postcondition first and says
"already done" rather than redoing work. Interrupting it and re-running is safe — cuRobo's
build fingerprint means even the expensive kernel compile is skipped when nothing changed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from tandem import resources
from tandem.cli import runtime as runtime_cli
from tandem.cli import theme
from tandem.core import paths, probe, profiles, secrets
from tandem.core import settings as settings_mod
from tandem.core.errors import TandemError, one_line
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
        help="Redo steps that are already done: rebuild the planner's runtime, and ask again for the key "
        "and the teleop settings. An existing profile is never touched by it.",
    ),
    profile_name: str = typer.Option("default", "--profile", help="Name for the profile to create."),
    import_from: Path = typer.Option(
        None,
        "--import-from",
        help="Import the setup from the planner's own older configuration (for TiPToP: a hitl-tamp-vla "
        "checkout).",
        exists=True,
        file_okay=False,
    ),
    planner: str = typer.Option(
        None,
        "--planner",
        help="The planner the profile plans with, and whose runtime is built (see `tandem planners list`). "
        "Default: the profile's own, or for a new one the machine's default (tiptop).",
    ),
    preset: str = typer.Option(
        None,
        "--preset",
        help="Lay a named preset over the profile init creates, such as `paper` (the paper's collection "
        "settings). `tandem profile presets` lists them. An existing profile is left as it is.",
    ),
) -> None:
    interactive = theme.is_tty() and not yes
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
    if not viz_only:
        _stop_on_blocking(checks, interactive=interactive)
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

    # ---- 4. the planner -----------------------------------------------------
    # Chosen before anything is built: the runtime step builds THIS planner's runtime, and the profile
    # step creates the profile with it. A laptop builds nothing, so it is only shown the choice it
    # made on the command line, if any.
    if not viz_only:
        theme.rule("planner")
        planner = _choose_planner(profile_name, planner, interactive=interactive)
        _planner_preflight(planner, interactive=interactive, repair=repair)
        theme.blank()

    # ---- 5. the planner's runtime -------------------------------------------
    if not viz_only:
        theme.rule("planner runtime")
        _build_runtime(profile_name, interactive=interactive, repair=repair, planner=planner)
        theme.blank()

    # ---- 6. the Gemini key --------------------------------------------------
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

    # ---- 7. the first profile -----------------------------------------------
    theme.rule("profile")
    # An existing profile is kept, --repair or not: it is somebody's collection setup (phase planning on,
    # a preset's settings, the rig's cameras), and rebuilding it from the template to repair a runtime
    # would lose all of it without a word. It is only pointed at the planner init was asked to set up.
    if profiles.exists(profile_name):
        theme.ok(f"Profile {profile_name!r} already exists", str(profiles.profiles_root() / profile_name))
        if planner is not None:
            _switch_planner(profile_name, planner)
        if preset:
            theme.warn(
                f"--preset {preset} was not applied to the existing profile {profile_name!r}",
                f"`tandem profile create NEW --from {profile_name} --preset {preset}` makes a copy with it",
            )
    else:
        _create_profile(
            profile_name,
            import_from=import_from,
            interactive=interactive,
            viz_only=viz_only,
            planner=planner,
            preset=preset,
        )
    cfg = settings_mod.load()
    cfg.active_profile = profile_name
    settings_mod.save(cfg)
    theme.blank()

    # ---- 8. teleop hand-off (optional) --------------------------------------
    if not viz_only:
        theme.rule("teleop hand-off  (optional)")
        _setup_teleop(interactive=interactive, repair=repair)
        theme.blank()

    # ---- 9. done ------------------------------------------------------------
    _summary(viz_only=viz_only, profile_name=profile_name)


# --------------------------------------------------------------------------- steps


def _choose_planner(profile_name: str, requested: str | None, *, interactive: bool) -> str:
    """Which planner this machine is set up for: the catalog shown, one chosen, and checked.

    ``--planner`` wins. Otherwise a profile that already exists keeps the planner it names -- init is
    re-run on a working machine, and must not quietly swap its planner -- and a new one gets the
    machine's default, or, at a terminal with more than one planner to choose from, the one picked.
    The choice is checked against the registry here, before twenty minutes go into a runtime.
    """
    from tandem.cli import planners as planners_cli
    from tandem.planners import registry

    payload = planners_cli.catalog_payload(profile_name=profile_name)
    planners_cli.render_catalog(payload["planners"])
    for row in payload["planners"]:
        if row["status"] == planners_cli.BROKEN:
            theme.fail(f"{row['name']}: {row['detail']}")

    # Read as written: a profile naming a planner this machine does not have is exactly the one --planner
    # is used to repair. Without --planner, that planner is what is checked below, and refused by name.
    existing = (
        profiles.load(profile_name, require_installed=False).planner.backend
        if profiles.exists(profile_name)
        else None
    )
    usable = [row["name"] for row in payload["planners"] if row["ok"]]
    if requested:
        choice = requested
    elif existing:
        choice = existing
    elif interactive and len(usable) > 1:
        default = payload["default_planner"] if payload["default_planner"] in usable else usable[0]
        choice = typer.prompt("  Planner", default=default).strip()
    else:
        choice = payload["default_planner"]

    info = registry.info(choice)  # unknown or broken: the registry's own error, with the nearest name
    detail = f"profile {profile_name!r} plans with it" if existing == choice else choice
    theme.ok(f"Planner: {info.title}", detail)
    if existing and existing != choice:
        theme.info(f"Profile {profile_name!r} plans with {existing} now; it will be switched to {choice}.")
    return choice


def _switch_planner(profile_name: str, planner: str) -> None:
    """Point an existing profile at the planner init was asked to set up, saying what changed."""
    from tandem.cli import planners as planners_cli

    result = planners_cli.use_planner(planner, profile_name=profile_name)
    if not result["changed"]:
        return
    theme.ok(f"Profile {profile_name!r} now plans with {result['display_name']}", f"was {result['previous']}")
    planners_cli.describe_switch(result)


def _build_runtime(profile_name: str, *, interactive: bool, repair: bool, planner: str | None = None) -> None:
    """Build ``planner``'s runtime -- by default the one the profile uses, or a new profile would get.

    The consent flow for pixi and the build itself are `tandem planners install`'s (``cli/runtime``),
    so the wizard and the command cannot drift apart. `init` is "accept every default" when it cannot
    ask, so without a terminal it installs pixi rather than failing.
    """
    from tandem.planners import registry

    planner, runtime = runtime_cli.planner_runtime(planner=planner, profile_name=profile_name)
    title = registry.info(planner).title
    if runtime is None:
        theme.ok(f"{title} is pure Python", "there is no runtime to build")
        return
    status = runtime.status()

    if status.installed and not repair:
        theme.ok("Runtime is already built", str(status.path))
        return

    if runtime_cli.needs_pixi(runtime):
        runtime_cli.ensure_pixi(title, ask=interactive, allowed=True)

    for note in getattr(getattr(runtime, "recipe", None), "notes", ()) or ():
        theme.info(note)
    if interactive and not typer.confirm("  Build it now?", default=True):
        theme.warn("Skipped", f"run `tandem planners install {planner}` when you are ready")
    else:
        runtime_cli.run_build(runtime, force=repair, planner=planner)



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


def _stop_on_blocking(checks: list[probe.Check], *, interactive: bool) -> None:
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
            + ("" if runtime_only else " On a laptop, --viz-only sets up for browsing only."),
        )


def _render_checks(checks: list[probe.Check]) -> None:
    table = theme.table("", "", "", box_style="none")
    table.columns[0].width = 3
    table.columns[1].width = 22
    table.columns[2].style = "faint"
    for check in checks:
        table.add_row(theme.status_glyph(check.state), check.name, check.detail)
    theme.console().print(table)


def _create_profile(
    name: str,
    *,
    import_from: Path | None,
    interactive: bool,
    viz_only: bool = False,
    planner: str | None = None,
    preset: str | None = None,
) -> None:
    from tandem.cli import planners as planners_cli
    from tandem.cli import profile as profile_cli
    from tandem.core import presets

    # Resolved before anything is written: a default naming a planner this machine no longer has
    # stops here, not in a profile that every later command refuses.
    planner = planners_cli.planner_for_new_profile(planner)
    calibration: dict = {}
    notes: list[str] = []

    # The planner's own importer, if it has older configuration to import from (TiPToP: a checkout of
    # the hitl-tamp-vla monorepo it came from).
    importer = registry.importer(planner)
    if import_from is None and interactive and importer is not None:
        theme.info("A profile holds the task, the cameras, the planner and the planner's settings.")
        guess = importer.find(Path.cwd())
        if guess is not None:
            prompt = f"  Import settings from {guess}?"
            if typer.confirm(prompt, default=True):
                import_from = guess

    if import_from is not None:
        if importer is None:
            raise TandemError(
                f"The {registry.info(planner).title} planner has nothing to import a profile from.",
                hint="Run `tandem init` without --import-from, or with --planner naming the planner whose setup it is.",
            )
        config = None
        options = importer.configs(import_from)
        if options and interactive:
            theme.info(f"{len(options)} task config(s) found in {importer.source}.")
            for i, path in enumerate(options[:12], 1):
                theme.console().print(f"    [accent]{i:>2}[/accent]  [faint]{path.stem}[/faint]")
            if len(options) > 12:
                theme.console().print(f"    [faint]… and {len(options) - 12} more[/faint]")
            answer = typer.prompt("  Import one? (number, or blank for none)", default="").strip()
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                config = options[int(answer) - 1]
        profile, calibration, notes = importer.build(name, source=import_from, config=config)
        origin = f"imported from {import_from}"
    else:
        profile = profiles.load_file(resources.path("profile_template.yml"), name=name)
        origin = "from the built-in template"

    if viz_only and import_from is None:
        # On a laptop a profile is just a folder of trajectories collected elsewhere. Keeping
        # the template's cameras would mean warning about extrinsics for hardware that is not
        # here and never will be.
        profile.cameras = profiles.CamerasSpec()
        profile.description = profile.description or "trajectories collected elsewhere"

    typed = None
    if interactive:
        typed = typer.prompt("  Task prompt", default=profile.task.prompt).strip()
        profile.task.prompt = typed

    # The planner init set this machine up for, whether the rest came from the template or an import:
    # a profile naming a different planner from the runtime just built would not collect. An import
    # is already that planner's, options and all; the template's options are TiPToP's, and go when
    # the planner is another.
    if profile.planner.backend != planner:
        profile.planner = profiles.planner_spec(planner, profile=name)
    # After the planner is settled, as `tandem profile create --preset` does: a preset is looked up for
    # the profile's planner. The prompt typed above is this profile's own, and wins over the preset.
    laid = None
    if preset:
        before = profile
        profile = presets.apply(profile, preset)
        changes = presets.differences(before.model_dump(mode="python"), profile.model_dump(mode="python"))
        laid = (presets.layers(preset, profile.planner.backend), changes)
        if typed:
            profile.task.prompt = typed
    path = profiles.save(profile)
    if calibration:
        profile.calibration_file().write_text(json.dumps(calibration, indent=2) + "\n")

    theme.ok(f"Created profile {name!r}", origin)
    theme.info(str(path))
    profile_cli.show_notes(notes)
    if laid is not None:
        profile_cli.show_preset(profile, *laid)
    profile_cli.warn_planner_profile_checks(profile)


def _setup_teleop(*, interactive: bool, repair: bool) -> None:
    cfg = settings_mod.load()
    if cfg.teleop.enabled and not repair:
        theme.ok("Teleop hand-off is configured", cfg.teleop.droid_dir)
        return

    theme.info('"Switch to teleop" lends the arm to a human mid-task and takes it back,')
    theme.info("without homing — the plan resumes from wherever they left it.")
    theme.info("It needs a DROID checkout and that environment's Python.")

    if not interactive:
        theme.info("Skipped (non-interactive). Configure it with `tandem config set teleop.enabled true`.")
        return
    if not typer.confirm("  Set it up now?", default=False):
        theme.info("Skipped — collection works without it; the hand-off button stays disabled.")
        return

    droid_dir = typer.prompt("  DROID checkout directory", default=cfg.teleop.droid_dir or "").strip()
    if not droid_dir or not Path(droid_dir).expanduser().is_dir():
        theme.warn("That directory does not exist — leaving teleop disabled.")
        return
    python = typer.prompt(
        "  Python for the DROID environment", default=cfg.teleop.python or sys.executable
    ).strip()
    if not Path(python).expanduser().is_file():
        theme.warn("That interpreter does not exist — leaving teleop disabled.")
        return

    cfg.teleop.enabled = True
    cfg.teleop.droid_dir = str(Path(droid_dir).expanduser().resolve())
    cfg.teleop.python = str(Path(python).expanduser().resolve())
    settings_mod.save(cfg)
    theme.ok("Teleop hand-off enabled", cfg.teleop.droid_dir)


def _summary(*, viz_only: bool, profile_name: str) -> None:
    cfg = settings_mod.load()
    theme.rule("ready")
    try:
        planner, runtime = runtime_cli.planner_runtime(profile_name=profile_name)
        where = runtime.status().path if runtime is not None else "none needed (pure Python)"
    except TandemError as exc:
        planner, where = "unknown", one_line(exc.message)
    theme.kv(
        [
            ("profile", profile_name),
            ("planner", planner),
            ("data root", cfg.resolved_data_root()),
            ("runtime", "not installed (visualization only)" if viz_only else where),
            ("config", paths.config_file()),
        ]
    )
    steps = [("tandem doctor", "confirm everything is wired up")]
    if not viz_only:
        steps.append(("tandem collect", "run a collection session in the terminal"))
    steps.append(("tandem ui", "browse and visualize trajectories in the browser"))
    theme.next_steps(steps)
    theme.blank()
