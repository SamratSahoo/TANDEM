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
from tandem.core import importers, paths, probe, profiles, secrets
from tandem.core import settings as settings_mod
from tandem.core.errors import TandemError


def init(
    viz_only: bool = typer.Option(
        False,
        "--viz-only",
        help="Set up for browsing and visualizing trajectories only — no GPU runtime, no robot.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Accept every default; ask nothing."),
    repair: bool = typer.Option(False, "--repair", help="Redo steps that are already done."),
    profile_name: str = typer.Option("default", "--profile", help="Name for the profile to create."),
    import_from: Path = typer.Option(
        None,
        "--import-from",
        help="Import robot, camera and TAMP settings from a hitl-tamp-vla checkout.",
        exists=True,
        file_okay=False,
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

    blocking = [c for c in checks if c.state == probe.FAIL]
    if blocking and not viz_only:
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
            raise TandemError(
                "Preflight found blocking problems: " + ", ".join(c.name for c in blocking),
                hint="Fix them and re-run `tandem init`, or use --viz-only for a laptop setup.",
            )
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

    # ---- 4. the GPU runtime -------------------------------------------------
    if not viz_only:
        theme.rule("gpu runtime")
        _build_runtime(profile_name, interactive=interactive, repair=repair)
        theme.blank()

    # ---- 5. the Gemini key --------------------------------------------------
    theme.rule("gemini api key")
    source = secrets.gemini_key_source()
    if source != "none" and not repair:
        theme.ok(f"Already set  {secrets.mask(secrets.gemini_api_key())}", f"from {source}")
    elif viz_only:
        theme.info("Not needed for visualization — skipping.")
    else:
        theme.info("Perception calls Gemini once per rollout to turn the task string")
        theme.info("into object bounding boxes and goal predicates.")
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

    # ---- 6. the first profile -----------------------------------------------
    theme.rule("profile")
    if profiles.exists(profile_name) and not repair:
        theme.ok(f"Profile {profile_name!r} already exists", str(profiles.profiles_root() / profile_name))
    else:
        _create_profile(
            profile_name, import_from=import_from, interactive=interactive, viz_only=viz_only
        )
    cfg = settings_mod.load()
    cfg.active_profile = profile_name
    settings_mod.save(cfg)
    theme.blank()

    # ---- 7. teleop hand-off (optional) --------------------------------------
    if not viz_only:
        theme.rule("teleop hand-off  (optional)")
        _setup_teleop(interactive=interactive, repair=repair)
        theme.blank()

    # ---- 8. done ------------------------------------------------------------
    _summary(viz_only=viz_only, profile_name=profile_name)


# --------------------------------------------------------------------------- steps


def _build_runtime(profile_name: str, *, interactive: bool, repair: bool) -> None:
    """Build the runtime of the planner this profile uses -- the one a new profile gets, before it exists."""
    from tandem.planners import registry
    from tandem.planners.runtime import RecipeRuntime

    planner, runtime = runtime_cli.planner_runtime(profile_name=profile_name)
    title = registry.info(planner).title
    if runtime is None:
        theme.ok(f"{title} is pure Python", "there is no runtime to build")
        return
    status = runtime.status()

    if status.installed and not repair:
        theme.ok("Runtime is already built", str(status.path))
        return

    needs_pixi = isinstance(runtime, RecipeRuntime) and runtime.recipe.environment is not None
    if needs_pixi and probe.find_pixi() is None:
        theme.info(f"pixi is the environment manager {title}'s planner stack needs.")
        theme.info("It installs to ~/.pixi and touches nothing else.")
        if interactive and not typer.confirm("  Install pixi now?", default=True):
            raise TandemError(
                "pixi is required to build the runtime.",
                hint="Install it from https://pixi.sh, then re-run `tandem init`.",
            )
        theme.busy("Installing pixi")
        runtime_cli.install_pixi(log=lambda _line: None)
        theme.ok("pixi installed")

    for note in runtime.recipe.notes if isinstance(runtime, RecipeRuntime) else ():
        theme.info(note)
    if interactive and not typer.confirm("  Build it now?", default=True):
        theme.warn("Skipped", "run `tandem runtime build` when you are ready")
    else:
        runtime_cli.run_build(runtime, force=repair)



def _preflight(*, viz_only: bool) -> list[probe.Check]:
    cfg = settings_mod.load()
    checks = [probe.check_python(), probe.check_platform()]
    if viz_only:
        checks.append(probe.check_ffmpeg())
        return checks

    checks += [
        probe.check_nvidia_driver(),
        probe.check_cuda_runtime(),
        probe.check_nvcc(),
        probe.check_disk(cfg.resolved_runtime_dir()),
        probe.check_pixi(),
        probe.check_ffmpeg(),
        probe.check_zed_sdk(),
    ]
    return checks


def _render_checks(checks: list[probe.Check]) -> None:
    table = theme.table("", "", "", box_style="none")
    table.columns[0].width = 3
    table.columns[1].width = 22
    table.columns[2].style = "faint"
    for check in checks:
        table.add_row(theme.status_glyph(check.state), check.name, check.detail)
    theme.console().print(table)


def _create_profile(
    name: str, *, import_from: Path | None, interactive: bool, viz_only: bool = False
) -> None:
    calibration: dict = {}
    notes: list[str] = []

    if import_from is None and interactive:
        theme.info("A profile holds the robot, the cameras, the task and the TAMP settings.")
        guess = _guess_monorepo()
        if guess is not None:
            prompt = f"  Import settings from {guess}?"
            if typer.confirm(prompt, default=True):
                import_from = guess

    if import_from is not None:
        tamp_config = None
        options = importers.list_tamp_configs(import_from)
        if options and interactive:
            theme.info(f"{len(options)} TAMP config(s) found in that checkout.")
            for i, path in enumerate(options[:12], 1):
                theme.console().print(f"    [accent]{i:>2}[/accent]  [faint]{path.stem}[/faint]")
            if len(options) > 12:
                theme.console().print(f"    [faint]… and {len(options) - 12} more[/faint]")
            answer = typer.prompt("  Import one? (number, or blank for none)", default="").strip()
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                tamp_config = options[int(answer) - 1]
        profile, calibration, notes = importers.build_profile(
            name, root=import_from, tamp_config=tamp_config
        )
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

    if interactive:
        profile.task.prompt = typer.prompt("  Task prompt", default=profile.task.prompt).strip()

    path = profiles.save(profile)
    if calibration:
        profile.calibration_file().write_text(json.dumps(calibration, indent=2) + "\n")

    theme.ok(f"Created profile {name!r}", origin)
    theme.info(str(path))
    for note in notes:
        theme.info(note)

    missing = profiles.missing_calibration(profile)
    if missing:
        theme.warn(
            f"No extrinsics for camera serial(s) {', '.join(missing)}",
            "collection will refuse to start until they exist",
        )


def _guess_monorepo() -> Path | None:
    """Look for a hitl-tamp-vla checkout next to the current directory.

    Only a suggestion — it is always confirmed before anything is read.
    """
    here = Path.cwd().resolve()
    for base in (here, *here.parents):
        for name in ("hitl-tamp-vla", "tamp-vla"):
            candidate = base / name
            if candidate.is_dir() and importers.find_sources(candidate)["tiptop_config"]:
                return candidate
    return None


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
    theme.kv(
        [
            ("profile", profile_name),
            ("data root", cfg.resolved_data_root()),
            ("runtime", "not installed (visualization only)" if viz_only else cfg.resolved_runtime_dir()),
            ("config", paths.config_file()),
        ]
    )
    steps = [("tandem doctor", "confirm everything is wired up")]
    if not viz_only:
        steps.append(("tandem collect", "run a collection session in the terminal"))
    steps.append(("tandem ui", "browse and visualize trajectories in the browser"))
    theme.next_steps(steps)
    theme.blank()
