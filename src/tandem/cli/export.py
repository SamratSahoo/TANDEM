"""`tandem export` — turn collected trajectories into training datasets."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from tandem.cli import theme
from tandem.core import paths, profiles, secrets, trajectories
from tandem.core import settings as settings_mod
from tandem.core.errors import TandemError

app = typer.Typer(no_args_is_help=True, help="Export trajectories to other dataset formats.")


@app.command("lerobot", help="Build a LeRobot v3.0 dataset from a profile's successful trajectories.")
def lerobot(
    profile_name: str = typer.Argument(None, help="Profile name (default: the active one)."),
    repo: str = typer.Option(None, "--repo", help="HuggingFace dataset repo, e.g. myorg/toys20."),
    out: Path = typer.Option(None, "--out", help="Where to write the dataset locally."),
    push: bool = typer.Option(False, "--push", help="Upload to the Hub after building."),
    private: bool = typer.Option(None, "--private/--public", help="Repo visibility when pushing."),
    max_episodes: int = typer.Option(None, "--max-episodes", "-n", help="Only export the first N."),
    force: bool = typer.Option(
        False, "--force", help="Replace what is at the destination even if tandem did not build it."
    ),
) -> None:
    """Matches `lerobot/droid_1.0.1`'s schema, so the result feeds a π₀.₅-DROID finetune."""
    try:
        from tandem.export import build as build_mod
    except ImportError as exc:
        raise TandemError(
            f"The export dependencies are not installed ({exc}).",
            hint=(
                "Add them with: pipx inject tandem-tamp av pyarrow huggingface_hub  "
                "(or `pip install 'tandem-tamp[export]'` inside a virtualenv)."
            ),
        ) from exc

    profile = profiles.load(profile_name, require_installed=False)
    cfg = settings_mod.load()

    repo_id = repo or profile.export.hf_repo
    if not repo_id:
        raise TandemError(
            f"Profile {profile.name!r} has no export.hf_repo and no --repo was given.",
            hint="Pass --repo myorg/dataset-name, or set it in the profile.",
        )
    if "/" not in repo_id:
        if not cfg.hf_org:
            raise TandemError(
                f"{repo_id!r} has no owner and no default is configured.",
                hint="Use --repo owner/name, or run `tandem config set hf_org <owner>`.",
            )
        repo_id = f"{cfg.hf_org}/{repo_id}"

    out_root = out or (cfg.resolved_data_root() / "exports")
    is_private = profile.export.private if private is None else private

    counts = trajectories.counts(profile)
    theme.blank()
    theme.heading("export · lerobot v3.0", repo_id)
    theme.kv(
        [
            ("profile", profile.name),
            ("source", f"{counts['success']} successful trajectories"),
            ("destination", out_root / repo_id),
            ("push", f"yes ({'private' if is_private else 'public'})" if push else "no"),
        ]
    )
    theme.blank()

    token = None
    if push:
        token = secrets.hf_token()
        if not token:
            raise TandemError(
                "Pushing needs a HuggingFace token and none is configured.",
                hint="Run `tandem config set-hf-token`, or drop --push to build locally.",
            )

    _configure_logging()

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    with Progress(
        SpinnerColumn(style="accent"),
        TextColumn("[bold]episodes"),
        BarColumn(bar_width=24, complete_style="accent", finished_style="ok"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[faint]{task.description}"),
        console=theme.console(),
    ) as progress:
        task_id = progress.add_task("", total=None)

        def on_episode(name: str, written: int, total: int, ok: bool) -> None:
            progress.update(task_id, total=total, completed=written, description=name)

        result = build_mod.build_dataset(
            profile,
            repo_id=repo_id,
            out_root=out_root,
            push=push,
            private=is_private,
            max_episodes=max_episodes,
            token=token,
            on_episode=on_episode,
            force=force,
        )

    theme.blank()
    theme.ok(f"Wrote {result['written']} of {result['considered']} episode(s)", result["dataset_root"])
    if result["skipped"]:
        theme.blank()
        theme.heading("skipped")
        table = theme.table("trajectory", "why")
        for name, reason in result["skipped"]:
            table.add_row(name, f"[faint]{reason}[/faint]")
        theme.console().print(table)
    if result["pushed"]:
        theme.ok("Pushed", f"https://huggingface.co/datasets/{repo_id}")
    elif push:
        theme.warn("Nothing to push", "no episodes were written")


@app.command("manifest", help="Write a JSON index of a profile's trajectories.")
def manifest(
    profile_name: str = typer.Argument(None, help="Profile name (default: the active one)."),
    out: Path = typer.Option(None, "--out", help="Where to write it (default: stdout)."),
) -> None:
    """A dependency-free description of what was collected — handy for custom pipelines."""
    profile = profiles.load(profile_name, require_installed=False)
    items = trajectories.list_all(profile, with_size=True)
    payload = {
        "profile": profile.model_dump(mode="json"),
        "counts": trajectories.counts(profile),
        "trajectories": [t.to_dict() for t in items],
    }
    text = json.dumps(payload, indent=2, default=str)
    if out:
        out.write_text(text + "\n")
        theme.ok(f"Wrote {len(items)} entries", str(out))
    else:
        typer.echo(text)


def _configure_logging() -> None:
    """Route the builder's warnings through Rich so they match everything else."""
    import logging

    from rich.logging import RichHandler

    logger = logging.getLogger("tandem.export")
    if logger.handlers:
        return
    handler = RichHandler(
        console=theme.console(), show_path=False, show_time=False, markup=False, rich_tracebacks=False
    )
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Keep the build log next to everything else for a post-mortem.
    paths.ensure_dir(paths.log_dir())
    file_handler = logging.FileHandler(paths.log_dir() / "export.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
