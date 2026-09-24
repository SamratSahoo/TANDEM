"""`tandem config` — global settings and credentials."""

from __future__ import annotations

import json
import sys

import typer

from tandem.cli import theme
from tandem.core import paths, secrets
from tandem.core import settings as settings_mod
from tandem.core.errors import TandemError

app = typer.Typer(no_args_is_help=True, help="Global settings and credentials.")


@app.command("list", help="Show every setting and where credentials come from.")
def list_settings(
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    cfg = settings_mod.load()
    flat = settings_mod.flatten(cfg)

    if as_json:
        payload = {
            "settings": {k: v for k, v in flat},
            "resolved": {
                "data_root": str(cfg.resolved_data_root()),
                "runtime_dir": str(cfg.resolved_runtime_dir()),
                "config_file": str(paths.config_file()),
                "credentials_file": str(paths.credentials_file()),
            },
            "credentials": {
                "gemini_api_key": {
                    "source": secrets.gemini_key_source(),
                    "masked": secrets.mask(secrets.gemini_api_key()),
                },
                "hf_token": {
                    "source": secrets.hf_token_source(),
                    "masked": secrets.mask(secrets.hf_token()),
                },
            },
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    theme.heading("settings", str(paths.config_file()))
    theme.kv(flat)

    theme.blank()
    theme.heading("resolved paths")
    theme.kv(
        [
            ("data root", cfg.resolved_data_root()),
            ("runtime", cfg.resolved_runtime_dir()),
            ("profiles", cfg.profiles_root()),
        ]
    )

    theme.blank()
    theme.heading("credentials", str(paths.credentials_file()))
    theme.kv(
        [
            ("gemini api key", f"{secrets.mask(secrets.gemini_api_key())}  ({secrets.gemini_key_source()})"),
            ("huggingface token", f"{secrets.mask(secrets.hf_token())}  ({secrets.hf_token_source()})"),
        ]
    )


@app.command("get", help="Print one setting.")
def get(key: str = typer.Argument(..., help="Dotted key, e.g. ui.port")) -> None:
    value = settings_mod.get_dotted(settings_mod.load(), key)
    typer.echo(str(value))


@app.command("set", help="Change one setting.")
def set_(
    key: str = typer.Argument(..., help="Dotted key, e.g. ui.port"),
    value: str = typer.Argument(..., help="New value"),
) -> None:
    cfg = settings_mod.load()
    updated = settings_mod.set_dotted(cfg, key, value)
    path = settings_mod.save(updated)
    theme.ok(f"{key} = {settings_mod.get_dotted(updated, key)}", str(path))


@app.command(
    "set-gemini-key",
    help="Store the Gemini API key: phase planning uses it, and so does a planner whose perception calls "
    "Gemini (TiPToP's does).",
)
def set_gemini_key(
    from_stdin: bool = typer.Option(
        False,
        "--stdin",
        help="Read the key from stdin instead of prompting, so it never enters shell history.",
    ),
    key: str = typer.Option("", "--key", help="Pass the key directly (visible in shell history — prefer --stdin)."),
) -> None:
    """Stored at ``~/.config/tandem/credentials.toml`` with mode 0600.

    Perception calls Gemini once per rollout to turn the task string into object bounding
    boxes and grounded predicates, so collection cannot run without this.
    """
    if from_stdin:
        value = sys.stdin.readline().strip()
        if not value:
            raise TandemError("Nothing arrived on stdin.", hint='Try: echo "$KEY" | tandem config set-gemini-key --stdin')
    elif key:
        value = key.strip()
    else:
        if not sys.stdin.isatty():
            raise TandemError(
                "No key given and stdin is not a terminal.",
                hint='Pipe it in: echo "$KEY" | tandem config set-gemini-key --stdin',
            )
        value = typer.prompt("Gemini API key", hide_input=True).strip()

    path = secrets.set_gemini_api_key(value)
    theme.ok(f"Gemini API key stored  {secrets.mask(value)}", str(path))
    theme.info("Get a key at https://aistudio.google.com/apikey")


@app.command("set-hf-token", help="Store the HuggingFace token used for dataset uploads.")
def set_hf_token(
    from_stdin: bool = typer.Option(False, "--stdin", help="Read the token from stdin."),
) -> None:
    if from_stdin:
        value = sys.stdin.readline().strip()
    else:
        if not sys.stdin.isatty():
            raise TandemError("No token given and stdin is not a terminal.", hint="Use --stdin.")
        value = typer.prompt("HuggingFace token", hide_input=True).strip()
    if not value:
        raise TandemError("Refusing to store an empty token.")
    path = secrets.set_hf_token(value)
    theme.ok(f"HuggingFace token stored  {secrets.mask(value)}", str(path))


@app.command("path", help="Print the config file path.")
def path_() -> None:
    typer.echo(str(paths.config_file()))


@app.command("edit", help="Open the config file in $EDITOR.")
def edit() -> None:
    path = paths.config_file()
    paths.ensure_dir(path.parent)
    if not path.is_file():
        settings_mod.save(settings_mod.load())
    from tandem.cli.editor import open_in_editor

    open_in_editor(path)
    # Re-read so a syntax error is reported now, not at the next command.
    settings_mod.load(force=True)
    theme.ok("config is valid", str(path))
