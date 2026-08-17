"""The single source of visual truth for the CLI.

The palette is the same one the web UI uses, so the terminal and the browser read as one
product rather than two tools that happen to ship together.

Glyph vocabulary — used consistently, and nowhere else:

    ✔  done        ✖  failed       ▸  in progress
    ◆  prompt      !  warning      ·  detail
"""

from __future__ import annotations

import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Shared with server/static/theme.css — keep the two in step.
ACCENT = "#4f9dff"
VIOLET = "#a371f7"
GREEN = "#3fb950"
AMBER = "#d29922"
RED = "#f85149"
MUTED = "#8b98a9"
FAINT = "#5c6875"

THEME = Theme(
    {
        "accent": ACCENT,
        "violet": VIOLET,
        "ok": GREEN,
        "warn": AMBER,
        "err": RED,
        "muted": MUTED,
        "faint": FAINT,
        "key": f"bold {ACCENT}",
        "value": "default",
        "heading": "bold",
        "hint": AMBER,
        "code": f"bold {VIOLET}",
    }
)

OK = "✔"
FAIL = "✖"
BUSY = "▸"
PROMPT = "◆"
WARN = "!"
DOT = "·"

_console: Console | None = None
_err_console: Console | None = None


def _no_color() -> bool:
    return bool(os.environ.get("NO_COLOR")) or os.environ.get("TANDEM_NO_COLOR") == "1"


def console() -> Console:
    global _console
    if _console is None:
        _console = Console(theme=THEME, no_color=_no_color(), soft_wrap=False, highlight=False)
    return _console


def err_console() -> Console:
    global _err_console
    if _err_console is None:
        _err_console = Console(theme=THEME, no_color=_no_color(), stderr=True, highlight=False)
    return _err_console


def set_no_color(value: bool) -> None:
    """Called by the root callback for --no-color, before anything renders."""
    global _console, _err_console
    if value:
        os.environ["TANDEM_NO_COLOR"] = "1"
    _console = None
    _err_console = None


# --------------------------------------------------------------------------- primitives


def ok(message: str, detail: str = "") -> None:
    _line(OK, "ok", message, detail)


def fail(message: str, detail: str = "") -> None:
    _line(FAIL, "err", message, detail)


def busy(message: str, detail: str = "") -> None:
    _line(BUSY, "accent", message, detail)


def warn(message: str, detail: str = "") -> None:
    _line(WARN, "warn", message, detail)


def info(message: str, detail: str = "") -> None:
    _line(DOT, "faint", message, detail)


def _line(glyph: str, style: str, message: str, detail: str) -> None:
    text = Text()
    text.append(f"  {glyph} ", style=style)
    text.append(message)
    if detail:
        text.append(f"  {detail}", style="faint")
    console().print(text)


def blank() -> None:
    console().print()


def rule(title: str) -> None:
    console().rule(Text(title, style="accent"), style="faint", align="left")


def heading(title: str, subtitle: str = "") -> None:
    text = Text()
    text.append(f"{PROMPT} ", style="violet")
    text.append(title, style="bold")
    if subtitle:
        text.append("  ")
        # Subtitles carry status chips ("[err]failure[/err]"), so parse markup rather than
        # printing the tags literally.
        text.append_text(Text.from_markup(subtitle, style="faint"))
    console().print(text)


def banner(subtitle: str = "") -> None:
    """The wordmark. Shown by `init` and `ui` only — chatty tools feel cheap."""
    word = Text()
    # A blue→violet ramp across the six letters, matching the web UI's brand-logo gradient.
    ramp = ["#4f9dff", "#5f93ff", "#7288fb", "#857df7", "#9578f7", "#a371f7"]
    for char, colour in zip("tandem", ramp, strict=True):
        word.append(char, style=f"bold {colour}")
    if subtitle:
        word.append(f"   {subtitle}", style="faint")
    console().print()
    console().print(Text("  ") + word)
    console().print(Text("  human-in-the-loop TAMP data collection", style="faint"))
    console().print()


def panel(body, *, title: str = "", style: str = "faint", subtitle: str = "") -> None:
    console().print(
        Panel(
            body,
            title=Text(title, style="accent") if title else None,
            subtitle=Text(subtitle, style="faint") if subtitle else None,
            border_style=style,
            padding=(1, 2),
            title_align="left",
            subtitle_align="right",
        )
    )


def error_panel(message: str, hint: str | None = None) -> None:
    body = Text(message)
    if hint:
        body.append("\n\n")
        body.append("hint  ", style="hint")
        body.append(hint, style="muted")
    err_console().print(
        Panel(body, title=Text(" error ", style="err"), border_style=RED, padding=(1, 2), title_align="left")
    )


def table(*columns: str, box_style: str = "simple") -> Table:
    from rich import box as rich_box

    boxes = {"simple": rich_box.SIMPLE, "rounded": rich_box.ROUNDED, "none": None}
    t = Table(
        box=boxes.get(box_style, rich_box.SIMPLE),
        header_style="faint",
        border_style="faint",
        pad_edge=False,
        show_edge=box_style != "simple",
    )
    for col in columns:
        t.add_column(col)
    return t


def kv(pairs: list[tuple[str, object]], *, title: str = "") -> None:
    t = table("", "", box_style="none")
    t.columns[0].style = "muted"
    t.columns[0].justify = "right"
    t.columns[1].style = "default"
    for key, value in pairs:
        t.add_row(key, _render_value(value))
    if title:
        heading(title)
    console().print(t)


def _render_value(value: object) -> Text:
    if value is None or value == "":
        return Text("—", style="faint")
    if value is True:
        return Text("yes", style="ok")
    if value is False:
        return Text("no", style="faint")
    if isinstance(value, (list, tuple)):
        return Text(", ".join(str(v) for v in value) if value else "—", style="default")
    return Text(str(value))


def status_glyph(state: str) -> Text:
    """Colour-coded glyph for a pass/warn/fail row in `tandem doctor`."""
    return {
        "ok": Text(OK, style="ok"),
        "warn": Text(WARN, style="warn"),
        "fail": Text(FAIL, style="err"),
        "skip": Text(DOT, style="faint"),
    }.get(state, Text(DOT, style="faint"))


def command(text: str) -> Text:
    return Text(text, style="code")


def next_steps(steps: list[tuple[str, str]]) -> None:
    """A closing 'what now' block: [(command, why)]."""
    t = table("", "", box_style="none")
    t.columns[0].style = "code"
    t.columns[1].style = "faint"
    for cmd, why in steps:
        t.add_row(cmd, why)
    blank()
    heading("next")
    console().print(t)


def is_tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()
