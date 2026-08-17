"""`tandem collect` — the human-in-the-loop rollout loop, in the terminal."""

from __future__ import annotations

import time
from collections import deque

import typer
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tandem.cli import theme
from tandem.cli.keys import KeyReader
from tandem.core import profiles
from tandem.core import runtime as runtime_mod
from tandem.core import session as session_mod
from tandem.core import settings as settings_mod
from tandem.core.errors import SessionConflict, TandemError
from tandem.core.session import State

# The stages a rollout moves through, as the operator experiences them.
PIPELINE = [
    ("warm", {State.WARMING}),
    ("perceive", {State.ROLLING}),
    ("execute", {State.ROLLING}),
    ("label", {State.AWAITING_LABEL, State.LABELING}),
    ("next", {State.AWAITING_TASK}),
]

LOG_ROWS = 14


def collect(
    profile_name: str = typer.Argument(None, help="Profile to collect under (default: the active one)."),
    task: str = typer.Option(None, "--task", "-t", help="Override the profile's task for this session."),
    episodes: int = typer.Option(None, "--episodes", "-n", help="Stop after this many labeled rollouts."),
    no_execute: bool = typer.Option(
        False,
        "--no-execute",
        help="Perceive and plan, but never move the robot. Useful for tuning TAMP settings.",
    ),
    no_record: bool = typer.Option(False, "--no-record", help="Skip camera recording."),
    web: bool = typer.Option(False, "--web", help="Drive the session from the browser instead."),
) -> None:
    """Warms up once, then loops rollouts against that warm state.

    While it runs:

      [key]s[/key] success   [key]f[/key] failure   [key]p[/key] preempt this rollout
      [key]t[/key] hand the arm to a human   [key]n[/key] new task   [key]q[/key] finish
    """
    profile = profiles.load(profile_name)
    cfg = settings_mod.load()

    if web:
        from tandem.cli.ui import serve

        theme.info(f"Starting the browser UI to collect under {profile.name!r}.")
        serve(profile_name=profile.name)
        return

    runtime = runtime_mod.Runtime(cfg.resolved_runtime_dir())
    # Fail before printing a session header for a session that cannot start.
    runtime.require_ready()

    theme.blank()
    theme.heading(f"collect · {profile.name}", profile.description)
    theme.kv(
        [
            ("task", task or profile.goal_or_prompt()),
            ("robot", f"{profile.robot.type} at {profile.robot.host}  ·  {profile.robot.time_dilation_factor:.0%} speed"),
            ("cameras", ", ".join(profile.cameras.configured())),
            ("execute", "no — planning only" if no_execute else "yes"),
            ("output", profile.trajectories_dir()),
        ]
    )
    if profile.tamp:
        theme.info(f"{len(profile.tamp)} TAMP override(s) active", "tandem profile show --tamp")
    if profile.hitl.enabled:
        theme.info(
            "phase planning is on",
            f"{profile.hitl.proposal_model} splits the task into robot and human steps",
        )
    theme.blank()
    theme.warn(
        "A preempt stops further plan steps, not the arm.",
        "the current motion segment finishes — use the E-stop for a hard halt",
    )
    theme.blank()

    manager = session_mod.manager()
    theme.busy("Warming up", "cuRobo, SAM2, cameras and the robot — this takes a minute")
    session = manager.create(
        profile,
        runtime,
        task=task,
        execute=not no_execute,
        record=not no_record,
        max_episodes=episodes,
    )

    try:
        _run_dashboard(session, profile)
    finally:
        if session.alive:
            theme.blank()
            theme.busy("Stopping the session and parking the arm")
            session.stop()
            session.wait(timeout=session_mod.HOME_EXIT_GRACE + 20)
        path = session_mod.write_session_log(session)
        _final_summary(session, profile, path)


def _run_dashboard(session, profile) -> None:
    logs: deque[str] = deque(maxlen=LOG_ROWS)
    session.subscribe(lambda msg: logs.append(_format_log(msg)) if msg.get("type") == "log" else None)
    for line in session.logs(limit=LOG_ROWS):
        logs.append(_format_log({"type": "log", **line}))

    status_note = {"text": "", "at": 0.0}

    def note(text: str) -> None:
        status_note["text"] = text
        status_note["at"] = time.time()

    with KeyReader() as keys, Live(
        console=theme.console(), refresh_per_second=8, screen=False, transient=False
    ) as live:
        last_state = None
        while session.alive:
            live.update(_render(session, profile, logs, status_note))

            key = keys.get(timeout=0.15)
            if key:
                try:
                    if not _handle_key(key, session, keys, note, live):
                        break
                except SessionConflict as exc:
                    note(f"[warn]{exc.message}[/warn]")
                except TandemError as exc:
                    note(f"[err]{exc.message}[/err]")

            # Prompt the operator when the driver reaches a decision point, so the footer is
            # not the only cue that it is waiting on a human.
            if session.state is not last_state:
                last_state = session.state
                if session.state is State.AWAITING_LABEL:
                    note("[accent]Did that work?  s = success   f = failure[/accent]")
                elif session.state is State.AWAITING_TASK:
                    note("[accent]Enter repeats the task   n = new task   q = finish[/accent]")
                elif session.state is State.TELEOP_HANDOFF:
                    note("[violet]The arm is yours. Press r to return control to TAMP.[/violet]")
                elif session.state is State.AWAITING_HUMAN_PHASE:
                    note("[violet]The plan needs you for this step — see below.[/violet]")

        live.update(_render(session, profile, logs, status_note))


def _handle_key(key: str, session, keys: KeyReader, note, live) -> bool:
    """Returns False to leave the loop."""
    state = session.state

    if state is State.AWAITING_HUMAN_PHASE:
        # A human phase has its own answers; the labeling keys would be ambiguous here.
        if key == "d":
            session.complete_human_phase()
            note("Checking that the step was done…")
        elif key == "t":
            session.request_teleop()
            note("[violet]Taking the arm — the driver hands it over from this prompt[/violet]")
        elif key == "a":
            session.abort_human_phase()
            note("[warn]Phase abandoned[/warn]")
        elif key == "q":
            return False
        elif key == "\x03":
            return False
        return True

    if key in ("s", "y") and state is State.AWAITING_LABEL:
        session.label(True)
        note("[ok]Labeled success[/ok]")
    elif key in ("f", "n") and state is State.AWAITING_LABEL:
        session.label(False)
        note("[warn]Labeled failure[/warn]")
    elif key in ("\r", "\n") and state is State.AWAITING_TASK:
        session.next_task()
        note("Repeating the task")
    elif key == "n" and state is State.AWAITING_TASK:
        live.stop()
        text = keys.read_line("  new task › ").strip()
        live.start()
        session.next_task(text or None)
        note(f"Task: {session.task}")
    elif key == "p":
        session.preempt()
        note("[warn]Preempting — the current motion segment still finishes[/warn]")
    elif key == "t":
        session.request_teleop()
        note("[violet]Hand-off armed — it happens at the next plan-step boundary[/violet]")
    elif key == "r" and state is State.TELEOP_HANDOFF:
        session.resume_from_teleop()
        note("Returning control to TAMP")
    elif key == "q":
        note("Finishing after this point")
        return False
    elif key == "\x03":  # Ctrl-C inside cbreak
        return False
    else:
        note("")
    return True


def _render(session, profile, logs: deque[str], status_note: dict) -> Panel:
    summary = session.summary()
    state = State(summary["state"])

    header = Table.grid(padding=(0, 2))
    header.add_column(style="muted", justify="right", width=8)
    header.add_column(ratio=1)
    header.add_row("task", Text(summary["task"], style="bold"))
    header.add_row("state", _pipeline(state))

    elapsed = time.time() - summary["started_at"]
    counts = Text()
    counts.append(f"{summary['success']} success", style="ok")
    counts.append("  ·  ", style="faint")
    counts.append(f"{summary['labeled'] - summary['success']} failure", style="err")
    counts.append("  ·  ", style="faint")
    counts.append(f"{summary['labeled']}/{summary['target']} labeled", style="faint")
    counts.append(f"   elapsed {_hms(elapsed)}", style="faint")
    header.add_row("", counts)

    progress = summary.get("phase_progress")
    if progress:
        header.add_row("phase", Text(f"{progress[0]} of {progress[1]}", style="violet"))
    if summary.get("handoff_error"):
        header.add_row("", Text(summary["handoff_error"], style="warn"))
    for clause in summary.get("unrepresented") or []:
        header.add_row(
            "",
            Text(
                f"not covered by the plan: {clause.get('clause') or clause}",
                style="warn",
            ),
        )
    if status_note["text"] and time.time() - status_note["at"] < 12:
        header.add_row("", Text.from_markup(status_note["text"]))

    log_panel = Group(*(Text.from_markup(line) for line in logs)) if logs else Text("…", style="faint")

    parts = [header, Text("")]
    phase_panel = _human_phase_panel(summary.get("human_phase"))
    if phase_panel is not None:
        parts += [phase_panel, Text("")]
    parts += [log_panel, Text(""), _footer(state, summary)]
    body = Group(*parts)
    return Panel(
        body,
        title=Text.from_markup(f"[accent]tandem[/accent] [faint]· collect ·[/faint] [bold]{profile.name}[/bold]"),
        title_align="left",
        subtitle=Text(str(profile.trajectories_dir()), style="faint"),
        subtitle_align="right",
        border_style="faint",
        padding=(1, 2),
    )


def _human_phase_panel(phase: dict | None) -> Panel | None:
    """What the person is being asked to do, and what will be checked afterwards.

    The expectations are shown because they are exactly the list the model is about to be
    asked about — being checked against a standard you were not told is the fastest way to
    make an operator distrust the whole thing.
    """
    if not phase:
        return None

    lines = Table.grid(padding=(0, 1))
    lines.add_column()

    title = phase.get("description") or "Your turn"
    if phase.get("total"):
        title = f"{title}   [faint]step {phase['index'] + 1} of {phase['total']}[/faint]"
    lines.add_row(Text.from_markup(f"[bold violet]{title}[/bold violet]"))

    if phase.get("instructions"):
        lines.add_row(Text(phase["instructions"]))

    if phase.get("expected"):
        lines.add_row(Text(""))
        lines.add_row(Text("When you are done, this should be true:", style="faint"))
        for item in phase["expected"]:
            lines.add_row(Text(f"  · {item}", style="muted"))

    if phase.get("missing"):
        lines.add_row(Text(""))
        lines.add_row(Text("That did not look done. Still expected:", style="warn"))
        for item in phase["missing"]:
            lines.add_row(Text(f"  · {item}", style="warn"))
    if phase.get("attempt", 1) > 1:
        lines.add_row(Text(f"attempt {phase['attempt']}", style="faint"))

    return Panel(
        lines,
        title=Text(" your turn ", style="violet"),
        title_align="left",
        border_style="violet",
        padding=(1, 2),
    )


def _pipeline(state: State) -> Text:
    """A lit strip showing where the rollout is."""
    out = Text()
    reached = False
    for i, (label, states) in enumerate(PIPELINE):
        if i:
            out.append("  ", style="faint")
        if state in states and not reached:
            out.append("◐ ", style="accent")
            out.append(label, style="bold accent")
            reached = True
        elif reached:
            out.append("○ ", style="faint")
            out.append(label, style="faint")
        else:
            out.append("● ", style="ok")
            out.append(label, style="muted")

    if state is State.HANDING_OFF:
        out = Text("◐ ", style="violet") + Text("handing the arm over…", style="bold violet")
    elif state is State.TELEOP_HANDOFF:
        out = Text("● ", style="violet") + Text("human has the arm", style="bold violet")
    elif state is State.AWAITING_HUMAN_PHASE:
        out = Text("◐ ", style="violet") + Text("waiting on you", style="bold violet")
    elif state is State.QUITTING:
        out = Text("◐ ", style="warn") + Text("finishing — parking the arm", style="warn")
    elif state is State.FAILED:
        out = Text("✖ ", style="err") + Text("failed", style="bold err")
    return out


def _footer(state: State, summary: dict) -> Text:
    keys: list[tuple[str, str]] = []
    if state is State.AWAITING_HUMAN_PHASE:
        keys = [("d", "I did it")]
        if summary.get("teleop_available"):
            keys.append(("t", "take the arm"))
        keys += [("a", "give up on this task"), ("q", "finish")]
        return _keys_text(keys)
    if state is State.AWAITING_LABEL:
        keys += [("s", "success"), ("f", "failure")]
    elif state is State.AWAITING_TASK:
        keys += [("↵", "repeat task"), ("n", "new task")]
    if summary.get("can_preempt") and state not in (State.AWAITING_TASK, State.AWAITING_LABEL):
        keys.append(("p", "preempt"))
    if state is State.TELEOP_HANDOFF:
        keys.append(("r", "return control"))
    elif summary.get("teleop_available") and summary.get("can_preempt"):
        keys.append(("t", "hand to human"))
    keys.append(("q", "finish"))
    return _keys_text(keys)


def _keys_text(keys: list[tuple[str, str]]) -> Text:
    out = Text()
    for i, (key, label) in enumerate(keys):
        if i:
            out.append("   ", style="faint")
        out.append(f" {key} ", style="reverse accent")
        out.append(f" {label}", style="faint")
    return out


def _format_log(msg: dict) -> str:
    text = msg.get("text", "")
    stream = msg.get("stream", "stdout")
    stamp = time.strftime("%H:%M:%S", time.localtime(msg.get("at", time.time())))
    text = _escape(text[:180])
    if stream == "tandem":
        return f"[faint]{stamp}[/faint]  [accent]{text}[/accent]"
    if "ERROR" in text or "Traceback" in text:
        return f"[faint]{stamp}[/faint]  [err]{text}[/err]"
    if "WARNING" in text:
        return f"[faint]{stamp}[/faint]  [warn]{text}[/warn]"
    return f"[faint]{stamp}[/faint]  {text}"


def _escape(text: str) -> str:
    return text.replace("[", "\\[")


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _final_summary(session, profile, log_path) -> None:
    summary = session.summary()
    theme.blank()
    theme.rule("session finished")
    theme.kv(
        [
            ("collected", f"{summary['success']} success, {summary['labeled'] - summary['success']} failure"),
            ("duration", _hms((summary.get("ended_at") or time.time()) - summary["started_at"])),
            ("trajectories", profile.trajectories_dir()),
            ("session log", log_path),
        ]
    )
    if summary.get("error"):
        theme.fail(summary["error"])
    theme.next_steps(
        [
            (f"tandem traj list {profile.name}", "see what was collected"),
            ("tandem ui", "play the videos and inspect the plots"),
        ]
    )
