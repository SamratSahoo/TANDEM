"""The events file the collection driver reports its progress through.

The session appends one JSON object per line to its events file, and so do the processes it
drives: a planner's sidecar (TiPToP's finds it through ``$TIPTOP_EVENTS_FILE``) and the teleop
driver. That file — not screen-scraping the driver's stdin prompts — is how we know what state
the session is in.
It is a deliberately dumb channel, which is why it survives the driver being preempted,
re-warmed, or handed off to a teleop process mid-task.

    {"event":"session_start"}
    {"event":"awaiting_task"}                          blocked on the "next task" prompt
    {"event":"rollout_start","dir":"<abs eval/ts>"}
    {"event":"rollout_saved","dir":"…","n_frames":123}
    {"event":"awaiting_label","dir":"…"}               blocked on the success/failure prompt
    {"event":"labeled","dir":"<abs success/ts>","success":true}
    {"event":"rollout_aborted"}                        a preempt unwound the rollout
    {"event":"session_end"}

    {"event":"teleop_switch_pending"}                  SIGUSR1 seen; finishing the plan step
    {"event":"teleop_handoff_start"}                   step done; releasing robot + cameras
    {"event":"teleop_handoff_warning","message":"…"}
    {"event":"awaiting_teleop_resume"}                 released; blocked on "resume"
    {"event":"teleop_handoff_done"}                    reconnected; about to replan

With phase planning on, tandem's own session file also says what the camera checks found and how
each trial ended (``tandem.core.phase_loop``, ``tandem.core.session``):

    {"event":"phase_preconditions_checked","phase_index":1,"what":"human phase",
     "ok":false,"enforced":false,"verdicts":[…]}      ok is null when the check could not run
    {"event":"human_phase_verified","phase_index":1,"attempt":1,"ok":true,"verdicts":[…]}
                                                       ok null + "skipped"/"unchecked" when not judged
    {"event":"phase_effects_checked","phase_index":0,"what":"robot leg","ok":true,…}
    {"event":"trial_outcome","outcome":"excluded","failure_stage":"verification","reason":"…"}
                                                       the loop ended the trial itself
    {"event":"trial_excluded","dir":"…","outcome":"excluded","filed_under":"failure",…}
                                                       filed without a label prompt
    {"event":"labeled",…,"outcome":"success","failure_stage":null}
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Every event name we act on. An unknown name is forwarded rather than dropped: the driver
# may be newer than we are, and a UI that shows an unrecognised event is better than one
# that silently swallows it.
KNOWN = frozenset(
    {
        "session_start",
        "session_end",
        "awaiting_task",
        "rollout_start",
        "rollout_saved",
        "rollout_aborted",
        "rollout_discarded",
        "awaiting_label",
        "labeled",
        "teleop_switch_pending",
        "teleop_handoff_start",
        "teleop_handoff_warning",
        "teleop_handoff_done",
        "awaiting_teleop_resume",
        "homing",
        "homed",
        # The phase loop's. A trial the method excludes never reaches `awaiting_label`, so
        # `trial_excluded` is the only event that tells a UI why it went back to the task prompt.
        "instruction_not_fully_represented",
        "awaiting_human_phase",
        "phase_preconditions_checked",
        "human_phase_verified",
        "phase_effects_checked",
        "phase_plan_failed",
        "phase_complete",
        "trial_outcome",
        "trial_excluded",
    }
)


@dataclass
class Event:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    at: float = 0.0

    @property
    def dir(self) -> str | None:
        value = self.payload.get("dir")
        return str(value) if value else None

    def to_dict(self) -> dict:
        return {"event": self.name, "at": self.at, **self.payload}


class EventTailer:
    """Follow an append-only JSONL file in a background thread.

    Tolerant on purpose: the file may not exist yet, may be truncated, and may lag the
    driver's stdout by a moment. None of those is an error worth surfacing to a user who is
    standing next to a robot.
    """

    def __init__(
        self,
        path: Path,
        on_event: Callable[[Event], None],
        *,
        poll_interval: float = 0.05,
    ) -> None:
        self.path = Path(path)
        self._on_event = on_event
        self._poll = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset = 0
        self._partial = ""

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=f"events:{self.path.name}", daemon=True)
        self._thread.start()

    def stop(self, *, drain: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if drain:
            # One last read: the driver's final session_end often lands between the last
            # poll and the process exiting.
            self._read_available()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._read_available()
            self._stop.wait(self._poll)

    def _read_available(self) -> None:
        try:
            if not self.path.is_file():
                return
            size = self.path.stat().st_size
            if size < self._offset:
                # Truncated (a re-created session file). Start over rather than read garbage.
                self._offset = 0
                self._partial = ""
            if size == self._offset:
                return
            with self.path.open("r", errors="replace") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
                self._offset = handle.tell()
        except OSError:
            return

        text = self._partial + chunk
        lines = text.split("\n")
        # The driver appends with flush, but a partial line is still possible between the
        # write and the flush; hold it until the newline arrives.
        self._partial = lines.pop()
        for line in lines:
            event = parse(line)
            if event is not None:
                self._on_event(event)


def parse(line: str) -> Event | None:
    line = line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.pop("event", None)
    if not name:
        return None
    at = payload.pop("ts", None)
    try:
        at = float(at) if at is not None else time.time()
    except (TypeError, ValueError):
        at = time.time()
    return Event(name=str(name), payload=payload, at=at)


def read_all(path: Path) -> list[Event]:
    """Every event in a finished session's file — for post-mortems."""
    if not Path(path).is_file():
        return []
    out: list[Event] = []
    for line in Path(path).read_text(errors="replace").splitlines():
        event = parse(line)
        if event is not None:
            out.append(event)
    return out
