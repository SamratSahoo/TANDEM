"""Session routes — the browser half of the human-in-the-loop.

The same ``Session`` object `tandem collect` drives from the terminal. Its callback bus is
bridged to Server-Sent Events here; the state machine itself lives in core and knows nothing
about HTTP.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tandem.core import profiles as profiles_mod
from tandem.core import session as session_mod
from tandem.core import settings as settings_mod
from tandem.planners.tiptop import runtime as runtime_mod

router = APIRouter(tags=["sessions"])


class CreateBody(BaseModel):
    profile: str | None = None
    task: str | None = None
    execute: bool = True
    record: bool | None = None
    episodes: int | None = None


class LabelBody(BaseModel):
    success: bool


class ContinueBody(BaseModel):
    task: str | None = None
    more: bool = True


@router.post("/sessions")
async def create_session(body: CreateBody) -> dict:
    profile = profiles_mod.load(body.profile)
    cfg = settings_mod.load()
    runtime = runtime_mod.Runtime(cfg.resolved_runtime_dir())
    session = session_mod.manager().create(
        profile,
        runtime,
        task=body.task,
        execute=body.execute,
        record=body.record,
        max_episodes=body.episodes,
    )
    return session.summary()


@router.get("/sessions")
async def list_sessions() -> dict:
    return {"sessions": [s.summary() for s in session_mod.manager().all()]}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    session = session_mod.manager().get(session_id)
    return {**session.summary(), "logs": session.logs(limit=500)}


@router.get("/sessions/{session_id}/stream")
async def stream_session(session_id: str) -> StreamingResponse:
    session = session_mod.manager().get(session_id)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=2000)

    def on_message(message: dict) -> None:
        # Called from the session's threads; hop back onto the event loop. A full queue
        # means a stalled browser — drop rather than block the thread driving a robot.
        try:
            loop.call_soon_threadsafe(queue.put_nowait, message)
        except (RuntimeError, asyncio.QueueFull):
            pass

    unsubscribe = session.subscribe(on_message)

    async def events() -> AsyncIterator[bytes]:
        try:
            # Replay enough history that a page opened mid-session is not blank.
            yield _sse({"type": "state", "state": session.state.value, **session.summary()})
            for line in session.logs(limit=300):
                yield _sse({"type": "log", **line})
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Keep proxies and browsers from dropping an idle stream.
                    yield b": keepalive\n\n"
                    if not session.alive and queue.empty():
                        break
                    continue
                yield _sse(message)
                if message.get("type") == "state" and message.get("state") in {"stopped", "failed"}:
                    # Let any trailing log lines drain before closing.
                    await asyncio.sleep(0.3)
                    while not queue.empty():
                        yield _sse(queue.get_nowait())
                    break
        finally:
            unsubscribe()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no", "connection": "keep-alive"},
    )


def _sse(message: dict) -> bytes:
    return f"data: {json.dumps(message, default=str)}\n\n".encode()


@router.post("/sessions/{session_id}/label")
async def label(session_id: str, body: LabelBody) -> dict:
    session = session_mod.manager().get(session_id)
    session.label(body.success)
    return session.summary()


@router.post("/sessions/{session_id}/continue")
async def continue_(session_id: str, body: ContinueBody) -> dict:
    session = session_mod.manager().get(session_id)
    if body.more:
        session.next_task(body.task)
    else:
        session.stop()
    return session.summary()


@router.post("/sessions/{session_id}/preempt")
async def preempt(session_id: str) -> dict:
    session = session_mod.manager().get(session_id)
    session.preempt()
    return session.summary()


@router.post("/sessions/{session_id}/stop")
async def stop(session_id: str) -> dict:
    session = session_mod.manager().get(session_id)
    session.stop()
    return session.summary()


@router.post("/sessions/{session_id}/force-stop")
async def force_stop(session_id: str) -> dict:
    session = session_mod.manager().get(session_id)
    session.force_stop()
    return session.summary()


@router.post("/sessions/{session_id}/teleop-switch")
async def teleop_switch(session_id: str) -> dict:
    """Ask for the arm: at the next plan-step boundary, or at once at a human phase's prompt.

    Between phases it lends the arm to a person through teleop. At a human phase's prompt it hands
    the step to the profile's human executor (``hitl.human_executor``), teleop unless it says
    otherwise; ``human_executor`` in the session summary says which, and whether it is ready.
    """
    session = session_mod.manager().get(session_id)
    session.request_teleop()
    return session.summary()


@router.post("/sessions/{session_id}/teleop-resume")
async def teleop_resume(session_id: str) -> dict:
    session = session_mod.manager().get(session_id)
    session.resume_from_teleop()
    return session.summary()


@router.post("/sessions/{session_id}/human-phase/done")
async def human_phase_done(session_id: str) -> dict:
    """The person did the step by hand. The driver checks it from a photo before carrying on.

    A 409 while recording, unless the profile sets ``hitl.allow_unrecorded_human_phase``: a step
    done by hand has no leg, and the episode would lack its demonstration while looking complete.
    The step is then handed over with ``teleop-switch``, which at a human phase's prompt runs the
    profile's human executor (``hitl.human_executor``) rather than waiting for a plan-step boundary.
    """
    session = session_mod.manager().get(session_id)
    session.complete_human_phase()
    return session.summary()


@router.post("/sessions/{session_id}/human-phase/abort")
async def human_phase_abort(session_id: str) -> dict:
    """Give up on this phase, and with it the task attempt."""
    session = session_mod.manager().get(session_id)
    session.abort_human_phase()
    return session.summary()
