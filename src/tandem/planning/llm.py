"""Vision-model calls that must come back as valid, well-formed JSON.

Two things are layered over the raw SDK call. Structured output (``response_schema``) fixes the SHAPE
of the reply, so nothing here has to cope with prose around a JSON blob. A reprompt loop fixes its
CONTENT: the parser raises ``ProposalError`` with a message written for the model, that message is
handed back, and the model gets another go. Almost every proposal failure in practice is semantic --
a predicate applied to the wrong number of objects, an object that is not in the scene -- and those
are exactly the ones a second attempt fixes.

The client is tandem's own now rather than the planner's, which is what lets a decomposition be
proposed and checked with no planner, no robot and no GPU anywhere in the process. See
``tandem plan``.

A third layer sits underneath both: a call the API itself failed TRANSIENTLY -- rate-limited (429),
overloaded (5xx), timed out -- is retried with a bounded backoff before anything is parsed. That is
kept apart from the repair attempts on purpose. A repair attempt is a second chance for the MODEL,
and ``max_attempts`` is the paper's "at most 3 attempts"; spending one on an HTTP 503 would leave a
plan that needed one repair with none, and fail a trial the method would have saved.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
import logging
import threading
from collections.abc import Awaitable, Callable, Coroutine
from functools import cache
from typing import Any, TypeVar

from tandem.core.errors import TandemError
from tandem.planning.cache import ProposalCache
from tandem.planning.record import active_recorder
from tandem.planning.symbols import ProposalError

_log = logging.getLogger(__name__)

_T = TypeVar("_T")

_REPROMPT = """\
Your previous response was rejected.

Your response was:
{response}

The problem was:
{error}

Try again, fixing exactly that problem and keeping everything else that was correct."""

# HTTP statuses that mean "the same request may well succeed shortly": request timeout, rate limit,
# and the server-side failures. Every other status -- a bad key (401/403), a malformed request (400),
# an unknown model (404) -- fails the same way however long one waits, so it is raised at once.
_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})

# How long to wait before each retry of a transiently failed call, in seconds; one retry per entry.
# Bounded, and much shorter than an offline evaluation can afford (LJ's waited up to two minutes a
# step): an operator is standing at the robot while this runs, and a rate limit that outlasts about
# a minute is not going to clear by waiting longer. Exhausting it raises a TandemError that says so.
TRANSIENT_DELAYS: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 30.0)

# The wait itself, as a module attribute so a test can take the time out of it.
_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


def is_transient(exc: BaseException) -> bool:
    """Whether ``exc`` is an API failure worth retrying unchanged, rather than a verdict on the call.

    The SDK's own errors carry the HTTP status as ``code`` (``google.genai.errors.APIError``); other
    clients use ``status_code``. Timeouts and dropped connections have no status at all, and come as
    the stdlib's exceptions or httpx's, depending on the transport the SDK was built with.
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and not isinstance(code, bool) and code in _TRANSIENT_STATUS:
        return True
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a dependency of google-genai
        return False
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError))


async def _generate(client: Any, *, model: str, contents: list, config: Any, label: str) -> Any:
    """One ``generate_content`` call, retried through transient API failures.

    Deliberately below the repair loop in ``query_json``: however many times this retries, it is one
    attempt as far as the model is concerned. Anything that is not transient is raised unchanged.
    """
    for retry in range(len(TRANSIENT_DELAYS) + 1):
        try:
            return await client.aio.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:
            if not is_transient(exc):
                raise
            if retry == len(TRANSIENT_DELAYS):
                raise TandemError(
                    f"The {label} request to {model} kept failing: {type(exc).__name__}: {exc}",
                    hint=(
                        f"The model API was unavailable or rate-limited for {len(TRANSIENT_DELAYS) + 1} "
                        "tries in a row. Wait a minute and try again, and check the key's quota if "
                        "it keeps happening."
                    ),
                ) from exc
            delay = TRANSIENT_DELAYS[retry]
            _log.warning(
                f"{label}: the model API failed transiently ({type(exc).__name__}: {exc}); "
                f"retrying in {delay:g}s ({retry + 1}/{len(TRANSIENT_DELAYS)})"
            )
            await _sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover - the loop always returns or raises


@cache
def gemini_client():
    """The Gemini client, built from tandem's own stored key.

    Cached because the SDK client is reusable and building one per query is pure latency. The key is
    resolved through ``tandem.core.secrets`` so ``tandem config set-gemini-key`` is the single place
    it is configured, and so the error when it is missing names that command.
    """
    from google import genai

    from tandem.core import secrets

    key = secrets.gemini_api_key()
    if not key:
        raise TandemError(
            "No Gemini API key is set, and phase planning needs one.",
            hint="Run `tandem config set-gemini-key`.",
        )
    return genai.Client(api_key=key)


# The one event loop every blocking model call runs on, on a daemon thread of its own.
#
# Not `asyncio.run` per call. The client above is cached for the process, and with it the SDK's httpx
# connection pool -- and a pooled connection belongs to the event loop that opened it. `asyncio.run`
# closes its loop on the way out, so the next call on a fresh loop picked that connection up and
# failed with "RuntimeError: Event loop is closed". The failure discarded the connection, so the call
# after it worked: every other query failed, a task decomposition one trial and a human step's check
# (which an error passes unchecked) the next. One loop for the life of the process keeps the pool on
# a loop that never closes.
_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _shared_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None:
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="tandem-llm-loop", daemon=True).start()
            _loop = loop
        return _loop


def run_blocking(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` to completion on tandem's shared model-call loop, blocking until it is done.

    What synchronous code calls in place of ``asyncio.run`` for anything that queries the model.
    The coroutine runs in a copy of the caller's context, as it would under ``asyncio.run``. Not
    for use from the shared loop's own thread, which would wait on itself forever.
    """
    loop = _shared_loop()
    context = contextvars.copy_context()
    done: concurrent.futures.Future = concurrent.futures.Future()

    def start() -> None:
        task = loop.create_task(coro, context=context)

        def finish(task: asyncio.Task) -> None:
            if task.cancelled():
                done.cancel()
            elif task.exception() is not None:
                done.set_exception(task.exception())
            else:
                done.set_result(task.result())

        task.add_done_callback(finish)

    loop.call_soon_threadsafe(start)
    return done.result()


async def query_json(
    prompt: str,
    parse: Callable[[Any], _T],
    *,
    model: str,
    schema: dict,
    image: Any | None = None,
    max_attempts: int = 3,
    temperature: float | None = None,
    label: str = "proposal",
    cache: ProposalCache | None = None,
) -> _T:
    """Ask for JSON matching ``schema`` and parse it, reprompting when ``parse`` objects.

    ``parse`` receives the decoded JSON and either returns a value or raises ``ProposalError`` with a
    message aimed at the model. The last error is re-raised once the attempts run out, so the caller
    sees why the proposal could not be used rather than a bare failure.

    ``cache`` is consulted and written only for responses that PARSED and VALIDATED, so a rejected
    proposal is never replayed. See ``cache.ProposalCache`` for why grounding never passes one.

    ``max_attempts`` counts answers the model gave. A call the API failed transiently is retried
    underneath (``TRANSIENT_DELAYS``) and never uses one up.
    """
    from google.genai import types

    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")

    if cache is not None:
        cached = cache.get(model, prompt, image)
        if cached is not None:
            try:
                parsed = parse(json.loads(cached))
                _log.info(f"{label}: reusing the cached response")
                recorder = active_recorder()
                if recorder is not None:
                    # On the trail like a live answer, and marked as a replay: it is the plan this
                    # trial runs on, and without it the trail cannot say what the proposer was asked
                    # or answered. Against the prompt the cache is keyed by -- the original one, even
                    # where the answer was first given to a repair of it -- and as attempt 1, since
                    # nothing was re-asked this time.
                    recorder.record(
                        label=label,
                        attempt=1,
                        model=model,
                        prompt=prompt,
                        response=cached,
                        image=image,
                        cached=True,
                    )
                return parsed
            except (ProposalError, json.JSONDecodeError) as exc:
                # The validator has changed since the entry was written; ask again rather than fail.
                _log.info(f"{label}: cached response no longer validates ({exc}); re-querying")

    client = gemini_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
    )
    attempt_prompt = prompt
    last_error: ProposalError | None = None
    for attempt in range(1, max_attempts + 1):
        contents: list = [image, attempt_prompt] if image is not None else [attempt_prompt]
        response = await _generate(client, model=model, contents=contents, config=config, label=label)
        text = (response.text or "").strip()
        recorder = active_recorder()
        try:
            if not text:
                raise ProposalError(
                    "The response was empty. Respond with the JSON object that was asked for."
                )
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ProposalError(f"The response was not valid JSON: {exc}") from exc
            parsed = parse(data)
            if recorder is not None:
                recorder.record(
                    label=label,
                    attempt=attempt,
                    model=model,
                    prompt=attempt_prompt,
                    response=text,
                    image=image,
                )
            if attempt > 1:
                _log.info(f"{label}: accepted on attempt {attempt}")
            if cache is not None:
                cache.put(model, prompt, image, text)
            return parsed
        except ProposalError as exc:
            last_error = exc
            _log.warning(f"{label} attempt {attempt}/{max_attempts} rejected: {exc}")
            if recorder is not None:
                # Recorded too, and marked as rejected: a proposal that had to be corrected is the
                # one worth looking at afterwards, and keeping only the accepted answer hides it.
                recorder.record(
                    label=label,
                    attempt=attempt,
                    model=model,
                    prompt=attempt_prompt,
                    response=text,
                    image=image,
                    rejected=str(exc),
                )
            attempt_prompt = f"{prompt}\n\n{_REPROMPT.format(response=text, error=exc)}"

    assert last_error is not None
    raise last_error
