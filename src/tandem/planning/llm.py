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
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
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
        response = await client.aio.models.generate_content(model=model, contents=contents, config=config)
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
