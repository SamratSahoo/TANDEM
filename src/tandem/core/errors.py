"""The one exception type the CLI knows how to render nicely."""

from __future__ import annotations


class TandemError(Exception):
    """A user-facing error.

    ``message`` says what went wrong; ``hint`` says what to do about it. The CLI's error
    boundary renders the pair in a panel and never shows a traceback unless --debug is on,
    so every raise site should be able to answer "and now what?".
    """

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class ProfileError(TandemError):
    """A profile is missing, malformed, or fails validation."""


class RuntimeNotReady(TandemError):
    """The GPU runtime `tandem init` builds is absent or incomplete."""


class SessionConflict(TandemError):
    """A session command is not legal in the session's current state."""
