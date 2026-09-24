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


class ProfileInvalid(ProfileError):
    """A profile that exists but does not validate.

    Its own type so the web API can answer 422 for it and keep 404 for a profile that is not there:
    an editor told "not found" about a file it can see has no reason to offer to fix it.
    """


def one_line(message: str) -> str:
    """``message`` as one line, for a listing, a table cell or a JSON field -- without losing the reason.

    Many errors put a header on the first line and the reasons on the next ones ("<path> is not a valid
    profile:" and then "  planner.backend: ..."). The first line alone is then a sentence that ends in a
    colon and says nothing; the header and the reasons joined with "; " are what a one-line slot can say.
    """
    lines = [line.strip() for line in str(message or "").splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) > 1 and lines[0].endswith(":"):
        return f"{lines[0]} {'; '.join(lines[1:])}"
    return lines[0]


class RuntimeNotReady(TandemError):
    """A planner's runtime (`tandem planners install`, which `tandem init` runs) is absent or incomplete."""


class SessionConflict(TandemError):
    """A session command is not legal in the session's current state."""
