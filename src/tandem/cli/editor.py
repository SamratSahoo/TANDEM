"""Opening a file in the person's editor: `tandem profile edit` and `tandem config edit`.

One helper, so the two commands cannot drift apart again: both used to hand the whole of $EDITOR to
the OS as a program name, and a perfectly ordinary ``EDITOR='code --wait'`` was then "No such file or
directory: 'code --wait'" -- as a traceback, with the backup made just before left behind.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from tandem.core.errors import TandemError


def editor_command() -> str:
    """$EDITOR, else $VISUAL, else vi -- as the person wrote it, arguments and all."""
    return os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"


def open_in_editor(path: Path) -> int:
    """Open ``path`` in the person's editor and wait for it. Returns the editor's exit code.

    $EDITOR is split the way a shell would split it, so ``code --wait`` and ``subl -w`` work. An editor
    that cannot be started is a TandemError naming it, never a traceback.
    """
    editor = editor_command()
    try:
        argv = shlex.split(editor)
    except ValueError as exc:
        raise TandemError(
            f"$EDITOR ({editor!r}) cannot be read as a command: {exc}.",
            hint="Set $EDITOR (or $VISUAL) to an editor on your PATH, e.g. `export EDITOR='code --wait'`.",
        ) from None
    if not argv:
        argv = ["vi"]
    try:
        return subprocess.call([*argv, str(path)])
    except OSError as exc:
        raise TandemError(
            f"Could not start your editor {editor!r}: {exc.strerror or exc}.",
            hint="Set $EDITOR (or $VISUAL) to an editor on your PATH, e.g. `export EDITOR='code --wait'`.",
        ) from exc
