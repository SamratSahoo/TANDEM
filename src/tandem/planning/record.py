"""Save every image sent to a vision model, and what it said about it, beside the episode.

Each query writes three things: the image exactly as it was sent, a rendered PNG pairing that image
with the response, and a line in ``index.jsonl`` carrying the full prompt and reply. The rendered PNG
is the one to open -- a run's worth of them scrolls past as "here is what the model saw, here is what
it decided", which is the only practical way to tell a bad classifier wording from a bad camera view.

Every ATTEMPT is recorded, rejected ones included. A proposal that had to be reprompted is exactly
the case worth looking at, and it is invisible if only the accepted answer is kept. So is an answer
replayed from the proposal cache (``cached: true``): it is the plan the trial ran on, and a trail
that began at the first camera check could not say what the proposer had been asked or answered.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# Set for the duration of one task's planning, so a query does not have to be handed a directory
# through five layers of call. One task is planned at a time in this process, which is what makes
# that safe; `recording_to` is the only supported way to set it, so it is always unset again.
_active: VLMRecorder | None = None

_PANEL_WIDTH = 900
_MARGIN = 16
_LINE_SPACING = 4
_BACKGROUND = (250, 250, 250)
_INK = (24, 24, 24)
_MUTED = (110, 110, 110)
_RULE = (200, 200, 200)


def _font(size: int):
    """A readable font, falling back to Pillow's built-in when no TTF is installed."""
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _slug(text: str) -> str:
    """A filename-safe version of a query label."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:60] or "query"


def _pretty(text: str) -> str:
    """Pretty-print a JSON response; leave anything else alone."""
    try:
        return json.dumps(json.loads(text), indent=2)
    except (json.JSONDecodeError, TypeError):
        return text


def _wrap(text: str, columns: int = 100) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(raw, columns) or [""])
    return lines


# The sequence number every file a query writes starts with: `003_classify-IsOpen-box_input.png`.
_SEQ_PREFIX = re.compile(r"^(\d+)_")


def _last_seq(directory: Path) -> int:
    """The highest sequence number already written into ``directory``, 0 when there is none.

    Read off the files rather than counted from ``index.jsonl``: a query whose image was saved and
    whose index line was not (a full disk, a render that failed) still owns its number, and a count
    of lines would hand that number out again.
    """
    try:
        names = [p.name for p in directory.iterdir()]
    except OSError:
        return 0
    return max((int(m.group(1)) for m in map(_SEQ_PREFIX.match, names) if m), default=0)


class VLMRecorder:
    """Writes the image/response pair for each query into one directory.

    Numbering CONTINUES from what the directory already holds rather than starting at 001. One
    directory is written to by many recorders: the phase loop opens one per model call site into the
    attempt's single ``vlm/`` directory -- the proposal, then every camera check -- and ``tandem plan
    --save-vlm-io DIR`` run twice opens two. Each used to count from 001, so the retry of a check
    wrote ``001_classify-IsOpen-box_input.png`` over the first attempt's frame, and the index then
    held two lines pointing at one file showing only the second image: the frame an operator was sent
    back over, and the verdict an excluded trial was excluded on, were gone.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._seq = _last_seq(self._dir)

    @property
    def directory(self) -> Path:
        return self._dir

    def record(
        self,
        *,
        label: str,
        attempt: int,
        model: str,
        prompt: str,
        response: str,
        image: Any | None,
        rejected: str | None = None,
        cached: bool = False,
    ) -> None:
        """Save one query. Never raises: an audit trail is not worth failing a rollout over.

        ``cached`` marks an answer replayed from the proposal cache rather than asked for: the plan
        the trial ran on still has to be on its trail, and it must not read as a live answer.
        """
        try:
            self._seq += 1
            self._dir.mkdir(parents=True, exist_ok=True)
            stem = f"{self._seq:03d}_{_slug(label)}"
            if attempt > 1:
                stem += f"_attempt{attempt}"

            input_path = None
            if image is not None:
                input_path = self._dir / f"{stem}_input.png"
                image.save(input_path)
            output_path = self._dir / f"{stem}_output.png"
            self._render(label, attempt, model, response, image, rejected, cached).save(output_path)

            with (self._dir / "index.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "seq": self._seq,
                            "label": label,
                            "attempt": attempt,
                            "model": model,
                            "input_image": input_path.name if input_path else None,
                            "output_image": output_path.name,
                            "rejected": rejected,
                            "cached": cached,
                            "prompt": prompt,
                            "response": response,
                        }
                    )
                    + "\n"
                )
        except Exception:
            _log.exception("could not record a model query; continuing")

    def _render(
        self,
        label: str,
        attempt: int,
        model: str,
        response: str,
        image: Any | None,
        rejected: str | None,
        cached: bool = False,
    ):
        """The image the model saw, above what it answered."""
        from PIL import Image, ImageDraw

        title = _font(19)
        small = _font(14)
        body = _font(14)

        thumbnail = None
        if image is not None:
            thumbnail = image.copy()
            thumbnail.thumbnail((_PANEL_WIDTH - 2 * _MARGIN, 520))

        header = f"{label}" + (f"   (attempt {attempt})" if attempt > 1 else "")
        subtitle = (
            model
            + ("   REJECTED" if rejected else "")
            + ("   CACHED (replayed, not asked)" if cached else "")
        )
        lines = _wrap(_pretty(response))
        if rejected:
            lines = ["Rejected: " + rejected, ""] + lines

        line_height = body.getbbox("Ay")[3] + _LINE_SPACING
        height = _MARGIN
        height += title.getbbox("Ay")[3] + 6 + small.getbbox("Ay")[3] + _MARGIN
        if thumbnail is not None:
            height += thumbnail.height + _MARGIN
        height += _MARGIN + len(lines) * line_height + _MARGIN

        canvas = Image.new("RGB", (_PANEL_WIDTH, height), _BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        y = _MARGIN
        draw.text((_MARGIN, y), header, font=title, fill=_INK)
        y += title.getbbox("Ay")[3] + 6
        draw.text((_MARGIN, y), subtitle, font=small, fill=(180, 60, 60) if rejected else _MUTED)
        y += small.getbbox("Ay")[3] + _MARGIN
        if thumbnail is not None:
            canvas.paste(thumbnail, (_MARGIN, y))
            y += thumbnail.height + _MARGIN
        draw.line([(_MARGIN, y), (_PANEL_WIDTH - _MARGIN, y)], fill=_RULE)
        y += _MARGIN
        for line in lines:
            draw.text((_MARGIN, y), line, font=body, fill=_INK)
            y += line_height
        return canvas


def active_recorder() -> VLMRecorder | None:
    return _active


@contextmanager
def recording_to(directory: Path | None):
    """Record every model query in this block into ``directory``; None turns recording off.

    A context manager rather than a bare setter because the previous recorder must come back even
    when the block raises -- a failed proposal that left the recorder pointed at a finished episode's
    directory would file the NEXT task's queries under it.
    """
    global _active
    previous = _active
    _active = VLMRecorder(directory) if directory is not None else None
    try:
        yield _active
    finally:
        _active = previous
