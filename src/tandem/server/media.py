"""Range-aware file serving.

Browsers will not scrub an mp4 that is served as a plain 200: seeking needs
``Accept-Ranges`` and a 206 with ``Content-Range``. Implemented explicitly rather than
leaning on FileResponse so the behaviour is the same on every Starlette version — a video
you cannot scrub makes the whole review UI useless.
"""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from starlette.requests import Request
from starlette.responses import FileResponse, Response, StreamingResponse

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
CHUNK = 512 * 1024


def guess_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def range_response(request: Request, path: Path, *, cache_seconds: int = 3600) -> Response:
    size = path.stat().st_size
    media_type = guess_type(path)
    headers = {
        "accept-ranges": "bytes",
        "cache-control": f"private, max-age={cache_seconds}",
    }

    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(path, media_type=media_type, headers=headers)

    match = _RANGE_RE.fullmatch(range_header.strip())
    if not match:
        return Response(status_code=416, headers={**headers, "content-range": f"bytes */{size}"})

    raw_start, raw_end = match.groups()
    if raw_start:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
    elif raw_end:
        # "bytes=-500" means the last 500 bytes.
        length = min(int(raw_end), size)
        start, end = size - length, size - 1
    else:
        return Response(status_code=416, headers={**headers, "content-range": f"bytes */{size}"})

    end = min(end, size - 1)
    if start > end or start >= size:
        return Response(status_code=416, headers={**headers, "content-range": f"bytes */{size}"})

    length = end - start + 1

    def stream():
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        stream(),
        status_code=206,
        media_type=media_type,
        headers={
            **headers,
            "content-range": f"bytes {start}-{end}/{size}",
            "content-length": str(length),
        },
    )
