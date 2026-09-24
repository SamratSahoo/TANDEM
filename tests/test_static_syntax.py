"""Every script the web UI ships parses as the ES module the browser loads it as, and is never run from a
stale cache.

The pages ship as package data with no build step, so nothing else ever parses them before a browser does:
a stray bracket is a blank page on the workstation. Needs node; skipped where there is none. (Plain `node
--check` guesses the module type and passed a file with an ESM syntax error; `--input-type=module` does not
guess.) And they import each other, so a browser that kept an old module after an upgrade ran the new pages
against it -- another blank page.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tandem
from tandem.server.app import create_app

STATIC = Path(tandem.__file__).parent / "server" / "static"
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("script", sorted(STATIC.rglob("*.js")), ids=lambda p: p.relative_to(STATIC).as_posix())
def test_the_script_parses_as_an_es_module(script):
    result = subprocess.run(
        [NODE, "--input-type=module", "--check"], input=script.read_text(), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_check_catches_a_syntax_error():
    result = subprocess.run(
        [NODE, "--input-type=module", "--check"], input="export const a = ;\n", capture_output=True, text=True
    )
    assert result.returncode != 0


def test_a_browser_asks_for_every_file_again_rather_than_run_a_stale_one():
    client = TestClient(create_app())
    for path in ("/", "/profiles", "/app.js", "/dom.js", "/pages/profiles.js", "/theme.css"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers["cache-control"] == "no-cache", path
        assert response.headers["etag"], "so asking again is a 304, not a download"
