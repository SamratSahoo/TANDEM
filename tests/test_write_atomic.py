"""``paths.write_atomic`` with more than one writer: each has a partial file of its own.

With one shared partial name, two writers at once (the web's rig card and `tandem rig set`, two `tandem
profile migrate` runs) truncated and interleaved into the same file, one renamed the mix into place, and the
other's rename raised FileNotFoundError.
"""

from __future__ import annotations

import os
import stat
import threading

from tandem.core import paths


def test_two_writers_at_once_leave_one_of_their_texts_whole_and_neither_raises(tmp_path):
    target = tmp_path / "rig.yml"
    texts = {name: (name * 20_000) + "\n" for name in ("a", "b")}
    errors: list[BaseException] = []
    start = threading.Barrier(2)

    def write(name: str) -> None:
        start.wait()
        try:
            for _ in range(60):
                paths.write_atomic(target, texts[name])
        except BaseException as exc:  # noqa: BLE001 - any failure is the finding
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(name,)) for name in texts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert target.read_text() in texts.values(), "one writer's text, whole, never a mix of both"
    assert not list(tmp_path.glob(".*.partial")), "no partial file is left behind"


def test_the_file_keeps_its_permissions_and_a_new_one_gets_the_umasks(tmp_path):
    kept = tmp_path / "kept.yml"
    kept.write_text("old\n")
    os.chmod(kept, 0o640)
    paths.write_atomic(kept, "new\n")
    assert kept.read_text() == "new\n"
    assert stat.S_IMODE(kept.stat().st_mode) == 0o640

    fresh = tmp_path / "fresh.yml"
    paths.write_atomic(fresh, "x\n")
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o666 & ~paths._umask(), "not mkstemp's owner-only 0600"
