"""Single-keystroke input.

Labeling a rollout should be one key, not `y` + Enter — the operator is standing at a robot
with one hand on the E-stop. Falls back cleanly when stdin is not a terminal (a pipe, CI, a
session driven from the browser), where there is nobody to press anything anyway.
"""

from __future__ import annotations

import queue
import sys
import threading


class KeyReader:
    """Read single characters from a terminal in a background thread."""

    def __init__(self) -> None:
        self.queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._saved = None
        self.available = self._can_raw()

    @staticmethod
    def _can_raw() -> bool:
        if not sys.stdin.isatty():
            return False
        try:
            import termios  # noqa: F401
            import tty  # noqa: F401
        except ImportError:
            return False
        return True

    def __enter__(self) -> KeyReader:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        if not self.available or self._thread is not None:
            return
        import termios
        import tty

        fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(fd)
        # cbreak, not raw: Ctrl-C still reaches us as a signal, which is what an operator
        # expects from a terminal next to a moving arm.
        tty.setcbreak(fd)
        self._thread = threading.Thread(target=self._run, name="keys", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._saved is not None:
            import termios

            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved)
            except Exception:
                pass
            self._saved = None
        self._thread = None

    def _run(self) -> None:
        import select

        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                char = sys.stdin.read(1)
            except (OSError, ValueError):
                return
            if char:
                self.queue.put(char)

    def get(self, timeout: float = 0.1) -> str | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_line(self, prompt: str = "") -> str:
        """Temporarily leave cbreak mode to read a whole typed line (a new task)."""
        import termios

        if not self.available:
            return input(prompt)
        fd = sys.stdin.fileno()
        raw_attrs = termios.tcgetattr(fd)
        try:
            if self._saved is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, self._saved)
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=0.5)
                self._thread = None
            return input(prompt)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, raw_attrs)
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="keys", daemon=True)
            self._thread.start()
