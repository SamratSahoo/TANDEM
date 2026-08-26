"""The GPU runtime — the pixi environment `tandem init` builds.

tandem itself is pure Python and installs with one pip command. Everything heavy lives here
instead: torch, the compiled cuRobo CUDA kernels, cuTAMP, tiptop, SAM-2, open3d and the ZED
bindings. That split is what lets a plain install plus `tandem ui` work on a laptop.

The runtime directory deliberately mirrors the source monorepo's relative layout::

    <runtime>/
        tiptop/     cuTAMP/     curobo/
        vae/checkpoints/vae_full_v2.pt
        rnd/checkpoints/rnd_droid.pt

because three separate modules resolve default asset paths by walking up from ``__file__``
to what they assume is a repo root:

    tiptop/tiptop/motion_planning.py                    parents[2]
    curobo/src/curobo/rollout/cost/vae_manifold_cost.py parents[5]
    curobo/src/curobo/rollout/cost/rnd_novelty_cost.py  parents[5]

Reproducing the layout makes all three resolve correctly with no patching, and means an
existing ``cfg/tamp/*.yml`` imports verbatim.

It must also be a WRITABLE COPY rather than a path into the installed wheel: cuRobo is
installed editable, so its compiled ``.so`` files and build fingerprint land inside the
source tree.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tandem.core import paths
from tandem.core.errors import RuntimeNotReady, TandemError

# Directory names inside the runtime. cuTAMP keeps its capital T because tiptop's
# install-cutamp.sh defaults to CUTAMP_DIR=../cuTAMP.
TIPTOP = "tiptop"
CUTAMP = "cuTAMP"
CUROBO = "curobo"

STAMP_FILE = ".tandem-runtime.json"


@dataclass
class RuntimeStatus:
    exists: bool = False
    sources_present: bool = False
    env_built: bool = False
    kernels_built: bool = False
    ready: bool = False
    vendor: dict | None = None
    built_at: str | None = None
    problems: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "exists": self.exists,
            "sources_present": self.sources_present,
            "env_built": self.env_built,
            "kernels_built": self.kernels_built,
            "ready": self.ready,
            "vendor": self.vendor,
            "built_at": self.built_at,
            "problems": self.problems or [],
        }


class Runtime:
    """Locate, inspect, build and invoke the runtime."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser()

    # ---- layout ------------------------------------------------------------

    @property
    def tiptop_dir(self) -> Path:
        return self.root / TIPTOP

    @property
    def cutamp_dir(self) -> Path:
        return self.root / CUTAMP

    @property
    def curobo_dir(self) -> Path:
        return self.root / CUROBO

    @property
    def stamp_file(self) -> Path:
        return self.root / STAMP_FILE

    @property
    def pixi_env(self) -> Path:
        return self.tiptop_dir / ".pixi" / "envs" / "default"

    # ---- status ------------------------------------------------------------

    def status(self) -> RuntimeStatus:
        problems: list[str] = []
        st = RuntimeStatus(exists=self.root.is_dir())
        if not st.exists:
            st.problems = ["the runtime has not been created yet"]
            return st

        sources = {
            "tiptop": (self.tiptop_dir / "pixi.toml").is_file(),
            "cuTAMP": (self.cutamp_dir / "cutamp" / "__init__.py").is_file(),
            "curobo": (self.curobo_dir / "src" / "curobo").is_dir(),
        }
        st.sources_present = all(sources.values())
        for name, present in sources.items():
            if not present:
                problems.append(f"{name} source is missing from {self.root}")

        st.env_built = (self.pixi_env / "bin" / "python").is_file()
        if not st.env_built:
            problems.append("the pixi environment has not been created")

        st.kernels_built = bool(list((self.curobo_dir / "src" / "curobo" / "curobolib").glob("*.so")))
        if st.sources_present and not st.kernels_built:
            problems.append("cuRobo's CUDA kernels have not been compiled")

        if self.stamp_file.is_file():
            try:
                stamp = json.loads(self.stamp_file.read_text())
                st.vendor = stamp.get("vendor")
                st.built_at = stamp.get("built_at")
            except (ValueError, OSError):
                pass

        st.ready = st.sources_present and st.env_built and st.kernels_built
        st.problems = problems
        return st

    def pending_vendor(self) -> dict | None:
        """What a build WOULD install, read from the shipped vendor manifest.

        Shown before anything is built so `tandem runtime status` can answer "which cuRobo
        am I about to compile?" without first spending twenty minutes compiling it.
        """
        return _read_vendor(paths.vendor_dir() / "VENDOR.toml")

    def is_ready(self) -> bool:
        return self.status().ready

    def require_ready(self) -> None:
        st = self.status()
        if st.ready:
            return
        detail = "\n".join(f"  · {p}" for p in (st.problems or []))
        raise RuntimeNotReady(
            f"The GPU runtime at {self.root} is not ready.\n{detail}",
            hint="Run `tandem init` to build it, or `tandem runtime build` to retry just this step.",
        )

    # ---- invocation --------------------------------------------------------

    def command(self, args: list[str]) -> list[str]:
        """Wrap a console-script invocation in `pixi run`, from the tiptop manifest dir."""
        from tandem.core.probe import find_pixi

        pixi = find_pixi()
        if pixi is None:
            raise RuntimeNotReady(
                "pixi is not installed, so the runtime cannot be entered.",
                hint="Run `tandem init`, or: curl -fsSL https://pixi.sh/install.sh | bash",
            )
        return [str(pixi), "run", "--manifest-path", str(self.tiptop_dir / "pixi.toml"), *args]

    def python(self) -> Path:
        path = self.pixi_env / "bin" / "python"
        if not path.is_file():
            raise RuntimeNotReady(f"No interpreter at {path}.", hint="Run `tandem runtime build`.")
        return path

    # ---- build -------------------------------------------------------------

    def materialize(self, vendor_root: Path, *, force: bool = False, log=None) -> None:
        """Copy the vendored sources into the runtime.

        Copied, not symlinked: cuRobo is installed editable and writes compiled objects back
        into its own tree, which must not mutate the installed package.
        """
        say = log or (lambda _msg: None)
        self.root.mkdir(parents=True, exist_ok=True)

        # What this runtime currently holds, versus what the installed wheel ships. "Already
        # present" is only a safe thing to say when they are the SAME sources -- and after a
        # re-vendor they are not. Skipping the copy while stamping the new commits anyway left a
        # runtime built from the old planner while claiming to be the new one, which surfaces as an
        # ImportError forty seconds into a warm-up with an operator standing next to the arm.
        installed = self.status().vendor or {}
        manifest = _read_vendor(vendor_root / "VENDOR.toml")
        incoming = manifest or {}

        for name in (TIPTOP, CUTAMP, CUROBO, "vae", "rnd"):
            src = vendor_root / name
            dst = self.root / name
            if not src.is_dir():
                if name in (TIPTOP, CUTAMP, CUROBO):
                    raise TandemError(
                        f"Vendored source {name!r} is missing from {vendor_root}.",
                        hint=(
                            "The install looks incomplete. Reinstall tandem, or point "
                            "$TANDEM_VENDOR_DIR at a checkout that has it."
                        ),
                    )
                continue
            here = (installed.get(name) or {}).get("commit")
            wanted = (incoming.get(name) or {}).get("commit")
            stale = bool(wanted) and here != wanted
            if dst.exists() and not force and not stale:
                say(f"{name}: already present")
                continue
            if dst.exists():
                if stale:
                    say(f"{name}: replacing {here or 'an unrecorded copy'} with {wanted}")
                shutil.rmtree(dst)
            else:
                say(f"{name}: copying")
            shutil.copytree(src, dst, symlinks=False, ignore=_ignore_build_junk)

        # Stamped only after the sources are actually on disk, so the record describes the tree
        # rather than the intention. `built_at` is cleared because a changed source means the
        # compiled kernels and the installed packages are stale too.
        stamp = {"vendor": manifest, "built_at": None}
        self.stamp_file.write_text(json.dumps(stamp, indent=2) + "\n")

    def build_env(self, *, log=None, extra_env: dict | None = None) -> None:
        """`pixi install` — solve and materialise the conda environment."""
        self._pixi(["install"], log=log, extra_env=extra_env, what="pixi install")

    def build_planners(self, *, log=None, extra_env: dict | None = None) -> None:
        """`pixi run setup-planners` — build cuRobo's CUDA kernels, then install cuTAMP.

        Order matters (cuTAMP imports cuRobo), and the pixi task already encodes it via
        depends-on. CUROBO_DIR / CUTAMP_DIR point the install scripts at our copies; their
        defaults (../curobo, ../cuTAMP) already match this layout, but being explicit means
        a renamed directory is a clear error instead of a confusing one.
        """
        env = {
            "CUROBO_DIR": str(self.curobo_dir),
            "CUTAMP_DIR": str(self.cutamp_dir),
            # cuRobo takes its version from setuptools_scm with no fallback, and a vendored tree has
            # no SCM metadata -- it is extracted with `git archive` precisely so no VCS state rides
            # along. Without this the editable install fails with "unable to detect version ... make
            # sure you're building from a fully intact git repository" before the 5-20 minute kernel
            # build even starts. tiptop's own pixi manifest pins itself the same way; cuTAMP carries
            # a static version and needs nothing.
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO": self._vendored_version(CUROBO),
            **(extra_env or {}),
        }
        self._pixi(["run", "setup-planners"], log=log, extra_env=env, what="pixi run setup-planners")
        self._touch_built()

    def _vendored_version(self, component: str) -> str:
        """A PEP 440 version for a vendored component, from what the manifest recorded.

        The commit is the honest answer to "which sources are these", but it is not a version, so
        it rides along as a local segment: ``0.0.0+g4db8f92``. Nothing resolves a dependency on
        these, so the number itself does not matter — only that one exists.
        """
        recorded = (self.status().vendor or {}).get(component) or {}
        described = str(recorded.get("version") or "").strip()
        return f"0.0.0+g{described}" if described else "0.0.0"

    def _touch_built(self) -> None:
        from datetime import datetime, timezone

        stamp = {}
        if self.stamp_file.is_file():
            try:
                stamp = json.loads(self.stamp_file.read_text())
            except (ValueError, OSError):
                stamp = {}
        stamp["built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.stamp_file.write_text(json.dumps(stamp, indent=2) + "\n")

    def _pixi(self, args: list[str], *, log, extra_env: dict | None, what: str) -> None:
        from tandem.core.probe import find_pixi

        pixi = find_pixi()
        if pixi is None:
            raise TandemError("pixi is not installed.", hint="Run `tandem init` to install it.")

        env = dict(os.environ)
        # The vendored tiptop has no .git, so setuptools_scm cannot infer a version and the
        # editable install fails. The vendored pixi.toml sets this too; belt and braces.
        env.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TIPTOP", "0.1.0")
        env.update(extra_env or {})

        # --manifest-path goes before the subcommand: after `run` it would be read as an
        # argument to the task rather than to pixi.
        cmd = [str(pixi), "--manifest-path", str(self.tiptop_dir / "pixi.toml"), *args]

        proc = subprocess.Popen(
            cmd,
            cwd=str(self.tiptop_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if log:
                log(line.rstrip("\n"))
        code = proc.wait()
        if code != 0:
            raise TandemError(
                f"{what} failed (exit {code}).",
                hint="The full build log is in " + str(paths.log_dir()) + ". Re-run `tandem runtime build`.",
            )


def _ignore_build_junk(directory: str, names: list[str]) -> set[str]:
    """Never copy VCS metadata, caches, or a previous build's artifacts."""
    drop = {".git", ".pixi", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
    out = {n for n in names if n in drop or n.endswith((".egg-info", ".pyc", ".so"))}
    out.discard("__init__.py")
    return out


def _read_vendor(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        import tomlkit

        return dict(tomlkit.parse(path.read_text()))
    except Exception:
        return None


def default() -> Runtime:
    from tandem.core import settings as settings_mod

    return Runtime(settings_mod.load().resolved_runtime_dir())
