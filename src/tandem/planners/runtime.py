"""A planner's runtime, declared rather than scripted: pinned sources, an environment, build steps.

A planner that needs more than pip -- torch, compiled kernels, a solver with an environment of its
own -- runs inside a runtime, and this module builds one from a ``RuntimeRecipe``:

- which source trees, each pinned to an exact commit, what to trim from each and which patches to
  apply to it;
- the environment manifest that solves the heavy dependencies (pixi);
- the build steps to run inside that environment once it is solved (compile kernels, install the
  sources editable);
- small files the planner package ships itself and the runtime needs at a fixed place.

Nothing here knows any particular planner. TiPToP's recipe is ``tandem/planners/tiptop/recipe.py``;
another planner declares its own the same way, and ``tandem planners install NAME`` installs it unchanged.

**Sources are fetched at install, not shipped in the wheel.** ``git fetch --depth 1 <url> <commit>``
into a scratch repository, then ``git archive`` of exactly that commit, so no VCS state and no build
artifact from anyone's working tree can ride along. A machine with no git gets GitHub's archive of
the same commit over HTTPS instead. Either way the commit is checked: git names the object it
fetched, and ``git archive`` stamps the commit into the tarball it makes -- GitHub's archive included.

**An offline machine installs from a directory instead.** ``TANDEM_PLANNER_SOURCES``, or
``tandem planners install NAME --sources DIR``, names a directory holding one checkout or export per source,
named as the recipe names them. A checkout is used as an object store: the pinned commit is exported
out of it, whatever its working tree holds. An export made by ``tools/bundle.py`` carries a marker
naming its commit, which has to be the pinned one, and a digest of its files, which have to be the
ones that were bundled (``tree_digest``). A bare directory with neither is taken on trust and
recorded as unverified. Nothing is ever fetched while a sources directory is in force: a source
missing from it is an error, because an air-gapped rig reaching for the network is a hang, not a
fallback.

**What was installed is written down**, in ``<runtime>/.tandem-runtime.json``: each tree's URL and
commit, where it came from, whether that commit was verified, what was trimmed from it and the
digest of every patch applied to it. ``status()`` compares that record against the recipe. A tandem
upgrade that moves a pin therefore reads "rebuild", instead of starting a session on a planner the
sidecar was never written against -- which is an ImportError forty seconds into a warm-up, with an
operator standing next to the arm.

Layout, for a recipe with sources ``a`` and ``b`` and a pixi environment whose manifest is in ``a``::

    <runtime>/
        a/  b/                    each source tree at its pinned commit, trimmed and patched
        a/.pixi -> ../env         where pixi looks for the environment
        env/                      the environment itself, OUTSIDE every source tree
        cache/a/...               what a tree writes into itself at run time, kept outside it too
        .tandem-runtime.json      what is installed, and from where
        .install.lock             held by the one install that may be running

The environment lives outside the trees on purpose. A tree is replaced wholesale when its pin moves,
and an environment inside it would be deleted with it and solved again from nothing -- a full torch
and CUDA download -- on every bump. pixi keeps a manifest's environment in ``<manifest dir>/.pixi``,
so that one path is a symlink: the tree is swapped underneath it and the environment is found again,
at the same absolute path it was built at. A directory a planner downloads into its own tree at run
time (``Source.persistent``: TiPToP's SAM-2 checkpoint) is kept out of the swap the same way, under
``cache/``.

**Only what tandem installed is ever replaced.** The root is a setting, and pointed at a workspace by
mistake -- the monorepo this layout mirrors, say -- a tree swap would delete someone's checkout, its
uncommitted work included. So a tree the record does not list is replaced only in a root that holds
tandem's record, and a git checkout never is: nothing tandem installs contains ``.git``.

Standard library and tandem's own light modules only: a planner's recipe is imported to list
planners, on a laptop that will never build one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from tandem.core import paths
from tandem.core.errors import RuntimeNotReady, TandemError
from tandem.planners.base import RuntimeStatus, SourcePin

#: What is installed, and from where. The name is the one the runtime's stamp has always had, so a
#: runtime built before sources were fetched is still read (see ``_read_manifest``).
MANIFEST_FILE = ".tandem-runtime.json"
#: Written into each export ``tools/bundle.py`` makes: the commit it is, so an offline install can
#: check it has been handed the pinned one.
SOURCE_MARKER = ".tandem-source.json"
#: Scratch space inside the runtime, so a half-fetched tree never sits where a finished one belongs
#: and a rename into place stays on one filesystem.
STAGING = ".staging"
#: Held for as long as an install runs (``RecipeRuntime._install_lock``).
LOCK_FILE = ".install.lock"
#: Where each source's ``persistent`` directories live, outside every tree.
PERSISTENT = "cache"
FORMAT = 2

Log = Callable[[str], None]

_COMMIT = re.compile(r"[0-9a-f]{40}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_PLACEHOLDER = re.compile(r"\{(root|source|version|commit)(?::([A-Za-z0-9_.-]+))?\}")
# Build junk a working tree accumulates and an install must never copy: VCS state, caches, and the
# artifacts of a previous build (a compiled extension from someone else's machine is worse than none).
_JUNK_NAMES = {".git", ".pixi", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", SOURCE_MARKER}
_JUNK_SUFFIXES = (".egg-info", ".pyc", ".so")


def _is_junk(name: str) -> bool:
    return name in _JUNK_NAMES or name.endswith(_JUNK_SUFFIXES)


def _quiet(_line: str) -> None:
    return None


# --------------------------------------------------------------------------- the recipe


@dataclass(frozen=True)
class Source:
    """One source tree a runtime is built from: a pinned commit, and what is done to it on the way in."""

    pin: SourcePin
    # Paths dropped from the tree, relative to its root. Every one should say why in the recipe.
    trim: tuple[str, ...] = ()
    # Unified diffs applied in order after trimming. Files the planner package ships, so a patch
    # that stops applying after a bump fails the install loudly instead of being silently lost.
    patches: tuple[Path, ...] = ()
    # A path inside the tree that exists only when the tree is really there (its manifest, a
    # package's __init__.py). Status trusts it over a directory that may be half-extracted.
    marker: str = ""
    # Directories inside the tree that the planner writes into at run time and that must outlive the
    # tree: a checkpoint it downloads into its own package, say. A replaced tree takes everything
    # inside it along, so each of these is kept under <runtime>/cache/<source>/ instead and the tree
    # holds a symlink to it, the way it holds one to the environment.
    persistent: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.pin.name


@dataclass(frozen=True)
class Asset:
    """A file the planner package ships, copied to a fixed place in the runtime at install."""

    source: Path
    # Relative to the runtime root.
    dest: str


@dataclass(frozen=True)
class PixiEnvironment:
    """A pixi environment, solved from a manifest that one of the sources carries.

    The manifest and its lock stay the planner's own -- tandem solves exactly what the planner
    pinned -- and only where the environment is kept changes (see the module docstring).
    """

    # Relative to the runtime root, and inside one of the source trees: "tiptop/pixi.toml".
    manifest: str
    # The environment's own directory, relative to the runtime root. Never inside a source tree.
    home: str = "env"
    # pixi's name for the environment.
    name: str = "default"
    # Set for every pixi call, unless the caller's own environment already sets it. Values may use
    # the placeholders ``BuildStep.env`` documents.
    env: Mapping[str, str] = field(default_factory=dict)
    # How a listing names it: "<label> built" / "<label> not built".
    label: str = "pixi env"

    @property
    def tool(self) -> str:
        return "pixi"


@dataclass(frozen=True)
class BuildStep:
    """Something run inside the solved environment: a task the environment manifest defines.

    ``env`` values may use placeholders, filled in at build time so a recipe never hard-codes a
    machine's paths: ``{root}`` is the runtime root, ``{source:NAME}`` a source tree's path,
    ``{commit:NAME}`` the commit it is installed at, and ``{version:NAME}`` that commit as a PEP 440
    version (``0.0.0+g4db8f92``) for a package whose version would otherwise come from the git
    metadata an exported tree does not have.
    """

    name: str
    task: str
    env: Mapping[str, str] = field(default_factory=dict)
    # Globs relative to the runtime root that match only once the step has run -- a compiled
    # extension, say. A step with none is judged by whether the last build finished.
    produces: tuple[str, ...] = ()
    # One line for a progress display while it runs.
    description: str = ""
    # How a listing names it: "<label> <done>" / "<label> <todo>".
    label: str = ""
    done: str = "built"
    todo: str = "not built"
    # The problem status reports while it has not run.
    problem: str = ""

    @property
    def title(self) -> str:
        return self.label or self.name


@dataclass(frozen=True)
class RuntimeRecipe:
    """Everything needed to build a planner's runtime from nothing, and nothing else.

    Declared, not scripted, so that a status can be computed from it without running anything, and
    so that a planner's runtime is a data file's worth of reading rather than an install script.
    """

    planner: str
    sources: tuple[Source, ...]
    title: str = ""
    environment: PixiEnvironment | None = None
    steps: tuple[BuildStep, ...] = ()
    assets: tuple[Asset, ...] = ()
    # Shown before a build starts: what it is about to install and how long it takes.
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate(self)

    @property
    def display_name(self) -> str:
        return self.title or self.planner

    @property
    def pins(self) -> tuple[SourcePin, ...]:
        return tuple(source.pin for source in self.sources)

    def source(self, name: str) -> Source:
        for source in self.sources:
            if source.name == name:
                return source
        raise KeyError(name)


def _validate(recipe: RuntimeRecipe) -> None:
    """Refuse a recipe that could not be installed, when it is declared rather than when it is built.

    Every check here is a mistake that would otherwise surface twenty minutes into an install, or
    worse, as an rmtree of the wrong directory.
    """

    def bad(problem: str) -> TandemError:
        return TandemError(
            f"The {recipe.planner or '?'} planner's runtime recipe is invalid: {problem}",
            hint="This is a bug in the planner package, not in your setup. Report it to its authors.",
        )

    if not recipe.planner:
        raise bad("it names no planner")
    if not recipe.sources:
        raise bad("it has no sources")
    reserved = {MANIFEST_FILE, STAGING, LOCK_FILE, PERSISTENT}
    names: set[str] = set()
    for source in recipe.sources:
        name = source.name
        if not _NAME.fullmatch(name):
            raise bad(f"{name!r} cannot be a directory name")
        if name in names:
            raise bad(f"two sources are named {name!r}")
        if name in reserved:
            raise bad(f"{name!r} is a name the runtime uses for itself")
        names.add(name)
        if not _COMMIT.fullmatch(source.pin.commit):
            # A branch or a short hash names a different tree tomorrow, and a dataset has to be
            # traceable to the planner that produced it.
            raise bad(f"{name} is pinned to {source.pin.commit!r}, which is not a full 40-character commit")
        if not source.pin.url:
            raise bad(f"{name} has no URL to fetch it from")
        for relative in (*source.trim, *source.persistent, *([source.marker] if source.marker else [])):
            if not _is_inside(relative):
                raise bad(f"{name}: {relative!r} is not a path inside the tree")

    env = recipe.environment
    if env is not None:
        manifest = PurePosixPath(env.manifest)
        if not _is_inside(env.manifest) or len(manifest.parts) < 2 or manifest.parts[0] not in names:
            raise bad(f"the environment manifest {env.manifest!r} is not inside one of the sources")
        if not _is_inside(env.home) or PurePosixPath(env.home).parts[0] in names | reserved:
            # Inside a source tree it would be deleted with that tree on the next bump -- the one
            # thing keeping it separate exists to prevent.
            raise bad(f"the environment's home {env.home!r} must be its own directory, outside every source")
        _check_placeholders(env.env, names, bad)

    steps: set[str] = set()
    for step in recipe.steps:
        if env is None:
            raise bad(f"step {step.name!r} runs in an environment, but the recipe declares none")
        if not _NAME.fullmatch(step.name) or step.name in steps or step.name in ("sources", "environment"):
            raise bad(f"step name {step.name!r} is unusable or used twice")
        steps.add(step.name)
        _check_placeholders(step.env, names, bad)
        for pattern in step.produces:
            if not _is_inside(pattern):
                raise bad(f"step {step.name!r}: {pattern!r} is not a path inside the runtime")

    for asset in recipe.assets:
        if not _is_inside(asset.dest):
            raise bad(f"asset destination {asset.dest!r} is not a path inside the runtime")
        top = PurePosixPath(asset.dest).parts[0]
        if top in reserved or (env is not None and top == PurePosixPath(env.home).parts[0]):
            raise bad(f"asset destination {asset.dest!r} is a place the runtime uses for itself")


def _check_placeholders(values: Mapping[str, str], names: set[str], bad) -> None:
    for value in values.values():
        for kind, name in _PLACEHOLDER.findall(str(value)):
            if kind == "root" and name:
                raise bad(f"{{root}} takes no source name, in {value!r}")
            if kind != "root" and name not in names:
                raise bad(f"{value!r} refers to a source the recipe does not have")


def _is_inside(relative: str) -> bool:
    path = PurePosixPath(relative)
    return bool(relative) and not path.is_absolute() and ".." not in path.parts and "\\" not in relative


# --------------------------------------------------------------------------- status


@dataclass(frozen=True)
class SourceState:
    """One source tree: what the recipe wants, and what the runtime holds."""

    name: str
    wanted: SourcePin
    present: bool
    # What the record says is installed. None when nothing is recorded for it.
    commit: str | None = None
    url: str | None = None
    origin: str | None = None
    verified: bool | None = None
    patches_match: bool = True

    @property
    def current(self) -> bool:
        return self.present and self.commit == self.wanted.commit and self.patches_match


@dataclass(frozen=True)
class RecipeStatus:
    """Everything a status display needs about a recipe's runtime, computed with stat calls only."""

    root: Path
    exists: bool
    sources: tuple[SourceState, ...] = ()
    environment_built: bool = False
    # step name -> whether it has run, in recipe order.
    steps: tuple[tuple[str, bool], ...] = ()
    assets_present: bool = True
    built_at: str | None = None
    # What keeps the runtime from running. Empty exactly when it is ready.
    problems: tuple[str, ...] = ()
    # Worth knowing, not worth refusing a session over: a tree taken on trust, say.
    notes: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.exists and not self.problems

    @property
    def sources_present(self) -> bool:
        return self.exists and all(s.present for s in self.sources)

    @property
    def steps_done(self) -> bool:
        return all(done for _, done in self.steps)


# --------------------------------------------------------------------------- the runtime


class RecipeRuntime:
    """A recipe's runtime at one root: inspect it, install it, enter it, delete it.

    Implements ``tandem.planners.base.BackendRuntime``, so a catalog installs any planner that
    declares a recipe the same way. The finer-grained operations (``fetch``, ``build_environment``,
    ``run_step``) are what ``install`` is made of, and are public for a caller that shows each as its
    own step.
    """

    def __init__(self, recipe: RuntimeRecipe, root: Path) -> None:
        self.recipe = recipe
        self.root = Path(root).expanduser()

    # ---- layout ------------------------------------------------------------

    @property
    def manifest_file(self) -> Path:
        return self.root / MANIFEST_FILE

    def source_dir(self, name: str) -> Path:
        self.recipe.source(name)  # a KeyError names a source the recipe does not have
        return self.root / name

    @property
    def env_home(self) -> Path | None:
        env = self.recipe.environment
        return None if env is None else self.root / env.home

    @property
    def env_prefix(self) -> Path | None:
        env = self.recipe.environment
        return None if env is None else self.root / env.home / "envs" / env.name

    @property
    def env_manifest(self) -> Path | None:
        env = self.recipe.environment
        return None if env is None else self.root / env.manifest

    @property
    def workdir(self) -> Path:
        """Where a command in the environment runs: the manifest's directory, as pixi expects."""
        manifest = self.env_manifest
        return manifest.parent if manifest is not None else self.root

    # ---- status ------------------------------------------------------------

    def inspect(self) -> RecipeStatus:
        """What is installed, against what the recipe wants. Stat calls and one small JSON read."""
        recipe = self.recipe
        if not self.root.is_dir():
            return RecipeStatus(
                root=self.root, exists=False, problems=("the runtime has not been created yet",)
            )

        record = _read_manifest(self.manifest_file)
        installed = record["sources"]
        problems: list[str] = []
        notes: list[str] = []

        states: list[SourceState] = []
        for source in recipe.sources:
            tree = self.root / source.name
            present = (tree / source.marker).exists() if source.marker else _non_empty(tree)
            entry = installed.get(source.name) or {}
            state = SourceState(
                name=source.name,
                wanted=source.pin,
                present=present,
                commit=entry.get("commit"),
                url=entry.get("url"),
                origin=entry.get("origin"),
                verified=entry.get("verified"),
                patches_match=_patches_match(entry.get("patches"), source.patches),
            )
            states.append(state)
            short = source.pin.commit[:7]
            if not present:
                problems.append(f"{source.name} source is missing from {self.root}")
            elif state.commit is None:
                problems.append(f"{source.name} is not recorded as installed at {short}")
            elif state.commit != source.pin.commit:
                problems.append(
                    f"{source.name} is at {state.commit[:7]}, but {recipe.display_name} now pins {short}"
                )
            elif not state.patches_match:
                problems.append(f"{source.name} was patched differently from what the recipe applies now")
            elif state.verified is False:
                notes.append(
                    f"{source.name} came from {state.origin or 'a directory'} and its commit was not verified"
                )

        env_built = False
        if recipe.environment is not None:
            env_built = self._env_python() is not None
            if not env_built:
                problems.append(f"the {recipe.environment.tool} environment has not been created")

        sources_present = all(s.present for s in states)
        steps: list[tuple[str, bool]] = []
        for step in recipe.steps:
            done = all(any(self.root.glob(pattern)) for pattern in step.produces)
            steps.append((step.name, done))
            # Only once the sources are there: "the kernels have not been compiled" says nothing
            # useful about a runtime with nothing to compile yet.
            if sources_present and not done:
                problems.append(step.problem or f"{step.title} {step.todo}")

        assets_present = True
        for asset in recipe.assets:
            if not (self.root / asset.dest).is_file():
                assets_present = False
                problems.append(f"{asset.dest} is missing")

        built = record.get("built") or {}
        failed = record.get("last_build") or {}
        builds = recipe.environment is not None or bool(recipe.steps)
        if builds and failed.get("ok") is False:
            # Said whatever else is wrong. A failed step can leave behind exactly what its `produces`
            # globs look for (one compiled extension of five), and without this the only problem
            # left to report would be the branch below's "not built since its sources changed" --
            # which blames sources nobody changed for a build that failed.
            at = f" ({failed['at']})" if failed.get("at") else ""
            problems.append(
                f"{failed.get('command') or 'the build'} did not finish{at}, and no build has since; "
                f"the build log in {paths.log_dir()} says why"
            )
        elif builds and not problems:
            # Everything is on disk -- but was it built for THESE sources? The environment outlives
            # a bump by design, so after one it is there, whole, and solved for the old commits.
            have = {s.name: s.commit for s in states}
            built_for = built.get("sources") or {}
            changed = sorted(n for n, c in have.items() if built_for.get(n) != c)
            if not built_for:
                problems.append("the runtime has not been built since its sources were installed")
            elif changed or not built.get("at"):
                problems.append(
                    "the runtime has not been built since "
                    + (", ".join(changed) if changed else "its sources")
                    + " changed"
                )

        return RecipeStatus(
            root=self.root,
            exists=True,
            sources=tuple(states),
            environment_built=env_built,
            steps=tuple(steps),
            assets_present=assets_present,
            built_at=built.get("at"),
            problems=tuple(problems),
            notes=tuple(notes),
        )

    def status(self) -> RuntimeStatus:
        st = self.inspect()
        return RuntimeStatus(
            installed=st.ready,
            path=str(self.root),
            pins=self._installed_pins(),
            detail=self.describe(st),
            problems=st.problems,
        )

    def describe(self, st: RecipeStatus | None = None) -> str:
        """One line of what is and is not there, for a listing."""
        st = st or self.inspect()
        if not st.exists:
            return "not created"
        parts = ["sources present" if st.sources_present else "sources missing"]
        env = self.recipe.environment
        if env is not None:
            parts.append(f"{env.label} {'built' if st.environment_built else 'not built'}")
        done = dict(st.steps)
        for step in self.recipe.steps:
            parts.append(f"{step.title} {step.done if done.get(step.name) else step.todo}")
        if st.built_at:
            parts.append(f"built {st.built_at}")
        return " · ".join(parts)

    def rows(self, st: RecipeStatus | None = None) -> list[tuple[str, str]]:
        """(label, state) pairs for a status table: the same facts as ``describe``, one per line."""
        st = st or self.inspect()
        rows = [("sources", "present" if st.sources_present else "missing")]
        env = self.recipe.environment
        if env is not None:
            rows.append((env.label, "built" if st.environment_built else "not built"))
        done = dict(st.steps)
        for step in self.recipe.steps:
            rows.append((step.title, step.done if done.get(step.name) else step.todo))
        rows.append(("built", st.built_at or "—"))
        return rows

    def is_ready(self) -> bool:
        return self.inspect().ready

    def require_ready(self) -> None:
        st = self.inspect()
        if st.ready:
            return
        detail = "\n".join(f"  · {p}" for p in st.problems)
        raise RuntimeNotReady(
            f"The {self.recipe.display_name} runtime at {self.root} is not ready.\n{detail}",
            hint=f"Run `tandem planners install {self.recipe.planner}` to build or repair it; every step "
            "already done is skipped.",
        )

    def record(self) -> dict:
        """The installed record (``.tandem-runtime.json``), normalised:
        ``{planner, sources, assets, built, last_build}``.

        ``sources`` maps each installed tree to ``{url, commit, origin, verified, trimmed, patches}``.
        ``built`` is ``{at, sources}``: when the build last finished (None since a tree changed) and
        the commits it was for. ``last_build`` is ``{step, command, ok: False, at}`` while a build
        that failed has not been followed by one that finished. Empty, never an error, for a runtime
        that has none yet.
        """
        return _read_manifest(self.manifest_file)

    def _installed_pins(self) -> tuple[SourcePin, ...]:
        """The pins the record says are installed: the recipe's sources first, in its order."""
        installed = self.record()["sources"]
        order = [s.name for s in self.recipe.sources] + [n for n in installed if n not in self.recipe_names]
        pins = []
        for name in order:
            entry = installed.get(name)
            if isinstance(entry, dict) and entry.get("commit"):
                pins.append(SourcePin(name, str(entry.get("url") or ""), str(entry["commit"])))
        return tuple(pins)

    @property
    def recipe_names(self) -> set[str]:
        return {s.name for s in self.recipe.sources}

    # ---- install -----------------------------------------------------------

    def plan(self, *, env_only: bool = False) -> list[tuple[str, str]]:
        """The steps ``install`` will announce, as (key, description), for sizing a progress display."""
        out = [("sources", "fetching the pinned sources")]
        if self.recipe.environment is not None:
            out.append(("environment", f"solving the {self.recipe.environment.tool} environment"))
        if not env_only:
            out += [(step.name, step.description or f"running {step.task}") for step in self.recipe.steps]
        return out

    def install(
        self,
        *,
        on_progress: Log | None = None,
        sources_dir: Path | None = None,
        force: bool = False,
        env_only: bool = False,
        on_step: Callable[[str, str], None] | None = None,
    ) -> None:
        """Fetch, patch, place, solve, build. Idempotent: a finished step is skipped or is a no-op.

        ``sources_dir`` installs from a directory of checkouts or exports instead of the network (it
        defaults to $TANDEM_PLANNER_SOURCES). ``force`` fetches every source again. ``env_only`` stops
        once the environment is solved. ``on_step(key, description)`` is called as each of
        ``plan()``'s steps starts; ``on_progress`` receives every line of output.
        """
        say = on_progress or _quiet
        announce = on_step or (lambda _key, _text: None)
        stages = dict(self.plan(env_only=env_only))
        if self.recipe.environment is not None and _find_pixi() is None:
            # Before fetching anything: the fetch would succeed, and the build fail a minute later.
            raise TandemError(
                f"pixi is not installed, and the {self.recipe.display_name} runtime is built in a pixi environment.",
                hint="Run `tandem init`, which installs it, or install it from https://pixi.sh.",
            )

        # Before the lock, whose file would be the first thing written into a directory that is not
        # tandem's. fetch checks again once it holds it, and checks the sources directory too.
        self._refuse_foreign_trees(None, force=force)
        with self._install_lock():
            announce("sources", stages["sources"])
            self.fetch(sources_dir=sources_dir, force=force, log=say)
            self.place_assets(log=say)

            env = self.recipe.environment
            if env is not None:
                announce("environment", stages["environment"])
                with self._build_stage("environment", f"{env.tool} install"):
                    self.build_environment(log=say)
            if env_only:
                return
            for step in self.recipe.steps:
                announce(step.name, stages[step.name])
                with self._build_stage(step.name, f"pixi run {step.task}"):
                    self.run_step(step, log=say)
            self.record_built()

        st = self.inspect()
        if not st.ready:
            raise TandemError(
                "The build finished but the runtime still looks incomplete: " + "; ".join(st.problems),
                hint="Run the install again; every finished step is skipped.",
            )

    @contextmanager
    def _build_stage(self, key: str, command: str) -> Iterator[None]:
        """Write down a build stage that did not finish, whatever stopped it, so status can say so.

        Interrupted counts: a build stopped with Ctrl-C is as unfinished as one that failed.
        ``record_built`` clears it once a whole build has run.
        """
        try:
            yield
        except BaseException:
            record = _read_manifest(self.manifest_file)
            record["last_build"] = {"step": key, "command": command, "ok": False, "at": _now()}
            _write_manifest(self.manifest_file, record)
            raise

    @contextmanager
    def _install_lock(self) -> Iterator[None]:
        """Hold this runtime's install lock for the duration: one install at a time, per runtime.

        Every install stages its trees in the same ``.staging`` and builds in the same environment.
        Two at once delete each other's staged trees -- the old tree kept to roll back to included --
        and run one CUDA build twice, concurrently, into one environment; the loser dies of a
        FileNotFoundError that names neither. The second install is refused rather than queued: it
        is usually the same command typed again in another terminal by someone who thought the first
        had hung, and a silent wait would look like exactly that.

        Re-entrant within a thread, since ``install`` holds it and calls ``fetch``, which takes it
        for a caller that fetches on its own. Another thread of this process is refused just as
        another process is: it would share the staging area just the same.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        key = str(self.root.resolve())
        me = threading.get_ident()
        with _LOCKS_GUARD:
            held = _LOCKS.get(key)
            if held is None:
                held = _Held(_acquire(self.root / LOCK_FILE, self.recipe.display_name), me)
                _LOCKS[key] = held
            elif held.owner != me:
                raise _busy(self.recipe.display_name, str(os.getpid()))
            held.depth += 1
        try:
            yield
        finally:
            with _LOCKS_GUARD:
                held.depth -= 1
                if held.depth == 0:
                    del _LOCKS[key]
                    if held.fd is not None:
                        os.close(held.fd)  # closing the descriptor releases the flock

    def fetch(
        self, *, sources_dir: Path | None = None, force: bool = False, log: Log | None = None
    ) -> list[str]:
        """Put every source tree in place at its pin. Returns the names that were (re)installed.

        A tree already recorded at its pin, with the recipe's patches, is left alone unless ``force``.
        Each tree is staged beside the runtime and renamed into place only once it is exported,
        trimmed and patched, so an interrupted fetch never leaves half a tree where a whole one was.
        A tree the record does not list is replaced only if tandem could have put it there
        (``_refuse_foreign_tree``); ``force`` does not change that.
        """
        say = log or _quiet
        self._check_shipped_files()
        override = self._sources_override(sources_dir)
        if override is not None:
            say(f"sources: installing from {override}, not fetching")
        # Before the lock, whose file would be the first thing written into a directory that is not
        # tandem's; checked again once it is held.
        self._refuse_foreign_trees(override, force=force)
        with self._install_lock():
            return self._fetch(override, force=force, say=say)

    def _sources_override(self, sources_dir: Path | None) -> Path | None:
        """The directory to install from instead of the network, checked: ``sources_dir``, else the
        environment's, else None."""
        override = (
            Path(sources_dir).expanduser() if sources_dir is not None else paths.planner_sources_override()
        )
        if override is None:
            return None
        if not override.is_dir():
            raise TandemError(
                f"The planner sources directory {override} does not exist.",
                hint="Point --sources or $TANDEM_PLANNER_SOURCES at a directory holding one checkout "
                "or export per source, or unset it to fetch them.",
            )
        if self.root.exists() and override.resolve() == self.root.resolve():
            # Every tree would be exported from itself, straight into its own replacement.
            raise TandemError(
                f"The planner sources directory {override} is the runtime itself.",
                hint="Point --sources or $TANDEM_PLANNER_SOURCES at the directory holding your checkouts "
                "or bundle, and the runtime somewhere else (`tandem runtime path` shows where it is).",
            )
        return override

    def _pending(self, record: dict, *, force: bool) -> list[tuple[Source, dict, Path, bool]]:
        """Every source as (source, its record entry, its tree, whether it has to be installed)."""
        out = []
        for source in self.recipe.sources:
            entry = record["sources"].get(source.name) or {}
            tree = self.root / source.name
            present = (tree / source.marker).exists() if source.marker else _non_empty(tree)
            current = (
                present
                and entry.get("commit") == source.pin.commit
                and _patches_match(entry.get("patches"), source.patches)
            )
            out.append((source, entry, tree, force or not current))
        return out

    def _refuse_foreign_trees(self, override: Path | None, *, force: bool) -> None:
        """``_refuse_foreign_tree`` for every tree a fetch would replace -- all of them before any is
        touched, so a workspace holding one checkout among three trees is refused whole rather than
        left with two of tandem's trees beside one of its own."""
        if not self.root.is_dir():
            return
        had_record = self.manifest_file.is_file()
        for source, entry, tree, due in self._pending(_read_manifest(self.manifest_file), force=force):
            if due:
                self._refuse_foreign_tree(source, entry, tree, override, had_record=had_record)

    def _fetch(self, override: Path | None, *, force: bool, say: Log) -> list[str]:
        staging = self.root / STAGING
        self._refuse_foreign_trees(override, force=force)
        record = _read_manifest(self.manifest_file)
        todo = []
        for source, entry, _tree, due in self._pending(record, force=force):
            if due:
                todo.append((source, entry))
            else:
                say(f"{source.name}: already at {source.pin.short()}")

        # Before any tree is replaced: a runtime built when the environment lived INSIDE the tree has
        # it moved out, and so does one whose trees still keep what they must not lose -- or the swap
        # would delete them.
        self._adopt_environment(say)
        self._link_persistent(say)

        changed: list[str] = []
        try:
            for source, entry in todo:
                if not self.manifest_file.is_file():
                    # Written before the first tree lands, so that a root with no record is one
                    # tandem never installed anything into -- which _refuse_foreign_tree relies on.
                    record["planner"] = self.recipe.planner
                    _write_manifest(self.manifest_file, record)
                if entry.get("commit") and entry.get("commit") != source.pin.commit:
                    say(f"{source.name}: replacing {str(entry['commit'])[:7]} with {source.pin.short()}")
                record["sources"][source.name] = self._install_source(source, override, staging, say)
                self._link_persistent(say)
                # The environment and anything built in it were built for the old tree. What it was
                # built for is kept, so status can name the tree that moved; a build that failed
                # before this is about trees that are gone.
                built_for = (record.get("built") or {}).get("sources") or {}
                record["built"] = {"at": None, "sources": built_for}
                record["last_build"] = None
                record["planner"] = self.recipe.planner
                # Written after every tree, so an install interrupted halfway resumes with what it has.
                _write_manifest(self.manifest_file, record)
                changed.append(source.name)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        self._link_environment()
        self._link_persistent(say)
        return changed

    def _refuse_foreign_tree(
        self, source: Source, entry: dict, tree: Path, override: Path | None, *, had_record: bool
    ) -> None:
        """Refuse to replace a tree at ``tree`` that tandem did not install.

        A tree the record lists is tandem's. One it does not list is ordinarily an install that was
        interrupted before its record was written -- and replacing it is how that install resumes --
        but the root is a setting, and this layout is the monorepo's: pointed at a workspace, the
        swap would delete a person's checkout, uncommitted work and all. Two things tell them apart:
        tandem writes its record into a root before the first tree lands there, and nothing it
        installs has a ``.git`` (``git archive`` has none, and a copy leaves it out).
        """
        if override is not None and (override / source.name).resolve() == tree.resolve():
            raise TandemError(
                f"{tree} is both the tree being installed and the one it would be installed from.",
                hint="Point --sources or $TANDEM_PLANNER_SOURCES at the directory holding your checkouts or "
                "bundle, not at the runtime.",
            )
        if entry.get("commit") or not (tree.is_symlink() or _non_empty(tree)):
            return
        if _is_repository(tree):
            why = "it is a git checkout, and nothing tandem installs has a .git"
        elif not had_record:
            why = f"{self.root} has no {MANIFEST_FILE}, so tandem never installed anything there"
        else:
            return
        raise TandemError(
            f"{tree} is not a tree tandem installed ({why}), so it was not replaced.",
            hint=f"The runtime is configured to be {self.root} (`tandem runtime path`). Point it at an empty "
            "directory, or at one tandem made. To build from checkouts you already have, name the directory "
            "holding them with --sources (or $TANDEM_PLANNER_SOURCES): they are only read, never changed. "
            f"If {tree} really is disposable, move or delete it by hand.",
        )

    def _install_source(self, source: Source, override: Path | None, staging: Path, say: Log) -> dict:
        staging.mkdir(parents=True, exist_ok=True)
        staged = staging / source.name
        shutil.rmtree(staged, ignore_errors=True)
        if override is not None:
            origin = export_from_directory(source.pin, override, staged, scratch=staging, log=say)
        else:
            origin = export_pinned(source.pin, staged, scratch=staging, log=say)
        trimmed = trim(staged, source.trim, name=source.name, log=say)
        applied = apply_patches(staged, source.patches, name=source.name, log=say)

        dest = self.root / source.name
        old = staging / f"{source.name}.old"
        if dest.exists() or dest.is_symlink():
            shutil.rmtree(old, ignore_errors=True)
            os.replace(dest, old)
        try:
            os.replace(staged, dest)
        except OSError:
            if old.exists():
                os.replace(old, dest)  # put the old tree back rather than lose it with the staging area
            raise
        # rmtree never follows a symlink, so the environment the old tree's .pixi pointed at survives.
        shutil.rmtree(old, ignore_errors=True)
        say(f"{source.name}: installed at {source.pin.short()}")
        return {
            "url": source.pin.url,
            "commit": source.pin.commit,
            **origin,
            "trimmed": trimmed,
            "patches": applied,
            "installed_at": _now(),
        }

    def place_assets(self, *, log: Log | None = None) -> None:
        """Copy the planner package's own files to where the runtime needs them. Skips an identical one."""
        say = log or _quiet
        self._check_shipped_files()
        self.root.mkdir(parents=True, exist_ok=True)
        record = _read_manifest(self.manifest_file)
        for asset in self.recipe.assets:
            dest = self.root / asset.dest
            digest = _sha256(asset.source)
            if (
                dest.is_file()
                and dest.stat().st_size == asset.source.stat().st_size
                and _sha256(dest) == digest
            ):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".partial")
            shutil.copyfile(asset.source, tmp)
            os.replace(tmp, dest)
            record["assets"][asset.dest] = {"sha256": digest}
            say(f"asset: {asset.dest}")
        if self.recipe.assets:
            _write_manifest(self.manifest_file, record)

    def build_environment(
        self, *, log: Log | None = None, extra_env: Mapping[str, str] | None = None
    ) -> None:
        """Solve and install the environment (``pixi install``). A no-op when it is already up to date."""
        env = self._require_environment()
        self._link_environment()
        self._pixi(["install"], log=log, extra_env=extra_env, what=f"{env.tool} install")

    def run_step(
        self, step: BuildStep, *, log: Log | None = None, extra_env: Mapping[str, str] | None = None
    ) -> None:
        """Run one build step's task inside the environment."""
        self._require_environment()
        variables = {key: self._expand(value) for key, value in step.env.items()}
        variables.update(extra_env or {})
        self._pixi(["run", step.task], log=log, extra_env=variables, what=f"pixi run {step.task}")

    def record_built(self) -> None:
        """Note that the environment and every step have run for the sources installed now."""
        record = _read_manifest(self.manifest_file)
        record["built"] = {
            "at": _now(),
            "sources": {
                name: (entry or {}).get("commit")
                for name, entry in record["sources"].items()
                if name in self.recipe_names
            },
        }
        record["last_build"] = None
        _write_manifest(self.manifest_file, record)

    # ---- entering it -------------------------------------------------------

    def command(self, args: Sequence[str]) -> list[str]:
        """``args`` wrapped to run inside the environment. Run it from ``workdir``."""
        manifest = self._require_environment_manifest()
        pixi = _find_pixi()
        if pixi is None:
            raise RuntimeNotReady(
                "pixi is not installed, so the runtime cannot be entered.",
                hint="Run `tandem init`, or: curl -fsSL https://pixi.sh/install.sh | bash",
            )
        return [str(pixi), "run", "--manifest-path", str(manifest), *args]

    def shell_command(self) -> list[str]:
        manifest = self._require_environment_manifest()
        pixi = _find_pixi()
        if pixi is None:
            raise RuntimeNotReady("pixi is not installed.", hint="Run `tandem init` to install it.")
        return [str(pixi), "shell", "--manifest-path", str(manifest)]

    def python(self) -> Path:
        path = self._env_python()
        if path is None:
            prefix = self.env_prefix
            raise RuntimeNotReady(
                f"No interpreter at {prefix / 'bin' / 'python' if prefix else self.root}.",
                hint=f"Run `tandem planners install {self.recipe.planner}`.",
            )
        return path

    @property
    def bin_dir(self) -> Path | None:
        """The built environment's bin directory -- its own ffmpeg, say -- or None before it is built."""
        python = self._env_python()
        return python.parent if python is not None else None

    def _env_python(self) -> Path | None:
        """The environment's interpreter, in its home -- or where a runtime built before the environment
        had a home keeps it, inside the manifest's tree. That runtime works as it stands, so it counts
        as built; the next install moves the environment out (``_adopt_environment``)."""
        env = self.recipe.environment
        if env is None:
            return None
        for prefix in (
            self.root / env.home / "envs" / env.name,
            (self.root / env.manifest).parent / ".pixi" / "envs" / env.name,
        ):
            if (prefix / "bin" / "python").is_file():
                return prefix / "bin" / "python"
        return None

    # ---- uninstall ---------------------------------------------------------

    def looks_like_a_runtime(self) -> bool:
        """Whether ``root`` holds a runtime tandem made -- or nothing at all -- and so may be deleted.

        Only tandem's own record says so. A directory holding something named like one of the
        recipe's sources does not: this layout is the monorepo's, and a workspace holding a
        ``tiptop/`` checkout is exactly the directory a mistyped setting points at. A git checkout
        anywhere a tree would be never is one, record or not -- nothing tandem installs has a .git.
        """
        root = self.root
        if not root.is_dir():
            return False
        if _is_repository(root) or any(_is_repository(root / s.name) for s in self.recipe.sources):
            return False
        # The lock an install takes before it writes anything is not something of anyone else's.
        if not any(p.name != LOCK_FILE for p in root.iterdir()):
            return True
        return self.manifest_file.is_file()

    def uninstall(self) -> None:
        root = self.root
        if not root.exists():
            return
        if not self.looks_like_a_runtime():
            # The root is a setting. Pointed somewhere else by mistake, an unconditional rmtree would
            # delete whatever is there.
            raise TandemError(
                f"{root} does not look like a {self.recipe.display_name} runtime, so it was not deleted.",
                hint="Check where the runtime is configured (`tandem runtime path`), and delete it by hand "
                "if it really is one.",
            )
        # Not from under an install that is still running.
        with self._install_lock():
            shutil.rmtree(root)

    # ---- internals ---------------------------------------------------------

    def _require_environment(self) -> PixiEnvironment:
        env = self.recipe.environment
        if env is None:
            raise TandemError(f"The {self.recipe.display_name} runtime has no environment to build or enter.")
        return env

    def _require_environment_manifest(self) -> Path:
        self._require_environment()
        manifest = self.env_manifest
        assert manifest is not None
        return manifest

    def _check_shipped_files(self) -> None:
        """Every patch and asset the recipe names has to be in the installed package. Loudly."""
        missing = [str(p) for s in self.recipe.sources for p in s.patches if not Path(p).is_file()]
        missing += [str(a.source) for a in self.recipe.assets if not Path(a.source).is_file()]
        if missing:
            raise TandemError(
                f"The {self.recipe.display_name} planner's package is missing files its runtime needs:\n  "
                + "\n  ".join(missing),
                hint="The install of tandem (or of the planner's package) is incomplete. Reinstall it.",
            )

    def _expand(self, value: str) -> str:
        record = None

        def fill(match: re.Match) -> str:
            nonlocal record
            kind, name = match.group(1), match.group(2)
            if kind == "root":
                return str(self.root)
            if kind == "source":
                return str(self.root / name)
            if record is None:
                record = _read_manifest(self.manifest_file)
            commit = str(
                (record["sources"].get(name) or {}).get("commit") or self.recipe.source(name).pin.commit
            )
            return commit if kind == "commit" else f"0.0.0+g{commit[:7]}"

        return _PLACEHOLDER.sub(fill, str(value))

    def _adopt_environment(self, say: Log) -> None:
        """Move an environment that lives inside a source tree out to its own home.

        Runtimes built before the environment had a home of its own kept it at ``<tree>/.pixi``. It is
        moved -- a rename, instant -- rather than rebuilt, and the symlink left in its place keeps every
        absolute path baked into it valid.
        """
        env = self.recipe.environment
        if env is None:
            return
        link = (self.root / env.manifest).parent / ".pixi"
        home = self.root / env.home
        if link.is_symlink() or not link.is_dir():
            return
        if home.exists() and any(home.iterdir()):
            raise TandemError(
                f"There are two {env.tool} environments for this runtime: {link} and {home}.",
                hint=f"Delete the one you do not want. {home} is the one tandem uses.",
            )
        if home.exists():
            home.rmdir()
        home.parent.mkdir(parents=True, exist_ok=True)
        os.replace(link, home)
        say(f"environment: moved out of the source tree to {home}")
        self._link_environment()

    def _link_environment(self) -> None:
        """Point ``<manifest dir>/.pixi`` at the environment's home, once the manifest's tree is there."""
        env = self.recipe.environment
        if env is None:
            return
        manifest_dir = (self.root / env.manifest).parent
        if not manifest_dir.is_dir():
            return
        home = self.root / env.home
        home.mkdir(parents=True, exist_ok=True)
        link = manifest_dir / ".pixi"
        target = os.path.relpath(home, manifest_dir)
        if link.is_symlink():
            if os.readlink(link) == target:
                return
            link.unlink()
        elif link.exists():
            self._adopt_environment(_quiet)
            return
        os.symlink(target, link, target_is_directory=True)

    def _link_persistent(self, say: Log) -> None:
        """Point each tree's ``persistent`` directories at their home under ``<runtime>/cache``.

        A tree that holds one as a real directory -- a runtime from before it had a home, or a tree
        that shipped the directory itself -- has what is in it moved out first, a rename, before the
        link takes its place. The home is created before the link: a planner that ``mkdir``s the
        directory with ``exist_ok`` still fails on a link to nothing.
        """
        for source in self.recipe.sources:
            tree = self.root / source.name
            if not tree.is_dir() or tree.is_symlink():
                continue
            for relative in source.persistent:
                link = tree / relative
                home = self.root / PERSISTENT / source.name / relative
                home.mkdir(parents=True, exist_ok=True)
                target = os.path.relpath(home, link.parent)
                if link.is_symlink():
                    if os.readlink(link) == target:
                        continue
                    link.unlink()
                elif link.is_dir():
                    for item in sorted(link.iterdir()):
                        if not (home / item.name).exists():
                            os.replace(item, home / item.name)
                    # Whatever is left the home already had, under the same name.
                    shutil.rmtree(link)
                    say(f"{source.name}: moved {relative} out of the tree, to {home}")
                elif link.exists():
                    raise TandemError(
                        f"{link} is a file, but the {self.recipe.display_name} recipe keeps a directory there.",
                        hint="This is a bug in the planner package's recipe, or the source moved on under it.",
                    )
                link.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(target, link, target_is_directory=True)

    def _pixi(
        self, args: list[str], *, log: Log | None, extra_env: Mapping[str, str] | None, what: str
    ) -> None:
        env_spec = self._require_environment()
        manifest = self._require_environment_manifest()
        pixi = _find_pixi()
        if pixi is None:
            raise TandemError("pixi is not installed.", hint="Run `tandem init` to install it.")

        env = dict(os.environ)
        for key, value in env_spec.env.items():
            env.setdefault(key, self._expand(value))
        env.update(extra_env or {})

        # --manifest-path is a per-subcommand option, not a global one: `pixi --manifest-path ...
        # install` is rejected outright. It goes after the subcommand but *before* any task name,
        # since `pixi run` treats everything from the task name onward as the task's own argv.
        cmd = [str(pixi), args[0], "--manifest-path", str(manifest), *args[1:]]
        _stream(cmd, cwd=manifest.parent, env=env, log=log, what=what)


# --------------------------------------------------------------------------- fetching


def export_pinned(pin: SourcePin, dest: Path, *, scratch: Path, log: Log | None = None) -> dict:
    """Put exactly ``pin.commit`` of ``pin.url`` at ``dest``: git if there is one, else GitHub's archive.

    Returns how it got there, for the record: ``{"origin": ..., "verified": True}``.
    """
    say = log or _quiet
    git_error = None
    if _which("git"):
        say(f"{pin.name}: fetching {pin.short()} from {pin.url}")
        try:
            _export_with_git(pin.url, pin.commit, dest, scratch=scratch, fetch=True)
            return {"origin": "git", "verified": True}
        except TandemError as exc:
            git_error = exc
            if _github_archive_url(pin.url, pin.commit) is None:
                raise
            say(
                f"{pin.name}: git could not fetch it ({exc.message.splitlines()[0]}); trying GitHub's archive"
            )
    archive = _github_archive_url(pin.url, pin.commit)
    if archive is None:
        raise TandemError(
            f"git is not installed, and {pin.url} is not on GitHub, so {pin.name} cannot be fetched without it.",
            hint="Install git, or install from a directory of sources (--sources / $TANDEM_PLANNER_SOURCES).",
        )
    say(f"{pin.name}: downloading {pin.short()} from {archive}")
    shutil.rmtree(dest, ignore_errors=True)  # whatever a failed git export left behind
    try:
        _export_from_archive(archive, pin.commit, dest, scratch=scratch)
    except TandemError as exc:
        if git_error is not None:
            raise TandemError(f"{git_error.message}\n{exc.message}", hint=exc.hint or git_error.hint) from exc
        raise
    return {"origin": "archive", "verified": True}


def export_from_directory(
    pin: SourcePin, directory: Path, dest: Path, *, scratch: Path, log: Log | None = None
) -> dict:
    """Put ``pin``'s tree at ``dest`` from ``directory/<name>``: a checkout, or an export.

    A checkout is an object store: the PINNED commit is exported out of it, whatever its working tree
    is at. An export is copied, its marker (from ``tools/bundle.py``) must name the pinned commit, and
    its files must be the ones the marker's digest was taken of. One with no marker -- or with a marker
    from before markers carried a digest -- is copied on trust and recorded as unverified.
    """
    entry = Path(directory) / pin.name
    if not entry.is_dir():
        raise TandemError(
            f"{pin.name} is not in the planner sources directory {directory}.",
            hint=f"It needs {entry}: a checkout of {pin.url} that has commit {pin.commit}, or an export of "
            "it. `python tools/bundle.py` makes a complete directory on a machine with network.",
        )
    return export_from_tree(pin, entry, dest, scratch=scratch, log=log)


def export_from_tree(
    pin: SourcePin, entry: Path, dest: Path, *, scratch: Path, log: Log | None = None
) -> dict:
    """``export_from_directory`` for one tree, wherever it is and whatever it is called."""
    say = log or _quiet
    entry = Path(entry)
    if _is_repository(entry) and _which("git"):
        say(f"{pin.name}: exporting {pin.short()} from the checkout at {entry}")
        _export_with_git(str(entry), pin.commit, dest, scratch=scratch, fetch=False)
        return {"origin": f"checkout {entry}", "verified": True}

    marker = entry / SOURCE_MARKER
    if marker.is_file():
        try:
            stated = json.loads(marker.read_text())
            if not isinstance(stated, dict):
                raise ValueError("it is not a JSON object")
        except (ValueError, OSError) as exc:
            raise TandemError(f"{marker} is unreadable: {exc}", hint="Make the bundle again.") from exc
        claimed = stated.get("commit")
        if claimed != pin.commit:
            raise TandemError(
                f"{entry} is {pin.name} at {str(claimed)[:7]}, but the recipe pins {pin.short()}.",
                hint="The bundle is for another version of tandem. Make it again with this one's "
                "`python tools/bundle.py`.",
            )
        say(f"{pin.name}: copying the export at {entry}")
        _copy_tree(entry, dest)
        # The marker is a file anyone can copy onto any tree, and a tree can be edited under it. Its
        # commit is only as good as the check that the files are still the ones it was written for.
        recorded = stated.get("sha256")
        if not recorded:
            say(
                f"{pin.name}: warning: the export at {entry} names its commit but records no digest of its "
                "files (a bundle made by an older tandem), so they are taken as that commit on trust. Make "
                "the bundle again to have them checked."
            )
            return {"origin": f"export {entry}", "verified": False}
        if tree_digest(dest) != recorded:
            raise TandemError(
                f"The export at {entry} is not the {pin.name} {pin.short()} it was bundled as: its files have "
                "changed since.",
                hint="The bundle was edited or damaged after it was made. Make it again with "
                f"`python tools/bundle.py`. To install these files on purpose, delete {marker}: they are "
                "then taken on trust, and recorded as unverified.",
            )
        return {"origin": f"export {entry}", "verified": True}

    say(
        f"{pin.name}: warning: {entry} is neither a git checkout nor a marked export, so there is no telling "
        f"which commit it is. Taking it as {pin.short()} on trust."
    )
    _copy_tree(entry, dest)
    return {"origin": f"directory {entry}", "verified": False}


def _export_with_git(repository: str, commit: str, dest: Path, *, scratch: Path, fetch: bool) -> None:
    """``git archive`` of exactly ``commit``, from a URL (fetched shallow first) or a local repository."""
    scratch.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="git-", dir=scratch))
    try:
        if fetch:
            repo = workdir / "repo.git"
            _git(["init", "--bare", "--quiet", str(repo)])
            _git(
                ["-C", str(repo), "fetch", "--depth", "1", "--no-tags", "--quiet", repository, commit],
                what=f"fetching {commit[:7]} from {repository}",
            )
        else:
            repo = Path(repository)
        # The object's own id is the verification: a fetch that handed back anything else would not
        # resolve to this commit.
        found = _git(
            ["-C", str(repo), "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"], check=False
        )
        if found.returncode != 0 or found.stdout.strip() != commit:
            raise TandemError(
                f"{repository} does not have commit {commit}.",
                hint="Fetch it there first (git fetch origin " + commit + "), or check the URL and the pin."
                if not fetch
                else "Check that the commit is pushed and reachable from a branch of that repository.",
            )
        archive = workdir / "tree.tar"
        _git(
            ["-C", str(repo), "archive", "--format=tar", "-o", str(archive), commit],
            what=f"exporting {commit[:7]}",
        )
        extract(archive, dest, commit=commit)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _export_from_archive(url: str, commit: str, dest: Path, *, scratch: Path) -> None:
    import urllib.error
    import urllib.request

    scratch.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="archive-", suffix=".tar.gz", dir=scratch)
    archive = Path(name)
    try:
        with os.fdopen(fd, "wb") as out:
            try:
                with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - https or a test's file://
                    shutil.copyfileobj(response, out, length=1 << 20)
            except (urllib.error.URLError, OSError) as exc:
                raise TandemError(
                    f"Could not download {url}: {exc}",
                    hint="Check the network, or install from a directory of sources (--sources).",
                ) from exc
        extract(archive, dest, commit=commit, strip=1)
    finally:
        archive.unlink(missing_ok=True)


def _github_archive_url(url: str, commit: str) -> str | None:
    match = re.match(r"^(?:https?://|git@|ssh://git@)github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if not match:
        return None
    owner, repo = match.groups()
    return f"https://github.com/{owner}/{repo}/archive/{commit}.tar.gz"


def extract(archive: Path, dest: Path, *, commit: str | None = None, strip: int = 0) -> None:
    """Unpack a tarball into ``dest``, refusing anything that would land outside it.

    ``commit`` is checked against the id ``git archive`` stamps into every tarball it makes (a pax
    global header), which GitHub's archives carry too. With ``strip=1`` the single top-level
    directory a GitHub archive wraps everything in is removed, and must itself name the commit when
    the header is absent. Written out rather than left to ``extractall(filter="data")``, which is
    missing from the earliest Python 3.10 releases tandem supports.

    "Outside" is decided on the disk, not on the names. A check of each entry's name alone passes a
    link ``c/d -> ..`` (it lands on the tree's own root), then a link ``e -> c/d/..`` (spelled, it is
    ``c``; followed, it is the directory ABOVE the tree), then a file ``e/x`` -- written through both,
    outside. So links are created last, after every file, directory and hard link, which means
    nothing is ever written through one; each is checked where it really points once they all exist;
    and a tree that fails any of it is deleted rather than left for a trim or a patch to follow.
    """
    import tarfile

    dest.mkdir(parents=True, exist_ok=True)
    try:
        tar = tarfile.open(archive)
    except (tarfile.TarError, OSError) as exc:
        raise TandemError(f"{archive.name} is not a readable archive: {exc}") from exc
    try:
        _extract(tar, archive, dest, commit=commit, strip=strip)
    except BaseException:
        # rmtree never follows a link, so whatever a refused link points at is not touched.
        shutil.rmtree(dest, ignore_errors=True)
        raise
    finally:
        tar.close()


def _extract(tar: Any, archive: Path, dest: Path, *, commit: str | None, strip: int) -> None:
    # The name checks compare as spelled; the disk checks compare real paths, on both sides --
    # resolving only one would turn every link in a tree under a symlinked directory (macOS's /tmp)
    # into one that looks like it escapes.
    base = os.path.normpath(os.path.abspath(dest))
    real_base = os.path.realpath(base)

    def inside(path: str | Path) -> bool:
        real = os.path.realpath(path)
        return real == real_base or real.startswith(real_base + os.sep)

    def place(member: Any, target: Path) -> None:
        """Refuse a target whose directory is not really inside the tree, or that is a link to write through."""
        if not inside(target.parent) or (not member.issym() and target.is_symlink()):
            raise TandemError(f"The archive has an entry outside its own tree: {member.name!r}")

    members = tar.getmembers()
    stamped = tar.pax_headers.get("comment")
    tops = {PurePosixPath(m.name).parts[0] for m in members if PurePosixPath(m.name).parts}
    if commit is not None:
        if stamped is not None and stamped != commit:
            raise TandemError(f"The archive is commit {stamped[:7]}, not the pinned {commit[:7]}.")
        if stamped is None and not (strip and len(tops) == 1 and next(iter(tops)).endswith(commit)):
            raise TandemError(
                f"Cannot confirm the archive is commit {commit[:7]}: it names no commit.",
                hint="Fetch it with git instead.",
            )
    if strip and len(tops) != 1:
        raise TandemError(f"{archive.name} does not have the single top-level directory it should.")

    hardlinks, symlinks = [], []
    for member in members:
        parts = PurePosixPath(member.name).parts[strip:]
        if not parts:
            continue
        if PurePosixPath(member.name).is_absolute() or ".." in parts:
            raise TandemError(f"The archive has an entry outside its own tree: {member.name!r}")
        target = Path(base).joinpath(*parts)
        if member.isdir():
            place(member, target)
            target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            place(member, target)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            assert source is not None
            with source, open(target, "wb") as out:
                shutil.copyfileobj(source, out, length=1 << 20)
            os.chmod(target, (member.mode & 0o777) | 0o600)
        elif member.issym():
            resolved = os.path.normpath(os.path.join(os.path.dirname(target), member.linkname))
            if os.path.isabs(member.linkname) or not (resolved + os.sep).startswith(base + os.sep):
                raise TandemError(f"The archive has a link out of its own tree: {member.name!r}")
            symlinks.append((member, target))
        elif member.islnk():
            hardlinks.append((member, target))
        # Devices and FIFOs have no business in a source tree; they are skipped.

    for member, target in hardlinks:
        link_parts = PurePosixPath(member.linkname).parts[strip:]
        origin = Path(base).joinpath(*link_parts)
        if ".." in link_parts or origin.is_symlink() or not origin.is_file() or not inside(origin):
            raise TandemError(f"The archive has a hard link to nothing it contains: {member.name!r}")
        place(member, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)

    # Last, so that nothing above was written through one.
    for member, target in symlinks:
        place(member, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(member.linkname, target)
        except FileExistsError:
            # Something else is already there: two entries of one name, or two whose names only a
            # case-insensitive filesystem confuses -- either way, not a tree git could have made.
            raise TandemError(
                f"The archive has two entries at {member.name!r}, one of them a link."
            ) from None
    # Checked once every link exists, because a link is only as contained as the links it goes
    # through, and those may come after it in the archive.
    for member, target in symlinks:
        if not inside(target):
            raise TandemError(f"The archive has a link out of its own tree: {member.name!r}")


# --------------------------------------------------------------------------- trimming and patching


def trim(tree: Path, relative: Sequence[str], *, name: str = "", log: Log | None = None) -> list[str]:
    """Delete these paths from ``tree``. Returns the ones that were there to delete."""
    say = log or _quiet
    dropped = []
    for rel in relative:
        target = tree / rel
        if target.is_symlink() or target.is_file():
            size = 0 if target.is_symlink() else target.stat().st_size
            target.unlink()
        elif target.is_dir():
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file() and not f.is_symlink())
            shutil.rmtree(target)
        else:
            continue
        dropped.append(rel)
        say(f"{name}: trimmed {rel} ({size / 1e6:.1f} MB)")
    return dropped


def apply_patches(
    tree: Path, patches: Sequence[Path], *, name: str = "", log: Log | None = None
) -> list[dict]:
    """Apply each patch to ``tree``, in order. Returns what was applied, with digests, for the record.

    Checked twice. A patch that does not apply is an error, not a warning: the pin moved and the patch
    no longer fits, and a runtime missing it is not the runtime the recipe describes. And a patch that
    applied to NOTHING is an error too -- see ``_check_applied``.
    """
    say = log or _quiet
    applied = []
    for patch in patches:
        patch = Path(patch)
        if _which("git"):
            _apply_with_git(tree, patch, name)
        elif _which("patch"):
            _apply_with_patch(tree, patch, name)
        else:
            raise TandemError(
                f"Neither git nor patch is installed, so {patch.name} cannot be applied to {name}.",
                hint="Install git (or patch) and run the install again.",
            )
        applied.append({"name": patch.name, "sha256": _sha256(patch)})
        say(f"{name}: applied {patch.name}")
    return applied


def _apply_with_git(tree: Path, patch: Path, name: str) -> None:
    # A tree that is not a repository must not be taken for part of one it happens to sit inside
    # (a runtime under a git checkout, say): run from inside a repository, `git apply` filters the
    # patch by the current prefix. The ceiling stops the search at the tree.
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(tree.resolve().parent)}
    result = subprocess.run(
        ["git", "apply", "--verbose", str(patch.resolve())],
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise TandemError(
            f"{patch.name} does not apply to {name}:\n{result.stderr.strip()}",
            hint="The pin moved and the patch no longer fits. Rewrite it against the new source, or drop "
            "it if upstream has taken it.",
        )
    _check_applied(result.stderr, patch.name, name)


def _check_applied(stderr: str, patch: str, name: str) -> None:
    """Exit 0 from `git apply` is NOT enough.

    Run from inside a repository, `git apply` filters a git-style diff (one carrying ``diff --git``
    headers) by the current prefix and quietly drops everything outside it -- reporting "Skipped
    patch" on stderr and success to the shell. A patch that went nowhere is exactly the failure this
    mechanism exists to make loud, and it has already shipped a tree missing a patch that was
    reported as applied.
    """
    skipped = [ln for ln in stderr.splitlines() if ln.startswith("Skipped patch")]
    if skipped:
        raise TandemError(
            f"{patch} was skipped rather than applied to {name}:\n" + "\n".join(f"  {ln}" for ln in skipped),
            hint="A plain unified diff (--- a/path, no `diff --git` header) is not filtered this way.",
        )
    if not any(ln.startswith("Applied patch") for ln in stderr.splitlines()):
        raise TandemError(f"{patch} applied to nothing in {name}.", hint="Check the paths the patch names.")


def _apply_with_patch(tree: Path, patch: Path, name: str) -> None:
    argv = ["patch", "-p1", "--forward", "--batch", "-i", str(patch.resolve())]
    for dry_run in (True, False):
        result = subprocess.run(
            argv + (["--dry-run"] if dry_run else []), cwd=tree, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise TandemError(
                f"{patch.name} does not apply to {name}:\n{(result.stdout + result.stderr).strip()}",
                hint="The pin moved and the patch no longer fits. Rewrite it against the new source.",
            )


# --------------------------------------------------------------------------- the record


def _read_manifest(path: Path) -> dict:
    """The installed record, in the current shape, whatever shape it was written in.

    A runtime built from the sources the wheel used to carry has a stamp of the form
    ``{"vendor": {name: {url, commit, version, trimmed, patches}}, "built_at": ...}``. Its trees came
    out of a ``git archive`` of exactly those commits, so they are read as verified, and a runtime
    whose commits still match the pins is current without fetching anything again.
    """
    record: dict[str, Any] = {
        "format": FORMAT,
        "planner": None,
        "sources": {},
        "assets": {},
        "built": None,
        "last_build": None,
    }
    if not path.is_file():
        return record
    try:
        raw = json.loads(path.read_text())
    except (ValueError, OSError):
        return record
    if not isinstance(raw, dict):
        return record
    if "vendor" in raw and "sources" not in raw:
        sources = {}
        for name, meta in (raw.get("vendor") or {}).items():
            if not isinstance(meta, dict) or not meta.get("commit"):
                continue
            sources[name] = {
                "url": meta.get("url"),
                "commit": meta.get("commit"),
                "origin": "vendored",
                "verified": True,
                "trimmed": list(meta.get("trimmed") or []),
                "patches": [{"name": str(p)} for p in meta.get("patches") or []],
            }
        record["sources"] = sources
        if raw.get("built_at"):
            record["built"] = {"at": raw["built_at"], "sources": {n: m["commit"] for n, m in sources.items()}}
        return record
    for key in ("planner", "sources", "assets", "built", "last_build"):
        if key in raw and raw[key] is not None:
            record[key] = raw[key]
    return record


def _write_manifest(path: Path, record: dict) -> None:
    record = {**record, "format": FORMAT}
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _patches_match(recorded: Any, wanted: Sequence[Path]) -> bool:
    """Whether the patches recorded against a tree are the ones the recipe applies now, in order.

    By content where the record has a digest, so an edited patch counts as a different patch; by name
    for a runtime from before digests were kept.
    """
    recorded = list(recorded or [])
    if len(recorded) != len(wanted):
        return False
    for entry, patch in zip(recorded, wanted, strict=True):
        entry = entry if isinstance(entry, dict) else {"name": str(entry)}
        if entry.get("name") != Path(patch).name:
            return False
        if entry.get("sha256") and Path(patch).is_file() and entry["sha256"] != _sha256(Path(patch)):
            return False
    return True


# --------------------------------------------------------------------------- helpers


def default_root(planner: str) -> Path:
    """Where a planner's runtime lives unless its factory says otherwise."""
    return paths.runtimes_dir() / planner


@dataclass
class _Held:
    """One runtime's install lock, as this process holds it."""

    fd: int | None
    owner: int
    depth: int = 0


# Runtime root -> the lock this process holds on it. A flock belongs to an open file, not to a process,
# so a second one taken here on a new descriptor would be refused by the first; re-entry goes through
# this table instead.
_LOCKS: dict[str, _Held] = {}
_LOCKS_GUARD = threading.Lock()


def _acquire(path: Path, title: str) -> int | None:
    """Take the install lock at ``path`` without waiting, and write this process's pid into it."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows, where no planner runtime is built (pixi envs are linux-64)
        return None
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            holder = os.read(fd, 64).decode(errors="replace").strip()
        finally:
            os.close(fd)
        raise _busy(title, holder) from None
    # The file stays after the lock is released -- deleting it would race the next install opening
    # it -- so the pid is only ever read while someone holds it.
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def _busy(title: str, holder: str) -> TandemError:
    return TandemError(
        f"Another install of the {title} runtime is running (pid {holder or '?'}).",
        hint="Wait for it to finish, or stop it, then run the install again: every step it finished is "
        "skipped.",
    )


def tree_digest(root: Path) -> str:
    """sha256 of a tree's contents: each file's bytes and each symlink's target, by relative path.

    What ``tools/bundle.py`` writes into an export's marker and an install from that export checks,
    so an export recorded as verified is the commit its marker names, not merely one that says so.
    Paths and contents only, not modes or times: a bundle carried on a stick keeps neither reliably,
    and neither changes which commit a tree is. The junk an install never copies (``_copy_tree``) is
    left out on both sides, the marker itself included.
    """
    root = Path(root)
    entries: list[tuple[str, str, Path]] = []
    for directory, dirnames, filenames in os.walk(root):
        here = Path(directory)
        descend = []
        for name in dirnames:
            if _is_junk(name):
                continue
            if (here / name).is_symlink():
                entries.append(((here / name).relative_to(root).as_posix(), "L", here / name))
            else:
                descend.append(name)
        dirnames[:] = descend
        for name in filenames:
            path = here / name
            if _is_junk(name) or not (path.is_symlink() or path.is_file()):
                continue
            entries.append((path.relative_to(root).as_posix(), "L" if path.is_symlink() else "F", path))

    digest = hashlib.sha256()
    for relative, kind, path in sorted(entries):
        if kind == "L":
            target = os.readlink(path).encode()
            digest.update(f"L {len(target)} {relative}\0".encode() + target)
            continue
        digest.update(f"F {path.stat().st_size} {relative}\0".encode())
        with open(path, "rb") as fh:
            for chunk in iter(lambda fh=fh: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _copy_tree(src: Path, dest: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {n for n in names if _is_junk(n)}

    shutil.copytree(src, dest, symlinks=True, ignore=ignore)


def _is_repository(path: Path) -> bool:
    return (path / ".git").exists() or ((path / "HEAD").is_file() and (path / "objects").is_dir())


def _non_empty(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _which(tool: str) -> str | None:
    """Indirection so a test can take git away without taking it from the rest of the machine."""
    return shutil.which(tool)


def _find_pixi() -> Path | None:
    from tandem.core.probe import find_pixi

    return find_pixi()


def _git(args: list[str], *, what: str = "", check: bool = True) -> subprocess.CompletedProcess:
    # Never prompt: a private or mistyped repository must fail, not wait for a password nobody is
    # there to type.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    result = subprocess.run(["git", *args], capture_output=True, text=True, env=env, check=False)
    if check and result.returncode != 0:
        raise TandemError(
            f"git failed{' ' + what if what else ''}:\n{result.stderr.strip()}",
            hint="Check the network and the URL, or install from a directory of sources (--sources).",
        )
    return result


def _stream(cmd: list[str], *, cwd: Path, env: dict, log: Log | None, what: str) -> None:
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
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
