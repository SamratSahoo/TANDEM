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
naming its commit, which has to be the pinned one. A bare directory with neither is taken on trust
and recorded as unverified. Nothing is ever fetched while a sources directory is in force: a source
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
        .tandem-runtime.json      what is installed, and from where

The environment lives outside the trees on purpose. A tree is replaced wholesale when its pin moves,
and an environment inside it would be deleted with it and solved again from nothing -- a full torch
and CUDA download -- on every bump. pixi keeps a manifest's environment in ``<manifest dir>/.pixi``,
so that one path is a symlink: the tree is swapped underneath it and the environment is found again,
at the same absolute path it was built at.

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
from collections.abc import Callable, Mapping, Sequence
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
FORMAT = 2

Log = Callable[[str], None]

_COMMIT = re.compile(r"[0-9a-f]{40}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_PLACEHOLDER = re.compile(r"\{(root|source|version|commit)(?::([A-Za-z0-9_.-]+))?\}")
# Build junk a working tree accumulates and an install must never copy: VCS state, caches, and the
# artifacts of a previous build (a compiled extension from someone else's machine is worse than none).
_JUNK_NAMES = {".git", ".pixi", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", SOURCE_MARKER}
_JUNK_SUFFIXES = (".egg-info", ".pyc", ".so")


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
    reserved = {MANIFEST_FILE, STAGING}
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
        for relative in (*source.trim, *([source.marker] if source.marker else [])):
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

        built = record.get("built") or None
        if (recipe.environment is not None or recipe.steps) and not problems:
            # Everything is on disk -- but was it built for THESE sources? The environment outlives
            # a bump by design, so after one it is there, whole, and solved for the old commits.
            have = {s.name: s.commit for s in states}
            built_for = (built or {}).get("sources") or {}
            changed = sorted(n for n, c in have.items() if built_for.get(n) != c)
            if not built or changed:
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
            built_at=(built or {}).get("at"),
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
        """The installed record (``.tandem-runtime.json``), normalised: ``{planner, sources, assets, built}``.

        ``sources`` maps each installed tree to ``{url, commit, origin, verified, trimmed, patches}``.
        Empty, never an error, for a runtime that has none yet.
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

        announce("sources", stages["sources"])
        self.fetch(sources_dir=sources_dir, force=force, log=say)
        self.place_assets(log=say)

        if self.recipe.environment is not None:
            announce("environment", stages["environment"])
            self.build_environment(log=say)
        if env_only:
            return
        for step in self.recipe.steps:
            announce(step.name, stages[step.name])
            self.run_step(step, log=say)
        self.record_built()

        st = self.inspect()
        if not st.ready:
            raise TandemError(
                "The build finished but the runtime still looks incomplete: " + "; ".join(st.problems),
                hint="Run the install again; every finished step is skipped.",
            )

    def fetch(
        self, *, sources_dir: Path | None = None, force: bool = False, log: Log | None = None
    ) -> list[str]:
        """Put every source tree in place at its pin. Returns the names that were (re)installed.

        A tree already recorded at its pin, with the recipe's patches, is left alone unless ``force``.
        Each tree is staged beside the runtime and renamed into place only once it is exported,
        trimmed and patched, so an interrupted fetch never leaves half a tree where a whole one was.
        """
        say = log or _quiet
        self._check_shipped_files()
        override = (
            Path(sources_dir).expanduser() if sources_dir is not None else paths.planner_sources_override()
        )
        if override is not None:
            if not override.is_dir():
                raise TandemError(
                    f"The planner sources directory {override} does not exist.",
                    hint="Point --sources or $TANDEM_PLANNER_SOURCES at a directory holding one checkout "
                    "or export per source, or unset it to fetch them.",
                )
            say(f"sources: installing from {override}, not fetching")

        self.root.mkdir(parents=True, exist_ok=True)
        staging = self.root / STAGING
        record = _read_manifest(self.manifest_file)
        # Before any tree is replaced: a runtime built when the environment lived INSIDE the tree
        # has it moved out, or the swap would delete it.
        self._adopt_environment(say)

        changed: list[str] = []
        try:
            for source in self.recipe.sources:
                entry = record["sources"].get(source.name) or {}
                tree = self.root / source.name
                present = (tree / source.marker).exists() if source.marker else _non_empty(tree)
                if (
                    not force
                    and present
                    and entry.get("commit") == source.pin.commit
                    and _patches_match(entry.get("patches"), source.patches)
                ):
                    say(f"{source.name}: already at {source.pin.short()}")
                    continue
                if entry.get("commit") and entry.get("commit") != source.pin.commit:
                    say(f"{source.name}: replacing {str(entry['commit'])[:7]} with {source.pin.short()}")
                record["sources"][source.name] = self._install_source(source, override, staging, say)
                # The environment and anything built in it were built for the old tree.
                record["built"] = None
                record["planner"] = self.recipe.planner
                # Written after every tree, so an install interrupted halfway resumes with what it has.
                _write_manifest(self.manifest_file, record)
                changed.append(source.name)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        self._link_environment()
        return changed

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
        """Whether ``root`` holds this recipe's runtime -- or nothing at all -- and so may be deleted."""
        if not self.root.is_dir():
            return False
        if not any(self.root.iterdir()):
            return True
        markers = [MANIFEST_FILE, *(s.name for s in self.recipe.sources)]
        if self.recipe.environment is not None:
            markers.append(self.recipe.environment.home)
        return any((self.root / marker).exists() for marker in markers)

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
    is at. An export is copied, and its marker (from ``tools/bundle.py``) must name the pinned commit.
    One with no marker is copied on trust and recorded as unverified.
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
            claimed = json.loads(marker.read_text()).get("commit")
        except (ValueError, OSError) as exc:
            raise TandemError(f"{marker} is unreadable: {exc}", hint="Make the bundle again.") from exc
        if claimed != pin.commit:
            raise TandemError(
                f"{entry} is {pin.name} at {str(claimed)[:7]}, but the recipe pins {pin.short()}.",
                hint="The bundle is for another version of tandem. Make it again with this one's "
                "`python tools/bundle.py`.",
            )
        say(f"{pin.name}: copying the export at {entry}")
        _copy_tree(entry, dest)
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
    """
    import tarfile

    dest.mkdir(parents=True, exist_ok=True)
    # Compared as spelled, not resolved: resolving one side and not the other turns every link in a
    # tree under a symlinked directory (macOS's /tmp) into one that looks like it escapes.
    base = os.path.normpath(os.path.abspath(dest))
    try:
        tar = tarfile.open(archive)
    except (tarfile.TarError, OSError) as exc:
        raise TandemError(f"{archive.name} is not a readable archive: {exc}") from exc
    with tar:
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

        hardlinks = []
        for member in members:
            parts = PurePosixPath(member.name).parts[strip:]
            if not parts:
                continue
            if PurePosixPath(member.name).is_absolute() or ".." in parts:
                raise TandemError(f"The archive has an entry outside its own tree: {member.name!r}")
            target = Path(base).joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
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
                target.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(member.linkname, target)
            elif member.islnk():
                hardlinks.append((member, target))
            # Devices and FIFOs have no business in a source tree; they are skipped.
        for member, target in hardlinks:
            link_parts = PurePosixPath(member.linkname).parts[strip:]
            origin = Path(base).joinpath(*link_parts)
            if ".." in link_parts or not origin.is_file():
                raise TandemError(f"The archive has a hard link to nothing it contains: {member.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)


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
    record: dict[str, Any] = {"format": FORMAT, "planner": None, "sources": {}, "assets": {}, "built": None}
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
    for key in ("planner", "sources", "assets", "built"):
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


def _copy_tree(src: Path, dest: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        out = {n for n in names if n in _JUNK_NAMES or n.endswith(_JUNK_SUFFIXES)}
        out.discard("__init__.py")
        return out

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
    # Decoded leniently: a compiler or a conda post-link script that prints one byte of Latin-1 raised
    # UnicodeDecodeError here, half-way through `tandem planners install`, with the build still running.
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
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
