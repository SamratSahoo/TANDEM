"""Which planner backends tandem knows about, by name.

Resolution is loud. A profile naming a backend that does not exist is a typo, and the failure mode
that must never happen is quietly falling back to a default -- a session that plans with a different
planner from the one the config asked for produces a dataset nobody can interpret afterwards. Same
discipline as the ``tamp`` keys: unknown name, clear error, nearest suggestion.

The registry maps a name to a *factory* (``tandem.planners.base.BackendFactory``) rather than an
instance, because building a backend opens cameras and a robot connection, and importing one may
need the heavy environment. Listing the available names must cost nothing.

A name comes from one of three places, and the first that has it wins:

1. ``register_backend(name, factory)``, called at runtime -- by a test, or by code embedding tandem;
2. the planners that ship inside tandem (``_BUILTIN``);
3. the ``tandem.planners`` entry-point group, which is how a planner in its own package plugs in::

       [project.entry-points."tandem.planners"]
       myplanner = "my_package.planner:MyPlanner"     # a tandem.planners.Planner subclass, or a factory

   after which ``planner: {backend: myplanner}`` in a profile is all it takes. Nothing in tandem
   changes, and nothing in tandem has to know the package exists.

Entry points are read by NAME when listing and imported only when one is actually used. A plugin
that fails to import is therefore not tandem's failure to start: it is reported against its own
name, in ``catalog()`` and in the error raised when something asks for it, and every other planner
keeps working.
"""

from __future__ import annotations

import difflib
import importlib
import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tandem.core import names
from tandem.core.errors import TandemError
from tandem.planners.base import (
    BackendContext,
    BackendFactory,
    BackendRuntime,
    Capabilities,
    PlannerInfo,
    TampBackend,
)

#: The entry-point group a package registers a planner under.
GROUP = "tandem.planners"

# The planners that ship inside tandem, by import path, so listing them imports nothing.
_BUILTIN: dict[str, str] = {
    "tiptop": "tandem.planners.tiptop:FACTORY",
}

# Registered at runtime: a factory, a factory class, or its "module:attribute" import path.
_registered: dict[str, Any] = {}

# Factories already loaded, by name, with what each was loaded FROM. A second lookup then returns the
# same object rather than a fresh instance of a registered class -- a factory is allowed to keep state,
# and must not lose it between calls -- and a name registered anew is never answered from the cache.
_loaded: dict[str, tuple[Any, BackendFactory]] = {}

# A planner's name is typed into YAML and onto a command line, and becomes part of paths. The same rule
# as a human executor's (tandem.core.names), so the two catalogs accept the same spellings.
_NAME = names.NAME

#: What an unknown name's hint points at: the listing that shows every planner, broken ones included.
LIST_COMMAND = "tandem planners list"

# What the session calls on a backend. A factory that returns something missing one of these fails at
# create() with the list, rather than at the first human phase with an AttributeError.
_BACKEND_MEMBERS = (
    "name",
    "capabilities",
    "require_ready",
    "warm",
    "close",
    "release_hardware",
    "reacquire_hardware",
    "capture_frame",
    "home",
    "perceive",
    "plan",
    "execute",
)
_FACTORY_MEMBERS = ("info", "capabilities", "create", "runtime")


# --------------------------------------------------------------------------- registering


def register_backend(name: str, factory: BackendFactory | type | str, *, replace: bool = False) -> None:
    """Make a planner available under ``name``.

    ``factory`` is a ``BackendFactory``, a class whose no-argument instance is one, a
    ``tandem.planners.Planner`` subclass (which is its own factory, and is used as the class, never
    instantiated here), or the ``"module:attribute"`` path of any of those -- the last is resolved
    only when the planner is used, so registering costs no import. Registering a name that is already
    taken, by anything, is an error unless ``replace=True``: two planners answering to one name is
    exactly the ambiguity this module exists to refuse.
    """
    _check_name(name)
    if isinstance(factory, str):
        _check_import_path(factory, what=f"planner {name!r}")
    existing = _registered.get(name)
    if existing is factory or (isinstance(factory, str) and existing == factory):
        return
    if not replace and name in available():
        raise TandemError(
            f"A planner named {name!r} is already registered.",
            hint="Choose another name, or pass replace=True to swap it deliberately.",
        )
    _registered[name] = factory
    _forget(name)


def unregister_backend(name: str) -> None:
    """Undo a ``register_backend``. A built-in or entry-point planner of that name comes back."""
    _registered.pop(name, None)
    _forget(name)


# --------------------------------------------------------------------------- looking up


def available() -> list[str]:
    """Every backend name, for an error message, a ``--help`` line or a profile check.

    Imports nothing: entry points are listed by name. A name here is one a profile may use, not a
    promise it will load -- ``catalog()`` answers that.
    """
    known = set(_registered) | set(_BUILTIN)
    known.update(ep.name for ep in _entry_points() if _NAME.match(ep.name))
    return sorted(known)


def factory(name: str) -> BackendFactory:
    """The factory registered as ``name``, loaded on demand."""
    key = (name or "").strip()
    origin, source = _source(key)
    return _materialise(key, origin, source)


def info(name: str) -> PlannerInfo:
    """What a catalog says about the planner, with nothing of it built or installed."""
    return factory(name).info


def origin(name: str) -> str:
    """Where the planner ``name`` comes from: "registered", "built-in" or "entry point (<dist> <version>)".

    Imports nothing. "unknown" for a name nothing provides.
    """
    return _origin_of((name or "").strip())


def capabilities(name: str) -> Capabilities:
    """What a backend can be asked for, without building it.

    This is what makes ``tandem plan`` work on a laptop: a decomposition can be proposed and checked
    against the real planner's goal language with no planner, no runtime and no robot in sight.
    """
    return factory(name).capabilities()


def runtime(name: str, settings: Any = None) -> BackendRuntime | None:
    """The planner's runtime on this machine, or None for a planner that is pure Python."""
    return factory(name).runtime(settings)


def create(name: str, ctx: BackendContext) -> TampBackend:
    """Build the backend a session drives. The one way a session gets a planner."""
    built = factory(name).create(ctx)
    missing = [member for member in _BACKEND_MEMBERS if not hasattr(built, member)]
    if missing:
        raise TandemError(
            f"The {name!r} planner's factory built a {type(built).__name__}, which is not a planner "
            f"backend: it has no {', '.join(missing)}.",
            hint="A backend implements tandem.planners.base.TampBackend in full.",
        )
    return built


# --------------------------------------------------------------------------- the catalog


@dataclass(frozen=True)
class CatalogEntry:
    """One planner as a listing shows it: what it is, or why it cannot be loaded."""

    name: str
    # "built-in", "registered", or "entry point (<distribution> <version>)".
    origin: str
    info: PlannerInfo | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "origin": self.origin,
            "ok": self.ok,
            "info": self.info.to_dict() if self.info is not None else None,
            "error": self.error,
        }


def catalog() -> list[CatalogEntry]:
    """Every planner tandem can see, loaded, with what is wrong with each one that will not load.

    Never raises for a single planner. This is what somebody runs to find out why their plugin is
    not being picked up, and an error that took the listing down with it would hide exactly the
    answer they came for. A plugin shadowed by a planner of the same name is listed too, with the
    reason it is not the one in use.
    """
    entries: list[CatalogEntry] = []
    try:
        declared = _entry_points(strict=True)
    except Exception as exc:
        declared = []
        entries.append(
            CatalogEntry(GROUP, "entry points", error=f"the installed packages could not be read: {exc}")
        )

    by_name: dict[str, list[importlib.metadata.EntryPoint]] = {}
    for ep in declared:
        by_name.setdefault(ep.name, []).append(ep)

    for name in sorted(set(_registered) | set(_BUILTIN) | set(by_name)):
        if not _NAME.match(name):
            for ep in by_name[name]:
                entries.append(
                    CatalogEntry(
                        name,
                        _ep_origin(ep),
                        error=f"{name!r} is not a usable planner name ({names.RULE})",
                    )
                )
            continue
        try:
            origin, source = _source(name)
            entries.append(CatalogEntry(name, origin, info=_materialise(name, origin, source).info))
        except TandemError as exc:
            entries.append(CatalogEntry(name, _origin_of(name), error=_one_line(exc)))
        except Exception as exc:  # a factory whose `info` raises is still only its own problem
            entries.append(CatalogEntry(name, _origin_of(name), error=f"{type(exc).__name__}: {exc}"))
        # A plugin that loses to a registered or built-in planner of the same name. Listed rather than
        # dropped, so "why is my plugin not used" has an answer; tandem's own entry point for a
        # built-in, which points at the very same factory, is not a conflict and is not listed.
        if name in _registered or name in _BUILTIN:
            winner = "registered" if name in _registered else "built-in"
            for ep in by_name.get(name, ()):
                if name in _BUILTIN and name not in _registered and ep.value == _BUILTIN[name]:
                    continue
                entries.append(
                    CatalogEntry(
                        name,
                        _ep_origin(ep),
                        error=f"not used: the {winner} planner {name!r} takes precedence over this entry point",
                    )
                )
    return entries


# --------------------------------------------------------------------------- internals


def _source(name: str) -> tuple[str, Any]:
    """Where ``name`` comes from and what to load for it, or a loud error naming the nearest match."""
    if name in _registered:
        return "registered", _registered[name]
    if name in _BUILTIN:
        return "built-in", _BUILTIN[name]
    declared = [ep for ep in _entry_points() if ep.name == name]
    # One distribution can be found twice on a path; the same target twice is one planner.
    distinct = {ep.value: ep for ep in declared}
    if len(distinct) == 1:
        (ep,) = distinct.values()
        return _ep_origin(ep), ep
    if len(distinct) > 1:
        raise TandemError(
            f"More than one installed package registers a planner named {name!r}: "
            + ", ".join(f"{_ep_origin(ep)} -> {ep.value}" for ep in distinct.values())
            + ".",
            hint="Uninstall all but one of them. tandem will not guess which one a profile means.",
        )
    known = available()
    suggestion = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
    hint = f"Did you mean {suggestion[0]!r}?" if suggestion else f"Known backends: {', '.join(known)}."
    # The listing, too: it is where a planner that is installed but will not load shows up with its
    # reason, which is the likeliest explanation for a name that "should" be here and is not.
    hint += f" `{LIST_COMMAND}` shows every planner, and why one will not load."
    raise TandemError(f"Unknown planner backend {name!r}.", hint=hint)


def _materialise(name: str, origin: str, source: Any) -> BackendFactory:
    """Load, instantiate and check a factory, or say exactly why it could not be."""
    if isinstance(source, importlib.metadata.EntryPoint):
        identity: Any = f"entry point {source.value}"
        loader: Callable[[], Any] = source.load
        target = source.value
    elif isinstance(source, str):
        identity = source
        loader = lambda: _import(source)  # noqa: E731
        target = source
    else:
        identity = source
        loader = lambda: source  # noqa: E731
        target = repr(source)

    cached = _loaded.get(name)
    if cached is not None and (cached[0] is identity or (isinstance(identity, str) and cached[0] == identity)):
        return cached[1]

    try:
        loaded = loader()
    except Exception as exc:
        raise TandemError(
            f"The planner backend {name!r} could not be loaded: {type(exc).__name__}: {exc}",
            hint=_load_hint(origin, target),
        ) from exc

    if isinstance(loaded, type) and _is_planner_class(loaded):
        # A Planner subclass is its own factory: info, capabilities(), create() and runtime() are all
        # class-level. Instantiating it here would build a BACKEND -- once, with no context -- and hand
        # that out as the factory of every session.
        if loaded._planner_base:
            unimplemented = sorted(getattr(loaded, "__abstractmethods__", ()))
            why = f"it leaves {', '.join(unimplemented)} unimplemented" if unimplemented else "it is declared abstract"
            raise TandemError(
                f"The planner backend {name!r} is registered as {target}, which is a base for planners, "
                f"not a planner: {why}.",
                hint="Register the concrete Planner subclass that implements perceive, plan and execute.",
            )
    elif isinstance(loaded, type):
        try:
            loaded = loaded()
        except Exception as exc:
            raise TandemError(
                f"The planner backend {name!r} could not be loaded: {target} is a class, and building "
                f"it with no arguments failed: {type(exc).__name__}: {exc}",
                hint="Point the registration at a factory instance instead of its class.",
            ) from exc

    missing = [member for member in _FACTORY_MEMBERS if not hasattr(loaded, member)]
    if missing:
        raise TandemError(
            f"The planner backend {name!r} is registered as {target}, which is not a planner factory: "
            f"it has no {', '.join(missing)}.",
            hint="A factory implements tandem.planners.base.BackendFactory: info, capabilities(), "
            "create(ctx) and runtime(settings) -- which returns None when there is nothing to install.",
        )
    declared = getattr(loaded.info, "name", None)
    if declared != name:
        raise TandemError(
            f"The planner registered as {name!r} describes itself as {declared!r}.",
            hint="A planner's info.name must be the name it is registered under, or a catalog and a "
            "profile would call one planner by two names.",
        )

    _loaded[name] = (identity, loaded)
    return loaded


def _is_planner_class(candidate: type) -> bool:
    # Imported here, not at the top: the SDK imports this module, and listing planners must not pay
    # for the SDK when no planner written with it is registered.
    from tandem.planners.sdk import Planner

    return issubclass(candidate, Planner)


def _entry_points(*, strict: bool = False) -> list[importlib.metadata.EntryPoint]:
    """The ``tandem.planners`` entry points of every installed package, read by name only.

    A metadata failure here is not allowed to break a lookup of a built-in planner, so outside
    ``catalog()`` -- which reports it -- it reads as "no plugins".
    """
    try:
        return list(importlib.metadata.entry_points(group=GROUP))
    except Exception:
        if strict:
            raise
        return []


def _import(path: str) -> Any:
    module_name, _, attribute = path.partition(":")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ImportError(f"module {module_name!r} has no attribute {attribute!r}") from exc


def _forget(name: str) -> None:
    _loaded.pop(name, None)


def _check_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME.match(name):
        raise TandemError(
            f"{name!r} is not a usable planner name.",
            hint=f"Use {names.RULE}: it is typed into profiles and onto command lines.",
        )


def _check_import_path(path: str, *, what: str) -> None:
    module_name, colon, attribute = path.partition(":")
    if not (module_name and colon and attribute):
        raise TandemError(
            f"The import path for {what} must be 'module:attribute', not {path!r}.",
            hint="For example 'my_package.planner:FACTORY'.",
        )


def _ep_origin(ep: importlib.metadata.EntryPoint) -> str:
    dist = getattr(ep, "dist", None)
    if dist is None:
        return "entry point"
    try:
        label = f"{dist.metadata['Name']} {dist.version}"
    except Exception:
        label = str(getattr(dist, "name", "") or "an installed package")
    return f"entry point ({label})"


def _origin_of(name: str) -> str:
    if name in _registered:
        return "registered"
    if name in _BUILTIN:
        return "built-in"
    for ep in _entry_points():
        if ep.name == name:
            return _ep_origin(ep)
    return "unknown"


def _load_hint(origin: str, target: str) -> str:
    if origin.startswith("entry point"):
        return (
            f"It is registered by an installed package's {GROUP!r} entry point ({origin}, {target}). "
            "Reinstall that package, or uninstall it if you no longer use this planner."
        )
    if origin == "built-in":
        return "This planner ships with tandem, so the install looks broken. Reinstall tandem-tamp."
    return f"It was registered as {target}; check that it imports."


def _one_line(exc: TandemError) -> str:
    return f"{exc.message} {exc.hint}".strip() if exc.hint else exc.message
