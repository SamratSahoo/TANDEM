"""Presets: named settings a new profile starts from (`tandem profile create NAME --preset paper`).

A preset is part of a profile, written the way a profile writes it. Its settings are laid over the
profile being created, and whatever it leaves out stays as the base had it: the template, a cloned
profile (``--from``) or an imported rig (``--import-from``). So a preset can say "collect the way the
paper did" without also saying which robot and which cameras, which belong to the rig and not to the
experiment.

Presets ship in two places, split the same way a profile is:

  * **tandem's own**, in ``tandem/resources/presets/<name>.yml``. These hold settings that mean the
    same whichever planner runs: phase planning (``hitl``), recording, the task. They may not state
    ``planner``, because a planner's options mean nothing to any other planner.
  * **a planner's own**, in the directory its factory names as ``presets_dir`` (TiPToP:
    ``planners/tiptop/presets/``). These hold ``planner.options``, and only that planner's. A
    planner's preset may say ``extends: <name>`` to build on one of tandem's, which is applied first.
    TiPToP's ``paper`` does this: the phase-planning half of the paper's settings is tandem's
    ``paper``, and TiPToP adds its own half, the TAMP and DATAFARM overrides.

A name is looked up in the profile's planner first and then in tandem. So ``--preset paper`` gives a
TiPToP profile both halves, and a profile using another planner tandem's half only -- which
`tandem profile create` says, rather than letting anything stand in for the half that planner does
not ship. A planner's preset with the name of one of tandem's must extend it (``available``), so that
tandem's half is never dropped for one planner without a word.

The file::

    title: The paper's collection settings        # required, one line
    summary: ...                                  # optional, one line for a listing
    caution: [...]                                # optional: what a person must know once it is applied
    extends: paper                                # optional: one of tandem's presets, applied first
    replace: [planner.options.tamp]               # optional: blocks substituted whole, not merged
    profile:                                      # the settings, as a profile spells them
      hitl: {...}

``replace`` exists for a block that is a complete specification on its own. TiPToP's ``tamp`` is
one. Its keys are overrides whose absence means "tiptop's own default", so merging the paper's into a
cloned profile's would keep any extra override that profile had. The result would still be called
the paper's settings, but it would plan differently.

Standard library, ruamel and tandem's light modules only: `tandem profile create` runs on a laptop.
"""

from __future__ import annotations

import copy
import difflib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tandem.core import names
from tandem.core.errors import TandemError

#: The origin of a preset tandem itself ships, where a planner's says the planner's name.
TANDEM_ORIGIN = "tandem"

#: The keys a preset file may have.
FILE_KEYS = ("title", "summary", "caution", "extends", "replace", "profile")

# What a preset may not state. The name, layout version and description belong to the profile being
# created. The planner is chosen with --planner or `tandem planners use`: a preset that switched it
# would carry options the chosen planner would refuse.
_NOT_IN_A_PRESET = {
    "name": "a profile's name is the one it is created with",
    "version": "the layout version is tandem's to write",
    "description": "the description says where the profile came from, which tandem fills in",
}

#: What `tandem profile presets` points at in an error.
LIST_COMMAND = "tandem profile presets"


@dataclass(frozen=True)
class Preset:
    """One preset file, read and checked. ``settings`` is only its own layer, without what it extends."""

    name: str
    title: str
    summary: str
    #: "tandem", or the name of the planner that ships it.
    origin: str
    path: Path
    settings: Mapping[str, Any]
    replace: tuple[str, ...] = ()
    extends: str | None = None
    #: Lines shown as warnings when the preset is applied: what it changes that a person has to know
    #: before the arm moves (TiPToP's paper preset runs planned motions at full speed).
    caution: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "summary": self.summary,
            "origin": self.origin,
            "path": str(self.path),
            "extends": self.extends,
            "replace": list(self.replace),
            "caution": list(self.caution),
            "settings": _plain(self.settings),
        }


# --------------------------------------------------------------------------- reading


def load(path: Path, *, origin: str) -> Preset:
    """Read and check one preset file. A problem is a TandemError naming the file.

    Checks the file's structure only: which keys it has, and that it states nothing a preset may not.
    Whether its values are valid is found out when it is applied to a profile. The profile's own
    validators, and the planner's for its options, decide that, as they do for any profile.
    """
    from ruamel.yaml import YAML

    path = Path(path)
    name = path.stem
    where = f"The {name!r} preset ({path})"
    if not names.is_valid(name):
        raise TandemError(f"{where} has a name that is not {names.RULE}.", hint="Rename the file.")
    try:
        with path.open() as fh:
            raw = YAML(typ="safe").load(fh)
    except Exception as exc:
        raise TandemError(f"{where} is not valid YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise TandemError(f"{where} must be a mapping with a `title:` and a `profile:`.")
    problems: list[str] = []
    for key in raw:
        if key not in FILE_KEYS:
            problems.append(f"unknown key {key!r}{_did_you_mean(str(key), FILE_KEYS)}")
    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        problems.append("`title:` must be one line saying what the preset is")
    summary = raw.get("summary") or ""
    if not isinstance(summary, str):
        problems.append("`summary:` must be one line of text")
    caution = raw.get("caution") or []
    if not isinstance(caution, list) or not all(isinstance(line, str) and line.strip() for line in caution):
        problems.append("`caution:` must be a list of lines of text")
        caution = []
    extends = raw.get("extends")
    if extends is not None:
        if origin == TANDEM_ORIGIN:
            problems.append("`extends:` is for a planner's preset building on one of tandem's")
        elif not names.is_valid(extends):
            problems.append(f"`extends:` must name one of tandem's presets, not {extends!r}")
    settings = raw.get("profile")
    if not isinstance(settings, Mapping) or not settings:
        problems.append("`profile:` must state at least one setting, spelled as a profile spells it")
        settings = {}
    sections = _profile_sections()
    for key in settings:
        if key in _NOT_IN_A_PRESET:
            problems.append(f"it states {key!r}, which a preset may not: {_NOT_IN_A_PRESET[key]}")
        elif key not in sections:
            problems.append(f"profile.{key} is not a profile setting{_did_you_mean(str(key), sections)}")
    if "planner" in settings:
        problems.extend(_planner_problems(settings["planner"], origin))
    replace = raw.get("replace") or []
    if not isinstance(replace, list) or not all(isinstance(p, str) for p in replace):
        problems.append("`replace:` must be a list of dotted paths, such as planner.options.tamp")
        replace = []
    for dotted in replace:
        if not isinstance(_lookup(settings, dotted), Mapping):
            problems.append(
                f"`replace:` names {dotted!r}, which is not a block this preset states; it can only "
                "substitute a block it sets"
            )
    if problems:
        raise TandemError(
            f"{where} is not a usable preset:\n" + "\n".join(f"  - {p}" for p in problems),
            hint="The layout is in tandem/core/presets.py.",
        )
    return Preset(
        name=name,
        title=title.strip(),
        summary=summary.strip(),
        caution=tuple(line.strip() for line in caution),
        origin=origin,
        path=path,
        settings=_plain(settings),
        replace=tuple(replace),
        extends=extends,
    )


def _planner_problems(planner: Any, origin: str) -> list[str]:
    if origin == TANDEM_ORIGIN:
        return [
            "it states `planner`, which only a planner's own preset may: options mean nothing to "
            "another planner. Ship them in that planner's presets_dir"
        ]
    if not isinstance(planner, Mapping):
        return ["`planner:` must be a mapping holding `options:`"]
    problems = []
    for key in planner:
        if key == "backend":
            problems.append(
                "it states planner.backend. A planner's preset applies to its own planner, which is "
                "chosen with --planner"
            )
        elif key != "options":
            problems.append(f"planner.{key} is not a setting; a preset states planner.options")
    return problems


def tandem_dir() -> Path:
    """Where tandem's own presets are: ``tandem/resources/presets``."""
    from tandem import resources

    return resources.path("presets")


def tandem_presets() -> dict[str, Preset]:
    """tandem's own presets, by name."""
    return read_dir(tandem_dir(), origin=TANDEM_ORIGIN)


def planner_presets(planner: str) -> dict[str, Preset]:
    """The presets the planner ``planner`` ships, by name. Raises the registry's error for one that won't load."""
    from tandem.planners import registry

    directory = registry.presets_dir(planner)
    return {} if directory is None else read_dir(Path(directory), origin=planner)


def read_dir(directory: Path, *, origin: str) -> dict[str, Preset]:
    """Every ``<name>.yml`` preset in ``directory``, read and checked, by name. None when it is not there."""
    if not directory.is_dir():
        return {}
    return {path.stem: load(path, origin=origin) for path in sorted(directory.glob("*.yml"))}


def available(planner: str) -> dict[str, Preset]:
    """Every preset a profile planning with ``planner`` may use, by name.

    Where the planner and tandem ship the same name, the planner's is the one listed. It is the one
    ``--preset NAME`` applies, and it must extend tandem's of that name, which is applied first.
    Otherwise tandem's half would be dropped without a word for this planner only, and "the paper's
    settings" would mean different phase planning depending on which planner ran it.
    """
    ours = tandem_presets()
    theirs = planner_presets(planner)
    for name, preset in theirs.items():
        if name in ours and preset.extends != name:
            raise TandemError(
                f"The {planner!r} planner's {name!r} preset ({preset.path}) has the name of one of tandem's "
                f"({ours[name].path}) without extending it, so tandem's settings in it would not apply.",
                hint=f"Add `extends: {name}` to the planner's preset, or give it a name of its own.",
            )
    return {**ours, **theirs}


def get(name: str, planner: str) -> Preset:
    """The preset ``name`` for a profile planning with ``planner``, or a TandemError naming the known ones."""
    known = available(planner)
    if name in known:
        return known[name]
    listing = ", ".join(sorted(known)) or "none"
    raise TandemError(
        f"There is no preset named {name!r} for a profile that plans with {planner!r}"
        f"{_did_you_mean(name, known)}.",
        hint=f"Presets for it: {listing}. `{LIST_COMMAND} --planner {planner}` says what each one sets.",
    )


def layers(name: str, planner: str) -> list[Preset]:
    """What applying ``name`` lays down, in order: what it extends first, then the preset itself."""
    preset = get(name, planner)
    if preset.extends is None:
        return [preset]
    base = tandem_presets().get(preset.extends)
    if base is None:
        raise TandemError(
            f"The {preset.name!r} preset ({preset.path}) extends {preset.extends!r}, which is not one of "
            f"tandem's presets{_did_you_mean(preset.extends, tandem_presets())}.",
            hint=f"tandem's presets are the files in {tandem_dir()}.",
        )
    return [base, preset]


# --------------------------------------------------------------------------- applying


def apply(profile: Any, name: str) -> Any:
    """``profile`` with the preset ``name`` laid over it, validated. The preset is looked up for its planner.

    Returns a new ``Profile``; the one passed in is not changed. The result is validated as any
    profile is -- tandem's sections by tandem, ``planner.options`` by the planner -- and a refusal is
    a TandemError naming the preset's files, since a preset is meant to be valid on any profile of
    its planner.
    """
    from tandem.core.profiles import Profile, format_errors

    planner = profile.planner.backend
    stack = layers(name, planner)
    data = profile.model_dump(mode="python")
    for preset in stack:
        data = overlay(data, preset.settings, replace=preset.replace)
    try:
        return Profile.model_validate(data)
    except Exception as exc:
        top = stack[-1]
        raise TandemError(
            f"The {top.name!r} preset does not make a valid {planner} profile:\n{format_errors(exc)}",
            hint=f"The preset is {' laid over '.join(str(p.path) for p in reversed(stack))}.",
        ) from exc


def overlay(base: Mapping[str, Any], over: Mapping[str, Any], *, replace: tuple[str, ...] = ()) -> dict:
    """``over`` merged into a copy of ``base``: mappings merged key by key, anything else substituted.

    A dotted path in ``replace`` names a mapping in ``over`` that is substituted whole instead of merged.
    Lists are substituted, never concatenated: ``blend_ops`` stated twice means the second list.
    """
    return _merge(dict(copy.deepcopy(base)), over, frozenset(replace), ())


def _merge(base: dict, over: Mapping[str, Any], replace: frozenset[str], path: tuple[str, ...]) -> dict:
    for key, value in over.items():
        here = (*path, str(key))
        current = base.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping) and ".".join(here) not in replace:
            base[key] = _merge(dict(current), value, replace, here)
        else:
            base[key] = copy.deepcopy(_plain(value))
    return base


def differences(
    before: Mapping[str, Any], after: Mapping[str, Any], path: tuple[str, ...] = ()
) -> dict[str, tuple]:
    """Every dotted path whose value differs between two nested mappings: ``{path: (before, after)}``.

    What `tandem profile create --preset` reports and what the tests pin: a preset is exactly the
    changes it makes to the profile it is applied to. A block present on one side only is reported
    setting by setting, as if the other side had it empty, so ``differences({}, settings)`` lists every
    setting a preset states.
    """
    out: dict[str, tuple] = {}
    for key in sorted(set(before) | set(after), key=str):
        here = (*path, str(key))
        old, new = before.get(key, _MISSING), after.get(key, _MISSING)
        if isinstance(old, Mapping) and new is _MISSING:
            new = {}
        elif isinstance(new, Mapping) and old is _MISSING:
            old = {}
        if isinstance(old, Mapping) and isinstance(new, Mapping):
            out.update(differences(old, new, here))
        elif old != new:
            out[".".join(here)] = (None if old is _MISSING else old, None if new is _MISSING else new)
    return out


_MISSING = object()


# --------------------------------------------------------------------------- helpers


def _profile_sections() -> list[str]:
    from tandem.core.profiles import Profile

    return list(Profile.model_fields)


def _lookup(mapping: Mapping[str, Any], dotted: str) -> Any:
    node: Any = mapping
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _plain(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _did_you_mean(name: str, known: Any) -> str:
    close = difflib.get_close_matches(name, list(known), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""
