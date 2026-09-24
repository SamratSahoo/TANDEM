"""The rig: this machine's robot, its cameras and their calibration. Every profile shares it.

A profile is a task -- its prompt, its phase planning, its TAMP settings -- and a task means the same on
whichever machine it is collected. What is bolted to that machine's table does not: the robot computer's
address, the arm, which camera serial sits on the wrist, where each camera looks from. Those are said
once per machine, here, and not copied into every profile (where editing one left the others stale)::

    <config dir>/rig.yml                 paths.rig_file(); `tandem init` writes it
        version: 1
        robot:       {type, host}              the arm, and the robot computer (the NUC)
        cameras:     {perception, hand, external, external_2}
        calibration: calibration.json          extrinsics keyed by camera serial, beside rig.yml
        planners:    {<name>: {...}}           each planner's own machine settings

``robot`` and ``cameras`` are tandem's: the teleop executor records from the cameras whichever planner
runs, their roles are the dataset's camera layout, and every planner that drives an arm needs its
address. What else a planner needs of the machine -- a robot shim's ports, a grasp server's address --
is its own to declare (``Planner.RIG_OPTIONS``) and to check (``validate_rig_options``), under
``planners.<name>``: a block per planner, so a profile switched to another planner and back loses
nothing, and a planner nobody here has installed keeps its block as written.

Read on demand and cached by the file's stamp, as the settings are: the web server is long-lived, and
must see a `tandem rig set` run in a terminal a minute ago. ``update`` rewrites the file through a
round trip, so the comments a person wrote into it survive a programmatic change.

Standard library, pydantic and ruamel only: `tandem rig show` runs on a laptop.
"""

from __future__ import annotations

import difflib
import io
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, ValidationError, field_validator, model_validator

from tandem.core import names, paths
from tandem.core.errors import RigInvalid, TandemError

#: What rig.yml is written as.
RIG_VERSION = 1

#: The camera roles, in the order a listing shows them. They are the dataset's camera layout.
ROLES = ("hand", "external", "external_2")

#: What a failure to read or change the file says to do.
EDIT_HINT = "`tandem rig edit` opens it."
NOTHING_WRITTEN = "Nothing was written; rig.yml is as it was. `tandem rig show` shows it."

_log = logging.getLogger(__name__)
# Planners already said to be absent from this machine, so a command that loads the rig ten times says it once.
_noticed: set[str] = set()


def _refuse_unknown(model: type[BaseModel], data: Any, what: str) -> Any:
    """Refuse a key ``model`` does not have, naming the nearest one it does: ``robot.hots`` is a typo."""
    if not isinstance(data, Mapping):
        return data
    known = list(model.model_fields)
    for key in data:
        if key not in known:
            close = difflib.get_close_matches(str(key), known, n=1, cutoff=0.6)
            meant = f" (did you mean {close[0]!r}?)" if close else ""
            raise ValueError(f"{key!r} is not {what}{meant}; the settings are {', '.join(known)}")
    return data


# --------------------------------------------------------------------------- the models


class RobotSpec(BaseModel):
    """The arm, and the computer it is reached through. What every planner that drives it needs."""

    model_config = {"extra": "forbid"}

    # The arm, by the name planners know it by (TiPToP: fr3_robotiq, panda_robotiq, panda, ur5). tandem
    # checks only that it is a name; whether a planner drives that arm is the planner's to say, and its
    # doctor rows say it.
    type: str = "fr3_robotiq"
    # The robot computer: for a Franka, the NUC running the bamboo-polymetis shim. An address and nothing
    # else -- the ports it is reached on are a planner's own settings.
    host: str = "172.16.0.2"

    @model_validator(mode="before")
    @classmethod
    def _known(cls, data: Any) -> Any:
        return _refuse_unknown(cls, data, "a robot setting")

    @field_validator("type")
    @classmethod
    def _arm_name(cls, v: str) -> str:
        if not names.is_valid(v):
            raise ValueError(f"{v!r} is not an arm's name ({names.RULE}), such as fr3_robotiq")
        return v

    @field_validator("host")
    @classmethod
    def _address(cls, v: str) -> str:
        v = v.strip()
        # One colon is a port ("172.16.0.2:5555"); an IPv6 address has several.
        if not v or any(c.isspace() for c in v) or "://" in v or "/" in v or v.count(":") == 1:
            raise ValueError(
                f"{v!r} must be a hostname or IP address, such as 172.16.0.2 (no http://, no port: ports are "
                "the planner's own settings)"
            )
        return v


class CameraSpec(BaseModel):
    model_config = {"extra": "forbid"}

    serial: str
    type: str = "zed"
    resolution: str = "HD720"
    # 15, not 30: three ZEDs at HD720@30 exceed the USB bandwidth budget and fail to open.
    # The LeRobot export resamples to 15 Hz anyway, so nothing is lost.
    fps: int = 15

    @model_validator(mode="before")
    @classmethod
    def _known(cls, data: Any) -> Any:
        return _refuse_unknown(cls, data, "a camera setting")

    @field_validator("serial", mode="before")
    @classmethod
    def _text(cls, v: Any) -> Any:
        # YAML reads an unquoted serial as a number, and a serial with a leading zero as a different one
        # (or as octal). Refused rather than converted, since the conversion is already lossy by then.
        if not isinstance(v, str):
            raise ValueError(f"a camera serial is text: quote it: '{v}'")
        if not v.strip():
            raise ValueError("a camera serial cannot be empty")
        return v.strip()


class CamerasSpec(BaseModel):
    model_config = {"extra": "forbid"}

    # Which camera the perception pipeline reads. "external" leaves the arm at its home pose (a static
    # third-person pose comes straight from calibration); "hand" drives the arm to a capture pose and
    # works out the camera's pose from the arm each time.
    perception: str = "external"
    hand: CameraSpec | None = None
    external: CameraSpec | None = None
    # Recorded as DROID exterior_2. Omit for a deliberate two-camera rig: when it IS configured and fails
    # to open, collection aborts before any rollout rather than record an episode with a camera missing.
    external_2: CameraSpec | None = None

    # Not checked here: that the camera perception reads is configured. `tandem rig set` changes one
    # camera at a time, and a rig passes through states (the wrist camera set, the external not yet)
    # that must be writable. A session refuses to start without it, and `tandem doctor` fails on it.

    @model_validator(mode="before")
    @classmethod
    def _known(cls, data: Any) -> Any:
        return _refuse_unknown(cls, data, "a camera role or setting")

    @field_validator("perception")
    @classmethod
    def _perception_choice(cls, v: str) -> str:
        if v not in {"hand", "external"}:
            raise ValueError("cameras.perception must be 'hand' or 'external'")
        return v

    @model_validator(mode="after")
    def _one_role_per_camera(self):
        seen: dict[str, str] = {}
        for role, cam in self.configured().items():
            if cam.serial in seen:
                raise ValueError(
                    f"camera {cam.serial!r} is both cameras.{seen[cam.serial]} and cameras.{role}; a camera fills "
                    "one role"
                )
            seen[cam.serial] = role
        return self

    def configured(self) -> dict[str, CameraSpec]:
        return {key: cam for key in ROLES if (cam := getattr(self, key)) is not None}

    def perception_missing(self) -> str | None:
        """Why perception has no camera to read, or None: the role it reads from is not configured."""
        if getattr(self, self.perception) is None:
            return f"cameras.perception is {self.perception!r} but cameras.{self.perception} is not configured"
        return None


class Rig(BaseModel):
    """This machine's rig, validated. Where it was read from is kept, for its relative calibration path."""

    model_config = {"extra": "forbid"}

    version: int = RIG_VERSION
    robot: RobotSpec = Field(default_factory=RobotSpec)
    cameras: CamerasSpec = Field(default_factory=CamerasSpec)
    # Extrinsics keyed by camera serial: relative to rig.yml's directory, or absolute.
    calibration: str = "calibration.json"
    # Each planner's machine settings, by planner name: what its RIG_OPTIONS declare.
    planners: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # The rig.yml this was read from; None for one built by hand, which resolves against the config dir.
    _source: Path | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _known(cls, data: Any) -> Any:
        return _refuse_unknown(cls, data, "a rig setting")

    @field_validator("version")
    @classmethod
    def _readable(cls, v: int) -> int:
        if v > RIG_VERSION:
            raise ValueError(
                f"version {v} was written by a newer tandem (this one reads version {RIG_VERSION}); upgrade tandem"
            )
        if v != RIG_VERSION:
            raise ValueError(f"must be {RIG_VERSION}")
        return v

    @field_validator("calibration")
    @classmethod
    def _a_path(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must name the calibration file, such as calibration.json")
        return v.strip()

    @field_validator("planners", mode="before")
    @classmethod
    def _planner_blocks(cls, v: Any) -> Any:
        if v is None:
            return {}
        if not isinstance(v, Mapping):
            raise ValueError("must be a mapping of planner name -> that planner's machine settings")
        checked: dict[str, dict[str, Any]] = {}
        for name, block in v.items():
            if not names.is_valid(name):
                raise ValueError(f"{name!r} is not a planner's name ({names.RULE})")
            if block is None:
                block = {}
            if not isinstance(block, Mapping):
                raise ValueError(f"planners.{name} must be a mapping of the {name} planner's machine settings")
            checked[str(name)] = _planner_block(str(name), dict(block))
        return checked

    # ---- the file and the calibration ---------------------------------------------------------

    def file(self) -> Path:
        return self._source if self._source is not None else paths.rig_file()

    def calibration_file(self) -> Path:
        candidate = Path(os.path.expanduser(self.calibration))
        return candidate if candidate.is_absolute() else self.file().parent / candidate

    def extrinsics(self) -> dict:
        """The calibration file's extrinsics, by camera serial. Empty when the file is not there yet.

        Not ``calibration()``: that name is the field saying where the file is."""
        path = self.calibration_file()
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise RigInvalid(
                f"{path} is not valid JSON: {exc}",
                hint="It holds each camera's extrinsics, keyed by serial. Fix it by hand, or calibrate again.",
            ) from exc
        if not isinstance(data, dict):
            raise RigInvalid(f"{path} must be a JSON object of extrinsics keyed by camera serial.")
        return data

    def missing_calibration(self) -> list[str]:
        """Configured camera serials with no extrinsics entry.

        Extrinsics are keyed by serial, and a planner that localises from them (TiPToP raises at warm-up
        for a serial it cannot find) is better told before a session starts.
        """
        known = set(self.extrinsics())
        return [cam.serial for cam in self.cameras.configured().values() if cam.serial not in known]

    def summary(self) -> str:
        return f"{self.robot.type} at {self.robot.host}"


def _planner_block(name: str, block: dict[str, Any]) -> dict[str, Any]:
    """``planners.<name>`` as that planner reads it, when it is installed here; as written otherwise.

    A planner that is not installed cannot check its block, and refusing the rig over it would stop every
    command on a machine that once had it. One that is installed but will not load keeps its block too,
    the way a profile keeps planner.options nobody can check: the session that builds it says why.
    """
    from tandem.planners import registry

    if name not in registry.available():
        if name not in _noticed:
            _noticed.add(name)
            _log.warning(
                "rig.yml has settings for the planner %r, which is not installed here; they are kept as written",
                name,
            )
        return block
    try:
        factory = registry.factory(name)
    except TandemError:
        return block
    try:
        return registry.rig_options_for(factory, block)
    except TandemError as exc:
        raise ValueError(f"planners.{name}: {exc.message} {exc.hint or ''}".rstrip()) from None
    except ValueError as exc:  # a pydantic ValidationError is one
        raise ValueError(_located(exc, f"planners.{name}")) from None


def _located(exc: Any, prefix: str) -> str:
    """A planner's complaint about its block, each line located under ``prefix`` for the reader."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        lines = []
        for err in errors():
            loc = ".".join(str(part) for part in err.get("loc", ()))
            msg = str(err.get("msg", "")).removeprefix("Value error, ")
            lines.append(f"{prefix}.{loc}: {msg}" if loc else f"{prefix}: {msg}")
        if lines:
            return "\n  ".join(lines)
    return f"{prefix}: {exc}"


def format_errors(exc: Exception) -> str:
    """A ValidationError of the rig's, one located line per problem."""
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return f"  {exc}"
    lines = []
    for err in errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = str(err.get("msg", "")).removeprefix("Value error, ")
        # A planner's block locates its own lines (planners.tiptop.perception.m2t2.url: ...).
        if (loc == "planners" and msg.startswith("planners.")) or not loc:
            lines.append(f"  {msg}")
        else:
            lines.append(f"  {loc}: {msg}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- reading


_cache: Rig | None = None
# What rig.yml was when _cache was read from it: its path, and its stamp (None: there was no file).
_stamp: tuple[str, tuple[int, int] | None] | None = None


def _file_stamp(path: Path) -> tuple[str, tuple[int, int] | None]:
    try:
        stat = path.stat()
    except OSError:
        return (str(path), None)
    return (str(path), (stat.st_mtime_ns, stat.st_size))


def exists() -> bool:
    return paths.rig_file().is_file()


def load(*, force: bool = False) -> Rig:
    """This machine's rig. Its defaults, with no cameras, when rig.yml is not there yet."""
    global _cache, _stamp
    path = paths.rig_file()
    stamp = _file_stamp(path)
    if _cache is not None and not force and stamp == _stamp:
        return _cache
    if stamp[1] is None:
        rig = Rig()
        rig._source = path
    else:
        try:
            text = path.read_text()
        except OSError as exc:
            raise RigInvalid(f"{path} could not be read: {exc}") from exc
        rig = parse_text(text, source=path, hint=EDIT_HINT)
    _cache, _stamp = rig, stamp
    return rig


def template() -> str:
    from tandem import resources

    return resources.read("rig_template.yml")


def read_text() -> str:
    """rig.yml as it is written, or the commented template when there is none yet."""
    path = paths.rig_file()
    return path.read_text() if path.is_file() else template()


def parse_text(text: str, *, source: Path | str, hint: str = EDIT_HINT) -> Rig:
    """``text`` as a rig, validated. ``RigInvalid``, with every problem located, when it is not one."""
    from ruamel.yaml import YAML

    try:
        data = YAML(typ="safe").load(text)
    except Exception as exc:
        raise RigInvalid(f"{source} is not valid YAML: {exc}", hint=hint) from exc
    if data is None:
        # Not "no settings": read as that, it would be a rig of defaults -- another robot's address --
        # without a word.
        raise RigInvalid(f"{source} is empty.", hint="`tandem rig edit` starts it from the template.")
    if not isinstance(data, dict):
        raise RigInvalid(f"{source} must be a mapping of rig settings, not a {type(data).__name__}.", hint=hint)
    try:
        rig = Rig.model_validate(data)
    except ValidationError as exc:
        raise RigInvalid(f"{source} is not a valid rig:\n{format_errors(exc)}", hint=hint) from None
    rig._source = Path(source)
    return rig


def planner_options(rig: Rig, name: str) -> dict[str, Any]:
    """``planners.<name>`` as the planner reads it: checked, defaults filled in, ``{}`` validated when absent.

    A planner that cannot be loaded here gets its block as written; the session that tries to build it
    says why it cannot.
    """
    from tandem.planners import registry

    block = dict(rig.planners.get(name) or {})
    try:
        factory = registry.factory(name)
    except TandemError:
        return block
    try:
        return registry.rig_options_for(factory, block)
    except TandemError as exc:
        raise RigInvalid(
            f"{rig.file()}: planners.{name} is not valid: {exc.message}",
            hint=exc.hint or f"`tandem rig set planners.{name}.KEY VALUE`, or {EDIT_HINT}",
        ) from None
    except ValueError as exc:
        raise RigInvalid(
            f"{rig.file()}: planners.{name} is not valid:\n  {_located(exc, f'planners.{name}')}",
            hint=f"`tandem rig set planners.{name}.KEY VALUE`, or {EDIT_HINT}",
        ) from None


# --------------------------------------------------------------------------- changing


def update(changes: Mapping[str, Any]) -> Rig:
    """Set dotted keys (``robot.host``, ``cameras.hand.serial``) and write rig.yml; None removes a key.

    A round trip, so the comments a person wrote survive; validated whole before anything is written, so a
    bad value leaves the file as it was; replaced atomically. The calibration file is created, empty, when
    it is not there yet: a planner's calibration script writes into it, and a session looks there.
    """
    path = paths.rig_file()
    text = path.read_text() if path.is_file() else template()
    doc = _new_yaml().load(text)
    if doc is None:
        from ruamel.yaml.comments import CommentedMap

        doc = CommentedMap()
    if not isinstance(doc, dict):
        raise RigInvalid(f"{path} must be a mapping of rig settings.", hint=EDIT_HINT)
    for key, value in changes.items():
        _set(doc, _parts(key), value)
    return write_text(_dump(doc))


def write_text(text: str) -> Rig:
    """Validate ``text`` as a rig and make it this machine's: `tandem rig edit`, the web's rig card."""
    path = paths.rig_file()
    parse_text(text, source=path, hint=NOTHING_WRITTEN)
    paths.ensure_dir(path.parent)
    paths.write_atomic(path, text)
    rig = load(force=True)
    ensure_calibration_file(rig)
    return rig


def ensure_calibration_file(rig: Rig) -> Path:
    path = rig.calibration_file()
    if not path.exists():
        paths.ensure_dir(path.parent)
        path.write_text("{}\n")
    return path


def _new_yaml():
    """A fresh round-trip YAML instance, one per dump (``profiles._new_yaml`` says why)."""
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 110
    return yaml


def _dump(doc: Any) -> str:
    buf = io.StringIO()
    _new_yaml().dump(doc, buf)
    return buf.getvalue()


def _set(doc: Any, parts: list[str], value: Any) -> None:
    from ruamel.yaml.comments import CommentedMap

    node = doc
    for depth, part in enumerate(parts[:-1]):
        child = node.get(part)
        if child is None:
            if value is None:
                return  # removing something under a section that is not there: nothing to do
            child = CommentedMap()
            node[part] = child
        elif not isinstance(child, dict):
            raise RigInvalid(
                f"{'.'.join(parts)} cannot be set: {'.'.join(parts[: depth + 1])} is {child!r}, not a section.",
                hint=NOTHING_WRITTEN,
            )
        # A section written inline ({serial: '1'}, or planners: {}) that gains a nested key reads better
        # as a block than as one long line.
        if hasattr(child, "fa"):
            child.fa.set_block_style()
        node = child
    if value is None:
        node.pop(parts[-1], None)
    else:
        node[parts[-1]] = _yamlable(value)


def _yamlable(value: Any) -> Any:
    """A value as it reads best in the file: a list of numbers on one line, a mapping as a block."""
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    if isinstance(value, Mapping):
        out = CommentedMap()
        for key, item in value.items():
            out[str(key)] = _yamlable(item)
        return out
    if isinstance(value, (list, tuple)):
        seq = CommentedSeq(_yamlable(item) for item in value)
        if all(not isinstance(item, (Mapping, list, tuple)) for item in value):
            seq.fa.set_flow_style()
        return seq
    return value


# --------------------------------------------------------------------------- `tandem rig set KEY VALUE`

#: Keys whose value is text as typed, never parsed as YAML: a serial typed as 14846828 stays that string.
_TEXT_KEYS = {"robot.type", "robot.host", "cameras.perception", "calibration"}
_TEXT_CAMERA_KEYS = {"serial", "type", "resolution"}


def _parts(key: str) -> list[str]:
    parts = [part.strip() for part in str(key).split(".")]
    if not key or any(not part for part in parts):
        raise TandemError(f"{key!r} is not a rig setting.", hint="Keys are dotted, such as robot.host.")
    return parts


def coerce(key: str, raw: str) -> Any:
    """``raw`` as the value of ``key``, as typed on a command line; ``null`` removes the key.

    The rig's text settings are taken as typed, so a serial stays a string; every other value is read as
    YAML (``[0, 1]``, ``true``, ``8123``). The key itself is checked here, with the nearest real one, so a
    typo is refused before anything is read or written.
    """
    parts = _parts(key)
    _check_key(parts)
    if raw.strip() == "null":
        return None
    dotted = ".".join(parts)
    if dotted in _TEXT_KEYS or (len(parts) == 3 and parts[0] == "cameras" and parts[2] in _TEXT_CAMERA_KEYS):
        return raw
    from ruamel.yaml import YAML

    try:
        return YAML(typ="safe").load(raw)
    except Exception as exc:
        raise TandemError(f"{raw!r} is not a value for {dotted}: {exc}") from None


def _check_key(parts: list[str]) -> None:
    top = parts[0]
    if top == "version":
        raise TandemError("version is tandem's to write.", hint="Leave it as it is.")
    sections: dict[str, type[BaseModel] | None] = {
        "robot": RobotSpec,
        "cameras": CamerasSpec,
        "calibration": None,
        "planners": None,
    }
    _known(top, list(sections), "")
    if top in ("robot", "calibration") and len(parts) > (2 if top == "robot" else 1):
        raise TandemError(f"{'.'.join(parts)} is not a rig setting.", hint=f"`tandem rig show` lists the {top} settings.")
    if top == "robot" and len(parts) == 2:
        _known(parts[1], list(RobotSpec.model_fields), "robot.")
    if top == "cameras" and len(parts) >= 2:
        _known(parts[1], list(CamerasSpec.model_fields), "cameras.")
        if len(parts) >= 3:
            if parts[1] == "perception" or len(parts) > 3:
                raise TandemError(f"{'.'.join(parts)} is not a rig setting.", hint="`tandem rig show` lists them.")
            _known(parts[2], list(CameraSpec.model_fields), f"cameras.{parts[1]}.")
    if top == "planners" and len(parts) >= 2:
        from tandem.planners import registry

        installed = registry.available()
        if parts[1] not in installed:
            close = difflib.get_close_matches(parts[1], installed, n=1, cutoff=0.6)
            raise TandemError(
                f"No planner named {parts[1]!r} is installed on this machine"
                + (f" (did you mean {close[0]!r}?)." if close else f"; it has {', '.join(installed)}."),
                hint="`tandem planners list` shows every planner; each one's machine settings go under "
                "planners.<name>.",
            )


def _known(key: str, known: list[str], where: str) -> None:
    if key in known:
        return
    close = difflib.get_close_matches(key, known, n=1, cutoff=0.6)
    section = f"{where.rstrip('.')} settings" if where else "rig's settings"
    raise TandemError(
        f"{where}{key} is not a rig setting" + (f" (did you mean {where}{close[0]}?)." if close else "."),
        hint=f"The {section} are: {', '.join(known)}. `tandem rig show` lists them.",
    )
