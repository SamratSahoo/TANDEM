"""The planner boundary: what a backend declares, and how tandem talks to one.

The point of this layer is that tandem drives a task and motion planner it has not modified. Two
things have to hold for that to be true, and both are tested here: the facts tandem knows about a
planner are a DECLARATION that can be checked against the real thing, and the channel to a hosted
backend survives what a real planner's process actually does to its stdout.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from tandem.core.errors import TandemError
from tandem.planners import registry
from tandem.planners.base import (
    VERBS,
    BackendError,
    Capabilities,
    GoalAtom,
    LegSpec,
    to_goal_atoms,
)
from tandem.planners.rpc import HostedBackendChannel
from tandem.planners.tiptop.backend import TiptopBackend, sidecar_path
from tandem.planning.symbols import Atom

FAKE_SIDECAR = Path(__file__).parent / "fake_sidecar.py"


# --- the registry ---------------------------------------------------------------------------------


def test_a_backend_resolves_without_importing_the_planner():
    # The whole laptop promise rests on this: knowing what a planner can be asked for must not need
    # the planner. `tandem plan` checks a decomposition against tiptop's real goal language on a
    # machine with no CUDA, no cuTAMP and no robot.
    caps = registry.capabilities("tiptop")
    assert caps.name == "tiptop"
    assert "cutamp" not in sys.modules and "torch" not in sys.modules


def test_an_unknown_backend_is_an_error_not_a_fallback():
    # A session that silently planned with a different planner from the one the config asked for
    # produces a dataset nobody can interpret afterwards. Same discipline as the `tamp` keys.
    with pytest.raises(TandemError) as excinfo:
        registry.capabilities("tiptopp")
    assert "tiptopp" in str(excinfo.value.message)
    assert "tiptop" in str(excinfo.value.hint), "a near miss should be suggested"

    with pytest.raises(TandemError) as excinfo:
        registry.capabilities("something-else")
    assert "Known backends" in str(excinfo.value.hint)


def test_the_tiptop_backend_satisfies_the_protocol():
    assert isinstance(TiptopBackend, type)
    assert issubclass(TiptopBackend, object)
    # runtime_checkable Protocols check method presence, which is what matters here: a backend that
    # forgot release_hardware would strand the arm at the first human phase.
    for verb in VERBS:
        assert hasattr(TiptopBackend, verb), f"TiptopBackend is missing {verb}"


# --- the declaration ------------------------------------------------------------------------------


def test_the_declaration_matches_the_real_cutamp_domain():
    """Declared on tandem's side, verified against the planner's when it happens to be reachable.

    tandem cannot import cuTAMP -- that is the point -- so the facts in `capabilities.py` are stated
    rather than computed, and a stale declaration would be silent: an atom the domain can no longer
    achieve would sail past the feasibility check and into a search with no bound.
    """
    vendored = Path(__file__).resolve().parents[1] / "src" / "tandem" / "_vendor" / "cuTAMP"
    if not (vendored / "cutamp" / "tamp_domain.py").is_file():
        pytest.skip("no vendored cuTAMP to check the declaration against")
    sys.path.insert(0, str(vendored))
    already = set(sys.modules)
    try:
        from cutamp.tamp_domain import (
            Movable,
            Surface,
            all_tamp_fluents,
            all_tamp_operators,
            get_initial_state,
        )
    except Exception as exc:  # pragma: no cover - the vendored tree needs no deps, but be kind
        pytest.skip(f"cuTAMP's symbolic layer is not importable here: {exc}")
    finally:
        sys.path.remove(str(vendored))
        # Leave sys.modules as it was found. This is the one test that imports a planner on purpose,
        # and a `cutamp` left behind makes any later "did that import a planner?" assertion depend
        # on test order.
        for name in set(sys.modules) - already:
            if name.split(".")[0] == "cutamp":
                del sys.modules[name]

    from tandem.planners.tiptop.capabilities import ACHIEVABLE, ALL_FLUENTS, CAPABILITIES

    assert ALL_FLUENTS == frozenset(f.name for f in all_tamp_fluents)
    initial = get_initial_state(movables=["m"], surfaces=["s"])
    assert ACHIEVABLE == (
        frozenset(f.name for op in all_tamp_operators for f in op.add_effects) | {a.name for a in initial}
    )
    assert CAPABILITIES.movable_type == Movable
    assert CAPABILITIES.surface_type == Surface
    # Everything statable as a goal must be something some operator can bring about, or a phase
    # asking for it would be accepted and then never satisfied.
    assert CAPABILITIES.goal_predicate_names() <= ACHIEVABLE


def test_a_goal_is_rendered_only_into_predicates_the_planner_reads():
    caps = registry.capabilities("tiptop")
    rendered = to_goal_atoms(
        [
            Atom("On", ("toy", "box")),
            Atom("Holding", ("toy",)),
            Atom("HandEmpty", ()),
            Atom("IsFolded", ("cloth",)),
        ],
        caps,
    )
    assert [a.to_dict() for a in rendered] == [
        {"predicate": "holding", "args": ["toy"]},
        {"predicate": "on", "args": ["toy", "box"]},
    ]


def test_a_backend_with_a_different_goal_language_needs_no_change_here():
    # The claim the whole refactor rests on. A planner that speaks a different vocabulary is a
    # different Capabilities, not a different phase planner.
    from tandem.planning.symbols import Parameter, Predicate

    caps = Capabilities(
        name="toy-planner",
        goal_predicates={
            "Inside": Predicate("Inside", (Parameter("obj", "thing"), Parameter("box", "container")))
        },
        goal_predicate_wire_names={"Inside": "inside"},
        achievable_predicates=frozenset({"Inside"}),
        movable_type="thing",
        surface_type="container",
        predicate_descriptions={"Inside": "{0} is inside {1}"},
    )
    assert caps.predicate_menu() == "- Inside(?obj: thing, ?box: container): {0} is inside {1}"
    assert [a.to_dict() for a in to_goal_atoms([Atom("Inside", ("ball", "bin"))], caps)] == [
        {"predicate": "inside", "args": ["ball", "bin"]}
    ]


# --- the sidecar ----------------------------------------------------------------------------------


def _sidecar_source() -> str:
    return sidecar_path().read_text()


def test_the_sidecar_ships_and_never_imports_tandem():
    """It runs in the planner's environment, where tandem is not installed and must not need to be.

    Adding tandem to that environment would mean installing tandem's own dependencies alongside the
    planner's for no reason, and would make the two versions something to keep in step.
    """
    assert sidecar_path().is_file()
    tree = ast.parse(_sidecar_source())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "tandem" not in imported, f"the sidecar imports tandem: {sorted(imported)}"


def test_the_sidecar_answers_exactly_the_protocol_verbs():
    """The verb list is written twice because the sidecar cannot import it. Pin the two together.

    A verb added on one side only is not a crash: the parent gets `unknown verb`, or the sidecar
    grows a method nothing ever calls. Both are the kind of thing that survives review.
    """
    tree = ast.parse(_sidecar_source())
    declared = None
    methods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "VERBS" for t in node.targets
        ):
            declared = tuple(el.value for el in node.value.elts)
        if isinstance(node, ast.ClassDef) and node.name == "Sidecar":
            methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
    assert declared == VERBS, "the sidecar's VERBS has drifted from tandem.planners.base.VERBS"
    assert set(VERBS) <= methods, f"the sidecar declares verbs it cannot answer: {set(VERBS) - methods}"


def test_the_sidecar_takes_stdout_away_from_the_libraries_it_imports():
    """The single thing that must be right or nothing works.

    A real planner's process prints on import -- CUDA banners, warp's version line, a stray print in
    a vendored tree. Any one of those on fd 1 lands in the middle of a JSON reply. The sidecar dups
    the real stdout for itself and points fd 1 at stderr before importing anything.
    """
    source = _sidecar_source()
    protocol_line = source.index("_PROTOCOL_OUT = os.fdopen(os.dup(1)")
    assert "os.dup2(2, 1)" in source
    # Before any planner import, not merely somewhere in the file.
    assert protocol_line < source.index("from tiptop"), "stdout must be secured before tiptop is imported"


# --- the channel ----------------------------------------------------------------------------------


def _channel(*extra, logs=None):
    return HostedBackendChannel(
        [sys.executable, str(FAKE_SIDECAR), *extra],
        on_log=(lambda stream, text: logs.append(text)) if logs is not None else None,
    )


def test_the_channel_carries_the_sub_goal_cycle():
    channel = _channel().start()
    try:
        scene = channel.call("perceive", task_hint="put the toy in the box", save_dir="/tmp/x")
        assert scene["object_labels"] == ["blue_toy", "white_box"]
        assert scene["table_label"] == "table"

        planned = channel.call(
            "plan",
            scene_id=scene["scene_id"],
            goal=[GoalAtom("on", ("blue_toy", "white_box")).to_dict()],
            surfaces=["white_box"],
            save_dir="/tmp/x",
        )
        assert planned["ok"] and planned["plan_handle"] == "p1"

        leg = LegSpec(trajectory_id="t-1", instruction="put the toy in the box", phase_index=0, n_phases=2)
        done = channel.call(
            "execute", plan_handle=planned["plan_handle"], leg=leg.to_dict(), save_dir="/tmp/x"
        )
        assert done["ok"] and done["n_frames"] == 42
    finally:
        channel.stop()


def test_library_noise_on_stdout_does_not_desynchronise_the_channel():
    # The failure this prevents is not a crash but a permanent one: one stray line and every
    # subsequent reply is read as the answer to the previous question.
    logs: list[str] = []
    channel = _channel("--noisy", logs=logs).start()
    try:
        for _ in range(3):
            scene = channel.call("perceive", task_hint="x", save_dir="/tmp/x")
            assert scene["scene_id"] == "s1"
        assert any("not json at all" in line for line in logs), "the noise should be forwarded, not dropped"
    finally:
        channel.stop()


def test_a_backend_that_cannot_start_says_so():
    channel = _channel("--fail-warm").start()
    try:
        with pytest.raises(BackendError, match="no CUDA device"):
            channel.call("warm", output_dir="/tmp/x", execute=True, record=True, cost_overrides=None)
    finally:
        channel.stop()


def test_a_child_that_never_announces_itself_is_not_waited_on_forever():
    import tandem.planners.rpc as rpc

    original = rpc.HANDSHAKE_TIMEOUT
    rpc.HANDSHAKE_TIMEOUT = 2.0
    try:
        with pytest.raises(BackendError, match="exited|did not answer"):
            _channel("--no-hello").start()
    finally:
        rpc.HANDSHAKE_TIMEOUT = original


def test_a_wedged_planner_is_reported_as_wedged_holding_the_robot():
    # An operator standing next to an arm needs to be told what is actually true, which is that the
    # planner still holds it -- not that a socket timed out.
    channel = _channel("--silent", "home").start()
    try:
        with pytest.raises(BackendError, match="wedged holding the robot"):
            channel.call("home", timeout=1.0)
    finally:
        channel.stop()


def test_calling_a_stopped_channel_is_an_error_not_a_hang():
    channel = _channel().start()
    channel.stop()
    with pytest.raises(BackendError, match="not running"):
        channel.call("home")


def test_the_backend_refuses_to_work_before_it_is_warm():
    backend = TiptopBackend(
        runtime=None,  # never touched on this path
        env={},
        output_dir=Path("/tmp"),
    )
    with pytest.raises(BackendError, match="has not been warmed"):
        backend.perceive(task_hint="x", save_dir=Path("/tmp"))


def test_every_symbol_the_sidecar_imports_exists_in_the_vendored_planner():
    """The sidecar cannot be exercised without a GPU, a robot and two cameras, so this is the check.

    It calls public functions of a planner tandem does not control. A re-vendor that moves or renames
    one of them would otherwise surface as an ImportError forty seconds into a warm-up, with an
    operator standing next to an arm — and only for whoever happened to run a session next.
    """
    vendor = Path(__file__).resolve().parents[1] / "src" / "tandem" / "_vendor"
    if not (vendor / "tiptop" / "tiptop" / "tiptop_run.py").is_file():
        pytest.skip("no vendored planner to check the sidecar against")
    if (vendor / "tiptop" / "tiptop" / "hitl").is_dir():
        # The vendored tree is still the fork this refactor exists to stop depending on, and the
        # sidecar is written against the clean upstream. Four of the functions it calls
        # (_planning_robot_types, home_all_arms, _execute_plan_recorded,
        # resolve_max_motion_refine_attempts) are upstream-only, so this check is meaningless until
        # the vendor swap. It starts enforcing itself the moment `tiptop/hitl/` stops being vendored.
        pytest.skip("the vendored planner is still the fork; the sidecar targets the clean upstream")

    def module_path(dotted: str) -> Path | None:
        root = vendor / ("tiptop" if dotted.startswith("tiptop") else "cuTAMP")
        rel = Path(*dotted.split("."))
        for candidate in (root / rel.with_suffix(".py"), root / rel / "__init__.py"):
            if candidate.is_file():
                return candidate
        return None

    def top_level_names(path: Path) -> set[str]:
        names: set[str] = set()
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.ImportFrom):
                names.update(a.asname or a.name for a in node.names)
            elif isinstance(node, ast.Import):
                names.update(a.asname or a.name.split(".")[0] for a in node.names)
        return names

    tree = ast.parse(sidecar_path().read_text())
    missing: list[str] = []

    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.module):
            continue
        if node.module.split(".")[0] not in ("tiptop", "cutamp"):
            continue
        path = module_path(node.module)
        if path is None:
            missing.append(f"module {node.module}")
            continue
        available = top_level_names(path)
        for alias in node.names:
            # `from tiptop import tiptop_run` is a submodule, not a name in the package's __init__.
            if module_path(f"{node.module}.{alias.name}") is not None:
                continue
            if alias.name not in available:
                missing.append(f"{node.module}.{alias.name}")

    # Module attributes reached directly, which is how the save pool is shared with the planner's own
    # rollout loop. Private, and therefore exactly the kind of thing that moves without warning.
    run_names = top_level_names(vendor / "tiptop" / "tiptop" / "tiptop_run.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "tiptop_run":
            if node.attr not in run_names:
                missing.append(f"tiptop_run.{node.attr}")

    assert not missing, "the sidecar names symbols the vendored planner does not have: " + ", ".join(
        sorted(set(missing))
    )


def _upstream_signatures(vendor: Path) -> dict[str, ast.arguments]:
    """Every function the sidecar could call, by the name it imports it under."""
    signatures: dict[str, ast.arguments] = {}
    for root in (vendor / "tiptop", vendor / "cuTAMP"):
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:  # pragma: no cover - a vendored tree we do not control
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    signatures.setdefault(node.name, node.args)
    return signatures


def test_the_sidecar_calls_the_planner_with_the_right_arguments():
    """Existence is not enough. A required parameter the sidecar does not pass is a TypeError forty
    seconds into a warm-up, on a machine none of this can be exercised on.

    That is not hypothetical: `resolve_time_dilation_factor(overrides, config_default)` was called
    with one argument here, which would have failed EVERY warm — and passing the wrong default would
    have been worse than the crash, since the planner ships 0.2 and 1.0 means five times the speed.
    """
    vendor = Path(__file__).resolve().parents[1] / "src" / "tandem" / "_vendor"
    if not (vendor / "tiptop" / "tiptop" / "tiptop_run.py").is_file():
        pytest.skip("no vendored planner to check the sidecar against")
    if (vendor / "tiptop" / "tiptop" / "hitl").is_dir():
        pytest.skip("the vendored planner is still the fork; the sidecar targets the clean upstream")

    tree = ast.parse(sidecar_path().read_text())

    # Only names the sidecar actually imported FROM the planner — anything else is its own.
    from_planner: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in ("tiptop", "cutamp"):
            from_planner.update(a.asname or a.name for a in node.names)

    signatures = _upstream_signatures(vendor)
    problems: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        name = node.func.id
        if name not in from_planner or name not in signatures:
            continue
        args = signatures[name]
        positional = args.posonlyargs + args.args
        required = len(positional) - len(args.defaults)
        supplied_kw = {k.arg for k in node.keywords if k.arg}
        supplied_pos = len(node.args)

        if any(k.arg is None for k in node.keywords) or any(isinstance(a, ast.Starred) for a in node.args):
            continue  # **kwargs / *args at the call site: nothing to check statically

        named_positionals = {p.arg for p in positional[supplied_pos:]}
        unfilled = [
            p.arg
            for p in positional[supplied_pos:required]
            if p.arg not in supplied_kw
        ]
        if unfilled:
            problems.append(f"{name}() is missing required argument(s): {', '.join(unfilled)}")
        if not args.vararg and supplied_pos > len(positional):
            problems.append(f"{name}() takes {len(positional)} positional argument(s), {supplied_pos} given")
        if not args.kwarg:
            known = named_positionals | {a.arg for a in args.kwonlyargs} | {p.arg for p in positional}
            unknown = sorted(supplied_kw - known)
            if unknown:
                problems.append(f"{name}() got unexpected keyword argument(s): {', '.join(unknown)}")

    assert not problems, "the sidecar calls the vendored planner wrongly:\n  " + "\n  ".join(problems)
