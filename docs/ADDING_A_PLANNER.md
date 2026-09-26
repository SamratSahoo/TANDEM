# Adding a planner

A planner is a Python package that tells tandem three things: what is in the scene (`perceive`), how to
reach one goal (`plan`), and how to run that plan on the robot while recording it (`execute`). tandem does
everything else: splitting the task into phases, teleop, checking a person's work and merging the legs.

Working examples: `src/tandem/planners/tiptop/` (the built-in planner) and `tests/toy_planner.py`.

## Quick start

Generate a package, install it where tandem runs, and run its tests:

```bash
tandem planners new shelfbot          # creates ./tandem-shelfbot (--dir to change)
cd tandem-shelfbot
pipx inject tandem-tamp --editable . pytest    # if tandem came from pipx
python -m pytest                               # passes as generated
tandem planners use shelfbot                   # the active profile now plans with it
```

The generated planner plans in a stand-in world, so its tests pass before you write any real code.
`planners new` prints the install command for your setup. With uv use
`uv tool install --reinstall git+https://github.com/SamratSahoo/tandem.git --with-editable . --with pytest`,
and in a virtualenv use `pip install -e ".[test]"`.

Then:

1. Fill in the `TODO`s in the generated README, in order, keeping `pytest` green.
2. Once `execute` records real legs, set `records_legs = True` in the test.
3. Check that `tandem plan --planner shelfbot --image workspace.png "<task>"` splits a real task into
   phases your planner can do ([Planning from a photo](USAGE.md#planning-from-a-photo)).
4. Run `tandem collect --no-execute`. It perceives and plans without moving the robot.

Use `tandem planners new NAME --sidecar` if the planner needs its own environment (torch, CUDA, a camera
SDK). See [Sidecars](#sidecars).

`tandem planners list` should now show `shelfbot` as `no runtime needed`. `tandem planners info shelfbot`
shows what tandem knows about it. What `planners use` changes in a profile is in
[Planner settings](CONFIGURATION.md#planner-settings).

### Names

A planner's name uses lowercase letters, digits, `_` and `-`, and starts with a letter.

## A minimal planner

This is the whole shape of a planner, trimmed from the scaffold. It can move a block onto a tray.

```python
import json, time, uuid
from pathlib import Path

from tandem.planners import (
    Capabilities, ExecuteResult, Parameter, Planner, PlannerInfo, PlanResult, Predicate, SceneView,
)

ON = Predicate("On", (Parameter("obj", "movable"), Parameter("place", "surface")))


class ShelfBot(Planner):
    info = PlannerInfo(name="shelfbot", display_name="ShelfBot", summary="Puts things on trays.")
    CAPABILITIES = Capabilities(
        name="shelfbot",                                # must equal info.name
        goal_predicates={"On": ON},                     # what a robot phase's goal may say
        robot_description="move an object onto a surface",
        goal_predicate_wire_names={"On": "on"},         # how plan() receives it
        achievable_predicates=frozenset({"On"}),
        reserved_predicate_names=frozenset({"On"}),
        movable_type="movable",
        surface_type="surface",
        predicate_descriptions={"On": "{0} is resting on top of {1}"},
        checkable_predicates=frozenset({"On"}),         # a camera can judge On from a photo
    )

    def perceive(self, *, task_hint, save_dir, reset_arm=True, open_gripper=False):
        scene_id = uuid.uuid4().hex[:8]
        return SceneView(
            object_labels=("block", "tray"),
            table_label="table",
            surface_labels=frozenset({"tray"}),
            scene_id=scene_id,
            rgb_path=str(Path(save_dir) / "perception_rgb.png"),   # write a real image here
        )

    def plan(self, scene_id, goal, *, surfaces=frozenset(), movables=None, return_home=True,
             save_dir, reuse_skeleton=None):
        for atom in goal:                               # e.g. GoalAtom("on", ("block", "tray"))
            if atom.predicate != "on":
                return PlanResult(ok=False, failure_reason=f"cannot plan {atom.predicate}")
        return PlanResult(ok=True, planning_seconds=0.0, plan_handle=uuid.uuid4().hex[:8],
                          task_plan=tuple(f"Place({a.args[0]}, {a.args[1]})" for a in goal))

    def execute(self, plan_handle, leg, *, save_dir, should_stop=None):
        start = time.time()
        # ... run the plan and record robot_state.npz and the camera clips ...
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        meta = {"trajectory_id": leg.trajectory_id, "segment_source": leg.segment_source,
                "instruction": leg.instruction, "record_start": start, "record_stop": time.time(),
                "fps": 15, "cameras": {}}
        if leg.phase_index is not None:
            meta.update(phase_index=leg.phase_index, n_phases=leg.n_phases,
                        phase_description=leg.phase_description)
        (Path(save_dir) / "_meta.json").write_text(json.dumps(meta))
        return ExecuteResult(ok=True, n_frames=0, rollout_dir=str(save_dir))
```

The rest of this page is the reference for each part.

## The protocol

A session calls a planner's verbs in this order:

```
factory.create(ctx) → require_ready() → warm()
    → ( perceive() → plan() → execute() )*       robot legs
    → release_hardware() … reacquire_hardware()   around every human leg
    → home() → close()                            when the session stops
```

Subclass [`Planner`](#the-planner-base-class), not `tandem.planners.base.TampBackend`. `Planner` supplies
every verb except `perceive`, `plan` and `execute`.

If a verb raises, the trial ends at that verb's stage. `perceive` and `plan` fail at `tamp_planning`, and
`execute` fails at `tamp_execution`.

| verb | what it must do |
|---|---|
| `capabilities()` | Return the capabilities. It must be static and cheap. |
| `require_ready()` | Raise `RuntimeNotReady` naming what is missing. |
| `warm()` | Open cameras, connect the robot and build solvers. It runs again after a verb raises, so it must be a no-op when already warm. |
| `perceive(*, task_hint, save_dir, reset_arm=True, open_gripper=False)` → `SceneView` | Look at the scene. `task_hint` is the whole instruction and only steers detection. |
| `plan(scene_id, goal, *, surfaces=frozenset(), movables=None, return_home=True, save_dir, reuse_skeleton=None)` → `PlanResult` | Plan one goal. If it can't, return `ok=False` with a `failure_reason` instead of raising. |
| `execute(plan_handle, leg, *, save_dir, should_stop=None)` → `ExecuteResult` | Run the plan and record one leg of `leg.trajectory_id` ([contract](#the-recording-contract)). `ok=False` ends the trial at `tamp_execution`. |
| `capture_frame(*, camera="external")` | Save one RGB frame from `camera` (set by `hitl.verification_camera`) and return its path. |
| `release_hardware()` | Block until the robot and cameras are free for a person. |
| `reacquire_hardware()` | Take them back, wherever the person left the arm. |
| `home()` | Park the arm. Don't open the gripper. |
| `close()` | Release everything. It must be safe to call twice, and before `warm`. |

Notes on the arguments:

- `reset_arm` is true only on an attempt's first leg. Never park an arm a person just handed back.
- `open_gripper` is true only on the first leg after a human phase. Open the hand and move nothing else.
- `goal` is a list of `GoalAtom(predicate, args)` in the wire spelling (for example `on`).
- `surfaces` is fixed for the whole task.

### `SceneView`

| field | meaning |
|---|---|
| `object_labels` | Every object seen, except the table. |
| `table_label` | The support surface. Defaults to `table`. |
| `surface_labels` | The objects that are surfaces. |
| `scene_id` | An opaque id that tandem passes back to `plan`. |
| `rgb_path` | An image of the scene. tandem splits the task and checks robot-leg preconditions from it. Without it, proposing a plan fails and the trial ends at `invention`. |
| `detected_goal` | The planner's own reading of the instruction as `GoalAtom`s. It is the leg's goal when phase planning is off. |

Labels may change between perception passes. tandem rebinds its plan to the new labels.

### `PlanResult`

Every field must be JSON-safe.

| field | meaning |
|---|---|
| `ok`, `failure_reason` | Whether it planned, and why not. |
| `planning_seconds` | How long planning took. |
| `plan_handle` | An opaque id that tandem passes back to `execute`. |
| `task_plan` | The operators with object arguments, e.g. `("Pick(bread)", "Place(bread, plate)")`. It goes into `hitl.json` and is never parsed. |
| `artifacts` | Role → path of each file written. |
| `skeleton`, `skeleton_reused` | Used only with `supports_skeleton_reuse`. |

## Capabilities

Capabilities are everything phase planning knows about your planner. They are checked when the class is
defined. [A minimal planner](#a-minimal-planner) shows the required fields. A planner usually adds a few
more, as the scaffold does:

```python
CAPABILITIES = Capabilities(
    ...,                               # the required fields, as in the minimal planner
    exclusive_arguments={"On": 0},     # an object rests on one thing at a time
    moved_arguments={"On": 0},         # On's first argument is the object that moves
    robot_operators=("Place(?obj: movable, ?place: surface)",),
    one_pick_per_object=True,
    supports_return_home=True,         # only once plan() honours return_home=False
)
```

When a proposed phase breaks one of these rules, tandem sends it back to the model to repair ("repaired"
below).

| field | meaning |
|---|---|
| `name` | Must equal `info.name`. |
| `goal_predicates` | Name → `Predicate(name, (Parameter(name, type), …))`, most important first. Robot phases can use only these. |
| `robot_description` | Required. One abstract sentence saying what the robot does. |
| `goal_predicate_wire_names` | How `plan(goal=)` spells each predicate, e.g. `{"On": "on"}`. Unlisted predicates are ones the planner supplies itself: they are dropped from goals, and a robot phase with only those is repaired. |
| `achievable_predicates` | What the planner's operators can make true, including every goal predicate. A robot phase asking for more is repaired. |
| `reserved_predicate_names` | Names the model may not invent, including every goal predicate. |
| `movable_type`, `surface_type` | The only two object types, and they must differ. Every goal-predicate and operator parameter uses one of them. |
| `moved_arguments` | Predicate → position of the moved object (`{"On": 0}`), which must be a `movable_type` parameter. Used for `movables=`, conjoining and wasted-move warnings. |
| `exclusive_arguments` | Predicate → the argument that can appear in only one atom at a time (`{"On": 0}`). It gives a free delete effect. A phase with two atoms in one slot is repaired. |
| `predicate_descriptions` | Templates shown to the operator and the camera check, e.g. `"{0} is resting on top of {1}"`. Use `{0}`, `{1}`, … within the arity, and `{{` for a literal brace. |
| `checkable_predicates` | Goal predicates a camera can judge from one photo. Invented predicates are always checkable. |
| `robot_operators` | Operator signatures, for the record only, e.g. `Pick(?obj: movable)`. They may use only the declared types. |
| `prompt_fragments` | Prompt paragraphs by slot: `placement_semantics`, `precondition_vocabulary`, `delete_effect_example`, `work_division`, `intermediate_state_example`, `robot_phase_rules`. Missing slots get generic text. |
| `one_pick_per_object` | A plan picks each object at most once. Defaults to `True`, the safe choice. |
| `initial_state_is_clean` | Every goal starts from the same clean state, which allows conjoining. Defaults to `False`. Set it only if the solver guarantees it. |
| `supports_movable_restriction` | `plan(movables=)` is honoured: only those objects are picked and the rest are obstacles. A goal that moves another object returns `ok=False`. Needs `moved_arguments`. |
| `supports_return_home` | `plan(return_home=False)` is honoured: the arm ends where the last operation leaves it. tandem uses this on every leg but the task's last. |
| `supports_cooperative_stop` | `execute` polls `should_stop` between steps. Without it, a preempt aborts the trial once the leg ends. |
| `supports_skeleton_reuse` | `plan` can reuse a previous `PlanResult.skeleton`. |

tandem passes `movables`, `return_home` and `should_stop` only when the matching capability is declared.
So `plan` may leave `movables` or `return_home` out of its signature if it doesn't declare them.

**Conjoining** (`hitl.conjoin_robot_phases`) plans consecutive robot phases as one goal from one
perception pass. The atoms are sorted, so the proposal's order between them is lost. It needs
`initial_state_is_clean`. It stops before a phase that moves an object again (with `one_pick_per_object`)
or refills an `exclusive_arguments` slot.

## The Planner base class

`tandem.planners.Planner` is an abstract base class, and it is also its own factory: the registry holds the
class itself.

| class attribute | required | what it is |
|---|---|---|
| `info` | yes | `PlannerInfo(name, display_name, summary, homepage, requires, sources)`, shown by `tandem planners list` and `info`. `requires` is free text. `sources` defaults to the recipe's. |
| `CAPABILITIES` | yes | See [Capabilities](#capabilities). |
| `recipe` | no | A [`RuntimeRecipe`](#a-runtime-recipe). `None` means pure Python. |
| `OPTIONS` | no | The task's settings: each key of a profile's `planner.options`, with a one-line description. |
| `RIG_OPTIONS` | no | Settings for this workstation, shared by all profiles: each key of `planners.<name>` in [rig.yml](CONFIGURATION.md#the-rig), with a one-line description. A key can't be in both. |

A bad declaration raises one `TandemError` at import that lists every problem. For a shared base class,
pass `abstract=True` (`class MyBase(Planner, abstract=True)`). It is not checked and can't be registered.

What you get without writing it:

| member | default |
|---|---|
| `warm`, `close`, `home`, `release_hardware`, `reacquire_hardware` | Do nothing. |
| `require_ready` | Checks that the recipe's runtime is installed. |
| `capture_frame`, `move_to_joints` | Raise `UnsupportedVerb`. A session won't start without `capture_frame` when phase planning is on with `hitl.check_human_effects`, `check_human_preconditions` or `check_tamp_effects`. |
| `create(ctx)` | Runs `validate_options(ctx.options)` and `validate_rig_options(ctx.rig_options)`, then `cls(ctx)`. |
| `runtime_env(*, rig, settings=None)` | Returns `{}`. It is what `tandem runtime run` and `shell` add to the environment of the planner's own scripts. [TiPToP](https://github.com/SamratSahoo/tiptop/tree/TANDEM)'s writes a `tiptop.yml` from the rig and points `$TIPTOP_CONFIG` at it. |
| `replay(rollout_dir, *, settings=None)` | Raises `UnsupportedVerb`, so `tandem traj open` isn't available. |
| `services(settings=None)` | Not defined, so the planner runs no helper servers. Define it to return servers such as TiPToP's [M2T2](https://github.com/SamratSahoo/M2T2/tree/TANDEM) and [FoundationStereo](https://github.com/SamratSahoo/FoundationStereo/tree/TANDEM), each with `name`, `title`, `runtime(settings)` (a `RecipeRuntime`), `url()`, `local()`, `healthy()`, `started_pid()`, `log_path`, `start()` and `stop()`. `tandem init` builds their runtimes and `tandem servers` manages them; the planner starts them for a session itself (TiPToP's backend does it in `warm` and `perceive`). |
| `runtime(settings)`, `runtime_root(settings)` | The runtime at `~/.local/share/tandem/runtimes/<name>` (or `$TANDEM_RUNTIMES_DIR`). `None` without a recipe. |

Inside a planner you can use:

| attribute | what it is |
|---|---|
| `self.ctx` | The session's `BackendContext`: `profile`, `session_dir`, `output_dir`, `execute`, `record`, `on_log`, `options`, `settings`, `session_id`, `task`, `events_file`, `runtime_dir`, `rig`, `rig_options`. |
| `self.options` | The validated task settings. |
| `self.rig_options` | The validated machine settings. |
| `self.rig` | The rig: `robot.host`, `robot.type`, `cameras`, `calibration_file()`. |
| `self.log(text)` | Writes a line to the session log. |

Any `tandem.planners.base.BackendFactory` also works in place of a `Planner` subclass. It needs `info`,
`capabilities()`, `create(ctx)` and `runtime(settings)`, plus any optional hooks. TiPToP is built this way.

## Sidecars

Use a sidecar when the planner needs torch, CUDA kernels, a camera SDK or a robot client. The planner then
runs as a script in its own interpreter, and tandem talks to it over JSON lines. `tandem planners new NAME
--sidecar` generates both halves.

The tandem half declares the planner and says which script to launch:

```python
# tandem_armsim/planner.py -- runs in tandem's process; keep it light
from tandem.planners import PlannerInfo, SidecarPlanner

class ArmSim(SidecarPlanner):
    info = PlannerInfo(name="armsim", display_name="ArmSim", summary="...")
    CAPABILITIES = CAPABILITIES      # as for any planner
    SIDECAR = "sidecar.py"           # next to this module
    recipe = None                    # its runtime; None runs the script with tandem's interpreter
    TIMEOUTS = {"warm": 600.0}       # seconds, overriding the defaults below

    def warm_args(self):
        # The sidecar can't read the rig, so send it what it needs.
        return {**super().warm_args(), "host": self.rig.robot.host, **self.rig_options}
```

The sidecar half is the planner itself, with one method per verb:

```python
# tandem_armsim/sidecar.py -- runs in the planner's own environment
from tandem_sidecar import log, serve

# isort: split


class World:
    def warm(self, *, output_dir=None, execute=True, record=True, **planner_specific):
        import torch                 # heavy imports go here, not at the top
        log("armsim: warm")
        return {}

    def perceive(self, *, task_hint, save_dir, reset_arm=True, open_gripper=False):
        return {"scene_id": "s1", "object_labels": ["block", "tray"], "table_label": "table",
                "surface_labels": ["tray"], "rgb_path": f"{save_dir}/perception_rgb.png"}

    def plan(self, *, scene_id, goal, surfaces, save_dir, movables=None, return_home=True,
             reuse_skeleton=None):
        # goal looks like [{"predicate": "at", "args": ["block", "tray"]}]
        return {"ok": True, "planning_seconds": 0.1, "plan_handle": "p1",
                "task_plan": ["Move(block, tray)"]}

    def execute(self, *, plan_handle, leg, save_dir):
        # record the leg and write _meta.json (see the recording contract)
        return {"ok": True, "n_frames": 0, "rollout_dir": save_dir}

    def capture_frame(self, *, camera="external"):
        return {"path": "/tmp/frame.png"}


if __name__ == "__main__":
    raise SystemExit(serve(World()))
```

Rules for the script:

- **Don't print to stdout.** tandem uses stdout to talk to the sidecar. Importing `tandem_sidecar` redirects
  stdout to stderr, which goes to the session log, so import it first and put `# isort: split` after it.
- **Import heavy libraries inside `warm()`.** Then a library that fails to import shows up as a failed
  warm-up with its error, instead of a sidecar that never starts. The conformance kit checks the import order.
- **Import only `tandem_sidecar` from tandem.** It is one standard-library file (Python 3.8+), and tandem puts
  it on the script's `PYTHONPATH`.
- **Append to `PYTHONPATH`; don't replace it.** If your [pixi](https://pixi.sh) environment sets `PYTHONPATH` (for example in
  `[activation.env]`), the sidecar dies with `No module named 'tandem_sidecar'`. To run a sidecar by hand,
  put tandem's `planners/sidecar_kit` directory on `PYTHONPATH` yourself.

Each method takes keyword arguments and returns a JSON-safe dict. `capture_frame` returns `{"path": ...}`.
A verb with no method gets `Planner`'s default. A sidecar missing `perceive`, `plan` or `execute` is refused
at `warm`. If a method raises, tandem reports `"<verb> failed -- <Type>: <message>"` and puts the traceback
in the session log. The wire protocol is documented in the docstring of
`src/tandem/planners/sidecar_kit/tandem_sidecar.py`.

### What tandem does with a sidecar

| behaviour | detail |
|---|---|
| Launch | Runs the runtime's `python` with `pixi run` (or tandem's interpreter without a runtime), from the runtime's working directory, in its own process group. |
| Optional arguments | `movables`, `return_home` and `reuse_skeleton` are sent only when declared. Passing an undeclared one is an error. |
| Output | Logs and stderr go to the session log. Events go to the events file (through `on_event`). |
| Crash or timeout | tandem sends SIGTERM, then SIGKILL, to the process group. The trial ends at that verb's stage, and the next `warm()` restarts the sidecar. |
| Cooperative stop | If declared, tandem polls `should_stop` and signals the sidecar through the file named in `TANDEM_SIDECAR_STOP_FILE`. |
| `close` | Asks the sidecar to quit, then ends its process group, helpers included. |

### Default timeouts

| verb | seconds |
|---|---|
| `warm` | 900 |
| `perceive` | 300 |
| `plan` | 900 |
| `execute` | 1800 |
| `capture_frame`, `home`, `release_hardware`, `reacquire_hardware` | 180 |

### `SidecarPlanner` methods you can override

| method | what it controls |
|---|---|
| `launch_command` | The command that starts the sidecar. |
| `launch_cwd` | Its working directory. |
| `launch_env` | Its environment. |
| `warm_args` | What `warm` sends. Defaults to `output_dir`, `execute` and `record`. |
| `on_event` | What happens to each event the sidecar sends. |
| `call(verb, **args)` | Not an override: calls a sidecar-only verb. |

### The `tandem_sidecar` kit

| function | what it does |
|---|---|
| `serve(handlers, verbs=None, on_exit=None)` | Answers requests until tandem says quit or closes stdin, then calls `on_exit` or the `close` handler. `handlers` is an object with one method per verb, or a verb → callable mapping. |
| `log(message, level="info")` | Writes to the session log. Safe from any thread. |
| `event(name, **fields)` | Sends an event. The fields can't be named `id`, `log` or `event`. |
| `should_stop()` | True once tandem asks for a cooperative stop. |

## A runtime recipe

A planner that needs more than pip declares its runtime as data. `tandem planners install NAME` builds it,
`tandem planners list` shows whether it is current, and `tandem planners remove NAME` deletes it. Here is a
recipe with one pinned source, a pixi environment and one build step:

```python
from tandem.planners import BuildStep, PixiEnvironment, RuntimeRecipe, Source, SourcePin

RECIPE = RuntimeRecipe(
    planner="myplanner",                     # must equal info.name
    title="MyPlanner",
    sources=(
        Source(
            SourcePin("mysolver", "https://github.com/me/mysolver.git",
                      "4db8f92c1e0a4b1f9d2e3c4b5a6978877665544a", ref="main"),
            marker="pixi.toml",              # a file that exists only in a complete tree
        ),
    ),
    environment=PixiEnvironment(manifest="mysolver/pixi.toml"),
    steps=(
        BuildStep(name="kernels", task="build-kernels", produces=("mysolver/build/*.so",),
                  description="compiling the CUDA kernels"),
    ),
)

class MyPlanner(Planner):
    recipe = RECIPE
    ...
```

TiPToP's full recipe is `src/tandem/planners/tiptop/recipe.py`. What it installs is described in
[The planner runtime](CONFIGURATION.md#the-planner-runtime).

| part | fields |
|---|---|
| `RuntimeRecipe` | `planner` (equals `info.name`), `title`, `sources`, `environment`, `steps`, `assets`, `notes`. |
| `Source` | `SourcePin(name, url, commit, ref=)`; `trim`: paths deleted after fetching; `patches`: applied in order, and a failure stops the install; `marker`: a path present only in a complete tree; `persistent`: directories written at run time that are kept across tree replacements. |
| `PixiEnvironment` | `manifest`: the planner's pixi manifest and lock, inside a source; `home`: where the environment lives, default `env`, outside every tree; `env`: environment variables. |
| `BuildStep` | `name`; `task` (a task in the manifest); `env`; `produces`: globs that exist once it has run; `description`; `optional`: the runtime works without it, so a failure doesn't fail the install and shows as a note; `requires`: absolute paths it needs, such as an SDK's installer, and it is skipped until they exist; `missing`: what won't work meanwhile, and the fix. TiPToP's ZED step uses the last three. |
| `Asset` | `(source, dest)`: a file from your package copied into the runtime. |

How it behaves:

- **Pins** are full 40-character commits. `ref` (the branch) is shown by `tandem planners info` and
  `tandem runtime status`, but never compared.
- **Fetching** is a shallow `git fetch` and `git archive` of the commit. If that fails it tries the `ref`
  branch, then GitHub's archive without git. The result is always checked against the pin.
- **Layout:** one directory per source, plus `env/` (the environment, linked as `.pixi` beside the
  manifest), `cache/` (the `persistent` directories) and `.tandem-runtime.json` (what is installed).
- **Moving a pin** replaces that tree without re-solving the environment. `persistent` directories survive.
  The installer never deletes a git checkout, and replaces unlisted trees only where `.tandem-runtime.json`
  exists.
- **Status** compares `.tandem-runtime.json` with the recipe. A moved pin shows `outdated`, with the command
  to rebuild.

Other details:

- **Placeholders** work in `env` values: `{root}`, `{source:NAME}` (a tree's path), `{commit:NAME}`, and
  `{version:NAME}`. `{version:NAME}` is the commit as a PEP 440 version (`0.0.0+g4db8f92`), for packages
  that take their version from git.
- **Offline installs** use `--sources DIR`, `$TANDEM_PLANNER_SOURCES` or `tandem planners bundle`
  ([Offline install](CONFIGURATION.md#offline-install)).
- **[ffmpeg](https://ffmpeg.org):** merging legs uses the environment's `bin/ffmpeg`, or the one on `PATH`.

## Options and doctor rows

A planner has two kinds of settings. Task settings live in each profile's `planner.options` and are declared
in `OPTIONS`. Machine settings, such as a server's address or a robot's ports, live in rig.yml under
`planners.<name>` and are declared in `RIG_OPTIONS`.

```python
from pydantic import BaseModel
from tandem.core.errors import TandemError

class _Options(BaseModel, extra="forbid"):
    speed: float = 0.5
    scene_file: str | None = None

class MyPlanner(Planner):
    OPTIONS = {"speed": "arm speed, 0-1", "scene_file": "the scene to load"}
    RIG_OPTIONS = {"server_url": "the solver server's address"}

    @classmethod
    def validate_options(cls, options):
        opts = _Options(**(options or {}))           # a ValidationError is reported by key
        if opts.scene_file is None:
            raise TandemError("planner.options.scene_file is required")
        return opts.model_dump(mode="json")          # plain data only
```

People set them like this:

```bash
tandem planners use myplanner --option scene_file=kitchen.yml   # a task setting
tandem rig set planners.myplanner.server_url http://HOST:9000 # a machine setting
```

tandem calls `validate_options(options)` when a profile loads, and `validate_rig_options(options)` when the
rig is read. The defaults refuse any key that isn't declared, suggest the nearest one, and say which file a
misplaced key belongs in. If you override either one, it must:

- Accept its own output, because the result is validated again on every read.
- Return plain data: string-keyed mappings, lists, strings, numbers, booleans and `None`. Use
  `model_dump(mode="json")`, never `Path`, `Enum` or numpy values.
- Raise `TandemError` or `ValueError` (pydantic's `ValidationError` counts). tandem reports it under
  `planner.options.` or `planners.<name>.`.

To make a setting required, refuse `{}` with a `TandemError` naming it. Put a machine setting in
`validate_rig_options` and a task setting in `validate_options`. `tandem init` fills a planner's machine
settings with whatever `validate_rig_options({})` returns.

### What people see

`describe_options(profile, *, settings=None)` returns an `OptionsView` with `summary`, `sections`,
`receives`, `receives_note` and `warnings`. `tandem profile show`, the web editor and the session header show
it. By default it lists each option as set. `tandem profile show NAME --planner` prints `receives`, which is
exactly what the planner gets.

### Doctor rows

`doctor_checks(profile, *, settings=None, probe_hardware=True)` adds rows to `tandem doctor`. It returns
`tandem.core.probe.Check(name, state, detail, hint, group)` rows, and by default returns none.

- `state` is `probe.OK`, `WARN`, `FAIL` or `SKIP`.
- With `profile=None` (the preflight of `tandem init`), check only the machine.
- With `probe_hardware=False` (`--no-hardware`), touch no network or bus.
- A FAIL stops a session. It also stops `tandem init` before it builds the runtime (interactively, it asks).
- tandem already reports the runtime, so don't repeat it.

## Registering it

The scaffold registers the planner with an entry point in `pyproject.toml`:

```toml
[project.entry-points."tandem.planners"]
arm = "tandem_arm.planner:ArmPlanner"     # a Planner subclass, or any BackendFactory
```

Then select it with `tandem planners use arm`, or `planner: {backend: arm}` in a profile.
`tandem planners default arm` makes it the planner new profiles get ([Commands](USAGE.md#commands)).

tandem looks up a name in this order, and the first match wins:

1. `tandem.register_backend(name, factory)` (alias `register_planner`), called at runtime. `factory` may be
   a lazily imported `"module:attribute"` string. A taken name is an error unless you pass `replace=True`.
2. The built-in planner, TiPToP.
3. The `tandem.planners` entry point.

`tandem planners list`, `info` and profile loading all import plugins. Keep the entry-point module light,
and import torch and robot clients in `warm()` or a sidecar.

When something goes wrong:

- A plugin that fails to import is listed as `broken`, with its error. Everything else still works, and a
  profile naming it still loads, with its options unchecked.
- A plugin shadowed by a registered or built-in planner is listed with the reason.
- Two installed packages claiming the same name is an error.

## The recording contract

`execute` records one leg of trial `leg.trajectory_id`. tandem's merge orders a trial's legs by their
recording windows and joins them into one episode. When `leg.record` is set, `save_dir` must hold three
things: `_meta.json`, `robot_state.npz` and the camera clips.

```
save_dir/
├── _meta.json
├── robot_state.npz
├── external_cam.mp4     external_cam_2.mp4 and hand_cam.mp4 too, if recorded
└── ...                  anything else is the planner's own
```

**`_meta.json`:**

```json
{
  "trajectory_id": "3f9c0a1b2d4e5f60", "segment_source": "tamp",
  "instruction": "place the bread inside the box",
  "phase_index": 0, "n_phases": 2, "phase_description": "put the bread in the box",
  "record_start": 1790270001.2, "record_stop": 1790270037.4, "fps": 15,
  "cameras": {"exterior_image_1_left": "external_cam.mp4", "wrist_image_left": "hand_cam.mp4"}
}
```

| key | value |
|---|---|
| `trajectory_id`, `segment_source` | Copied from `leg` (`"tamp"` for a planner leg). The merge finds the leg by them. |
| `instruction` | `leg.instruction`: the whole task, which becomes the dataset's language label. |
| `phase_index`, `n_phases`, `phase_description` | Copied from `leg` when `leg.phase_index` is not `None`. |
| `record_start`, `record_stop` | Epoch seconds. Legs are ordered by them. |
| `fps` | Frames per second. |
| `cameras` | Dataset key → clip file, e.g. `{"exterior_image_1_left": "external_cam.mp4"}`. |
| `plan_file` (optional) | The saved plan's bare file name (default `tiptop_plan.json`), for `tandem traj show` and the web UI. |

**`robot_state.npz`** holds each array in `tandem.core.merge.STATE_KEYS`, one row per frame, and optionally
`action_joint_velocity` (`[F,7]`). Nothing else may be in it. `cmd_*` arrays are commanded values and the
rest are measured.

| array | shape |
|---|---|
| `joint_position`, `cmd_joint_position`, `cmd_joint_velocity` | `[F,7]` |
| `gripper_position` | `[F]`, in `[0,1]` |
| `cmd_gripper` | `[F]`, binary. [The export](USAGE.md#exporting) skips an episode where it isn't. |
| `frame_time` | `[F]`, wall-clock time, float64 |

Record measured arrays from the robot, not copies of the commands. Otherwise a policy trained on the data
learns to echo its commands.

**Camera clips:** every clip that `cameras` names must exist, and there must be at least one even if
`cameras` names none. Name each one `external_cam.mp4`, `external_cam_2.mp4` or `hand_cam.mp4`, whatever its
dataset key. Only those names are merged, viewed and exported. Only cameras that every leg recorded are
joined.

`tandem.core.trajectories.is_complete(leg_dir)` checks a leg against this contract. Any other files belong to
the planner, and the merge copies the first planner leg's into the episode. The merged layout is in
[Episode layout](DATA.md#episode-layout).

Edge cases:

- Write `_meta.json` even when execution fails part-way or records nothing (`n_frames=0`, as the scaffold
  does). Otherwise the leg becomes an episode of its own. Always return `rollout_dir` (usually `save_dir`)
  and `n_frames`.
- Return `ExecuteResult(stopped_early=True)` when you honoured a cooperative stop (a preempt or a session
  stop). The leg then never advances the plan, whatever `ok` says. After a preempt the trial is filed as
  aborted. A stop nobody asked for is a `tamp_execution` failure.

## The conformance kit

`tandem.planners.testing` runs the whole protocol against your planner with no GPU, robot or session. Your
package's test subclasses it:

```python
from tandem.planners.testing import PlannerConformance
from tandem_arm.planner import ArmPlanner

class TestArmPlanner(PlannerConformance):
    planner = ArmPlanner          # a Planner subclass, a BackendFactory or a registered name
    records_legs = True           # False checks only the _meta.json stamp
    options = {}                  # its planner.options
    rig_options = {}              # its machine settings (rig.yml planners.<name>)
    task_hint = "put one thing where it belongs"
```

It tests the declarations, options, machine settings, doctor rows, sidecar script, lifecycle and every verb.
With `records_legs`, it also tests the recording. It skips anything the planner doesn't declare or ship.

The backend is always built for a stand-in machine (`stand_in_rig`): the default arm at 172.16.0.2, a hand
camera and an external camera, and no extrinsics. It never reads your rig.yml. So a planner that
reads `self.rig.robot.host` runs, and the tests don't depend on the machine.

Two switches hold the planner to what phase planning needs. Both are on by default, and the scaffold passes
both with stand-in images. Turn one off only for a planner that is never run with phase planning.

| switch | requires |
|---|---|
| `phase_planning = True` | Every scene's `rgb_path` opens as an image. |
| `verifies_human_phases = True` | `capture_frame(camera="external")` returns an image. |

Hooks to override:

| hook | use it to |
|---|---|
| `rig(tmp_path)` | Test against a different stand-in rig. |
| `goal(scene, caps)` | Supply a plannable goal when the kit can't guess one. |
| `make_backend(tmp_path)` | Build the backend differently. |

Each check is also a plain function that raises `ConformanceError`: `check_declarations`, `check_protocol`,
`check_scene`, `check_plan_result`, `check_leg` and `check_sidecar_script`.
