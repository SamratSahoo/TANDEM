# Adding a planner

A planner is a Python package that declares its goal language and implements `perceive`, `plan` and
`execute`. tandem does the rest: phases, hand-offs, camera checks and merging legs. A planner is only
ever asked to achieve one goal in one scene and record what it did as one [leg](README.md#terms).

On this page: [Quick start](#quick-start) · [Protocol](#the-protocol) · [Capabilities](#capabilities) ·
[Planner base class](#the-planner-base-class) · [Sidecars](#sidecars) · [Runtime recipe](#a-runtime-recipe) ·
[Options, presets, doctor](#options-presets-and-doctor-rows) · [Registering](#registering-it) ·
[Recording contract](#the-recording-contract) · [Conformance kit](#the-conformance-kit) ·
[Before collecting](#before-collecting)

## Quick start

1. Generate a package:

   ```bash
   tandem planners new shelfbot              # runs in tandem's own process
   tandem planners new armsim --sidecar      # runs as a script in an environment of its own
   ```

   It is written to `./tandem-NAME` (`--dir` to choose). As generated it plans a stand-in world (one
   block, one tray) in memory and passes the [conformance kit](#the-conformance-kit):

   ```
   tandem-shelfbot/
   ├── pyproject.toml                  the package and its tandem.planners entry point
   ├── README.md                       the TODOs, in order
   ├── src/tandem_shelfbot/
   │   ├── __init__.py
   │   ├── planner.py                  CAPABILITIES and the ShelfbotPlanner class
   │   └── sidecar.py                  (--sidecar only) the script the planner's environment runs
   └── tests/test_conformance.py       the conformance kit, run against it
   ```

2. Install it into the environment tandem runs in. tandem reads only its own environment's entry
   points, and the package depends on `tandem-tamp`, which is not on PyPI. `planners new` prints the
   right command; by install method:

   ```bash
   cd tandem-shelfbot
   pipx inject tandem-tamp --editable . pytest                   # tandem installed with pipx
   uv tool install --reinstall git+https://github.com/SamratSahoo/tandem.git \
       --with-editable . --with pytest                           # tandem installed with uv tool
   pip install -e ".[test]"                                      # tandem in an active virtualenv
   ```

3. Check it, using that environment's Python (`planners new` prints its path; a `pytest` on your
   `PATH` may belong to an environment without tandem):

   ```bash
   python -m pytest                # the conformance kit: green as generated
   tandem planners list            # shelfbot is listed as "no runtime needed"
   tandem planners info shelfbot   # its goal language, as phase planning sees it
   tandem planners use shelfbot    # the active profile now plans with it
   ```

   What `planners use` changes in the profile: [CONFIGURATION.md](CONFIGURATION.md#planner-settings).

4. Replace each `TODO` in the order the generated README lists them, keeping `pytest` green:
   1. The goal language: [`CAPABILITIES`](#capabilities).
   2. `perceive`: object labels, the table, which objects are surfaces, a scene id, an image.
   3. `plan`: one goal planned, or `ok=False` with the reason.
   4. `execute`: run the plan and record the leg; then set `records_legs = True` in the test file.
   5. `capture_frame`: a current frame from the camera asked for.
   6. The `supports_*` switches, each turned on once honoured, and `initial_state_is_clean` /
      `one_pick_per_object` set to what the solver assumes (the scaffold starts at the safe values).
   7. A [runtime recipe](#a-runtime-recipe), if the planner needs more than pip.
5. Work through [Before collecting](#before-collecting).

### Names

A planner's name is lowercase letters, digits, `_` and `-`, starting with a letter
(`src/tandem/core/names.py`). Human executors use the same rule.

## The protocol

tandem drives a planner through `tandem.planners.base.TampBackend`. Don't implement it by hand:
subclass [`Planner`](#the-planner-base-class), which supplies every verb except `perceive`, `plan` and
`execute`. A session calls:

```
factory.create(ctx) → require_ready() → warm()
    → ( perceive() → plan() → execute() )*       robot legs
    → release_hardware() … reacquire_hardware()   around every human leg
    → home() → close()                            when the session stops
```

| verb | when | contract |
|---|---|---|
| `capabilities()` | Any time, before anything is built | Declared, cheap, static. |
| `require_ready()` | Before `warm` | Raise `RuntimeNotReady` naming what is missing. |
| `warm()` | Once per session, and again after a verb raised | Open cameras, connect the robot, build solvers. On a planner that is already warm, do nothing. |
| `perceive(*, task_hint, save_dir, reset_arm=True, open_gripper=False)` | Before every robot leg | `task_hint` is the whole instruction and steers detection only. `reset_arm` is True only for an attempt's first leg: never park an arm a person just handed back. `open_gripper` is True only for the first leg after a human phase; open the hand and move nothing else. Returns a `SceneView`. |
| `plan(scene_id, goal, *, surfaces=frozenset(), movables=None, return_home=True, save_dir, reuse_skeleton=None)` | After `perceive` | `goal` is a list of `GoalAtom(predicate, args)` in the planner's wire spelling. `surfaces` fixes which objects are surfaces for the whole task. `movables` and `return_home` are passed only when [declared](#capabilities). A goal it cannot plan is `ok=False` with `failure_reason`, not an exception. Returns a `PlanResult`. |
| `execute(plan_handle, leg, *, save_dir, should_stop=None)` | After a plan succeeds | Run it and record it as one leg of `leg.trajectory_id` ([recording contract](#the-recording-contract)). Returns an `ExecuteResult`; `ok=False` ends the trial at `tamp_execution`. |
| `capture_frame(*, camera="external")` | For every camera check of a human phase or a robot leg's effects | Write one RGB frame to a file and return its path. `camera` is `hitl.verification_camera`. A robot leg's preconditions are checked on that pass's `rgb_path` instead. |
| `release_hardware()` / `reacquire_hardware()` | Around every human leg | Release blocks until the robot and every camera are really free. Reacquire takes them back from wherever the person left the arm. |
| `home()` | When the session stops | Park the arm. Don't open the gripper: it may be holding something. |
| `close()` | Last | Release everything. Safe twice, and safe on a planner that never warmed. |

A verb that raises ends the trial at its stage (`tamp_planning` for `perceive` and `plan`,
`tamp_execution` for `execute`), and the session calls `warm()` again before the next task.

**`SceneView`**, what `perceive` returns:

| field | meaning |
|---|---|
| `object_labels` | Every object seen, not including the table. |
| `table_label` | The support surface (default `table`). |
| `surface_labels` | The objects that are surfaces. |
| `scene_id` | Opaque; handed back to `plan`. |
| `rgb_path` | An image of what was seen. With phase planning on, the task is split into phases from it, and a pass that must propose the plan (the first, or the first after a `replan`) without one ends the trial at `invention`. |
| `detected_goal` | The planner's own reading of the instruction, as `GoalAtom`s. With phase planning off, this is the leg's goal. |

Labels may differ from pass to pass; tandem rebinds its plan to them.

**`PlanResult`**, what `plan` returns. It must be JSON-safe: a sidecar sends it over the wire and it is
written into the trial's record.

| field | meaning |
|---|---|
| `ok`, `failure_reason` | Whether it planned, and why not. |
| `planning_seconds` | Time spent planning. |
| `plan_handle` | Opaque; handed back to `execute`. |
| `task_plan` | The operators the plan runs, object arguments only, e.g. `("Pick(bread)", "Place(bread, plate)")`. Written into `hitl.json`; nothing parses it. |
| `artifacts` | Role → path of files the planner wrote. |
| `skeleton`, `skeleton_reused` | Only with `supports_skeleton_reuse`. |

## Capabilities

`Capabilities` is everything phase planning reads about a planner. `Planner` checks the declaration
when the class is defined. TiPToP's is `src/tandem/planners/tiptop/capabilities.py`; `tests/toy_planner.py`
is a second example (`InBin(?obj: item, ?bin: container)`, no table, no exclusivity).

| field | meaning | TiPToP |
|---|---|---|
| `name` | Must equal `info.name`. | `"tiptop"` |
| `goal_predicates` | Name → `Predicate(name, (Parameter(name, type), …))`: the goal language shown to the model. A robot phase may use nothing else. Shown in declaration order, so put the most important first. | `On(?obj: movable, ?surface: surface)`, `Holding(?obj: movable)`, `HandEmpty()` |
| `robot_description` | One abstract sentence of what the robot does, read by the model instead of operator signatures. Required. | `"pick an object up and place it on a surface"` |
| `goal_predicate_wire_names` | How each goal predicate is spelled in `plan(goal=)`. A predicate left out is one the planner supplies itself: it is dropped from goals, and a robot phase stating nothing else is sent back for repair. | `{"On": "on", "Holding": "holding"}` |
| `achievable_predicates` | Everything some operator can make true. A robot phase asking for anything else is refused and repaired in the proposal's repair loop, before any robot leg is planned. Must include every goal predicate. | cuTAMP's add effects, plus what a fresh scene holds |
| `reserved_predicate_names` | Names the model may not invent. Must include every goal predicate. | all 19 cuTAMP fluents |
| `movable_type`, `surface_type` | The only two object types; must differ. tandem types every perceived object as one of them, so every parameter of a goal predicate and of a robot operator must use one. | `movable`, `surface` |
| `moved_arguments` | Predicate → position of the argument naming the object a robot phase moves. Must point at a `movable_type` parameter. Used for `movables=`, for conjoining, and for the wasted-move warning. | `{"On": 0, "Holding": 0}` |
| `exclusive_arguments` | Predicate → the argument that can hold in only one atom at a time. The contract check reads it as a free delete effect, and a phase asking for two atoms in one slot is sent back for repair. | `{"On": 0}` |
| `predicate_descriptions` | `{0}`-templates saying what each goal predicate means, for the operator and the camera. Only `{0}`, `{1}`, … within the arity; write a literal brace as `{{`. | `"{0} is resting on top of {1}"`, … |
| `checkable_predicates` | Goal predicates a camera can judge from one photo. Invented predicates are always checkable. | `{"On"}` (the gripper is usually out of shot) |
| `robot_operators` | One signature per operator, for the record only: `Pick(?obj: movable)` or `Pick(obj: movable)`, any spacing, written to the record as `Pick(obj: movable)`. Types must be the declared ones. | `Pick(?obj: movable)`, `Place(?obj: movable, ?surface: surface)` |
| `prompt_fragments` | Planner-specific paragraphs of the phase-planning prompt, by slot: `placement_semantics`, `precondition_vocabulary`, `delete_effect_example`, `work_division`, `intermediate_state_example`, `robot_phase_rules`. A missing slot gets a generic paragraph, so `{}` is valid. | all six, the paper's Appendix B wording |
| `one_pick_per_object` | One plan picks each object at most once. Default `True` (safe: it only keeps phases apart). | `True` (cuTAMP deletes `HasNotPickedUp`, so `On(toy, table)` and `On(toy, shelf)` in one plan can't both be met) |
| `initial_state_is_clean` | Every goal is planned from the same clean state, so consecutive robot phases may be conjoined. Default `False`: every robot phase is its own leg. Set it only when the solver really makes that promise. | `True` |
| `supports_movable_restriction` | `plan(movables=)` is honoured: only those objects may be picked and every other one is an obstacle. A goal moving anything else is `ok=False` naming it. Requires `moved_arguments`. | `True` |
| `supports_return_home` | `plan(return_home=False)` ends the leg where its last operation leaves the arm. tandem passes `False` for every leg except the task's last. | `True` |
| `supports_cooperative_stop` | `execute` polls `should_stop` at step boundaries. Without it, the leg runs to its end and a preempt then aborts the trial. | `False` |
| `supports_skeleton_reuse` | `plan` can reuse a previous `PlanResult.skeleton`. | `False` |

**Only what is declared is passed.** tandem passes `movables` only with `supports_movable_restriction`,
`return_home` only with `supports_return_home`, and `should_stop` only with `supports_cooperative_stop`.
A planner declaring neither of the first two may leave them out of its `plan` signature.

**Conjoining** (`hitl.conjoin_robot_phases`) plans consecutive robot phases as one goal, with one
perception pass; their atoms are sorted, and the proposal's order between them is dropped. It happens
only when `initial_state_is_clean` is `True`, and stops before a phase that moves an object already moved
in the run (with `one_pick_per_object`) or fills a slot of `exclusive_arguments` already filled.

## The Planner base class

`tandem.planners.Planner` (`src/tandem/planners/sdk.py`) is an abstract base class and also its own
factory: the registry uses the class, and an instance is the backend a session builds.

A declaration with a goal language of its own (the scaffold's `planner.py` is a complete, runnable one):

```python
from tandem.planners import Capabilities, Parameter, Planner, PlannerInfo, Predicate

IN_BIN = Predicate("InBin", (Parameter("obj", "item"), Parameter("bin", "container")))

class BinPlanner(Planner):
    info = PlannerInfo(name="bins", display_name="Bin sorter", summary="Drops items into bins.")
    CAPABILITIES = Capabilities(
        name="bins", goal_predicates={"InBin": IN_BIN}, robot_description="drop an item into a bin",
        goal_predicate_wire_names={"InBin": "in_bin"}, achievable_predicates=frozenset({"InBin"}),
        reserved_predicate_names=frozenset({"InBin"}), movable_type="item", surface_type="container",
        predicate_descriptions={"InBin": "{0} is inside {1}"}, checkable_predicates=frozenset({"InBin"}),
        moved_arguments={"InBin": 0}, one_pick_per_object=False,  # an item may be dropped twice in one plan
    )
    OPTIONS = {"bins": "the bin names, left to right"}

    def perceive(self, *, task_hint, save_dir, reset_arm=True, open_gripper=False): ...        # -> SceneView
    def plan(self, scene_id, goal, *, surfaces=frozenset(), save_dir, reuse_skeleton=None): ...  # -> PlanResult
    def execute(self, plan_handle, leg, *, save_dir, should_stop=None): ...                      # -> ExecuteResult
```

**Class attributes:**

| attribute | required | what it is |
|---|---|---|
| `info` | yes | `PlannerInfo(name, display_name, summary, homepage, requires, sources)`, shown by `tandem planners list` and `info`. `requires` is human-readable lines, never checked. `sources` is filled in from `recipe` when left empty. |
| `CAPABILITIES` | yes | [Above](#capabilities). |
| `recipe` | no | A [`RuntimeRecipe`](#a-runtime-recipe), when the planner needs more than pip. `None`: pure Python. |
| `OPTIONS` | no | `planner.options` key → one line saying what it does. Any other key is refused when a profile loads. |
| `importer` | no | A `ProfileImporter` (`source`, `find`, `configs`, `build`) for `--import-from`. `build` returns the profile, the extrinsics and notes; a note starting with `WARNING_NOTE` (`"warning: "`) is shown as a warning. |
| `presets_dir` | no | A directory of `<name>.yml` [presets](#options-presets-and-doctor-rows) for this planner's options. |

**Checked when the class is defined.** Every *must* in the tables above is checked, along with a wire
name or description for an undeclared predicate, a misspelt prompt slot and a recipe for another planner.
All problems are reported together, in one `TandemError` at import. A base class for other planners
passes `abstract=True` (`class MyBase(Planner, abstract=True)`) and is neither checked nor registrable.

**Defaults for everything else:**

| member | default |
|---|---|
| `warm`, `close`, `home`, `release_hardware`, `reacquire_hardware` | Do nothing (right for a planner that holds no hardware). |
| `require_ready` | Checks the recipe's runtime is installed. No recipe: nothing to check. |
| `capture_frame`, `move_to_joints` | Raise `UnsupportedVerb`. With phase planning on and `hitl.check_human_effects`, `check_human_preconditions` or `check_tamp_effects` on, a session refuses to start on a planner without `capture_frame`. |
| `create(ctx)` (classmethod) | `validate_options(ctx.options)`, then `cls(ctx)`. Override when construction needs more. |
| `validate_options(options)` | Refuses any key not in `OPTIONS`, suggesting the nearest. |
| `describe_options(profile, *, settings=None)` | `OptionsView.generic`: each option as set. |
| `doctor_checks(profile, *, settings=None, probe_hardware=True)` | `[]`. |
| `replay(rollout_dir, *, settings=None)` | Raises `UnsupportedVerb`: no viewer for `tandem traj open`. |
| `runtime(settings)` / `runtime_root(settings)` | A runtime at `~/.local/share/tandem/runtimes/<name>` (`$TANDEM_RUNTIMES_DIR` overrides). `None` without a recipe. |

Inside the class, `self.ctx` is the `BackendContext` (`profile`, `session_dir`, `output_dir`,
`execute`, `record`, `on_log`, `options`, `settings`, `session_id`, `task`, `events_file`,
`runtime_dir`), `self.options` is `planner.options` as validated, and `self.log(text)` writes a line
to the session log the operator watches.

A factory that isn't a `Planner` also works: any object with `info`, `capabilities()`, `create(ctx)`
and `runtime(settings)`, plus any of the optional hooks above (`tandem.planners.base.BackendFactory`).
TiPToP's is one (`src/tandem/planners/tiptop/factory.py`).

## Sidecars

A planner that needs torch, CUDA kernels, a camera SDK or a robot client can't run in tandem's
process. It runs as a **sidecar**: a script launched with the planner's own interpreter, answering
the protocol one JSON object per line. tandem's side is a `tandem.planners.SidecarPlanner` subclass
(`src/tandem/planners/sidecar.py`):

```python
class ArmPlanner(SidecarPlanner):
    info = PlannerInfo(name="arm", ...)
    CAPABILITIES = Capabilities(name="arm", ...)
    recipe = RECIPE                    # the environment the sidecar runs in; None: tandem's interpreter
    SIDECAR = "sidecar.py"             # relative to this module's directory; checked at class definition
    TIMEOUTS = {"warm": 600.0}         # overrides sidecar.DEFAULT_TIMEOUTS
```

| behaviour | detail |
|---|---|
| Launch | The runtime's `python` via `pixi run`, from the runtime's working directory; tandem's own interpreter when there is no runtime. The sidecar gets its own process group, and `tandem_sidecar` goes first on its `PYTHONPATH`. |
| Verbs | Every verb becomes a request, and each reply becomes tandem's type. A verb the sidecar doesn't list in its hello gets `Planner`'s default. A sidecar that doesn't answer `perceive`, `plan` and `execute` is refused at `warm`. |
| Declared-only arguments | `movables`, `return_home` and `reuse_skeleton` go on the wire only when declared. Passing one to a planner that didn't declare it is an error. |
| Output | Log lines and stderr go to the session log; events go to the session's events file (`on_event`). Pipes are decoded leniently and drained until they close. |
| Timeouts | Defaults: `warm` 900 s, `perceive` 300, `plan` 900, `execute` 1800, `capture_frame`, `home`, `release_hardware`, `reacquire_hardware` 180 each. A sidecar that doesn't answer in time gets SIGTERM, then SIGKILL, to its whole process group. A timeout or crash ends the trial at its stage, and the next `warm()` starts a fresh sidecar. |
| Cooperative stop | When declared, `should_stop` is polled in tandem and passed to the sidecar as a stop file named by `TANDEM_SIDECAR_STOP_FILE`. |
| `close` | Asks the sidecar to quit, then makes sure its whole process group is gone, including helpers it started. |

Override `launch_command`, `launch_cwd`, `launch_env`, `warm_args` (by default `output_dir`,
`execute`, `record`) or `on_event` when a default is wrong for your planner. `call(verb, **args)`
reaches a verb of the sidecar's own. TiPToP's backend is this class plus its launch details
(`src/tandem/planners/tiptop/backend.py`).

**The script** imports nothing from tandem, only `tandem_sidecar`
(`src/tandem/planners/sidecar_kit/tandem_sidecar.py`), a single standard-library file for Python 3.8+
that tandem puts on the script's path. Its shape (the scaffold's `sidecar.py` is a complete one):

```python
from tandem_sidecar import log, serve          # FIRST: it takes stdout for the protocol

# isort: split
# the planner's own imports below here, or better, inside warm()

class World:                                   # one method per verb, each returning a JSON-safe dict
    def warm(self, *, output_dir=None, execute=True, record=True, **planner_specific): ...
    def perceive(self, *, task_hint, save_dir, reset_arm=True, open_gripper=False): ...  # scene_id, object_labels, rgb_path, …
    def plan(self, *, scene_id, goal, surfaces, save_dir): ...                            # ok, plan_handle, task_plan, …
    def execute(self, *, plan_handle, leg, save_dir): ...                                 # ok, n_frames, rollout_dir, …
    def capture_frame(self, *, camera): ...                                               # path

if __name__ == "__main__":
    raise SystemExit(serve(World()))
```

| `tandem_sidecar` | what it does |
|---|---|
| `serve(handlers, verbs=None, on_exit=None)` | `handlers` is an object (every protocol verb it has a method for) or a mapping of verb → callable. Answers until tandem says quit or closes stdin, then calls `on_exit` or the `close` handler. |
| `log(message, level="info")` | One line in the session log. Safe from any thread. |
| `event(name, **fields)` | One event in the session's events file. `id`, `log` and `event` can't be field names. |
| `should_stop()` | True once tandem asks the execution in flight to stop. |
| a handler that raises | Reported to tandem as `"<verb> failed -- <Type>: <message>"`, with the traceback in the session log. |

The wire protocol, and what each verb is sent and returns, is in that file's docstring.

**stdout belongs to the protocol.** Importing `tandem_sidecar` takes the real stdout and points fd 1
at stderr, so a CUDA banner or a stray `print()` lands in the session log instead of corrupting a
reply. Import it before anything that might print; the conformance kit refuses a script that imports
a non-standard-library module first.

**`PYTHONPATH`:**

- tandem adds the kit's directory to the sidecar's `PYTHONPATH`. If your environment's activation
  *sets* `PYTHONPATH` (for example a pixi `[activation.env]`), the kit is lost and the sidecar dies at
  start with `ModuleNotFoundError: No module named 'tandem_sidecar'`. Append to `PYTHONPATH` instead.
- To run a sidecar by hand, put the kit on the path yourself. For TiPToP's:

  ```bash
  PYTHONPATH=<tandem>/planners/sidecar_kit \
      pixi run --manifest-path <runtime>/tiptop/pixi.toml python <tandem>/planners/tiptop/sidecar.py
  ```

  `<tandem>` is the installed package's directory
  (`python -c "import tandem, os; print(os.path.dirname(tandem.__file__))"`), and `<runtime>` is what
  `tandem runtime path` prints.

## A runtime recipe

A planner that needs more than pip declares its runtime as data. `tandem planners install NAME`
builds it, `tandem planners list` says whether it is current, and `tandem planners remove NAME`
deletes it. TiPToP's is `src/tandem/planners/tiptop/recipe.py`; installing and updating it is in
[CONFIGURATION.md](CONFIGURATION.md#the-planner-runtime).

```python
from pathlib import Path
from tandem.planners import Asset, BuildStep, PixiEnvironment, RuntimeRecipe, Source, SourcePin

HERE = Path(__file__).parent

RECIPE = RuntimeRecipe(
    planner="arm",                                   # must equal info.name
    title="Arm",
    sources=(
        Source(
            SourcePin("arm", "https://github.com/you/arm.git", "<40-hex commit>"),
            trim=("docs/videos",),                   # dropped after fetching; say why in a comment
            patches=(HERE / "patches" / "0001-fix.patch",),   # applied in order; one that fails stops the install
            marker="pixi.toml",                      # exists only when the tree is really there
            persistent=("arm/.cache",),              # written at run time, must outlive a new tree
        ),
        Source(SourcePin("solver", "https://github.com/you/solver.git", "<40-hex commit>", ref="main")),
    ),
    environment=PixiEnvironment(
        manifest="arm/pixi.toml",                    # inside one of the sources: the planner's own manifest and lock
        home="env",                                  # the environment, outside every source tree
        env={"SOME_VAR": "{source:solver}"},
    ),
    steps=(
        BuildStep(
            name="kernels",
            task="build-kernels",                    # a task the manifest defines
            env={"SOLVER_DIR": "{source:solver}", "SETUPTOOLS_SCM_PRETEND_VERSION": "{version:solver}"},
            produces=("solver/build/*.so",),         # globs that exist only once the step has run
            description="compiling the solver's kernels (about 10 minutes)",
        ),
    ),
    assets=(Asset(HERE / "assets" / "weights.pt", "weights/weights.pt"),),   # files the package ships
    notes=("Building Arm: torch, the solver's CUDA kernels.",),
)
```

- **Pins are full 40-character commits**; the recipe refuses anything else when it is declared.
  `ref` names the branch the commit came from: it is shown next to the commit (`tandem planners info`,
  `tandem runtime status`) and recorded, but never compared.
- **Fetching** is `git fetch --depth 1 <url> <commit>`, then `git archive`, so no VCS state or build
  artefact comes along. A server that won't hand out a bare commit (`uploadpack.allowReachableSHA1InWant`
  off) is asked for the pin's `ref` branch instead. Without git, GitHub's archive of the commit is used.
  The commit is always checked.
- **Layout:**

  ```
  <runtime>/
      arm/  solver/            each source at its pinned commit, trimmed and patched
      arm/.pixi -> ../env      where pixi looks for the environment
      env/                     the environment, outside every tree
      cache/arm/arm/.cache/    each persistent directory; arm/arm/.cache links here
      .tandem-runtime.json     what is installed (commit and branch), from where, verified or not, patch digests
      .install.lock            held by the install that is running
  ```

  The environment sits outside the trees, so moving a pin replaces a tree without solving torch and CUDA
  again. `persistent` keeps what the planner downloads into its own tree at run time (TiPToP's SAM-2
  checkpoint) across that swap. The installer never deletes a git checkout, and replaces an unlisted tree
  only inside a runtime that has `.tandem-runtime.json`.
- **Placeholders** in `env` values are filled at build time:

  | placeholder | value |
  |---|---|
  | `{root}` | the runtime root |
  | `{source:NAME}` | a source tree's path |
  | `{commit:NAME}` | its commit |
  | `{version:NAME}` | that commit as a PEP 440 version (`0.0.0+g4db8f92`), for a package that reads its version from git metadata an exported tree lacks |

- **Status** compares `.tandem-runtime.json` with the recipe; a moved pin shows as `outdated`, with the
  command that rebuilds it.
- **Sources from a directory** (`--sources DIR`, `$TANDEM_PLANNER_SOURCES`, `tandem planners bundle`):
  [offline install](CONFIGURATION.md#offline-install).
- **Assets** are small files the planner package ships as package data and the runtime needs at a
  fixed path (TiPToP ships its two DATAFARM checkpoints this way).
- **Tools.** The merge joins legs' videos with the `ffmpeg` in the built environment's `bin/`, and
  falls back to the one on `PATH`.
- The build log is listed in [DATA.md](DATA.md#logs-and-session-files).

## Options, presets and doctor rows

**Options.** A profile's `planner.options` block belongs to the planner, and `validate_options`
checks it when the profile loads. The default refuses any key not in `OPTIONS`; override it for
options with structure (a pydantic model works well).

- What it returns is stored in the profile and validated again on every read, so it must accept its
  own output unchanged.
- It must return plain data: mappings with string keys, lists, strings, numbers, booleans, `None`.
  From pydantic, return `model_dump(mode="json")`. A `Path`, an `Enum` or a numpy value is refused.
- Raise `TandemError` or `ValueError` (a pydantic `ValidationError` is one). Errors are located under
  `planner.options.`, so an error at `tamp` reads as `planner.options.tamp`.
- A setting with no sensible default (a robot's address) may be required: refuse `{}` with a
  `TandemError` naming it, and a person supplies it with `tandem planners use NAME --option KEY=VALUE`.

**Showing them.** `describe_options(profile, *, settings=None)` returns an `OptionsView` (`summary`,
`sections`, `receives`, `receives_note`, `warnings`), used by `tandem profile show`, the web editor and
the session header. `receives` is exactly what the planner will be handed, resolved; it is what
`tandem profile show NAME --planner` prints.

**Presets.** A planner may ship `<name>.yml` presets for its options in `presets_dir`.
`tandem profile create NAME --preset PRESET` applies one, and `tandem profile presets --planner NAME`
lists them. What tandem's own `paper` preset sets is in [CONFIGURATION.md](CONFIGURATION.md#presets).
The file format (`src/tandem/core/presets.py`):

```yaml
title: One line                       # required
summary: One line for a listing       # optional
caution: [what a person must know]    # optional, shown as warnings once applied
extends: paper                        # optional: one of tandem's presets, applied first
replace: [planner.options.tamp]       # optional: blocks substituted whole, not merged
profile:                              # settings, spelled as a profile spells them
  planner:
    options: {...}                    # a planner's preset states planner.options and nothing else
```

- No preset may state `name`, `version` or `description`.
- A planner's preset may state only `planner.options`: not `planner.backend`, `hitl`, `cameras` or the
  task. tandem's own presets may not state `planner` at all.
- A planner's preset with the same name as one of tandem's (tandem ships `paper`) must `extends:` it.
- `replace` may name only blocks the preset itself sets.
- Ship the directory as package data, e.g. in `pyproject.toml` under `[tool.setuptools.package-data]`:
  `"tandem_arm" = ["presets/*.yml"]`. The scaffold doesn't add this.

**Doctor rows.** `doctor_checks(profile, *, settings=None, probe_hardware=True)` returns
`tandem.core.probe.Check(name, state, detail, hint, group)` rows for `tandem doctor`. `state` is
`probe.OK`, `WARN`, `FAIL` or `SKIP`.

- `profile` is `None` during `tandem init`'s preflight: check the machine only.
- `probe_hardware=False` (`--no-hardware`): touch nothing on the network or the bus.
- A FAIL is something that stops a session. `tandem init` stops on one (or asks, when interactive)
  before building the runtime.
- tandem already reports every planner's runtime; don't repeat it.

## Registering it

A name resolves in this order, first match wins:

1. **`register_backend(name, factory)`** (alias `register_planner`) at runtime, from a script, a test
   or code embedding tandem: `tandem.register_backend("arm", ArmPlanner)`. The factory may be a
   `"module:attribute"` string, imported only when the planner is used. A taken name is an error
   unless `replace=True`.
2. **Built in**: the `_BUILTIN` table in `src/tandem/planners/registry.py`. TiPToP is the only one.
3. **The `tandem.planners` entry point**, for a planner in its own package. The scaffold writes it:

   ```toml
   [project.entry-points."tandem.planners"]
   arm = "tandem_arm.planner:ArmPlanner"     # a Planner subclass, or any BackendFactory
   ```

How plugins behave:

- `tandem planners list` and `info` import every plugin to describe it, and loading a profile imports
  its planner to check `planner.options`. Keep the module the entry point names light: import torch and
  robot clients inside `warm()`, or in a sidecar.
- A plugin that fails to import, or calls `sys.exit()` at import, is listed as `broken` with its
  error. Every other planner keeps working, and a profile naming it still loads, its options unchecked.
- A plugin that loses to a registered or built-in planner of the same name is listed with the reason
  it isn't used.
- Two installed packages registering the same name is an error naming both.
- A package installed while `tandem ui` runs, editable ones included, appears on the next listing.

Then `planner: {backend: arm}` in a profile, or `tandem planners use arm`, selects it.
`tandem planners default arm` makes it the planner new profiles get ([commands](USAGE.md#commands)).
The planner endpoints of the HTTP API are in [USAGE.md](USAGE.md#http-api).

## The recording contract

`execute` records one leg of the trial `leg.trajectory_id`. tandem's merge (`src/tandem/core/merge.py`)
finds a trial's legs by that id, orders them by recording window, and joins them into one episode.
When `leg.record` is set, `save_dir` must hold the three items below.

**`_meta.json`:**

| key | value |
|---|---|
| `trajectory_id`, `segment_source` | Copied from `leg` (`segment_source` is `"tamp"` for a planner's leg). This is how the merge finds the leg. |
| `instruction` | `leg.instruction`: the whole task, which is the dataset's language label. |
| `phase_index`, `n_phases`, `phase_description` | Copied from `leg` when `leg.phase_index` is not `None`. |
| `record_start`, `record_stop` | Epoch seconds. Legs are ordered by them. |
| `fps` | Frames per second. |
| `cameras` | Dataset key → clip file name, e.g. `{"exterior_image_1_left": "external_cam.mp4"}`. |
| `plan_file` (optional) | A bare file name for the planner's saved plan (e.g. `"plan.json"`), read by `tandem traj show` and the web UI's "plan: recorded". Without it they look for `tiptop_plan.json`. |

**`robot_state.npz`:** every array in `tandem.core.merge.STATE_KEYS`, one row per frame, and nothing
else except `OPTIONAL_STATE_KEYS` (`action_joint_velocity`, `[F,7]`).

| array | shape | |
|---|---|---|
| `joint_position` | `[F,7]` | measured |
| `gripper_position` | `[F]` | measured, in `[0,1]` |
| `cmd_joint_position` | `[F,7]` | commanded |
| `cmd_joint_velocity` | `[F,7]` | commanded |
| `cmd_gripper` | `[F]` | commanded, binary ([the export](USAGE.md#exporting) skips an episode where it isn't) |
| `frame_time` | `[F]` | wall clock, float64 |

Record the measured arrays from the robot, never as a copy of the command: a policy trained on a
lagged copy of its own action learns to echo it.

**Camera clips:** each named `external_cam.mp4`, `external_cam_2.mp4` or `hand_cam.mp4`
(`tandem.core.trajectories.CAMERA_FILES`), whatever its dataset key: the merge, the viewer and the export
read only these names. Every clip `cameras` names must exist; if it names none, at least one of the three
must. Only cameras every leg of a trial recorded are joined.

`tandem.core.trajectories.is_complete(leg_dir)` is the check. Any other file in the leg is the
planner's own; the merge copies the first planner leg's to the top of the episode.

**Edge cases:**

- Stamp `_meta.json` even when execution fails part-way: a leg without its trajectory id is filed as
  an episode of its own. Return `rollout_dir` (usually `save_dir`) and `n_frames`.
- Stamp it even when the leg records nothing (`n_frames=0`), as the scaffold does until recording is
  wired in.
- `ExecuteResult.stopped_early=True` means a cooperative stop was honoured (`should_stop` is true once
  the operator preempts or the session stops). A leg that stopped early never advances the plan,
  whatever `ok` says. After a preempt the trial is filed as aborted; a stop nothing asked for is a
  `tamp_execution` failure.

What a merged episode looks like on disk is in [DATA.md](DATA.md#episode-layout).

## The conformance kit

`tandem.planners.testing` (`src/tandem/planners/testing.py`) runs the whole protocol against a
planner, with no GPU, no robot and no tandem session:

```python
from tandem.planners.testing import PlannerConformance
from tandem_arm.planner import ArmPlanner

class TestArmPlanner(PlannerConformance):
    planner = ArmPlanner          # a Planner subclass, any BackendFactory, or a registered name
    records_legs = True           # False: only the _meta.json stamp is checked, not the recording
    options = {}                  # the planner.options it is built with
    task_hint = "put one thing where it belongs"
```

Each test builds the planner through its factory into pytest's `tmp_path`, warms it, and always
closes it. The tests hold it to this page: the declarations, options, presets, doctor rows and sidecar
script; the lifecycle (closing twice or before warming, hardware released and reacquired twice); and
every verb, including the recording contract when `records_legs` is set. A test for something the planner
doesn't declare or ship is skipped, with the reason.

Override `goal(scene, caps)` when the kit can't guess a plannable goal from your scene, and
`make_backend(tmp_path)` to build the backend another way. Each check is also a plain function that
raises `ConformanceError` listing every problem: `check_declarations`, `check_protocol`, `check_scene`,
`check_plan_result`, `check_leg`, `check_sidecar_script`, `check_presets`.

The kit checks the protocol, not your planner's quality. Two class switches, on by default, hold it
to what phase planning needs:

- **`phase_planning = True`**: every scene's `rgb_path` must open as an image. Without one, every trial
  with `hitl.enabled` ends at `invention`.
- **`verifies_human_phases = True`**: `capture_frame(camera="external")` must return an image. A
  planner that raises `UnsupportedVerb` fails, since no human phase could be verified.

Turn one off only for a planner that is never run that way. The scaffold returns a stand-in image for
both until you wire in real cameras.

## Before collecting

1. `python -m pytest` is green with `records_legs = True`.
2. `tandem planners info NAME` shows the goal language you meant.
3. `tandem plan --planner NAME --image workspace.png "<task>"` splits a real instruction, on a photo of
   your workspace, into phases your planner can carry out ([USAGE.md](USAGE.md#planning-from-a-photo)).
4. `tandem collect --no-execute` runs a session that perceives and plans without moving the robot.
