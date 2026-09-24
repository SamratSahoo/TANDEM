# Adding a planner

TANDEM "only requires an interface for specifying subgoals and executing the resulting plans"
(paper, Sec. IV-D). This is that interface, and the kit for writing against it. A new task and
motion planner is a package that declares what it can be asked for and implements three verbs.
Nothing in tandem changes, and nothing in tandem has to know the package exists.

- [What tandem does, and what a planner does](#what-tandem-does-and-what-a-planner-does)
- [Start here: `tandem planners new`](#start-here-tandem-planners-new)
- [The protocol](#the-protocol)
- [`Capabilities`, field by field](#capabilities-field-by-field)
- [The `Planner` base class](#the-planner-base-class)
- [In tandem's process, or in a sidecar](#in-tandems-process-or-in-a-sidecar)
- [A runtime recipe](#a-runtime-recipe)
- [Options, presets and doctor rows](#options-presets-and-doctor-rows)
- [Registering it](#registering-it)
- [The recording contract](#the-recording-contract)
- [The conformance kit](#the-conformance-kit)
- [Checklist](#checklist)

---

## What tandem does, and what a planner does

| tandem | the planner |
|---|---|
| Splits the instruction into ordered robot and human phases, invents predicates and magic operators, checks the plan hangs together, and repairs it with the model | Declares its goal language (`Capabilities`), which is all the phase planner reads about it |
| Decides who does each phase, and when | Perceives the workspace: object labels, the table, an image |
| Hands the arm to a person for a human phase, and checks their work from a photo | Plans one goal in one scene, and says why when it cannot |
| Mints the trajectory id, merges every leg into one episode, and writes `hitl.json` | Executes that plan on the robot and records it as one leg, stamped with what tandem asked |
| Everything in the session: prompts, labels, the UI, the events file | Lets go of the robot and cameras when asked, and takes them back |

A planner never sees a phase list, an invented predicate or a person. It is asked, over and over:
*achieve this goal in this scene, and record what you did.*

---

## Start here: `tandem planners new`

```bash
tandem planners new shelfbot                 # a planner that runs in tandem's own process
tandem planners new armsim --sidecar         # one that runs as a script in an environment of its own
```

Each writes a package (default `./tandem-NAME`; `--dir` to choose) that passes tandem's conformance
kit as generated. It plans in a stand-in world, one block and one tray, and "executes" in memory:

```
tandem-shelfbot/
├── pyproject.toml                    the package, and its tandem.planners entry point
├── README.md                         what to fill in, in order
├── src/tandem_shelfbot/
│   ├── __init__.py
│   ├── planner.py                    CAPABILITIES and the ShelfbotPlanner class
│   └── sidecar.py                    (--sidecar only) the script the planner's environment runs
└── tests/test_conformance.py         tandem's conformance kit, run against it
```

Then:

```bash
cd tandem-shelfbot
pytest                          # green as generated: `pythonpath = ["src"]` means no install is needed
pip install -e ".[test]"        # into the environment tandem runs in; the entry point is what tandem reads
tandem planners list            # shelfbot is listed, "no runtime needed"
tandem planners info shelfbot   # its goal language, as the phase planner will see it
tandem planners use shelfbot    # the active profile now plans with it
```

`tandem planners use` switches the profile's `planner.backend` and removes the previous planner's
`planner.options`, naming them, since the new planner would refuse them. Switching back gives the old
planner its defaults. It warns, and does not refuse, when the planner is not installed yet.

Then replace each `TODO`, in the order the generated README lists them, and keep `pytest` green:

1. **The goal language** (`CAPABILITIES`).
2. **`perceive`**: every object's label, the table's, which objects are surfaces, a scene id, and
   an image.
3. **`plan`**: one goal, planned; or `ok=False` with the reason.
4. **`execute`**: run it, record the leg. Once legs are really recorded, set
   `records_legs = True` in `tests/test_conformance.py` so the kit checks the recording too.
5. **The switches**: turn on `supports_movable_restriction`, `supports_return_home` and
   `supports_cooperative_stop` as the planner learns to honour each. The kit checks every one
   declared.
6. **A runtime recipe**, if the planner needs more than pip.

A name is `lowercase letters, digits, _ and -, starting with a letter` (`tandem/core/names.py`). The
same rule holds for human executors. It is typed into YAML and onto command lines, and it becomes a
directory name.

---

## The protocol

`tandem.planners.base.TampBackend`. You do not implement it by hand: subclass `Planner` (next
sections), which supplies every verb but three. The lifecycle a session drives is:

```
factory.create(ctx) → require_ready() → warm()
    → ( perceive() → plan() → execute() )*       robot legs
    → release_hardware() … reacquire_hardware()   around every human leg
    → home() → close()                            when the session stops
```

| verb | when tandem calls it | contract |
|---|---|---|
| `capabilities()` | Any time, before anything is built | Declared, cheap, static. |
| `require_ready()` | Before `warm` | Raise `RuntimeNotReady` naming what is missing. |
| `warm()` | Once per session | Open cameras, connect the robot, build solvers. |
| `perceive(*, task_hint, save_dir, reset_arm=True, open_gripper=False)` | Before every robot leg, never before a person's step | `task_hint` is the whole instruction and steers detection only. `reset_arm` is True only for the attempt's first leg: never park an arm a person just handed back. `open_gripper` is True only for the first leg after a human phase. Returns a `SceneView`. |
| `plan(scene_id, goal, *, surfaces=frozenset(), movables=None, return_home=True, save_dir, reuse_skeleton=None)` | After `perceive` | `goal` is a list of `GoalAtom(predicate, args)` in the planner's wire spelling. `surfaces` pins which objects are surfaces for the whole task. `movables` and `return_home` are passed **only** when declared (below). Returns a `PlanResult`. An unplannable goal is `ok=False` with `failure_reason`, not an exception. |
| `execute(plan_handle, leg, *, save_dir, should_stop=None)` | After a plan succeeds | Run it and record it as one leg of `leg.trajectory_id` ([the recording contract](#the-recording-contract)). Returns an `ExecuteResult`. `ok=False` ends the trial (`tamp_execution`). |
| `capture_frame(*, camera="external")` | For every check of a human phase, and of a robot leg's effects | One RGB frame written to a file; return its path. `camera` is `hitl.verification_camera`. (A robot leg's preconditions are checked on that pass's perception image instead.) |
| `release_hardware()` / `reacquire_hardware()` | Around every human leg | Release blocks until the robot and every camera are really free. A person's executor opens them next. |
| `home()` | When the session stops | Park the arm. Do not open the gripper: it may be holding something. |
| `close()` | Last | Release everything. Safe twice, and safe on a planner that never warmed. |

**What `perceive` returns.** A `SceneView` carries:

- `object_labels`, not including the table;
- `table_label`;
- `surface_labels`: the objects that are surfaces;
- `scene_id`, handed back to `plan`;
- `rgb_path`: an image of what was seen;
- `detected_goal`: the planner's own reading of the instruction, as `GoalAtom`s.

With phase planning on, `rgb_path` is what the task is decomposed from. A pass with no image ends
the trial at `invention`. With phase planning off, the leg's goal is `detected_goal`. Labels may
differ from pass to pass; tandem rebinds its plan to them.

**What `plan` returns.** A `PlanResult` carries:

- `ok`, and `failure_reason` when it failed;
- `planning_seconds`;
- `plan_handle`: opaque, handed back to `execute`;
- `task_plan`: the operators the plan runs, object arguments only, e.g.
  `("Pick(bread)", "Place(bread, plate)")`. Written into `hitl.json`; nothing parses it;
- `artifacts`: role → path;
- `skeleton` and `skeleton_reused`: only with `supports_skeleton_reuse`.

It must be JSON-safe: a sidecar sends it over the wire, and it is written into the record.

---

## `Capabilities`, field by field

The phase planner reads nothing else about a planner. A field it gets wrong is a planner that
loads, lists, and then plans the wrong thing, so `Planner` checks the declaration when the class is
defined (`sdk.capability_problems`).

TiPToP's declaration is `src/tandem/planners/tiptop/capabilities.py`. The toy planner the test suite
drives (`tests/toy_planner.py`) drops items into bins, with no table and no exclusivity.

| field | what it is, and who reads it | TiPToP | toy |
|---|---|---|---|
| `name` | Must equal `info.name`. | `"tiptop"` | `"toy"` |
| `goal_predicates` | Ψ₀: name → `Predicate(name, (Parameter(name, type), …))`. The goal language shown to the model; a robot phase may use nothing else. Declaration order is shown order: put the load-bearing one first. | `On(?obj: movable, ?surface: surface)`, `Holding(?obj: movable)`, `HandEmpty()` | `InBin(?obj: item, ?bin: container)` |
| `robot_description` | One abstract sentence of what the robot does. The model reads it instead of real operator signatures. | `"pick an object up and place it on a surface"` | `"drop an item into a bin"` |
| `goal_predicate_wire_names` | How each goal predicate is spelled in `plan(goal=)`. A predicate **absent** here is one the planner supplies itself: it may be stated, and is dropped from goals. | `{"On": "on", "Holding": "holding"}` (no `HandEmpty`) | `{"InBin": "in_bin"}` |
| `achievable_predicates` | Everything some operator can make true. A robot phase asking for anything else is refused, and repaired, before perception is paid for. Must include every goal predicate. | cuTAMP's add effects, plus what a fresh scene holds | `{"InBin"}` |
| `reserved_predicate_names` | Names the model may not invent. Must include every goal predicate. | all 19 cuTAMP fluents | `{"InBin"}` |
| `movable_type`, `surface_type` | The two object types, and which is which. Must differ. | `movable`, `surface` | `item`, `container` |
| `predicate_descriptions` | `{0}`-templates saying what each goal predicate means, for the operator and for the camera. | `"{0} is resting on top of {1}"`, … | `"{0} is inside {1}"` |
| `checkable_predicates` | Goal predicates a camera can judge from one photo. Invented predicates are always checkable. | `{"On"}` (the gripper is usually out of shot) | `{"InBin"}` |
| `one_pick_per_object` | One plan picks each object at most once, so two phases moving the same object are never conjoined. | `True` (cuTAMP deletes `HasNotPickedUp`) | `False` |
| `initial_state_is_clean` | Every goal is planned from the same clean state, so consecutive robot phases may be conjoined into one goal (`hitl.conjoin_robot_phases`). | `True` | `False`: every robot phase is its own leg |
| `supports_cooperative_stop` | `execute` polls `should_stop` at step boundaries. Otherwise a preempt is an abort. | `False` | `True` |
| `supports_skeleton_reuse` | `plan` can reuse a previous `PlanResult.skeleton`. | `False` | `False` |
| `prompt_fragments` | Planner-specific paragraphs of the segmentation prompt, by slot (`tandem.planning.prompts.PROMPT_SLOTS`): `placement_semantics`, `precondition_vocabulary`, `delete_effect_example`, `work_division`, `intermediate_state_example`, `robot_phase_rules`. A slot left out gets a generic paragraph rendered from the goal predicates, so `{}` is a complete declaration. | all six, the paper's Appendix-B wording (pinned byte for byte by a golden test) | `{}` |
| `exclusive_arguments` | Predicate → the argument that can hold in one atom at a time. The contract check reads it as the delete effect a placement gets for free. | `{"On": 0}` | `{}` |
| `moved_arguments` | Predicate → the argument naming the object a robot phase moves. Must point at a `movable_type` parameter. Read for `movables=`, for conjoining, and for the wasted-move warning. | `{"On": 0, "Holding": 0}` | `{"InBin": 0}` |
| `robot_operators` | Ω₀, one signature each, for the record only. `Pick(?obj: movable)` or `Pick(obj: movable)`; types must be declared ones. | `("Pick(?obj: movable)", "Place(?obj: movable, ?surface: surface)")` | `("Drop(?obj: item, ?bin: container)",)` |
| `supports_movable_restriction` | `plan(movables=)` is honoured: only those objects may be picked, and every other one is an obstacle. A goal that moves anything else is `ok=False` naming it. Requires `moved_arguments`. | `True` | `True` |
| `supports_return_home` | `plan(return_home=False)` ends the leg where its last operation leaves the arm. tandem passes `False` for every leg but the task's last. | `True` | `True` |

tandem passes `movables` only when `supports_movable_restriction` is set, and `return_home` only
when `supports_return_home` is set. A planner declaring neither is never handed either, and may
leave both out of its `plan` signature.

---

## The `Planner` base class

`tandem.planners.Planner` (`src/tandem/planners/sdk.py`) is an abstract base class, and it is also
its own factory. The registry uses the class itself and never instantiates it to get a factory. An
instance is a backend, built once per session.

```python
from tandem.planners import (
    Capabilities, ExecuteResult, Parameter, Planner, PlannerInfo, PlanResult, Predicate, SceneView,
)

IN_BIN = Predicate("InBin", (Parameter("obj", "item"), Parameter("bin", "container")))

class BinPlanner(Planner):
    info = PlannerInfo(name="bins", display_name="Bin sorter", summary="Drops items into bins.")
    CAPABILITIES = Capabilities(
        name="bins",
        goal_predicates={"InBin": IN_BIN},
        robot_description="drop an item into a bin",
        goal_predicate_wire_names={"InBin": "in_bin"},
        achievable_predicates=frozenset({"InBin"}),
        reserved_predicate_names=frozenset({"InBin"}),
        movable_type="item",
        surface_type="container",
        predicate_descriptions={"InBin": "{0} is inside {1}"},
        checkable_predicates=frozenset({"InBin"}),
        moved_arguments={"InBin": 0},
    )
    OPTIONS = {"bins": "the bin names, left to right"}

    def perceive(self, *, task_hint, save_dir, reset_arm=True, open_gripper=False) -> SceneView: ...
    def plan(self, scene_id, goal, *, surfaces=frozenset(), save_dir, reuse_skeleton=None) -> PlanResult: ...
    def execute(self, plan_handle, leg, *, save_dir, should_stop=None) -> ExecuteResult: ...
```

**Declarations** (class attributes):

| attribute | required | what it is |
|---|---|---|
| `info` | yes | `PlannerInfo(name, display_name, summary, homepage, requires, sources)`. What `tandem planners list/info` show. `requires` is human-readable lines, shown and never checked. `sources` is filled in from `recipe` when left empty. |
| `CAPABILITIES` | yes | Above. `CAPABILITIES.name` must equal `info.name`. |
| `recipe` | no | A `RuntimeRecipe`, when the planner needs more than pip. None: pure Python. |
| `OPTIONS` | no | `planner.options` key → one line saying what it does. Any other key is refused when a profile loads. |
| `importer` | no | A `ProfileImporter` (`find`, `configs`, `build`), for `tandem profile create --import-from` and `tandem init --import-from`. TiPToP's reads a hitl-tamp-vla checkout. `build` returns the profile, the extrinsics and notes; a note starting with `tandem.planners.base.WARNING_NOTE` (`"warning: "`) is shown as a warning. |
| `presets_dir` | no | A directory of `<name>.yml` presets for this planner's options. |

**Checked when the class is defined.** Every problem is listed in one `TandemError` at import:

- an unachievable or unreserved goal predicate;
- a moved argument that points at a surface;
- a wire name for a predicate that does not exist;
- a misspelt prompt slot;
- a pin that is not a full 40-character commit;
- a recipe for another planner;
- `info.name` and `CAPABILITIES.name` that disagree.

A base class for other planners passes `abstract=True` (`class MyBase(Planner, abstract=True)`), and
is then neither checked nor registrable.

**Defaults for everything else:**

| member | default |
|---|---|
| `warm`, `close`, `home`, `release_hardware`, `reacquire_hardware` | Do nothing: right for a planner that holds no hardware. |
| `require_ready` | Checks the declared recipe's runtime. No recipe: nothing to check. |
| `capture_frame`, `move_to_joints` | Raise `UnsupportedVerb`. There is no honest default for a camera frame: a verifier shown a made-up one would pass or fail a person's work on nothing. Without `capture_frame`, human phases cannot be verified, so a session with `hitl.check_human_effects`, `check_human_preconditions` or `check_tamp_effects` on refuses to start on the planner; implement it, or turn those off. |
| `create(ctx)` (classmethod) | `validate_options(ctx.options)`, then `cls(ctx)`. Override when construction needs more. |
| `validate_options(options)` | Refuses any key not in `OPTIONS`, with a difflib hint. |
| `describe_options(profile, *, settings=None)` | `OptionsView.generic`: each option, as set. |
| `doctor_checks(profile, *, settings=None, probe_hardware=True)` | `[]`. |
| `replay(rollout_dir, *, settings=None)` | Raises `UnsupportedVerb`: no viewer (`tandem traj open`). |
| `runtime(settings)` / `runtime_root(settings)` | A `RecipeRuntime` at `<runtimes dir>/<name>` (`~/.local/share/tandem/runtimes/<name>` on Linux, `$TANDEM_RUNTIMES_DIR` overrides). None without a recipe. |

**In your implementation:** `self.ctx` is the `BackendContext`: `profile`, `session_dir`,
`output_dir`, `execute`, `record`, `options`, `settings`, `session_id`, `task`, `events_file` and
`runtime_dir`. `self.options` is `planner.options` as validated. `self.log(text)` writes a line to
the session log the operator watches.

A factory that is not a `Planner` works too: any object with `info`, `capabilities()`,
`create(ctx)` and `runtime(settings)`, plus any of the optional hooks above
(`tandem.planners.base.BackendFactory`). TiPToP's is one (`planners/tiptop/factory.py`), because its
backend predates the SDK.

---

## In tandem's process, or in a sidecar

tandem installs with pip and runs on a laptop. A planner that needs torch, CUDA kernels, a camera
SDK or a robot client cannot live in tandem's process. It runs as a **sidecar**: a script launched
with the planner's own interpreter, answering the protocol one JSON object per line.

```
tandem (pure Python)                              the planner's environment (pixi)
  MyPlanner(SidecarPlanner).plan(goal)  ──JSON──►   my_sidecar.py:  plan(scene_id, goal, ...)
                                        ◄──JSON──   {"ok": true, "plan_handle": ..., "task_plan": [...]}
```

**The class** (`tandem.planners.SidecarPlanner`, `src/tandem/planners/sidecar.py`):

```python
class ArmPlanner(SidecarPlanner):
    info = PlannerInfo(name="arm", ...)
    CAPABILITIES = Capabilities(name="arm", ...)
    recipe = RECIPE                    # the environment the sidecar runs in; None: this interpreter
    SIDECAR = "sidecar.py"             # relative to this module's directory; checked at class definition
    TIMEOUTS = {"warm": 600.0}         # over sidecar.DEFAULT_TIMEOUTS
```

What it does for you:

- **Launches** the script with the runtime's `python` (via `pixi run`, from the runtime's working
  directory), or with tandem's own interpreter when there is no runtime. The sidecar gets its own
  process group, and `tandem_sidecar` goes first on its `PYTHONPATH`.
- **Maps every verb** onto a request, and each reply back onto tandem's types.
- **Sends only what is declared.** `movables`, `return_home` and `reuse_skeleton` go on the wire
  only when declared. A caller that passes one to a planner that did not declare it gets an error,
  not a quietly unrestricted plan.
- **Falls back for unanswered verbs.** A verb the sidecar does not list in its hello gets
  `Planner`'s default. A sidecar that does not answer `perceive`, `plan` and `execute` is refused at
  `warm`.
- **Streams output.** Log lines and stderr go into the session log. Events go into the session's
  events file (`on_event`).
- **Times out** every verb. Defaults: `warm` 900 s, `perceive` 300, `plan` 900, `execute` 1800,
  `capture_frame`, `home`, `release_hardware` and `reacquire_hardware` 180 each. A crash is
  reported with its exit code, and a dead sidecar is relaunched at the next `warm`.
- **Stops cooperatively** when declared. While `execute` runs, `should_stop` is polled in tandem
  and passed to the sidecar as a stop file (`TANDEM_SIDECAR_STOP_FILE`).
- **`close`** asks the sidecar to quit, then makes sure the whole process group is gone, so nothing
  is left holding a camera.

Override `launch_command`, `launch_cwd`, `launch_env`, `warm_args` (what the sidecar's `warm` is
handed; by default `output_dir`, `execute`, `record`) or `on_event` when the defaults are wrong for
your planner. `call(verb, **args)` reaches a verb of the sidecar's own. TiPToP's backend is this class
plus its launch details (`planners/tiptop/backend.py`).

**The script** imports nothing from tandem, only `tandem_sidecar`
(`src/tandem/planners/sidecar_kit/tandem_sidecar.py`). That is one standard-library file, written
for Python 3.8 and later, which tandem puts on the script's path:

```python
from tandem_sidecar import log, serve          # FIRST: it takes stdout for the protocol

# isort: split
# the planner's own imports below here -- or better, inside warm()

class World:
    def warm(self, *, output_dir=None, execute=True, record=True, **planner_specific): ...
    def perceive(self, *, task_hint, save_dir, reset_arm=True, open_gripper=False):
        return {"scene_id": "s1", "object_labels": ["block", "tray"], "table_label": "table",
                "surface_labels": ["tray"], "rgb_path": "/path/to/rgb.png"}
    def plan(self, *, scene_id, goal, surfaces, save_dir):
        return {"ok": True, "plan_handle": "p1", "task_plan": ["Move(block, tray)"]}
    def execute(self, *, plan_handle, leg, save_dir):
        return {"ok": True, "n_frames": 120, "rollout_dir": save_dir}
    def capture_frame(self, *, camera):
        return {"path": "/path/to/frame.png"}

if __name__ == "__main__":
    raise SystemExit(serve(World()))
```

- `serve(handlers)` takes an object (answering every protocol verb it has a method for) or a
  mapping of verb → callable. It answers until tandem says quit or closes stdin, then calls
  `close` (or `on_exit`).
- `log(message, level="info")` writes a line to the session log, safely from any thread.
- `event(name, **fields)` writes one event to the session's events file.
- `should_stop()` is true once tandem asks the execution in flight to stop.
- A handler that raises is reported to tandem as `"<verb> failed -- <Type>: <message>"`, with the
  traceback in the session log.

The full wire protocol, and what each verb is sent and returns, is in that file's docstring.

**stdout belongs to the protocol.** Importing `tandem_sidecar` takes the real stdout and points fd 1
at stderr. From then on, a CUDA banner or a stray `print()` lands in the session log instead of
corrupting a reply. So import it before anything that might print. The conformance kit refuses a
script that imports a non-standard-library module before it.

**Two things to know about `PYTHONPATH`:**

- tandem adds the kit's directory to the sidecar's `PYTHONPATH` when it launches one. If your
  environment's own activation *replaces* `PYTHONPATH` (a pixi `[activation.env]` that sets it, for
  example), the kit is lost, and the sidecar dies at start with
  `ModuleNotFoundError: No module named 'tandem_sidecar'` in the session log. Append to
  `PYTHONPATH` rather than setting it. TiPToP's manifest sets none.
- To run a sidecar by hand, outside tandem, put the kit on the path yourself. TiPToP's, for
  example:

  ```bash
  PYTHONPATH=<tandem>/planners/sidecar_kit \
      pixi run --manifest-path <runtime>/tiptop/pixi.toml python <tandem>/planners/tiptop/sidecar.py
  ```

  `<tandem>` is the installed package's directory (`python -c "import tandem, os;
  print(os.path.dirname(tandem.__file__))"`), and `<runtime>` is what `tandem runtime path` prints.

---

## A runtime recipe

A planner that needs more than pip declares the runtime it runs in, as data. `tandem planners install
NAME` builds it, `tandem planners list` says whether it is current, and `tandem planners remove NAME`
deletes it. TiPToP's is `src/tandem/planners/tiptop/recipe.py`.

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
            patches=(HERE / "patches" / "0001-fix.patch",),   # applied in order; a patch that fails stops the install
            marker="pixi.toml",                      # exists only when the tree is really there
        ),
        Source(SourcePin("solver", "https://github.com/you/solver.git", "<40-hex commit>")),
    ),
    environment=PixiEnvironment(
        manifest="arm/pixi.toml",                    # inside one of the sources: the planner's own manifest and lock
        home="env",                                  # the environment, OUTSIDE every source tree
        env={"SOME_VAR": "{source:solver}"},
    ),
    steps=(
        BuildStep(
            name="kernels",
            task="build-kernels",                    # a task the manifest defines
            env={"SOLVER_DIR": "{source:solver}", "SETUPTOOLS_SCM_PRETEND_VERSION": "{version:solver}"},
            produces=("solver/build/*.so",),         # globs that exist only once the step has run
            description="compiling the solver's kernels — about 10 minutes",
        ),
    ),
    assets=(Asset(HERE / "assets" / "weights.pt", "weights/weights.pt"),),   # files the package ships
    notes=("Building Arm: torch, the solver's CUDA kernels.",),
)
```

- **Pins are full commits.** A branch names a different planner next week, and a dataset has to be
  traceable to the planner that produced it. The recipe refuses anything else when it is declared.
- **Fetching.** Each pin is fetched with `git fetch --depth 1 <url> <commit>` and exported with
  `git archive`, so no VCS state or build artifact from anyone's working tree comes along. Without
  git, GitHub's archive of the commit is used. Either way the commit is checked.
- **Layout:**

  ```
  <runtime>/
      arm/  solver/            each source at its pinned commit, trimmed and patched
      arm/.pixi -> ../env      where pixi looks for the environment
      env/                     the environment, outside every tree
      .tandem-runtime.json     what is installed, from where, verified or not, with each patch's digest
  ```

  The environment lives outside the trees on purpose: moving a pin replaces a tree without solving
  torch and CUDA again from nothing.
- **Placeholders** in `env` values are filled at build time:
  - `{root}`: the runtime root;
  - `{source:NAME}`: a tree's path;
  - `{commit:NAME}`: its commit;
  - `{version:NAME}`: that commit as a PEP 440 local version (`0.0.0+g4db8f92`), for a package whose
    version would otherwise come from git metadata an exported tree does not have.
- **Status** compares `.tandem-runtime.json` with the recipe. A tandem upgrade that moves a pin shows
  the planner as `outdated` in `tandem planners list`, with the command that rebuilds it. That is
  better than an ImportError forty seconds into a warm-up.
- **Offline installs.** A workstation with no network installs from a directory holding one
  checkout or export per source, named as the recipe names them:
  `tandem planners install NAME --sources DIR`, or `$TANDEM_PLANNER_SOURCES=DIR`. From a checkout of
  the tandem repository, on a machine that can reach the sources, `python tools/bundle.py --planner
  NAME --out DIR` makes one; each export carries a marker naming its commit, which the install
  checks. While a sources directory is in force nothing is fetched: a missing source is an error,
  not a hang.
- **Assets** are small files the planner package itself ships (as package data) and the runtime
  needs at a fixed path. TiPToP ships two DATAFARM checkpoints this way, because their source
  repository is private.
- **Tools.** The merge joins legs' videos with the ffmpeg in the built environment's `bin/`
  (`registry.tools_dir`), the build that recorded the clips, and falls back to the one on `PATH`.

`tandem planners install` asks before installing pixi into your home directory (`--yes` to accept),
and writes the full build log to `~/.local/state/tandem/logs/runtime-build-<time>.log`.

---

## Options, presets and doctor rows

**Options.** A profile's `planner.options` block belongs to the planner. The planner checks it
when the profile loads (`validate_options`), so a mistake is found when the profile is edited, not
when the arm is about to move. What `validate_options` returns is what the profile stores, and it is
validated again on every read, so it must accept its own output unchanged. For options with
structure, override it; a pydantic model is the natural tool. Raise `TandemError`, or `ValueError`
(a pydantic `ValidationError` is one). Error locations are shown under `options.`, so a pydantic
error at `tamp` reads as `planner.options.tamp`.

**Showing them.** `describe_options(profile, *, settings=None)` returns an `OptionsView`, which
`tandem profile show`, the web editor and the session header all use:

- `summary`: one line;
- `sections`: titled rows;
- `receives`: exactly what the planner will be handed, resolved. This is what
  `tandem profile show NAME --planner` prints: the answer to "did my setting apply?";
- `receives_note`, and `warnings`.

Over HTTP, `GET /api/profiles/{name}` carries the view as `planner_view`, and every profile card
carries its one-line `planner_summary` next to `planner`.

**Presets.** A planner may ship `<name>.yml` presets for its options in `presets_dir`.
`tandem profile create NAME --preset PRESET` lays one over a new profile, and `tandem profile
presets` lists them. The file (`tandem/core/presets.py`):

```yaml
title: One line                       # required
summary: One line for a listing       # optional
caution: [what a person must know]    # optional, shown as warnings once applied
extends: paper                        # optional: one of tandem's presets, applied first
replace: [planner.options.tamp]       # optional: blocks substituted whole, not merged
profile:                              # settings, spelled as a profile spells them
  planner:
    options: {...}                    # under planner:, only options: -- the planner is chosen with --planner
```

A preset may not state `name`, `version` or `description`, and a planner's preset may not state
`planner.backend`. tandem's own presets may not state `planner` at all. A planner's preset with the
same name as one of tandem's (tandem ships `paper`) must extend it, so tandem's half is never
silently dropped for one planner. Ship the directory as package data: in
`pyproject.toml`, `[tool.setuptools.package-data]` with a glob such as `"tandem_arm" =
["presets/*.yml"]`. The scaffold does not add this for you.

**Doctor rows.** `doctor_checks(profile, *, settings=None, probe_hardware=True)` returns
`tandem.core.probe.Check(name, state, detail, hint, group)` rows for `tandem doctor`. `state` is one
of `probe.OK`, `WARN`, `FAIL` or `SKIP`. `profile` is None during `tandem init`'s preflight: check the
machine only. `probe_hardware=False` (`--no-hardware`) means touch nothing on the network or the bus.
A FAIL is something that stops a session, and `tandem init` stops on one before building the
runtime. tandem already reports every planner's runtime; do not repeat it.

---

## Registering it

Three ways, the first match winning:

1. **`register_backend(name, factory)`** at runtime, from a script, a test or code embedding
   tandem: `tandem.register_backend("arm", ArmPlanner)`. The factory may also be given as a
   `"module:attribute"` string, imported only when the planner is used. Registering a taken name is
   an error unless `replace=True`.
2. **Built in**: the `_BUILTIN` table in `src/tandem/planners/registry.py`, for a planner that
   ships inside tandem. TiPToP is the only one.
3. **The `tandem.planners` entry point**, for a planner in its own package. This is what the
   scaffold writes:

   ```toml
   [project.entry-points."tandem.planners"]
   arm = "tandem_arm.planner:ArmPlanner"     # a Planner subclass, or any BackendFactory
   ```

Entry points are read by name when listing and imported only when used. So keep the module the
entry point names light: torch and robot clients go inside `warm()`, or into a sidecar. A plugin
that fails to import does not stop tandem. It is listed as `broken` with its error, and every other
planner keeps working. A plugin that loses to a registered or built-in planner of the same name is
listed with the reason it is not used. A package installed while `tandem ui` runs is seen on the
next listing.

Then `planner: {backend: arm}` in a profile (or `tandem planners use arm`) is all it takes.
`tandem planners use arm --default` also makes it the planner every new profile gets
(`settings.default_planner`). Over HTTP, `GET /api/planners` and `GET /api/planners/{name}` return
what `tandem planners list/info --json` print. `POST /api/planners/{name}/use` and
`POST /api/planners/{name}/default` do what `use` and `--default` do. Installing is deliberately not
an endpoint: every catalog row carries the `install_command` to run in a terminal.

---

## The recording contract

`execute` records one **leg** of the trajectory `leg.trajectory_id`. tandem joins a trial's legs,
the planner's and a person's, into one episode (`tandem/core/merge.py`). It finds them by that id,
orders them by their recording windows, and requires exactly this in `save_dir` when `leg.record`
is set:

- **`_meta.json`**:
  - `trajectory_id` and `segment_source`, copied from `leg`. This is how merging finds the leg.
    `segment_source` is `"tamp"` for a planner's leg.
  - `phase_index`, `n_phases` and `phase_description`, copied from `leg` when `leg.phase_index` is
    not None. This is how the merged episode says which frames were which phase.
  - `instruction`: `leg.instruction`, the whole task and the dataset's language label.
  - `record_start` and `record_stop`: epoch seconds. Legs are ordered by them.
  - `fps`.
  - `cameras`: dataset key → clip file name, e.g. `{"exterior_image_1_left": "external_cam.mp4"}`.
- **`robot_state.npz`** with every array in `tandem.core.merge.STATE_KEYS`, one row per frame, and
  nothing else except `OPTIONAL_STATE_KEYS` (`action_joint_velocity`):

  | array | shape | |
  |---|---|---|
  | `joint_position` | `[F,7]` | measured |
  | `gripper_position` | `[F]` | measured, in `[0,1]` |
  | `cmd_joint_position` | `[F,7]` | commanded |
  | `cmd_joint_velocity` | `[F,7]` | commanded |
  | `cmd_gripper` | `[F]` | commanded, binary; the export skips an episode where it is not |
  | `frame_time` | `[F]` | wall clock, float64 |

- **The camera clips** `cameras` names. With none named, at least one of `external_cam.mp4`,
  `external_cam_2.mp4` and `hand_cam.mp4`. Every clip is named from those three
  (`trajectories.CAMERA_FILES`), whatever its dataset key: they are the only names the merge joins,
  the viewer lists and the export decodes, and a person's teleop legs always use them. Only the
  cameras every leg of a trial recorded are joined.

`tandem.core.trajectories.is_complete(leg_dir)` is the check. It does not ask for a planner's own
plan file. `tiptop_plan.json` is TiPToP's, and any other file a leg holds is the recorder's own; the
merge surfaces the first planner leg's files at the top of the episode. A planner that saves its
plan can say where with `plan_file` in `_meta.json` (a bare file name, e.g. `"plan.json"`), which is
what `tandem traj show` and the web UI's "plan: recorded" read; without it they look for
`tiptop_plan.json`.

**Stamp `_meta.json` even when execution fails part-way.** A leg on disk without its trajectory id
is filed as an episode of its own. Return `rollout_dir` (usually `save_dir`) and `n_frames`.
`ExecuteResult.stopped_early=True` means a cooperative stop was honoured: tandem passes `should_stop`
only to a planner that declares `supports_cooperative_stop`, true once the operator preempts or the
session stops. A leg that stopped early never advances the plan, whatever `ok` says, and the trial
is filed as aborted rather than as a `tamp_execution` failure.

A planner that records nothing (`n_frames=0`) still stamps `_meta.json`, so the leg directory is
kept. That is what the scaffold does until real recording is wired in.

---

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

It builds the planner through its factory into pytest's `tmp_path`, warms it, and always closes it.
It has 16 tests:

- the declarations;
- every protocol keyword, with its default;
- that the backend declares what its factory does;
- that `validate_options` accepts its own output;
- that every preset gives options the planner accepts;
- that doctor rows are `Check`s;
- that a sidecar script can run without tandem;
- closing twice, and before warming;
- a well-formed scene;
- a JSON-safe plan that says what it runs;
- the recorded leg, stamped as asked (and, with `records_legs`, meeting the recording contract);
- a stop honoured when promised;
- picks restricted when promised;
- a leg that does not go home;
- hardware handed over and back, twice;
- a frame for the verifier that is an image on disk.

A test for something the planner does not declare is skipped with the reason.

Override `goal(scene, caps)` when the kit cannot guess a plannable goal from your scene, and
`make_backend(tmp_path)` to build the backend another way. Each check is also a plain function that
raises `ConformanceError` listing every problem found: `check_declarations`, `check_protocol`,
`check_scene`, `check_plan_result`, `check_leg`, `check_sidecar_script`, `check_presets`.

The kit checks the protocol, not your planner's quality, and it does not require two things phase
planning needs:

- **an image from `perceive`**: without `rgb_path` a task cannot be decomposed, and the trial ends
  at `invention`;
- **`capture_frame`**: without it no human phase can be verified, and a session with the camera
  checks on refuses to start.

The scaffold returns neither until you wire them in. Before collecting with phase planning on, run a
session with `--no-execute`, and run `tandem plan --backend NAME` on a photo of your workspace.

---

## Checklist

- [ ] `info.name` and `CAPABILITIES.name` are the same valid name, and nothing else claims it
      (`tandem planners list`).
- [ ] The goal language is the smallest set of predicates a phase needs. Every goal predicate is
      achievable and reserved, and `predicate_descriptions` read well to a person.
- [ ] `goal_predicate_wire_names` omits exactly the predicates the planner supplies itself.
- [ ] `checkable_predicates` lists only what a third-person photo can settle.
- [ ] `moved_arguments` point at the moved object. `exclusive_arguments` declares any "one place at
      a time" predicate.
- [ ] `one_pick_per_object` and `initial_state_is_clean` say what the solver really assumes. When
      in doubt, `initial_state_is_clean=False`: every robot phase then gets its own leg.
- [ ] `perceive` honours `reset_arm` and `open_gripper`, and returns an `rgb_path`.
- [ ] `plan` returns `ok=False` with a reason instead of raising, and fills `task_plan`.
- [ ] `execute` meets the recording contract, stamps `_meta.json` on every path, and
      `records_legs = True` in the kit.
- [ ] `capture_frame` returns an image from the camera it is asked for.
- [ ] `release_hardware` blocks until the robot and every camera are free, and
      `reacquire_hardware` takes them back from wherever a person left the arm.
- [ ] Each `supports_*` flag is declared only once it is honoured, and the kit's test for it passes.
- [ ] Nothing heavy is imported by the module the entry point names.
- [ ] A sidecar imports `tandem_sidecar` first and never imports tandem. Its environment does not
      overwrite `PYTHONPATH`.
- [ ] Recipe pins are full commits, and every trim and patch says why.
- [ ] `OPTIONS` or `validate_options` refuses what the planner does not read.
- [ ] `pytest` is green. `tandem planners info NAME` shows what you meant. `tandem plan --backend
      NAME` decomposes a real instruction into phases your planner can carry out.
