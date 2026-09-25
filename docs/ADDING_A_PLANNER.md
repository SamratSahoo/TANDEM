# Adding a planner

A planner is a Python package that declares a goal language and implements `perceive`, `plan` and
`execute`: plan one goal in one scene, record one [leg](README.md#terms). Examples:
`src/tandem/planners/tiptop/`, `tests/toy_planner.py`.

## Quick start

1. Generate a package in `./tandem-NAME` (`--dir` to change). It plans a stand-in world and passes
   the [conformance kit](#the-conformance-kit).

   ```bash
   tandem planners new shelfbot              # in tandem's process
   tandem planners new armsim --sidecar      # a script in its own environment
   ```

2. Install it into tandem's own environment; `planners new` prints the command:

   ```bash
   cd tandem-shelfbot
   pipx inject tandem-tamp --editable . pytest                   # tandem from pipx
   uv tool install --reinstall git+https://github.com/SamratSahoo/tandem.git \
       --with-editable . --with pytest                           # tandem from uv tool
   pip install -e ".[test]"                                      # tandem in an active virtualenv
   ```

3. Check it with that environment's Python (printed by `planners new`):

   ```bash
   python -m pytest                # green as generated
   tandem planners list            # shelfbot: "no runtime needed"
   tandem planners info shelfbot
   tandem planners use shelfbot
   ```

   What `planners use` changes: [CONFIGURATION.md](CONFIGURATION.md#planner-settings).

4. Fill in the README's `TODO`s in order, keeping `pytest` green; once `execute` records legs, set
   `records_legs = True` in the test.
5. Before collecting, check that `tandem plan --planner NAME --image workspace.png "<task>"` splits a
   real task into phases your planner can do ([USAGE.md](USAGE.md#planning-from-a-photo)) and
   `tandem collect --no-execute` perceives and plans without moving the robot.

### Names

Lowercase letters, digits, `_` and `-`, starting with a letter.

## The protocol

Subclass [`Planner`](#the-planner-base-class), not `tandem.planners.base.TampBackend`: it supplies
every verb but `perceive`, `plan` and `execute`. A session calls:

```
factory.create(ctx) → require_ready() → warm()
    → ( perceive() → plan() → execute() )*       robot legs
    → release_hardware() … reacquire_hardware()   around every human leg
    → home() → close()                            when the session stops
```

| verb | contract |
|---|---|
| `capabilities()` | Static, cheap. |
| `require_ready()` | Raise `RuntimeNotReady` naming what is missing. |
| `warm()` | Open cameras, connect the robot, build solvers; rerun after a verb raises (no-op if warm). |
| `perceive(*, task_hint, save_dir, reset_arm=True, open_gripper=False)` → `SceneView` | `task_hint` (the whole instruction) only steers detection. `reset_arm` only on an attempt's first leg: never park an arm a person handed back. `open_gripper` only on the first leg after a human phase: open the hand, move nothing else. |
| `plan(scene_id, goal, *, surfaces=frozenset(), movables=None, return_home=True, save_dir, reuse_skeleton=None)` → `PlanResult` | `goal`: wire-spelled `GoalAtom(predicate, args)`s. `surfaces`: fixed per task. Can't plan: `ok=False` with `failure_reason`, don't raise. |
| `execute(plan_handle, leg, *, save_dir, should_stop=None)` → `ExecuteResult` | Record one leg of `leg.trajectory_id` ([contract](#the-recording-contract)). `ok=False` ends the trial at `tamp_execution`. |
| `capture_frame(*, camera="external")` | Save an RGB frame of `camera` (`hitl.verification_camera`); return its path. |
| `release_hardware()` / `reacquire_hardware()` | Release blocks until robot and cameras are free; reacquire wherever the person left the arm. |
| `home()` | Park the arm; don't open the gripper. |
| `close()` | Release everything; safe twice and before `warm`. |

A verb that raises ends the trial at its stage: `tamp_planning` (`perceive`, `plan`) or
`tamp_execution` (`execute`).

| `SceneView` field | meaning |
|---|---|
| `object_labels` | Every object seen but the table. |
| `table_label` | The support surface (default `table`). |
| `surface_labels` | Objects that are surfaces. |
| `scene_id` | Opaque; passed to `plan`. |
| `rgb_path` | Scene image to split the task and check robot-leg preconditions. Missing when proposing the plan: the trial ends at `invention`. |
| `detected_goal` | The planner's reading of the instruction as `GoalAtom`s; the leg's goal with phase planning off. |

Labels may change between passes; tandem rebinds its plan to them.

| `PlanResult` field (must be JSON-safe) | meaning |
|---|---|
| `ok`, `failure_reason` | Planned or not, and why. |
| `planning_seconds` | Planning time. |
| `plan_handle` | Opaque; passed to `execute`. |
| `task_plan` | Operators with object arguments, e.g. `("Pick(bread)", "Place(bread, plate)")`, for `hitl.json`; never parsed. |
| `artifacts` | Role → path of files written. |
| `skeleton`, `skeleton_reused` | Only with `supports_skeleton_reuse`. |

## Capabilities

All phase planning knows about a planner, checked at class definition.

| field | meaning |
|---|---|
| `name` | Equals `info.name`. |
| `goal_predicates` | Name → `Predicate(name, (Parameter(name, type), …))`, most important first; robot phases use nothing else. |
| `robot_description` | Required: one abstract sentence of what the robot does. |
| `goal_predicate_wire_names` | `plan(goal=)` spelling, e.g. `{"On": "on"}`; unlisted ones (the planner supplies them) are dropped, and a robot phase of only those is repaired. |
| `achievable_predicates` | What operators can make true, every goal predicate included; robot phases asking for more are repaired. |
| `reserved_predicate_names` | Names the model may not invent, every goal predicate included. |
| `movable_type`, `surface_type` | The only two object types, distinct; every goal-predicate and operator parameter uses one. |
| `moved_arguments` | Predicate → position of the moved object (`{"On": 0}`), a `movable_type` parameter, for `movables=`, conjoining and wasted-move warnings. |
| `exclusive_arguments` | Predicate → argument held in one atom at a time (`{"On": 0}`); a free delete effect. A phase with two atoms in one slot is repaired. |
| `predicate_descriptions` | Templates for the operator and camera, e.g. `"{0} is resting on top of {1}"` (`{0}`, `{1}`, … within arity; `{{` for a brace). |
| `checkable_predicates` | Goal predicates a camera can judge from one photo (invented ones always are). |
| `robot_operators` | Signatures, for the record only, e.g. `Pick(?obj: movable)`; declared types only. |
| `prompt_fragments` | Prompt paragraphs by slot: `placement_semantics`, `precondition_vocabulary`, `delete_effect_example`, `work_division`, `intermediate_state_example`, `robot_phase_rules`; missing slots get generic text. |
| `one_pick_per_object` | A plan picks each object at most once. Default `True` (safe). |
| `initial_state_is_clean` | Every goal starts from the same clean state, allowing conjoining. Default `False`; set only if the solver guarantees it. |
| `supports_movable_restriction` | Honours `plan(movables=)`: pick only those, the rest are obstacles; a goal moving another is `ok=False`. Needs `moved_arguments`. |
| `supports_return_home` | Honours `plan(return_home=False)`: end where the last operation leaves the arm (all but the task's last leg). |
| `supports_cooperative_stop` | `execute` polls `should_stop` between steps; otherwise a preempt aborts the trial once the leg ends. |
| `supports_skeleton_reuse` | `plan` can reuse a previous `PlanResult.skeleton`. |

`movables`, `return_home` and `should_stop` are passed only when declared; `plan` may leave an undeclared
`movables` or `return_home` out of its signature.

**Conjoining** (`hitl.conjoin_robot_phases`) plans consecutive robot phases as one goal from one
perception pass; their atoms are sorted, losing the proposal's order. It needs `initial_state_is_clean`, and
stops before a phase that re-moves an object (with `one_pick_per_object`) or refills an
`exclusive_arguments` slot.

## The Planner base class

`tandem.planners.Planner` is an abstract base class and its own factory (the registry holds the
class).

| class attribute | required | what it is |
|---|---|---|
| `info` | yes | `PlannerInfo(name, display_name, summary, homepage, requires, sources)` for `tandem planners list`/`info`; `requires` is free text, `sources` defaults to the recipe's. |
| `CAPABILITIES` | yes | [Above](#capabilities). |
| `recipe` | no | A [`RuntimeRecipe`](#a-runtime-recipe); `None`: pure Python. |
| `OPTIONS` | no | The task's settings: a profile's `planner.options` key → one-line description. |
| `RIG_OPTIONS` | no | This machine's settings, shared by every profile: a `planners.<name>` key in [rig.yml](CONFIGURATION.md#the-rig) → one-line description (a server's address, a robot's ports). No key in both. |

A bad declaration raises one `TandemError` at import, listing every problem. A base class passes
`abstract=True` (`class MyBase(Planner, abstract=True)`): unchecked, unregistrable.

| member | default |
|---|---|
| `warm`, `close`, `home`, `release_hardware`, `reacquire_hardware` | No-op. |
| `require_ready` | Checks the recipe's runtime is installed. |
| `capture_frame`, `move_to_joints` | Raise `UnsupportedVerb`; with phase planning and `hitl.check_human_effects`, `check_human_preconditions` or `check_tamp_effects` on, a session won't start without `capture_frame`. |
| `create(ctx)` | `validate_options(ctx.options)` and `validate_rig_options(ctx.rig_options)`, then `cls(ctx)`. |
| `runtime_env(*, rig, settings=None)` | `{}`: what `tandem runtime run` and `shell` add to the environment of the planner's own scripts. TiPToP's writes a `tiptop.yml` from the rig and points `$TIPTOP_CONFIG` at it. |
| `replay(rollout_dir, *, settings=None)` | Raises `UnsupportedVerb` (no `tandem traj open`). |
| `runtime(settings)`, `runtime_root(settings)` | `~/.local/share/tandem/runtimes/<name>` (or `$TANDEM_RUNTIMES_DIR`); `None` without a recipe. |

`self.ctx` is the session's `BackendContext` (`profile`, `session_dir`, `output_dir`, `execute`, `record`,
`on_log`, `options`, `settings`, `session_id`, `task`, `events_file`, `runtime_dir`, `rig`, `rig_options`),
`self.options` and `self.rig_options` the validated settings, `self.rig` the machine's rig (`robot.host`,
`robot.type`, `cameras`, `calibration_file()`); `self.log(text)` writes to the session log. Any `tandem.planners.base.BackendFactory`
(`info`, `capabilities()`, `create(ctx)`, `runtime(settings)`, optional hooks) also works, like TiPToP's.

## Sidecars

A planner needing torch, CUDA kernels, a camera SDK or a robot client runs as a **sidecar**: a script
in its own interpreter, speaking the protocol as JSON lines. tandem's side:

```python
class ArmPlanner(SidecarPlanner):      # tandem.planners.SidecarPlanner; info, CAPABILITIES as usual
    recipe = RECIPE                    # its environment; None: tandem's interpreter
    SIDECAR = "sidecar.py"             # relative to this module; checked at definition
    TIMEOUTS = {"warm": 600.0}         # seconds, over the defaults below
```

| behaviour | detail |
|---|---|
| Launch | `pixi run` of the runtime's `python` (else tandem's interpreter), from the runtime's working directory, in its own process group. |
| Verbs | Verbs its hello doesn't list get `Planner`'s defaults; one lacking `perceive`, `plan` or `execute` is refused at `warm`. `movables`, `return_home`, `reuse_skeleton` are sent only when declared; passing an undeclared one is an error. |
| Output | Logs and stderr → session log; events → events file (`on_event`). |
| Timeouts | `warm` 900 s, `perceive` 300, `plan` 900, `execute` 1800; `capture_frame`, `home`, `release_hardware`, `reacquire_hardware` 180. Then SIGTERM, SIGKILL to the process group. A timeout or crash ends the trial at its stage; the next `warm()` restarts the sidecar. |
| Cooperative stop | If declared, tandem polls `should_stop` and signals via the file in `TANDEM_SIDECAR_STOP_FILE`. |
| `close` | Asks the sidecar to quit, then ends its process group, helpers included. |

Overridable: `launch_command`, `launch_cwd`, `launch_env`, `warm_args` (default `output_dir`,
`execute`, `record`), `on_event`; `call(verb, **args)` calls a sidecar-only verb.

**The script** imports only `tandem_sidecar` (one standard-library file, Python 3.8+) from tandem;
the scaffold's `sidecar.py` is complete. It has one method per verb (keyword arguments in, JSON-safe
dict out; `capture_frame` returns `{"path": ...}`) and ends `raise SystemExit(serve(World()))`.
`serve(handlers, verbs=None, on_exit=None)` takes that object or a verb → callable mapping, answers until
tandem says quit or closes stdin, then calls `on_exit` or the `close` handler. The kit also
has `log(message, level="info")` (any thread), `event(name, **fields)` (no `id`, `log` or `event` fields)
and `should_stop()`. A raising handler is reported as `"<verb> failed -- <Type>: <message>"`, traceback in
the session log. Wire protocol: the docstring of `src/tandem/planners/sidecar_kit/tandem_sidecar.py`.

**stdout belongs to the protocol:** importing `tandem_sidecar` points fd 1 at stderr, sending stray
output to the session log. Import it first (then `# isort: split`) and the planner's modules in
`warm()`; the conformance kit checks.

**`PYTHONPATH`:** tandem puts the kit (`planners/sidecar_kit` in its package) first on it. If your
environment's activation *sets* it (e.g. pixi `[activation.env]`), the sidecar dies with
`No module named 'tandem_sidecar'`; append instead. To run a sidecar by hand, prepend the kit yourself.

## A runtime recipe

A planner needing more than pip declares its runtime as data, `recipe = RuntimeRecipe(...)` (from
`tandem.planners`). `tandem planners install NAME` builds it, `list` shows if it is current,
`remove NAME` deletes it. TiPToP's: [CONFIGURATION.md](CONFIGURATION.md#the-planner-runtime).

| part | fields |
|---|---|
| `RuntimeRecipe` | `planner` (equals `info.name`), `title`, `sources`, `environment`, `steps`, `assets`, `notes`. |
| `Source` | `SourcePin(name, url, commit, ref=)`, `trim` (paths deleted after fetching), `patches` (in order; a failure stops the install), `marker` (only in a complete tree), `persistent` (run-time directories kept across trees). |
| `PixiEnvironment` | `manifest` (the planner's pixi manifest and lock, in a source), `home` (default `env`, outside every tree), `env`. |
| `BuildStep` | `name`, `task` (from the manifest), `env`, `produces` (globs present once run), `description`; `optional` (the runtime works without it: a failure doesn't fail the install, and it shows as a note), `requires` (absolute paths it needs, such as an SDK's installer; skipped until they exist) and `missing` (what won't work meanwhile, and the fix). TiPToP's ZED step is one. |
| `Asset` | `(source, dest)`: a package file copied into the runtime. |

- **Pins** are full 40-character commits; `ref` (the branch) is shown by `tandem planners info` and
  `tandem runtime status`, never compared.
- **Fetching:** a shallow `git fetch` and `git archive` of the commit (else its `ref` branch, or
  GitHub's archive without git), always checked.
- **Layout:** a directory per source; `env/` (the environment, linked as `.pixi` beside the manifest);
  `cache/` (`persistent` directories); `.tandem-runtime.json` (what is installed).
- **Moving a pin** swaps that tree without re-solving the environment; `persistent` directories survive.
  The installer never deletes a git checkout, and replaces unlisted trees only where
  `.tandem-runtime.json` exists.
- **Placeholders** in `env` values: `{root}`, `{source:NAME}` (a tree's path), `{commit:NAME}`,
  `{version:NAME}` (that commit as a PEP 440 version, `0.0.0+g4db8f92`, for packages versioned from git).
- **Status** compares `.tandem-runtime.json` with the recipe; a moved pin shows `outdated` with the
  rebuild command.
- **Offline:** `--sources DIR`, `$TANDEM_PLANNER_SOURCES`, `tandem planners bundle`
  ([offline install](CONFIGURATION.md#offline-install)).
- **ffmpeg:** the merge joins videos with the environment's `bin/ffmpeg`, else the one on `PATH`.

## Options and doctor rows

**Options** are of two kinds. The task's, in each profile's `planner.options`, are declared in `OPTIONS` and
checked by `validate_options(options)` when a profile loads. This machine's, in rig.yml's `planners.<name>`
(`tandem rig set planners.NAME.KEY VALUE`), are declared in `RIG_OPTIONS` and checked by
`validate_rig_options(options)` when the rig is read. The defaults refuse keys not declared, suggesting the
nearest, and say where a key put in the wrong one belongs. Overrides (pydantic works well) must:

- Accept their own output (it is re-validated on every read).
- Return plain data (string-keyed mappings, lists, strings, numbers, booleans, `None`), e.g.
  `model_dump(mode="json")`; not `Path`, `Enum` or numpy values.
- Raise `TandemError` or `ValueError` (including pydantic's `ValidationError`), reported under
  `planner.options.` or `planners.<name>.`.

For a required setting, refuse `{}` with a `TandemError` naming it: a machine setting (a robot's address) in
`validate_rig_options`, set with `tandem rig set`; a task setting in `validate_options`, set with
`tandem planners use NAME --option KEY=VALUE`. `tandem init` fills a planner's machine settings with what
`validate_rig_options({})` returns.

`describe_options(profile, *, settings=None)` returns the `OptionsView` (`summary`, `sections`,
`receives`, `receives_note`, `warnings`) shown by `tandem profile show`, the web editor and the session
header (default: each option as set). `tandem profile show NAME --planner` prints `receives`: exactly what
the planner gets.

**Doctor rows.** `doctor_checks(profile, *, settings=None, probe_hardware=True)` returns
`tandem.core.probe.Check(name, state, detail, hint, group)` rows (`state`: `probe.OK`, `WARN`,
`FAIL`, `SKIP`) for `tandem doctor`; default none. `profile=None` (`tandem init`'s preflight): check
the machine only. `probe_hardware=False` (`--no-hardware`): touch no network or bus. A FAIL stops a
session, and stops `tandem init` before it builds the runtime (interactive: it asks). tandem already
reports the runtime; don't repeat it.

## Registering it

First match wins:

1. **`tandem.register_backend(name, factory)`** (alias `register_planner`), at runtime; `factory` may
   be a lazily imported `"module:attribute"`. A taken name errors unless `replace=True`.
2. **Built in:** TiPToP.
3. **The `tandem.planners` entry point** (the scaffold writes it):

   ```toml
   [project.entry-points."tandem.planners"]
   arm = "tandem_arm.planner:ArmPlanner"     # a Planner subclass, or any BackendFactory
   ```

`tandem planners list`, `info` and profile loading import plugins, so keep the entry-point module
light: import torch and robot clients in `warm()` or a sidecar.

A plugin that fails to import is listed `broken` with its error; the rest work, and a profile naming it
still loads, options unchecked. One shadowed by a registered or built-in planner is listed with why. Two
installed packages claiming one name is an error.

Select it with `planner: {backend: arm}` or `tandem planners use arm`; `tandem planners default arm`
makes it new profiles' planner ([commands](USAGE.md#commands), [HTTP API](USAGE.md#http-api)).

## The recording contract

`execute` records one leg of trial `leg.trajectory_id`; the merge orders a trial's legs by recording
window and joins them into one episode. With `leg.record` set, `save_dir` must hold:

**`_meta.json`:**

| key | value |
|---|---|
| `trajectory_id`, `segment_source` | From `leg` (`"tamp"` for a planner leg); how the merge finds the leg. |
| `instruction` | `leg.instruction`: the whole task, the dataset's language label. |
| `phase_index`, `n_phases`, `phase_description` | From `leg`, when `leg.phase_index` is not `None`. |
| `record_start`, `record_stop` | Epoch seconds; legs are ordered by them. |
| `fps` | Frames per second. |
| `cameras` | Dataset key → clip file, e.g. `{"exterior_image_1_left": "external_cam.mp4"}`. |
| `plan_file` (optional) | The saved plan's bare file name (default `tiptop_plan.json`), for `tandem traj show` and the web UI. |

**`robot_state.npz`:** each `tandem.core.merge.STATE_KEYS` array, one row per frame, optionally
`action_joint_velocity` (`[F,7]`), nothing else. `cmd_*` are commanded, the rest measured.

| array | shape |
|---|---|
| `joint_position`, `cmd_joint_position`, `cmd_joint_velocity` | `[F,7]` |
| `gripper_position` | `[F]`, in `[0,1]` |
| `cmd_gripper` | `[F]`, binary ([the export](USAGE.md#exporting) skips an episode where it isn't) |
| `frame_time` | `[F]`, wall clock, float64 |

Record measured arrays from the robot, not command copies (a policy would learn to echo them).

**Camera clips:** every clip `cameras` names must exist (at least one if it names none), each called
`external_cam.mp4`, `external_cam_2.mp4` or `hand_cam.mp4` whatever its dataset key: only these are
merged, viewed or exported. Only cameras every leg recorded are joined.

`tandem.core.trajectories.is_complete(leg_dir)` is the check. Other files are the planner's; the
merge copies the first planner leg's into the episode.

**Edge cases:**

- Stamp `_meta.json` even if execution fails part-way or records nothing (`n_frames=0`, like the
  scaffold), or the leg becomes its own episode. Return `rollout_dir` (usually `save_dir`) and
  `n_frames`.
- `ExecuteResult.stopped_early=True`: a cooperative stop (preempt or session stop) was honoured; the
  leg never advances the plan, whatever `ok` says. After a preempt the trial is filed aborted; an
  unrequested stop is a `tamp_execution` failure.

Merged episode layout: [DATA.md](DATA.md#episode-layout).

## The conformance kit

`tandem.planners.testing` runs the whole protocol with no GPU, robot or session:

```python
from tandem.planners.testing import PlannerConformance
from tandem_arm.planner import ArmPlanner

class TestArmPlanner(PlannerConformance):
    planner = ArmPlanner          # a Planner subclass, a BackendFactory or a registered name
    records_legs = True           # False: check only the _meta.json stamp
    options = {}                  # its planner.options
    rig_options = {}              # its machine settings (rig.yml planners.<name>)
    task_hint = "put one thing where it belongs"
```

Tests cover the declarations, options, machine settings, doctor rows, sidecar script, lifecycle, every verb and
(with `records_legs`) the recording, skipping anything not declared or shipped.

Override `goal(scene, caps)` if the kit can't guess a plannable goal, and `make_backend(tmp_path)` to
build the backend differently. Each check is also a function raising `ConformanceError`:
`check_declarations`, `check_protocol`, `check_scene`, `check_plan_result`, `check_leg`,
`check_sidecar_script`.

Two switches, on by default, hold it to phase planning's needs (turn one off only for a planner
never run that way; the scaffold passes both with stand-in images):

- **`phase_planning = True`**: every scene's `rgb_path` must open as an image.
- **`verifies_human_phases = True`**: `capture_frame(camera="external")` must return an image.
