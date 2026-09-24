<div align="center">

<h1>TANDEM</h1>

**Task and Motion Planning with As-Needed Demonstrations**

A planner does the parts of a task it can. A person does the rest, only where needed. Every trial
becomes one demonstration for fine-tuning a vision-language-action model.

[Paper website](https://prpl-group.com/tandem/) ·
[Method, in code](docs/METHOD.md) ·
[Add a planner](docs/ADDING_A_PLANNER.md) ·
[Add a human executor](docs/ADDING_A_HUMAN_EXECUTOR.md)

[![CI](https://github.com/SamratSahoo/tandem/actions/workflows/ci.yml/badge.svg)](https://github.com/SamratSahoo/tandem/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-4f9dff)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20NVIDIA-a371f7)](NOTICE)
[![Platform](https://img.shields.io/badge/platform-linux%20%C2%B7%20cuda%2012-3fb950)](#requirements)

```bash
pipx install git+https://github.com/SamratSahoo/tandem.git && tandem init
```

</div>

This is the code for *TANDEM: Task and Motion Planning with As-Needed Demonstrations for Efficient
Vision-Language-Action Model Fine-tuning* (Samrat Sahoo, Liang Ji, Tom Silver, Yixuan Huang), as a
pip-installable package, `tandem-tamp`, with one command, `tandem`.

---

## What it is

Human teleoperators spend substantial time demonstrating things a robot can already do on its own.
A task and motion planner (TAMP) can do those parts autonomously. But its fixed planning domain
rarely covers every stage of a long-horizon task, so on its own it cannot finish one.

TANDEM treats human help as **an on-demand planning capability**. Given an instruction and a photo
of the workspace, a vision-language model (VLM) looks at what the planner can do. It invents the
predicates and the human-executed **magic operators** the planner's domain is missing. Then it plans
the task as an ordered list of phases, each one either the planner's or a person's. The planner runs
its phases, and a person teleoperates the rest. After every human phase, a fresh photo checks that
the step really happened. The whole trial is recorded as **one demonstration**, which is what a VLA
is fine-tuned on.

```
   "place the bread inside the box"       the bread sits on the box's closed lid
                   │
                   ▼
   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │ phase 0  robot   On(bread, plate)          TAMP plans and executes it           │
   │ phase 1  human   LiftOpen(box)             magic operator:                      │
   │                    pre  HandEmpty()          invented predicate IsOpen(box):    │
   │                    add  IsOpen(box)          "the lid of {0} is lifted open"    │
   │                  → a person teleoperates; a new photo must show IsOpen(box)     │
   │ phase 2  robot   On(bread, box)            TAMP, planned from a fresh look      │
   └─────────────────────────────────────────────────────────────────────────────────┘
                   │
                   ▼
   one episode: [ planner leg | teleop leg | planner leg ],  each stretch labeled with its phase
```

**tandem plans the task; a planner plans the motion.** The model decides what each phase must
achieve and in what order. The planner decides how the robot's phases are carried out. tandem
decides who does what, hands the arm over, checks the person's work, and records it. None of that
lives inside the planner. It is asked for one thing, *achieve this goal in this scene and record
what you did*, so another planner can be swapped in without touching the method
([ADDING_A_PLANNER.md](docs/ADDING_A_PLANNER.md)). [TiPToP](https://github.com/SamratSahoo/tiptop)
(cuTAMP and cuRobo on a Franka) is the planner that ships.

---

## The method, in brief

The full mapping from the paper to the code is [docs/METHOD.md](docs/METHOD.md).

- **Phases.** A task is an ordered list of phases φ_k = (γ_k, e_k): a sub-goal and who carries it
  out, the robot or a person. The ordering runs both ways. "Put the toy on the cloth, then fold it"
  needs the person last; "open the box, then put the toy in" needs the robot last. Some tasks need an
  intermediate state that no final-state goal can express.
- **Invented predicates.** Where the planner's predicates cannot say what a step achieves, the model
  invents one: a name, typed arguments, and a sentence saying what must be *visible* for it to hold,
  such as `IsOpen(x)`: "the lid of {0} is lifted open". That sentence is the whole definition. It is
  what a person is asked to bring about, and what the VLM classifier checks in a photo afterwards.
- **Magic operators.** Every human phase is an explicit operator with **preconditions**, **add
  effects** and **delete effects**, carried out by a person through teleoperation: `LiftOpen(box)`,
  pre `HandEmpty()`, add `IsOpen(box)`. The planner never sees one. It is the phase's contract.
- **The plan is checked before the arm moves.** Each proposal is validated. Every atom must name a
  detected object, every robot phase must ask only for what the planner can achieve, and every
  operator must be coherent. Then the operators are checked against each other, symbolically: no
  phase may need something an earlier phase made false. A rejected proposal goes back to the model
  with the reason, **at most 3 attempts** in all.
- **Execution.** A robot phase goes to the planner with the current scene, and the scene is
  perceived again before every robot leg. A human phase is shown to the operator in words, together
  with what will be checked.
- **Every human phase is verified.** After the step, a fresh image is taken. Its add effects must
  hold, and its delete effects must no longer hold. A failed check gets one more try by default.
- **A trial that fails verification is excluded.** It is terminated and kept out of the dataset:
  filed under `failure/`, marked `excluded: true`, with its failing verdicts and raw legs kept for
  inspection, and never offered for a label.
- **One merged demonstration per trial.** Every leg of a successful trial, the planner's and the
  person's, is joined into one episode, τ = ((τ₁, φ₁), …, (τ_N, φ_N)). Each stretch of frames says
  which phase it carried out.
- **DATAFARM alignment.** For TiPToP, the paper's preset pulls planned motions toward the DROID
  teleoperation distribution with a VAE-manifold cost.

---

## Install

`tandem` is a command-line tool, so install it with [pipx](https://pipx.pypa.io). That gives it a
private environment and puts only the `tandem` command on your PATH:

```bash
pipx install git+https://github.com/SamratSahoo/tandem.git
tandem init
```

`uv tool install git+https://github.com/SamratSahoo/tandem.git` does the same with uv.

> **Not on PyPI yet**, so the install is from git. Plain `pip install` works *inside a
> virtualenv*, but on Debian and Ubuntu it fails against the system Python with
> `externally-managed-environment`. pipx exists for exactly this.

`tandem` itself is pure Python. The heavy stack a planner needs (torch, cuRobo's compiled CUDA
kernels, cuTAMP, tiptop) is built on your machine into a self-contained **runtime**, and none of it is
in the package. `tandem planners install tiptop`, which `tandem init` runs for you, fetches the exact
commits this version of tandem pins:

| source | branch | pinned commit |
|---|---|---|
| [SamratSahoo/tiptop](https://github.com/SamratSahoo/tiptop/tree/TANDEM) | `TANDEM` | `6820474` |
| [SamratSahoo/cuTAMP](https://github.com/SamratSahoo/cuTAMP/tree/TANDEM) | `TANDEM` | `fc8f233` |
| [SamratSahoo/curobo](https://github.com/SamratSahoo/curobo) | `main` | `3a90ff4` |

The `TANDEM` branches are each fork's `main` plus LJ1356's surface-fitted placement, ported so that
the paper's placement tasks can be reproduced ([below](#surface-fitted-placement-placement_)). All of
it is off until a profile turns it on: without those settings the two branches plan exactly as the
mains do. The branch is recorded with each commit, in `tandem planners info tiptop` and in the
runtime's own record.

Each is fetched with `git fetch --depth 1` and exported with `git archive` (GitHub's archive of the
commit when there is no git), and the commit is checked. The install then applies one small patch,
solves tiptop's own pixi environment, and compiles cuRobo's kernels: 5–20 minutes the first time. It
writes down exactly what it installed. When a tandem upgrade moves a pin, `tandem planners list`
says the runtime is `outdated` instead of letting a session fail forty seconds into warm-up.

**A workstation that cannot reach GitHub** takes the sources from a bundle made elsewhere, by the
same version of tandem, on a machine that can:

```bash
tandem planners bundle tiptop --out /media/usb/planner-sources
```

then, on the workstation, `tandem planners install tiptop --sources /media/usb/planner-sources`, or
set `TANDEM_PLANNER_SOURCES`. With a sources directory in force no source is fetched, and each export
is checked against the pinned commit, and its files against the digest taken when it was bundled. A
bundle carries the sources only: `pixi install` still downloads the environment from conda-forge and
PyPI (and builds SAM-2 from GitHub), and TiPToP's first warm-up downloads the SAM-2 checkpoint, so the
workstation still needs those.

<table>
<tr><td width="50%">

**On the robot workstation**

```bash
tandem init
```

Checks the machine and the GPU, installs [pixi](https://pixi.sh) if you agree, builds the planner's
runtime, takes your Gemini key, creates a profile, and optionally sets up the teleop driver. It is
idempotent: re-run it any time.

</td><td width="50%">

**On a laptop**

```bash
tandem init --viz-only
tandem ui
```

No GPU, no robot, no cameras. Browse and review trajectories collected elsewhere: point the data
root at them. `tandem plan` works here too, with a Gemini key.

</td></tr>
</table>

Working on tandem itself? An editable install in a virtualenv:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[export,dev]'
pytest -q
```

The LeRobot export needs the `export` extra: `pipx inject tandem-tamp av pyarrow huggingface_hub`.

Already have a `hitl-tamp-vla` checkout? Import its setup instead of retyping it:

```bash
tandem init --import-from ~/hitl-tamp-vla
tandem profile create bread-box --import-from ~/hitl-tamp-vla --tamp-config ~/hitl-tamp-vla/data-collection/cfg/tamp/4_bread_box_v3.yml
```

What comes across as a profile:

- the robot config, camera serials and extrinsics, per-workspace layers included;
- a task config's TAMP settings, the `placement_*` keys of the puzzle and bread/box configs
  included, under the same names;
- its `hitl:` block (`HITL_KEYS` in `src/tandem/planners/tiptop/importers.py` says what each key
  becomes).

What tandem cannot do is refused rather than swapped for something it can:

- A config whose human phases a learned policy carries out (`policy_type: diffusion` or `act`, the
  HITL-TAMP baseline) is not imported with phase planning on, and the refusal names the same task's
  config for a person.
- Settings only tiptop's own rollout loop reads (`auto_mode`, `reset_placement_region`,
  `clear_goal_surfaces`) are left out, with a warning.

Every imported config also gets three switches it never names: what LJ1356's tiptop, which it ran
on, always did and the pinned TiPToP does only when asked
([below](#surface-fitted-placement-placement_)). The import sets them, with a note saying so, so the
profile plans as the config's runs did.

---

## Quickstart

```bash
tandem init                                    # a workstation: runtime, Gemini key, a first profile
tandem doctor                                  # every check, what it found, what to do about it

# A profile that collects the way the paper did, cloned from the rig `init` set up:
tandem profile create bread-box --from default --preset paper --prompt "place the bread inside the box"
tandem profile use bread-box

# Check the decomposition from a photo before the arm moves:
tandem plan "place the bread inside the box" --image workspace.png

tandem collect                                 # collect in the terminal ...
tandem ui                                      # ... or in the browser, and review what was collected
tandem export lerobot --repo myorg/bread-box --push
```

`--preset paper` turns phase planning on, with the paper's models, checks and retries, and TiPToP's
TAMP and DATAFARM settings. It keeps the rig's robot, cameras and perception. It also runs planned
motions at the speed blending gives them, not slowed by `robot.time_dilation_factor`. It says so
when applied: watch the first rollouts with a hand on the stop.

**While a session runs** (`tandem collect`), the footer shows the keys the current state accepts:

| state | keys |
|---|---|
| planning or executing | `p` preempt · `t` hand the arm to a person (when teleop is set up) |
| a human phase | `t` take the arm (or run the configured executor) · `d` I did it (only when allowed) · `a` give up on this task |
| an operator hand-off | `r` return control |
| the label prompt | `s` success · `f` failure |
| the task prompt | `↵` repeat the task · `n` new task |

`q` finishes the session from any state, parking the arm first. It is a step boundary like a
preempt: nothing more is perceived, planned or executed, the trial in flight is filed as aborted,
and one waiting at the label prompt is left unlabeled in `eval/` with its `hitl.json`
(`tandem traj merge <trajectory id> --status success` files it later).

Three things a person can do during a session:

- **Preempt.** Abandon the attempt in flight. The session stays warm, and you are back at the task
  prompt in a second; what was recorded is filed as aborted. It does not stop the arm mid-motion
  (see [Troubleshooting](#troubleshooting)).
- **Hand off.** Take the arm between phases. The planner lets go of the robot and cameras at the
  next boundary. When you hand back, the same phase is perceived and planned again from wherever you
  left the arm: no homing, no dropped object. Every leg merges into **one** trajectory.
- **Label.** Mark the trial success or failure while watching its video. Only a trial whose plan ran
  to the end is offered for a label (plus a failed check, with `on_verification_failure: label`). One
  the loop ended itself -- excluded, failed at a TAMP or human-policy stage, or aborted -- is filed
  under `failure/` without one, and the prompt says why.

---

## The planner catalog

```console
$ tandem planners list
    planner                   status          summary
───────────────────────────────────────────────────────────────────────────────────────────────
●   tiptop  TiPToP  default   not installed   GPU task and motion planning with cuTAMP and cuRobo, perceiving…

  · profile 'default' plans with tiptop
  · new profiles plan with tiptop  `tandem planners default NAME`
```

| | |
|---|---|
| `tandem planners list` | Every planner tandem can see: built in, or registered by an installed package. Each is `installed`, `not installed`, `outdated`, `no runtime needed` or `broken` (with why). `●` marks the one the active profile uses. |
| `tandem planners info NAME` | What it is and needs, the commits it pins (and the branch each is taken from) against what is installed, its goal language as the phase planner sees it, what it supports, the `planner.options` it reads, and its presets. |
| `tandem planners install NAME` | Fetch its pinned sources and build its runtime. Safe to re-run: an installed, current runtime returns at once, and an outdated one is rebuilt. `--sources DIR` takes the sources from DIR instead of GitHub; `--force` rebuilds; `--yes` asks nothing. |
| `tandem planners use NAME` | Make a profile plan with it (`--profile P`; `--option KEY=VALUE` for a setting it needs; `--default` for every new profile too). The old planner's `planner.options` leave the profile, named, and are kept beside it in `planner-options.<planner>.yml`: switching back restores them. It also repairs a profile naming a planner this machine no longer has. |
| `tandem planners default NAME` | The planner new profiles get. Changes no profile. |
| `tandem planners bundle NAME --out DIR` | Its pinned sources in DIR, for `install --sources DIR` on a machine that cannot fetch them. |
| `tandem planners remove NAME` | Delete its runtime. It stays listed and can be installed again. |
| `tandem planners new NAME [--sidecar]` | Scaffold a package for a planner of your own. It passes tandem's conformance kit as generated. |
| `tandem executors list` | Who can carry out a human phase (`hitl.human_executor`), whether each is ready here, and what one still needs. |
| `tandem executors use NAME` | Make a profile's human phases run with it. |

TiPToP is the only planner that ships. Another is a package that registers under the
`tandem.planners` entry point, after which `planner: {backend: NAME}` in a profile is all it takes;
nothing in tandem changes. [docs/ADDING_A_PLANNER.md](docs/ADDING_A_PLANNER.md) walks through it,
from `tandem planners new` to a planner that passes the conformance kit. A human phase is carried out
by `teleop`, a person driving the arm, unless a package adds another executor
([docs/ADDING_A_HUMAN_EXECUTOR.md](docs/ADDING_A_HUMAN_EXECUTOR.md)).

---

## Plan a task from a photo

The phase planner is tandem's own code. It needs no planner, no GPU and no robot, only the planner's
declared goal language and a Gemini key. So you can check a decomposition before collecting:

```console
$ tandem plan "place the bread inside the box" --image workspace.png -o bread -o box -o plate
  · objects  box, bread, plate

◆ place the bread inside the box  3 phase(s) · planner: tiptop
  0  robot  move the bread off the box's lid onto the plate
        · bread is resting on top of plate
        goal: [{"predicate": "on", "args": ["bread", "plate"]}]
  1  human  lift the box's lid open
        · the lid of box is lifted open, so its inside is visible
        Lift the lid of the box until it stays open.
        operator LiftOpen(x0: surface)  as LiftOpen(box)
          preconditions  HandEmpty()
          add effects    IsOpen(box)
          delete effects none
  2  robot  put the bread inside the box
        · bread is resting on top of box
        goal: [{"predicate": "on", "args": ["bread", "box"]}]

  ✔ the phases' contracts hang together  checked in the repair loop

invented predicates ─────────────────────────────────────────────────────────────
  IsOpen(surface)
      the lid of {0} is lifted open, so its inside is visible
```

It prints:

- the ordered phases, who does each, and the goal the planner would be handed;
- each human phase's operator;
- whether the operators hang together;
- the invented predicates and their classifier sentences;
- loudly, any clause of the instruction the model could not express.

That last one matters: the usual cause is an object the instruction names that was not detected.
The remedy, putting it on the table or rewording the task, is only available *before* you start
collecting.

- `--object` / `-o LABEL`, repeated, pins the object labels. Without it, a vision model names the
  objects in the photo first.
- `--profile P` takes the planning settings from a profile, and its planner.
- `--planner NAME` plans in another planner's goal language (`--backend` is its older name).
- `--json` prints the plan record, the same structure `hitl.json` has.
- `--save-vlm-io DIR` keeps every image and reply.

The same thing from Python, for scripts and notebooks:

```python
import tandem

plan = tandem.plan_task("place the bread inside the box", "workspace.png", planner="tiptop",
                        objects=["bread", "box", "plate"])
for i, phase in enumerate(plan.phases):
    print(i, phase.executor, phase.description, sorted(map(str, phase.atoms)))
# 0 robot move the bread off the box's lid onto the plate ['On(bread, plate)']
# 1 human lift the box's lid open ['IsOpen(box)']
# 2 robot put the bread inside the box ['On(bread, box)']
```

`plan_task` runs exactly the steps `tandem plan` runs, and returns the `PhasePlan` a session would
walk: its phases, operators, invented predicates, and `to_json()`. `plan_task_async` is the same for
code already inside an event loop. `import tandem` imports nothing heavy: the whole public surface
(`plan_task`, `PhasePlan`, `PlanningConfig`, the planner SDK's `Planner`, `SidecarPlanner`,
`Capabilities`, `PlannerInfo`, `Predicate`, `Parameter` and `RuntimeRecipe`, `register_backend` (or
`register_planner`, the same function), `register_human_executor`, `TandemError`) is resolved on first use.

---

## Configuration

A **profile** is one collection setup and everything collected under it:

- the task;
- the cameras;
- phase planning (`hitl:`);
- the planner and its own settings (`planner:`);
- the trajectories.

```
~/tandem-data/profiles/bread-box/
├── profile.yml            the whole setup
├── calibration.json       camera extrinsics, keyed by serial
└── trajectories/
    ├── eval/              recorded, not yet labeled
    ├── success/
    └── failure/           failures, and excluded trials
```

```bash
tandem profile create bread-box --prompt "place the bread inside the box"   # from the template
tandem profile create bread-paper --from bread-box --preset paper           # clone, then lay a preset over it
tandem profile use bread-box
tandem profile edit bread-box          # $EDITOR, validated on save
tandem profile show bread-box          # the whole profile
tandem profile presets                 # the presets --preset can lay over a new profile
```

Switching profiles re-points collection, inspection and export in one move. Two robots, two tasks,
or two TAMP regimes you want to compare: each is a profile.

**Unknown keys are rejected at load time, with a suggestion.** This is deliberate. In the system
tandem is extracted from, a config shipped `blend_ops: [Pick, MoveFree. MoveHolding]`, one typo'd
period, and the planner silently ignored the whole list for months. A setting that quietly does
nothing is the one failure that looks exactly like success.

### `hitl:`, phase planning

Off by default. Disabled, nothing in `tandem.planning` runs, and a session plans each attempt with
the planner's own reading of the instruction. `--preset paper` turns it on with the paper's settings,
which are these defaults apart from `enabled`. What each one changes in the trial loop is in
[docs/METHOD.md §5](docs/METHOD.md#5-every-hitl-setting-and-what-it-changes).

```yaml
hitl:
  enabled: false                      # phase planning on or off
  proposal_model: gemini-2.5-pro      # splits the task, invents predicates and operators
  vlm_model: gemini-2.5-flash         # answers each per-atom check
  max_attempts: 3                     # answers per proposal, each rejection fed back; also the re-plan budget
  classify_initial: false             # measure the invented predicates on the first image
  verify_retries: 1                   # extra goes at a human phase whose check failed
  verify_enforced: true               # false: record a failed check and carry on
  on_verification_failure: exclude    # exclude (the paper) | label (ask the operator anyway)
  verify_final_phase: true            # check the last phase too, when it is a person's
  check_human_effects: true           # add effects must hold, delete effects must not
  check_human_preconditions: false    # check a human phase's preconditions before the hand-off
  check_tamp_preconditions: false     # check what earlier phases left, before a robot leg
  check_tamp_effects: false           # check a robot leg's effects (recorded, never enforced)
  precondition_enforced: false        # an unmet precondition stops the trial
  check_plan_effects: true            # the symbolic contract check, in the repair loop
  save_vlm_io: true                   # keep every image sent to a model and its reply, as vlm/
  cache_path: null                    # SQLite cache for proposals only
  on_robot_phase_failure: abort       # abort (the paper) | teleop | replan
  conjoin_robot_phases: true          # consecutive robot phases as one goal, where sound
  human_executor: teleop              # who carries out a human phase
  human_executor_options: {}          # each executor's own settings, under its name
  allow_unrecorded_human_phase: false # while recording, accept a human step done off the record
  verification_camera: external       # external | hand | perception
```

- **A phase the planner cannot plan ends the trial**, the way the paper counts it
  (`on_robot_phase_failure: abort`). `teleop` describes the sub-goal to you instead: you carry it out,
  the same check verifies it, and the task carries on. `replan` proposes the task again, telling the
  model why the planner could not plan that phase. A leg that was planned but failed to *execute*
  always ends the trial: the arm is somewhere no plan put it.
- **Your part is recorded like the robot's.** A human phase is carried out by `hitl.human_executor`,
  and its leg is stamped with the phase it carried out. Answering "I did it" without teleoperating
  would leave that phase with no demonstration while the episode looks complete. So while recording,
  it is refused unless `allow_unrecorded_human_phase: true`, and a step accepted that way says so in
  `hitl.json` (`phases[k].carried_out`, `checks.unrecorded_human_phases`). `tandem executors list` shows what the
  executor still needs on this machine, and `tandem doctor` says so before you start rather than at
  the first human step.

### `planner:`, the planner and its own settings

Everything only the planner reads lives in its own block, `planner.options`. The planner checks it
when the profile loads. `tandem planners info NAME` lists what a planner reads. For TiPToP that is
`robot`, `perception` and `tamp`. The `tamp` block goes straight to cuTAMP and cuRobo using
**tiptop's own key names**, so anything documented upstream works verbatim, and an existing
`cfg/tamp/*.yml` can be pasted in unchanged.

```yaml
planner:
  backend: tiptop
  options:
    robot: {host: 172.16.0.2, time_dilation_factor: 0.2}   # and the rest of the arm
    tamp:
      num_particles: 256              # cuTAMP coverage per skeleton
      opt_steps_per_skeleton: 250
      traj_length_norm: inf           # charge moves the infinity-norm, not Euclidean
      grasp_pose_change_weight: 0.1   # prefer grasps that reorient the wrist less
      vae_manifold_weight: 25000      # pull trajectories toward the DROID motion manifold
      blend_trajectory: true          # one continuous stroke per operation
      blend_ops: [Pick, Place, GoToInitial]
      blend_boundary_speed: 0.3       # never fully stop at gripper events
```

`tandem profile show NAME --planner` prints exactly what the planner will receive: the answer to "did
my setting apply?".

#### Surface-fitted placement (`placement_*`)

By default cuTAMP may put an object down anywhere in a surface's bounding box, with the object's
bottom at the height of the surface's highest point. That is right for a slab and wrong for anything
with structure: for an open box it is the top of the folded-back lid, for a plate its rim. In
"Store Bread in Closed Box" the bread was released about 19 cm up, out over the box's far wall, and
fell. `placement_support: true` fits the region to what the camera saw of the surface instead: the
level patches that would hold this object's footprint, at that patch's own height.

| key | default | what it does |
|---|---|---|
| `placement_support` | `false` | Turns it on. The six keys below are read only when it is on; set without it, `tandem doctor` says they do nothing. |
| `placement_support_margin` | `0.01` | Surface the object must keep around its footprint, in metres. `>= 0`. |
| `placement_flatness_tol` | `0.008` | How much the surface under a footprint may vary and still count as one level patch, in metres. `> 0`. It absorbs stereo noise too, so a noisy reconstruction needs it raised; it is also how much real slope a placement may sit on. |
| `placement_support_required` | `true` | When no patch of a goal surface would hold the object, the plan fails with that reason. `false` falls back to the bounding box. |
| `placement_into_surface` | `true` | A placed object may overlap the surface it was placed on in the collision check. Placing *into* a container needs it: perception reconstructs one as a filled hull. |
| `placement_fill_occluded` | `false` | The unobserved cells inside a surface's outline count as floor, for a camera that looks across a box and cannot see its floor. The one setting that places onto surface nobody saw. |
| `placement_min_seen_frac` | `0.25` | The fraction of every footprint that must really have been observed: the guard on `placement_fill_occluded`. In `[0, 1]`. |

Two of the paper's tasks use it. **Solve Constrained Puzzle** (`1_toy_puzzle_v3.yml`) sets
`placement_support`, `placement_support_required` and `placement_into_surface`, so the toy rests on
the cloth rather than back on the puzzle board. **Store Bread in Closed Box** (`4_bread_box.yml`,
`4_bread_box_v3.yml`) sets all seven: margin `0.005` (the tray is barely wider than the bread),
flatness `0.012`, `placement_fill_occluded: true` with `placement_min_seen_frac: 0.25`. Import either
config with `--tamp-config` and they come across as they are.

When no surface can hold the object, the leg is an ordinary plan failure, so
`hitl.on_robot_phase_failure` decides what happens next: `abort` (the default) ends the trial,
`replan` asks for another plan, `teleop` hands the phase to the operator, as LJ1356's tiptop did.

Those configs were tuned on LJ1356's tiptop, which also always did three things the pinned TiPToP
does only when asked. Importing a config (or `--preset paper`) sets all three, and a key the config
states itself keeps its own value; set them by hand in any other profile that should plan as those
runs did:

| key | what it switches on |
|---|---|
| `table_plane_support_vote` | Pick the table among RANSAC's planes by the objects resting ON each one, not by any object within 3 cm of it either side (which counts objects below a plane too). |
| `disjoint_object_masks` | Build object meshes and point clouds from disjoint masks, every pixel two masks claim going to the smaller object, so a container's hull stops at what rests on it. (The placement fit always uses disjoint masks.) |
| `blend_stretch_to_caps` | With `blend_trajectory` on, slow a stroke that cannot be re-timed inside the velocity and acceleration caps until it fits, instead of running it at the plan's own timing. It can make a stroke many times slower. |

Profiles written before profile version 2 had `robot:`, `perception:` and `tamp:` at the top level.
They still load, with a one-line notice, and are written in the new layout the next time they are
saved. `tandem profile migrate` rewrites them all at once, keeping each old file as
`profile.yml.v1.bak` and leaving a current profile byte for byte as it is. A version-1 profile says
`on_robot_phase_failure: teleop` (version 1's default, whether or not anyone chose it); it is kept, and
named in the notice, since the default is now `abort`. An older tandem cannot read a version-2
profile.

### Machine settings

`~/.config/tandem/config.toml`, edited with `tandem config list | get | set | edit`:

| key | default | |
|---|---|---|
| `active_profile` | `default` | Set by `tandem profile use`. |
| `data_root` | `~/tandem-data` | Where profiles and trajectories live. `$TANDEM_DATA_ROOT` wins. |
| `runtime_dir` | `~/.local/share/tandem/runtime` | TiPToP's runtime (about 25 GB). `$TANDEM_RUNTIME_DIR` wins. Other planners' live in `~/.local/share/tandem/runtimes/NAME` (`$TANDEM_RUNTIMES_DIR`). |
| `default_planner` | `tiptop` | The planner new profiles get (`tandem planners default NAME`). |
| `hf_org` | | The default Hugging Face owner for `tandem export lerobot`. |
| `teleop.enabled`, `teleop.droid_dir`, `teleop.python`, `teleop.device` (`vr` or `spacemouse`), `teleop.controller` (`right` or `left`) | off | The teleop driver: a DROID checkout and its environment's interpreter. |
| `ui.host`, `ui.port`, `ui.open_browser` | `127.0.0.1`, `8787`, `true` | `tandem ui`. |

Credentials go in `credentials.toml` beside it, through `tandem config set-gemini-key` (`--stdin`
keeps the key out of your shell history) and `tandem config set-hf-token`. `GEMINI_API_KEY` or
`GOOGLE_API_KEY` in the environment wins over a stored key.

---

## What a trajectory looks like

Every leg of a trial is recorded under `eval/` as it happens. When the trial is labeled, or filed
without a label, its legs are merged into one episode and filed:

```
trajectories/success/2026-08-16_21-14-02/
├── external_cam.mp4  external_cam_2.mp4  hand_cam.mp4    every leg's clips, joined
├── robot_state.npz        every leg's per-frame arrays, joined
├── _meta.json             instruction, fps, timing, lineage, and segments[]
├── hitl.json              the phase plan and everything that happened to it   (phase planning)
├── vlm/                   every image sent to a model, and what it said        (phase planning)
├── tiptop_plan.json       the first TiPToP leg's plan
└── segments/
    ├── 00_tamp_2026-08-16_21-14-02/       each raw leg, as recorded
    ├── 01_teleop_2026-08-16_21-15-40/
    └── 02_tamp_2026-08-16_21-16-55/
```

- **`_meta.json`** carries `segments[]`: one entry per leg, with its `source` (`tamp` or `teleop`),
  its frames and its span of the video. It also carries `phase_index` and `phase_description`, which
  is how τ = ((τ₁, φ₁), …) is read back off one episode. The UI draws it as a ribbon under the
  videos.
- **`hitl.json`** holds:
  - the instruction and the planner;
  - the phases, with each robot phase's goal and the operators the planner ran, and each human
    phase's instructions and magic operator;
  - the invented predicates;
  - which clause of the instruction each phase covers, and what was left out;
  - which checks ran, and every verdict, failing ones included;
  - how the trial ended: `outcome` (`success`, `failure`, `excluded` or `aborted`), `failure_stage`
    (`invention`, `tamp_planning`, `tamp_execution`, `verification`, `human_policy`) and
    `excluded`;
  - after an `on_robot_phase_failure: replan`, every plan the trial replaced (`superseded_plans`,
    each with its phases, verdicts and why it was given up). Each segment then also says which plan
    its phase belongs to (`plan_generation`).

  The full schema is in [docs/METHOD.md §6](docs/METHOD.md#hitljson).
- **`vlm/`** holds each image sent to a model, a rendered PNG of what it answered (rejected
  attempts included, and a proposal replayed from the cache, marked as such), and `index.jsonl` with
  every prompt and reply. When a run goes wrong, the
  question is always "what did the model see, and what did it decide". That is unanswerable
  afterwards without it.

| array | shape | what it is |
|---|---|---|
| `joint_position` | `[F,7]` | **measured** arm encoders: no lead, no lag, no plan fallback |
| `gripper_position` | `[F]` | **measured** gripper closedness, continuous in `[0,1]` |
| `cmd_joint_position` | `[F,7]` | **commanded** joint targets |
| `cmd_joint_velocity` | `[F,7]` | **commanded** joint velocities |
| `cmd_gripper` | `[F]` | the commanded gripper, **binary** 0 or 1 |
| `frame_time` | `[F]` | wall clock, float64 (float32 would collapse every frame to one timestamp) |
| `action_joint_velocity` | `[F,7]` | optional: the DROID joint-velocity action, when a TiPToP leg recorded it |

Proprioception and action are **decoupled on purpose**. When the action is a lagged copy of the
measured state, a policy learns to echo it, and a fine-tuned policy that has learned to echo the
gripper never closes it. `tandem export lerobot` skips an episode whose `cmd_gripper` is not binary,
loudly, rather than write a dataset with that defect in it. It exports `success/` only, so failed
and excluded trials never reach a dataset -- and it also skips a trial under `success/` whose
`hitl.json` says the method settled it (excluded, aborted, or failed part-way), however it got there.

The on-disk format is unchanged from the system tandem was extracted from, so data moves between
the two in either direction.

---

## The UI

`tandem ui` opens a browser view of everything a profile has collected. The package serves it
itself: no Node, no build step, no CDN.

- **Trajectories**: every rollout, with its camera videos and per-frame plots. Click a chart to
  seek every video to that frame. A merged trajectory gets a ribbon showing which stretch the planner
  drove and which a person did; click a stretch to jump to it.
- **Collect**: run a session from the browser, with the same controls as the terminal. A human
  phase shows its instructions and what will be checked. A clause the plan leaves out is shown before
  the arm moves. The label prompt has an inline review, so you decide while looking at the rollout
  rather than from memory. An excluded trial says so, and is never offered for a label.
- **Profiles**: edit a profile and see exactly what its planner will receive.
- **Settings**: credentials and paths; the planners and human executors (which is installed, which
  the profile uses, a button to switch, and the command that installs one); runtime status; and the
  full diagnostic report.

The plots show something worth knowing: the shaded bands are the frames that π₀.₅-DROID's
**non-idle filter will throw away at training time**. A rollout that looks fine can be 40% idle.
tandem runs the real filter, at full resolution, on the clipped action the dataset actually stores,
so what you see is what training sees.

---

## Commands

| | |
|---|---|
| `tandem init` | Set up this machine: checks, the planner (`--planner NAME`) and its runtime, the Gemini key, a profile (`--import-from DIR`, `--preset NAME`), teleop. `--viz-only` for a laptop. Idempotent; `--repair` redoes the runtime, key and teleop steps and never touches an existing profile. |
| `tandem doctor` | Every check, what it found, and what to do about it. `--no-hardware` skips the robot, camera and grasp-server probes. |
| `tandem collect [profile]` | Run a session in the terminal. `--task`, `--episodes N`, `--no-execute` (plan without moving), `--no-record`, `--web`. |
| `tandem plan "<task>" --image photo.png` | Decompose a task into phases from a photo, with no robot and no GPU. |
| `tandem ui` | Serve the browser UI. |
| `tandem profile list \| show \| create \| presets \| migrate \| use \| edit \| delete \| path` | Manage profiles. `show --planner` prints what the planner receives. |
| `tandem planners list \| info \| install \| use \| default \| remove \| bundle \| new` | The planner catalog (above). |
| `tandem executors list \| use` | The human executors. |
| `tandem traj list \| show \| open \| relabel \| rm \| merge \| copy \| path` | Inspect trajectories. `open` replays one in its planner's own viewer (TiPToP's: Rerun, inside the runtime). `merge` re-joins a trial's legs if the automatic merge failed. |
| `tandem export lerobot \| manifest` | Build a LeRobot v3.0 dataset from `success/` (and push it to the Hub), or write a JSON index. A rebuild replaces the last dataset only once the new one is complete, and only one tandem built: `--force` replaces anything else at the destination. |
| `tandem config list \| get \| set \| set-gemini-key \| set-hf-token \| path \| edit` | Machine settings and credentials. |
| `tandem runtime status \| build \| shell \| python \| run \| clean \| path` | The runtime of the active profile's planner (`--planner NAME` for another). |

`doctor`, `plan`, `profile list|show|presets`, `traj list|show`, `planners list|info|use`,
`executors list|use`, `runtime status` and `config list` take `--json`. `NO_COLOR` is honoured.

---

## Requirements

**To collect with TiPToP**

- Linux, an NVIDIA GPU with CUDA 12 or newer, a recent driver
- pixi (`tandem init` installs it if you agree) and about 25 GB free disk for the runtime
- An arm TiPToP drives, named by `robot.type` in the profile's `planner.options`: a Franka FR3 with a
  Robotiq 2F-85 (`fr3_robotiq`), a Franka Panda with a Robotiq 2F-85 (`panda_robotiq`) or with the
  Franka Hand (`panda`), reachable over the bamboo-polymetis shim, started with `--state-port` so
  encoders stay readable while the arm moves; or a UR5 with a Robotiq gripper (`ur5`), over ur_rtde,
  which is tiptop's `ur5` extra and is not installed by the runtime
- 2–3 ZED cameras and the [ZED SDK](https://www.stereolabs.com/developers/release)
- An M2T2 grasp server
- A [Gemini API key](https://aistudio.google.com/apikey)
- ffmpeg, to join the legs' videos
- For human phases: a DROID checkout and its environment, and a VR headset and controller or a
  SpaceMouse

**To plan from a photo**: Python 3.10+ and a Gemini key.

**To visualize**: Python 3.10+. That is all.

`tandem doctor` checks what it can of these without moving anything, and says what is missing and
what to do about it.

---

## Troubleshooting

<details>
<summary><b>A preempt didn't stop the arm</b></summary>

It can't, and no software button can. Unless the planner declares cooperative stop, it is handed
a whole trajectory segment in one request and has no abort, so the motion runs to the end of that
segment; one that does stops at its next step boundary. Preempt stops *further plan steps*. **The
physical E-stop is the only instant stop.**
</details>

<details>
<summary><b>A camera won't open / shows serial number 0</b></summary>

Serial `0` means "not yours yet": another process still holds it. After a teleop hand-off the
cameras take about 15 seconds to release, because the save workers inherited the device descriptors
and have to exit first. Wait, then retry. If it persists, another tandem or tiptop process is still
running.
</details>

<details>
<summary><b>"No extrinsics for camera serial …"</b></summary>

Extrinsics are keyed by camera serial, and a serial with no entry aborts at warm-up. Add it to the
profile's `calibration.json`, or import from a checkout that has it:
`tandem profile create <name> --import-from <path>`.
</details>

<details>
<summary><b>A trial was excluded</b></summary>

A human phase still failed its check after its retries, so the trial was kept out of the dataset,
as the paper does. Open its `hitl.json`. The failing verdicts under `verifications` say which effect
the camera did not see, and why; `satisfied: false` is the one to read. `vlm/` shows the image each
verdict was made on. If the classifier was wrong rather than the person, set
`hitl.on_verification_failure: label` while you calibrate it: every disagreement between you and the
check then becomes a labeled data point. `tandem traj relabel <id> success` refuses an excluded
trial; `--force` (a confirm in the web UI) overrules the check on purpose, and its `hitl.json` then
says so under `overruled` -- the one way such a trial reaches the export.
</details>

<details>
<summary><b>The plan leaves part of the instruction out</b></summary>

`tandem plan`, the session log and the UI list each clause the model could not express, with its
reason. It is almost always an object the instruction names that perception did not detect. Put it
on the table, or reword the task, before collecting: the dataset is labeled with the whole
instruction either way.
</details>

<details>
<summary><b>"I did it" is refused at a human step</b></summary>

While recording, a human phase has to be carried out through the executor (`t`, to take the arm) so
the episode has that stretch of demonstration. Set `hitl.allow_unrecorded_human_phase: true` to
accept a step done off the record, or run with `--no-record`. `tandem executors list` says whether
teleop is ready on this machine.
</details>

<details>
<summary><b>The planner shows as outdated</b></summary>

This version of tandem pins other commits than the ones the runtime was built from. Run
`tandem planners install tiptop`. It replaces only the source trees whose pin moved, then rebuilds
against the pixi environment already on disk rather than solving one from nothing.
</details>

<details>
<summary><b>The runtime build failed</b></summary>

The full log is `~/.local/state/tandem/logs/runtime-build-<time>.log`. The usual causes are no
`nvcc`, a torch/CUDA mismatch, or running out of disk mid-compile. `tandem planners install tiptop`
retries; a step already done is skipped.
</details>

<details>
<summary><b>A leg fails with "No level patch of … is large enough"</b></summary>

That is `placement_support` saying no observed patch of the goal surface would hold the object with
`placement_support_margin` around it. From the camera's side a box's near wall and lid can hide most
of its floor: `placement_fill_occluded: true` counts the hidden floor, and a noisy floor needs a larger
`placement_flatness_tol`. `placement_support_required: false` places on the bounding box instead,
which is what the setting exists to avoid. The full reason, with the object's footprint and the
margin, is in the leg's `metadata.json` and in the session log.
</details>

<details>
<summary><b>My TAMP setting seems to do nothing</b></summary>

Run `tandem profile show <name> --planner`. That is exactly what the planner receives: if your key
is not in it, it never applied. Unknown keys are rejected when the profile loads, so a typo shows up
as an error rather than silence.
</details>

<details>
<summary><b>Where is everything, after a session that went wrong?</b></summary>

- `~/.local/state/tandem/logs/session-<id>.json`: the session's summary and log.
- `~/.local/state/tandem/sessions/<profile>/<id>/events.jsonl`: one line per event, in order. The
  event types are in [docs/METHOD.md](docs/METHOD.md#the-events-file).
- Beside it: each perception pass that recorded nothing, and `vlm/<trajectory id>/` for a trial that
  was never filed.
</details>

<details>
<summary><b>The videos won't scrub in the browser</b></summary>

They are served with HTTP Range support, so this should not happen. If it does, check that nothing
is proxying `/api/media/` without passing Range headers through.
</details>

---

## How it is put together

```
src/tandem/
├── __init__.py, api.py   the library surface: tandem.plan_task, and the SDK's names
├── planning/        THE METHOD: proposal, invented predicates, magic operators, the contract check,
│                    verification. Pure Python, no planner, no robot.
├── planners/        the planner protocol, and the kit to write one
│   ├── base.py      what tandem needs from a planner, and nothing more
│   ├── sdk.py       Planner: the base class a new planner is written against
│   ├── sidecar.py   SidecarPlanner: a planner that runs in its own environment, over JSON lines
│   ├── sidecar_kit/ tandem_sidecar, the stdlib-only helper every such sidecar is written with
│   ├── runtime.py   a planner's runtime from a recipe: pinned sources, an environment, build steps
│   ├── testing.py   the conformance kit a planner's own test suite subclasses
│   ├── registry.py  planners by name: built in, registered, or a `tandem.planners` entry point
│   └── tiptop/      TiPToP: its declaration, recipe, options, presets, importer and sidecar
├── executors/       who carries out a human phase: the protocol, the registry, teleop
├── core/            the session and its trial loop (phase_loop.py), episodes, merge, profiles
├── cli/             the command tree (Typer + Rich)
├── server/          FastAPI + a no-build single-page app
├── export/          the LeRobot v3.0 writer
├── teleop/          the hand-off driver, run under a DROID environment
└── resources/       the annotated profile template, tandem's presets, the planner scaffold
```

The **session engine** (`core/session.py`) owns the state machine, the prompts and the label.
The walk itself is `core/phase_loop.py`. It decides who does each phase, calls the planner for the
robot's, lends the arm to the human executor for a person's, checks the step from a photo, and
stamps every leg with the trajectory id that joins them into one episode. The session survives being
preempted, re-warmed and handed over mid-task. It appends a line per event to a JSONL file, so a
session that went wrong can be read off disk after the process is gone. The same object backs both
`tandem collect` and the browser UI, so the state machine exists once.

A planner needs torch, CUDA kernels, a camera SDK and a robot client; tandem needs none of those and
never will. So a hosted planner runs as a child process inside its own runtime and answers verbs over
newline-delimited JSON:

```
tandem (pure Python, no CUDA)               the planner's runtime (pixi, CUDA)
  planning/            the phases              planners/tiptop/sidecar.py   ← tandem's file
  core/phase_loop.py   the trial               │  import tiptop, cutamp
  planners/tiptop/backend.py  ──JSON──►        │  perceive / plan / execute / record
                              ◄──JSON──        │
```

`sidecar.py` is tandem's own file, executed by the runtime's interpreter. It imports nothing from
`tandem`, speaks through `tandem_sidecar`, and calls **public functions of an unmodified planner**.
Goals reach cuTAMP through `run_perception`'s existing `goal_builder` hook, so there is no
planner-side change to keep alive. That is the difference from the design this replaces, where the
phase planner lived inside a fork of the planner, and every planner tandem wanted to drive had to be
forked with it.

A planner's runtime is **declared, not shipped**. TiPToP's recipe
(`src/tandem/planners/tiptop/recipe.py`):

- pins tiptop, cuTAMP and cuRobo to exact commits;
- lists what to trim from each, and the one patch to apply;
- names tiptop's own pixi manifest;
- names the build step that compiles cuRobo's kernels.

`planners/runtime.py` does the rest, for any planner that declares a recipe, and records what it
installed in `<runtime>/.tandem-runtime.json`. The pixi environment lives beside the sources rather
than inside them, so moving a pin replaces a tree without re-solving 20 GB of torch and CUDA. The
runtime keeps the source monorepo's directory layout on purpose. Three modules resolve default asset
paths by walking up from `__file__`, and that shape makes all three resolve with no patching.

---

## Citation

```bibtex
@misc{sahoo2026tandem,
  title  = {{TANDEM}: Task and Motion Planning with As-Needed Demonstrations for Efficient
            Vision-Language-Action Model Fine-tuning},
  author = {Sahoo, Samrat and Ji, Liang and Silver, Tom and Huang, Yixuan},
  year   = {2026}
}
```

---

## Credits & license

Built on work by others:

- **[TiPToP](https://github.com/SamratSahoo/tiptop)**: the real-robot TAMP pipeline tandem drives as
  its default planner, used unmodified apart from one small patch. MIT. William Shen, Nishanth Kumar,
  and contributors.
- **The phase planner** is tandem's own (`src/tandem/planning/`). Its design and its prompts (the
  paper's Appendix B) began life as `tiptop.hitl` on the `feat/hitl-phase-planning` branch of
  [LJ1356/tiptop](https://github.com/LJ1356/tiptop).
- **[cuTAMP](https://github.com/SamratSahoo/cuTAMP)**: GPU-parallel task and motion planning.
  NVIDIA License.
- **[cuRobo](https://github.com/NVlabs/curobo)**: GPU motion generation and collision-aware IK.
  NVIDIA License. NVIDIA Seattle Robotics Lab.
- **DATAFARM**: the VAE-manifold and RND-novelty checkpoints tandem ships for TiPToP's runtime.

tandem's own code is MIT. **cuRobo and cuTAMP are under NVIDIA's source-available license, whose use
limitation is research and evaluation only. tandem does not redistribute them, but running it with
the TiPToP planner is research and evaluation use only.** See [NOTICE](NOTICE) for the details.
