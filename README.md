<div align="center">

<h1>tandem</h1>

**Human-in-the-loop TAMP data collection for real robots.**

Plan with a GPU TAMP solver, watch it run, step in when it goes wrong, and keep the data.

[![Python](https://img.shields.io/badge/python-3.10%2B-4f9dff)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20NVIDIA-a371f7)](NOTICE)
[![Platform](https://img.shields.io/badge/platform-linux%20%C2%B7%20cuda%2012-3fb950)](#requirements)

```bash
pip install tandem-tamp && tandem init
```

</div>

---

## What it is

`tandem` collects real-robot manipulation trajectories where a **planner does the work and a
human stays in the loop**. You give it a task in plain language. It perceives the scene,
searches for a task-and-motion plan on the GPU, and executes it on the arm — while you watch,
preempt a bad rollout, take the arm yourself when the plan can't finish, and label what
happened.

```
   "put the toys on the plate"
              │
              ▼
   ┌──────────────────────┐
   │  perceive            │   ZED stereo → Gemini → SAM2 → M2T2 grasps
   ├──────────────────────┤
   │  plan                │   cuTAMP task+motion search  ·  cuRobo trajectories
   ├──────────────────────┤
   │  execute             │   Franka FR3 + Robotiq, encoders sampled at 30 Hz
   ├──────────────────────┤        ▲                        │
   │  label  ✔ / ✖        │        │  you take over ────────┘
   └──────────────────────┘        └─── replan from where you left the arm
              │
              ▼
   trajectories/success/2026-08-16_21-14-02/
```

The loop **warms up once** — solver, segmentation, cameras, robot — and then runs rollout
after rollout against that warm state. A bad episode costs you one preempt, not a two-minute
restart.

**Three things a human can do, at any point:**

| | |
|---|---|
| **Preempt** | Abort the rollout in flight. The session stays warm; you are back at the task prompt in a second. |
| **Hand off** | Take the arm mid-task. The planner parks at a plan-step boundary, releases the robot and cameras, and waits. When you hand back, it replans the *same* task from wherever you left the arm — no homing, no dropped object. All the legs merge into **one** trajectory. |
| **Label** | Mark the rollout success or failure while watching the video of it. |

### Phase planning

The loop above still leaves the human's part *outside* the system: you have to notice the
planner cannot fold a cloth, carve that clause out of the instruction by hand, and remember to
press the button. Nothing knows your part was ever part of the task, and nothing checks it
happened.

Turn on `hitl.enabled` and a VLM does that carving itself. It breaks the instruction into an
**ordered list of phases** — each one either a sub-goal for the planner or something only a
person can do — invents the predicate it needs to describe your part, hands that phase over
with written instructions, and verifies from a photo that you did it.

```
   "put the toy on the cloth, then fold it"
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
   phase 1  robot            phase 2  human
   On(toy, cloth)            Folded(cloth)
   → cuTAMP plans it         → "Fold the near edge of the cloth
                                over the toy so it is covered."
                             → checked from a photo afterwards
```

Phases rather than one final-state goal because **the ordering runs both ways**: that task
needs the human last, "open the box, then put the toy in" needs the robot last, and some tasks
need an intermediate state no final-state goal can express at all.

If the check says it did not happen you are told what is still missing and given another go,
rather than losing the demonstration to one bad classifier call. Every rollout drops a
`hitl.json` — the phases, the invented predicates, which clauses of the instruction each phase
covered, and every verdict — plus a `vlm/` folder holding each image sent to the model and a
rendered PNG of what it said, rejected attempts included. When a run goes wrong the question is
always "what did the model see, and what did it decide", and that is unanswerable afterwards
without it.

Off by default, and disabled the package is never even imported.

---

## Install

```bash
pip install tandem-tamp
tandem init
```

That is the whole install. `tandem` itself is pure Python — the heavy stack (torch, cuRobo's
compiled CUDA kernels, cuTAMP, tiptop) is built by `tandem init` into a self-contained
runtime under `~/.local/share/tandem/`. The sources for all three ship inside the package, so
the build needs no network and no `git`.

<table>
<tr><td width="50%">

**On the robot workstation**

```bash
pip install tandem-tamp
tandem init
```

Probes the GPU, installs [pixi](https://pixi.sh) if needed, compiles the planner
(5–20 min the first time), takes your Gemini key, and creates a profile.

</td><td width="50%">

**On a laptop**

```bash
pip install tandem-tamp
tandem init --viz-only
tandem ui
```

No GPU, no robot, no cameras. Browse and visualize trajectories collected elsewhere —
just point the data root at them.

</td></tr>
</table>

Already have a [`hitl-tamp-vla`](https://github.com/SamratSahoo/tamp-vla) checkout? Import its
setup instead of retyping it:

```bash
tandem init --import-from ~/hitl-tamp-vla
```

Robot config, camera serials, extrinsics (including the per-workspace layers) and any
`cfg/tamp/*.yml` come across as a profile.

---

## Quickstart

```console
$ tandem init
  ✔ nvidia driver          NVIDIA GeForce RTX 5090 · driver 580.95 · 32607 MiB
  ✔ cuda runtime           12.8
  ✔ pixi                   pixi 0.70.2
  ▸ pixi env               solving
  ▸ planners               compiling cuRobo CUDA kernels — 5–20 minutes the first time
  ✔ Runtime built          ~/.local/share/tandem/runtime
  ◆ Gemini API key ›       ••••••••••••••••
  ✔ Created profile 'default'

$ tandem collect
╭─ tandem · collect · default ────────────────────────────────────────────╮
│ task   place the toys on the plate with no collisions                   │
│ state  ●warm  ●perceive  ◐plan  ○execute  ○label        elapsed 00:41   │
│                                                                          │
│ 3 success  ·  1 failure  ·  4/20 labeled                                │
├─────────────────────────────────────────────────────────────────────────┤
│ 21:14:02  cuTAMP: skeleton 3/12  cost 0.83  particles 256               │
│ 21:14:07  motion_gen: 4 segments, 6.2 s, tdf 0.20                       │
├─────────────────────────────────────────────────────────────────────────┤
│  s success   f failure   p preempt   t hand to human   q finish         │
╰─────────────────────────────────────────────────────────────────────────╯

$ tandem ui
  ✔ Serving   http://localhost:8787
```

---

## The UI

`tandem ui` opens a browser view of everything a profile has collected. It is served by the
package itself — no Node, no build step, no CDN.

- **Trajectories** — every rollout, with its camera videos and per-frame plots. Click a chart
  to seek all three videos to that frame. A merged hand-off trajectory gets a ribbon showing
  which stretch the planner drove and which stretch you did; click a stretch to jump to it.
- **Collect** — run a session from the browser, with the same preempt / hand-off / label
  controls as the terminal, and an inline review at the label prompt so you decide while
  looking at the rollout rather than from memory.
- **Profiles** — edit a profile and see exactly what the planner will receive.
- **Settings** — credentials, paths, runtime status, and the full diagnostic report.

The plots show something worth knowing: the shaded bands are the frames that π₀.₅-DROID's
non-idle filter will **throw away at training time**. A rollout that looks fine can be 40%
idle. `tandem` runs the real filter, at full resolution, on the clipped action the dataset
actually stores — so what you see is what training sees.

---

## Profiles

A **profile** is one collection setup and everything collected under it: the robot, the
cameras, the task, the TAMP settings, and the trajectories.

```
~/tandem-data/profiles/fold-cloth/
├── profile.yml            the whole setup
├── calibration.json       camera extrinsics, keyed by serial
└── trajectories/
    ├── eval/              collected, not yet labeled
    ├── success/
    └── failure/
```

```bash
tandem profile create fold-cloth --prompt "place the toy on the cloth and fold it"
tandem profile use fold-cloth
tandem profile edit fold-cloth        # $EDITOR, validated on save
```

Switching profiles re-points collection, inspection and export in one move. Two robots, two
tasks, or two TAMP regimes you want to compare — each is a profile.

### TAMP settings

The `tamp:` block goes straight to the planner, using **tiptop's own key names**, so anything
documented upstream works verbatim and an existing `cfg/tamp/*.yml` can be pasted in
unchanged.

```yaml
tamp:
  num_particles: 256              # cuTAMP coverage per skeleton
  opt_steps_per_skeleton: 250
  traj_length_norm: inf           # charge moves the infinity-norm, not Euclidean
  grasp_pose_change_weight: 0.1   # prefer grasps that reorient the wrist less
  vae_manifold_weight: 25000      # pull trajectories toward the DROID motion manifold
  joint_density_weight: 5000
  blend_trajectory: true          # one continuous stroke per operation
  blend_ops: [Pick, Place, GoToInitial]
  blend_boundary_speed: 0.3       # never fully stop at gripper events
```

**Unknown keys are rejected at load time, with a suggestion.** This is deliberate. In the
system tandem is extracted from, a config shipped `blend_ops: [Pick, MoveFree. MoveHolding]`
— one typo'd period — and the planner silently ignored the whole list for months. A setting
that quietly does nothing is the one failure mode that looks exactly like success.

Phase planning is configured separately, because it changes what a dataset *contains* rather
than how the arm moves:

```yaml
hitl:
  enabled: true
  verify_retries: 1        # extra goes at a step the check says did not happen
  verify_enforced: true    # false records the verdict and carries on
```

```console
$ tandem profile show fold-cloth --tamp
◆ planner overrides  passed as --curobo-overrides
{
  "blend_ops": ["Pick", "Place", "GoToInitial"],
  "blend_trajectory": true,
  "traj_length_norm": "inf",
  ...
}
```

---

## Commands

| | |
|---|---|
| `tandem init` | Set up this machine. Idempotent — re-run it any time. |
| `tandem doctor` | Every check, what it found, and what to do about it. |
| `tandem collect [profile]` | Run a session in the terminal. |
| `tandem ui` | Serve the browser UI. |
| `tandem profile list \| show \| create \| use \| edit \| delete` | Manage profiles. |
| `tandem traj list \| show \| open \| relabel \| rm` | Inspect trajectories. `open` is a 3D replay in Rerun. |
| `tandem export lerobot` | Build a LeRobot v3.0 dataset and optionally push it to the Hub. |
| `tandem config set-gemini-key` | Store the Gemini key. `--stdin` keeps it out of shell history. |
| `tandem runtime status \| build \| shell \| run` | The GPU runtime. |

Every read command takes `--json`. `NO_COLOR` is honoured.

---

## What a trajectory looks like

The on-disk format is unchanged from the system tandem was extracted from, so data moves
between the two in either direction.

```
trajectories/success/2026-08-16_21-14-02/
├── external_cam.mp4  external_cam_2.mp4  hand_cam.mp4
├── tiptop_plan.json        the TAMP plan that was executed
├── robot_state.npz         the per-frame arrays below
├── _meta.json              instruction, fps, timestamps, lineage
├── hitl.json               the phase plan and its verdicts   (phase planning only)
└── vlm/                    every image sent to the model, and what it said
```

| array | shape | what it is |
|---|---|---|
| `joint_position` | `[F,7]` | **measured** arm encoders — no lead, no lag, no plan fallback |
| `gripper_position` | `[F]` | **measured** gripper closedness, continuous in `[0,1]` |
| `cmd_joint_position` | `[F,7]` | **commanded** joint targets from the plan |
| `cmd_joint_velocity` | `[F,7]` | **commanded** joint velocities from the plan |
| `cmd_gripper` | `[F]` | the plan's gripper command, **binary** 0 or 1 |
| `frame_time` | `[F]` | wall clock, float64 (float32 would collapse every frame to one timestamp) |

Proprioception and action are **decoupled on purpose**. When the action is a lagged copy of
the measured state, a policy learns to echo it — and a fine-tuned policy that has learned to
echo the gripper never closes it. `tandem export lerobot` refuses an episode whose
`cmd_gripper` is not binary, rather than write a dataset with that defect in it.

---

## Requirements

**To collect**

- Linux, NVIDIA GPU with CUDA 12, a recent driver
- ~25 GB free disk for the runtime
- Franka FR3 (or UR5) with a Robotiq 2F-85, reachable over the bamboo-polymetis shim —
  started with `--state-port` so encoders stay readable while the arm moves
- 2–3 ZED cameras and the [ZED SDK](https://www.stereolabs.com/developers/release)
- An M2T2 grasp server
- A [Gemini API key](https://aistudio.google.com/apikey)

**To visualize** — Python 3.10+. That is all.

`tandem doctor` checks every one of these and tells you which are missing.

---

## Troubleshooting

<details>
<summary><b>A preempt didn't stop the arm</b></summary>

It can't, and no software button can. The controller is handed a whole trajectory segment in
one request and has no abort, so the motion runs to the end of that segment. Preempt stops
*further plan steps*. **The physical E-stop is the only instant stop.**
</details>

<details>
<summary><b>A camera won't open / shows serial number 0</b></summary>

Serial `0` means "not yours yet" — another process still holds it. After a teleop hand-off the
cameras take about 15 seconds to release, because the save workers inherited the device
descriptors and have to exit first. Wait, then retry. If it persists, another tandem or tiptop
process is still running.
</details>

<details>
<summary><b>"No extrinsics for camera serial …"</b></summary>

Extrinsics are keyed by camera serial, and a serial with no entry aborts at warmup. Add it to
the profile's `calibration.json`, or import from a checkout that has it:
`tandem profile create <name> --import-from <path>`.
</details>

<details>
<summary><b>The cuRobo build failed</b></summary>

The full log is under `~/.local/state/tandem/logs/`. The usual causes are no `nvcc`, a
torch/CUDA mismatch, or running out of disk mid-compile. `tandem runtime build` retries; the
build fingerprint means an unchanged, already-compiled kernel is skipped.
</details>

<details>
<summary><b>My TAMP setting seems to do nothing</b></summary>

Run `tandem profile show <name> --tamp`. That is exactly the JSON the planner receives — if
your key is not in it, it never applied. Unknown keys are rejected at load time, so a typo
shows up as an error rather than silence.
</details>

<details>
<summary><b>The videos won't scrub in the browser</b></summary>

They are served with HTTP Range support, so this should not happen. If it does, check that
nothing is proxying `/api/media/` without passing Range headers through.
</details>

---

## How it is put together

```
src/tandem/
├── cli/           the command tree (Typer + Rich)
├── core/          profiles, trajectories, the session state machine, the runtime
├── server/        FastAPI + a no-build single-page app
├── export/        LeRobot v3.0 writer
└── _vendor/       tiptop · cuTAMP · cuRobo, pinned and trimmed
```

The **session engine** (`core/session.py`) drives the collection process over a deliberately
dumb channel — a JSONL events file, stdin, and POSIX signals — which is what lets it survive
being preempted, re-warmed, and handed to a teleop process mid-task. The same object backs
both `tandem collect` and the browser UI, so the state machine exists once.

The **runtime** mirrors the source monorepo's directory layout on purpose. Three separate
modules resolve default asset paths by walking up from `__file__` to what they assume is a
repo root; reproducing that shape makes them all resolve correctly with no patching, and
means an imported legacy config works unchanged.

Vendored sources are pinned by commit in `src/tandem/_vendor/VENDOR.toml`, with the trimmed
paths and applied patches recorded alongside. Re-vendor with:

```bash
python tools/vendor.py --source /path/to/hitl-tamp-vla
```

---

## Credits & license

Built on work by others:

- **[TiPToP](https://github.com/LJ1356/tiptop)** — the real-robot TAMP pipeline, and the
  `tiptop.hitl` phase planner behind the phase mode above. Vendored from the
  `feat/hitl-phase-planning` branch. MIT. William Shen, Nishanth Kumar, and contributors.
- **[cuTAMP](https://github.com/SamratSahoo/cuTAMP)** — GPU-parallel task-and-motion planning.
  NVIDIA License.
- **[cuRobo](https://github.com/NVlabs/curobo)** — GPU motion generation and collision-aware IK.
  NVIDIA License. NVIDIA Seattle Robotics Lab.

tandem's own code is MIT. **cuRobo and cuTAMP are under NVIDIA's source-available license,
whose use limitation is research and evaluation only — which means tandem as distributed is
too.** See [NOTICE](NOTICE) for the details and for how to build without them.
