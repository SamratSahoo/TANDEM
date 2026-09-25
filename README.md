# TANDEM

**TANDEM** (Task and Motion Planning with As-Needed Demonstrations) collects demonstrations for fine-tuning
vision-language-action (VLA) models. A vision-language model splits a task into robot phases and human phases.
The robot does its phases with task and motion planning (TAMP), and a person teleoperates the rest. Each trial
is recorded as one demonstration.

The planner is pluggable. [TiPToP](https://github.com/SamratSahoo/tiptop/tree/TANDEM) is built in.

[Paper website](https://prpl-group.com/tandem/) · [Docs](docs/README.md)

## Setup

You need:

- a Linux x86-64 workstation with an NVIDIA GPU, CUDA 12, ffmpeg and about 25 GB of free disk;
- a Franka FR3 or Panda with a Robotiq 2F-85 gripper, and its polymetis NUC;
- 2–3 ZED cameras and the [ZED SDK](https://www.stereolabs.com/developers/release);
- [pipx](https://pipx.pypa.io) or [uv](https://docs.astral.sh/uv/), and a [Gemini API key](https://aistudio.google.com/apikey);
- for human phases, a [DROID fork](https://github.com/SamratSahoo/droid) checkout and environment, with a VR
  headset or a SpaceMouse.

Every command below runs on the workstation unless it says otherwise.

### 1. Install

```bash
pipx install git+https://github.com/SamratSahoo/tandem.git
pipx inject tandem-tamp av pyarrow huggingface_hub   # only needed for `tandem export lerobot`
# or, with uv, both at once:
uv tool install git+https://github.com/SamratSahoo/tandem.git --with av --with pyarrow --with huggingface_hub
```

### 2. Initialize

Install the ZED SDK first. Otherwise the runtime still builds, but ZED cameras won't open until you install the
SDK and run `tandem planners install tiptop` again.

```bash
tandem init
tandem init -y --robot-host 172.16.0.2 --camera hand=SERIAL --camera external=SERIAL   # without a terminal
```

`tandem init` builds TiPToP's runtime (5–20 minutes) and asks for your Gemini key, the robot's address (the NUC),
the arm type and the camera serials. It also adds the paper's five tasks as profiles and offers to set up teleop
(step 6). It is safe to re-run, and each step can be redone alone: `tandem planners install tiptop`,
`tandem rig set KEY VALUE` or `tandem config set-gemini-key`.

On a laptop used only to review trajectories, run `tandem init --viz-only` and point it at them with
`tandem config set data_root DIR`. `tandem plan` there also needs `tandem config set-gemini-key`.

### 3. Robot

```bash
# on the NUC
python scripts/server/run_server.py   # terminal 1: in the DROID checkout and its polymetis environment
python bamboo_polymetis_shim.py       # terminal 2
```

Start DROID's server first. It launches polymetis's robot server (port 50051) and gripper server (port 50052),
killing any already running, and teleop drives the arm through it. Without teleop,
`droid/franka/launch_robot.sh` and `launch_gripper.sh` are enough.

Then copy TiPToP's
[`bamboo_polymetis_shim.py`](https://github.com/SamratSahoo/tiptop/blob/682047493b88e5301c6b2b49da914ea4f173e5d9/bamboo_polymetis_shim.py)
to the NUC. Run it in an environment with polymetis, pyzmq, msgpack, numpy and scipy. Its log should show
`PolymetisGripper connected to localhost:50052`.

tandem reaches the NUC at the address you gave `tandem init`. To change it: `tandem rig set robot.host 172.16.0.5`.

### 4. Perception servers

TiPToP needs an M2T2 grasp server and a FoundationStereo depth server. Install each as TiPToP's
[installation guide](https://github.com/SamratSahoo/tiptop/blob/682047493b88e5301c6b2b49da914ea4f173e5d9/docs/installation.md#installing-m2t2)
describes, and run each in its own terminal:

```bash
git clone https://github.com/williamshen-nz/M2T2.git && cd M2T2
pixi run setup && pixi run download-weights
pixi run server   # http://localhost:8123
```

```bash
git clone https://github.com/williamshen-nz/FoundationStereo.git && cd FoundationStereo
pixi run setup && pixi run download-checkpoints
pixi run server   # http://localhost:1234
```

`tandem doctor` checks both. For a server on another machine, run
`tandem rig set planners.tiptop.perception.m2t2.url http://HOST:8123` (or `...foundation_stereo.url`).

### 5. Cameras and calibration

The robot and cameras are this machine's [rig](docs/CONFIGURATION.md#the-rig), shared by every profile.
`tandem init` already asked for them. Change one setting at a time:

```bash
tandem rig show                                   # the robot, the cameras, and which have extrinsics
tandem rig set robot.type panda_robotiq           # a Panda
tandem rig set cameras.external_2.serial SERIAL   # roles: hand, external, external_2
```

**Extrinsics.** Every configured camera needs extrinsics before a session can start. `tandem rig show` and
`tandem doctor` list the missing ones. They go in the rig's `calibration.json` (`tandem rig path --calibration`),
one entry per camera, keyed by serial:

```json
{
  "14846828": {"pose": [0.0266, 0.0705, -0.1392, -0.4509, -0.0045, -1.5620]},
  "32439448": {"pose": [0.1532, -0.5827, 0.4410, -2.0843, 0.0126, 0.1964]}
}
```

`pose` is `[x, y, z, roll, pitch, yaw]` in meters and radians (`xyz` Euler). For the wrist camera it is
relative to the end effector; for an external camera it is relative to the robot's base. The full format is in
[CONFIGURATION.md](docs/CONFIGURATION.md#the-rig).

To fill it in:

- **External cameras:** calibrate each from DROID's GUI with the ChArUco board, as DROID's
  [calibration guide](https://droid-dataset.github.io/droid/example-workflows/calibrating-cameras.html)
  describes. DROID writes `droid/calibration/calibration_info.json` in the same format. Copy each
  `"<serial>_left"` entry into the rig's file under the bare serial (`"32439448_left"` becomes `"32439448"`).
- **Wrist camera:** run `tandem runtime run calibrate-wrist-cam`, following TiPToP's
  [guide](https://github.com/SamratSahoo/tiptop/blob/682047493b88e5301c6b2b49da914ea4f173e5d9/docs/getting-started.md)
  but skipping its Bamboo controller step (the shim replaces it). It writes the entry for you.
- **Any other tool:** write the entry yourself. A 4×4 transform `T` becomes `T[:3, 3]` followed by
  `Rotation.from_matrix(T[:3, :3]).as_euler("xyz")`.

Check the result with `tandem runtime run viz-calibration` (the wrist camera) or
`tandem runtime run viz-calibration --camera external`. `tandem runtime run` points TiPToP's scripts at your
rig: your NUC, your cameras and this calibration file. Recalibrate a camera whenever it moves.

### 6. Teleop

```bash
tandem config set teleop.enabled true
tandem config set teleop.droid_dir /path/to/droid
tandem config set teleop.python /path/to/droid/env/bin/python
tandem config set teleop.device spacemouse   # default: vr
tandem executors list                        # teleop should say `ready`
```

`tandem init` offers to do this for you. Human phases run TANDEM's teleop driver in the DROID fork's
environment, through the DROID server from step 3. With VR, `teleop.controller left|right` picks the hand.

The fork's `droid/misc/parameters.py` must name your NUC: set `nuc_ip` to the rig's `robot.host`.

### 7. Check the setup

```bash
tandem doctor   # every check, and what to do about each; --no-hardware skips the robot, camera and server probes
```

Common problems are in [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Usage

### 1. Choose a task

```bash
tandem profile use store-bread-in-closed-box
```

A profile is one task in one YAML file, `~/tandem-data/profiles/<name>.yml`. `tandem init` added the paper's
five, each with the settings the paper used: `store-bread-in-closed-box`, `cover-bread-rolls`,
`solve-constrained-puzzle`, `sort-and-cover-snacks` and `open-obstructed-book`.

To make your own with the paper's settings:

```bash
tandem profile create my-task --prompt "put the cup on the plate" --use
tandem profile edit my-task      # optional; validated on save
```

`--from PROFILE` copies an existing profile instead. See [every setting](docs/CONFIGURATION.md#profiles).

> **Keep a hand on the E-stop.** The paper's settings run planned motions at their own pace. The rig's
> `time_dilation_factor` does not slow them down.

### 2. Check the plan

```bash
tandem plan "place the bread inside the box" --image workspace.png
```

This needs only the photo and the Gemini key. It prints the phases, who does each, each human phase's magic
operator and the invented predicates. `-o LABEL` (repeatable) pins object labels.

It also lists any part of the task the plan can't express. That is usually an object that wasn't detected: put
it on the table, or reword the task.

### 3. Collect

```bash
tandem collect   # in the terminal
tandem ui        # or in the browser, at http://127.0.0.1:8787
```

`tandem collect` also takes `--episodes N`, `--task "..."`, `--no-execute` (plan only) and `--no-record`. The
session warms the planner once, then waits for you. The footer shows the keys each state accepts:

| state | keys |
|---|---|
| task prompt | `Enter` runs the task, `n` types a new one |
| planning or executing | `p` preempts, `t` lends you the arm at the next plan-step boundary |
| human phase | `t` takes the arm to teleoperate, `a` gives up, `d` marks it done by hand (refused while recording, by default) |
| you have the arm | `r` gives it back |
| label prompt | `s` success, `f` failure |
| any | `q` ends the session and parks the arm |

> **`p` does not stop the arm.** The current motion segment still finishes. Only the E-stop stops it at once.

After you give the arm back from a `t` hand-off, the planner replans from where you left it. At a human phase,
the screen says what to do. When you give the arm back, a fresh camera image must show the step done. If it
doesn't, you get one more try; a trial that still fails is saved as excluded and never exported.

A robot phase the planner can't plan ends the trial. More in [USAGE.md](docs/USAGE.md#collecting).

### 4. Review and export

```bash
tandem traj list                                        # newest first; --status eval|success|failure
tandem traj relabel <id> failure                        # move one between success, failure and eval
tandem export lerobot --repo <hf-user>/my-task          # LeRobot v3.0 from success/, in ~/tandem-data/exports/
tandem export lerobot --repo <hf-user>/my-task --push   # and upload it
```

Each trial becomes one episode, with its robot and human legs merged ([on-disk format](docs/DATA.md)).
`--push` needs a Hugging Face token: `tandem config set-hf-token`, or `HF_TOKEN`.

## Adding a planner

A planner perceives the scene, plans one goal in it, and executes and records that plan. TANDEM does the rest.

```bash
tandem planners new myplanner       # scaffolds ./tandem-myplanner (--sidecar: runs in its own environment)
# run the two commands it prints: install the package where tandem runs, then run the conformance kit
tandem planners list                # myplanner is now listed
tandem planners install myplanner   # builds its runtime, if it declares one
tandem planners use myplanner       # the active profile now plans with it
```

A planner subclasses `tandem.planners.Planner`, registers under the `tandem.planners` entry point, and passes the
conformance kit (`tandem.planners.testing.PlannerConformance`). See [ADDING_A_PLANNER.md](docs/ADDING_A_PLANNER.md).

Human phases run the `teleop` executor unless the profile names another (`tandem executors use NAME`). A package
can register its own: see [ADDING_A_HUMAN_EXECUTOR.md](docs/ADDING_A_HUMAN_EXECUTOR.md).

## Citation

```bibtex
@misc{sahoo2026tandem,
  title  = {{TANDEM}: Task and Motion Planning with As-Needed Demonstrations for Efficient
            Vision-Language-Action Model Fine-tuning},
  author = {Sahoo, Samrat and Ji, Liang and Silver, Tom and Huang, Yixuan},
  year   = {2026}
}
```

## License and acknowledgements

This repository is released under the [MIT License](LICENSE).
