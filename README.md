# TANDEM

**TANDEM** (Task and Motion Planning with As-Needed Demonstrations) collects demonstrations for fine-tuning
vision-language-action (VLA) models. A vision-language model splits an instruction into robot and human phases,
inventing predicates and human-executed "magic operators" for what the planner cannot do. The robot runs its
phases with task and motion planning (TAMP), a person teleoperates the rest, each human phase is verified from a
fresh image, and every trial is recorded as one demonstration. The planner is pluggable, and
[TiPToP](https://github.com/SamratSahoo/tiptop/tree/TANDEM) is built in.

[Paper website](https://prpl-group.com/tandem/) · [How it works](docs/METHOD.md) · [Docs](docs/README.md)

## Setup

**Requirements:** a Linux x86-64 workstation with an NVIDIA GPU, CUDA 12 and about 25 GB of free disk; a Franka
FR3 or Panda with a Robotiq 2F-85 and its polymetis NUC; 2–3 ZED cameras and the
[ZED SDK](https://www.stereolabs.com/developers/release); ffmpeg; [pipx](https://pipx.pypa.io) or
[uv](https://docs.astral.sh/uv/); a [Gemini API key](https://aistudio.google.com/apikey); and, for human phases,
a [DROID fork](https://github.com/SamratSahoo/droid) checkout and environment with a VR headset or a SpaceMouse.

Unless noted, every code block runs on the workstation.

### 1. Install

```bash
pipx install git+https://github.com/SamratSahoo/tandem.git
pipx inject tandem-tamp av pyarrow huggingface_hub   # only for `tandem export lerobot`
# or, with uv, both at once:
uv tool install git+https://github.com/SamratSahoo/tandem.git --with av --with pyarrow --with huggingface_hub
```

### 2. Initialize

```bash
tandem init                      # builds TiPToP's runtime (5–20 min), asks for the Gemini key, creates the
                                 # `default` profile and offers teleop (step 6); safe to re-run
tandem runtime run install-zed   # the ZED Python API, into the runtime (needs the ZED SDK in /usr/local/zed)
# to redo one step: tandem planners install tiptop, or tandem config set-gemini-key
```

On a laptop, `tandem init --viz-only` sets up only for reviewing trajectories
(`tandem config set data_root DIR` points it at them); `tandem plan` there also needs
`tandem config set-gemini-key`.

### 3. Robot

On the NUC, start DROID's server before the shim: it launches polymetis's robot server (port 50051) and gripper
server (port 50052), killing any already running, and teleop drives the arm through it. Then copy TiPToP's
[`bamboo_polymetis_shim.py`](https://github.com/SamratSahoo/tiptop/blob/682047493b88e5301c6b2b49da914ea4f173e5d9/bamboo_polymetis_shim.py)
to the NUC and run it in an environment with polymetis, pyzmq, msgpack, numpy and scipy:

```bash
# on the NUC
python scripts/server/run_server.py   # in the DROID checkout and its polymetis environment; without teleop,
                                      # droid/franka/launch_robot.sh and launch_gripper.sh are enough
python bamboo_polymetis_shim.py       # in a second terminal
```

Check that the shim's log shows `PolymetisGripper connected to localhost:50052`.

### 4. Perception servers

TiPToP expects an M2T2 grasp server at `http://localhost:8123` (on another machine, set
`planner.options.perception.m2t2.url`) and a FoundationStereo depth server at `http://localhost:1234`, on the
workstation itself (`tandem doctor` does not check this one). Install each as TiPToP's
[installation guide](https://github.com/SamratSahoo/tiptop/blob/682047493b88e5301c6b2b49da914ea4f173e5d9/docs/installation.md#installing-m2t2)
describes, and run each in its own terminal:

```bash
git clone https://github.com/williamshen-nz/M2T2.git && cd M2T2
pixi run setup && pixi run download-weights
pixi run server   # port 8123
```

```bash
git clone https://github.com/williamshen-nz/FoundationStereo.git && cd FoundationStereo
pixi run setup && pixi run download-checkpoints
pixi run server   # port 1234
```

### 5. Cameras and calibration

Set your rig in the `default` profile `tandem init` created (a profile made `--from default` copies its
cameras and calibration):

```bash
tandem profile edit default   # opens $EDITOR, validates on save
# cameras: your ZED serials under hand, external and optionally external_2
# planner.options.robot: host if the NUC is not at 172.16.0.2; type: panda_robotiq for a Panda
```

Extrinsics go in the profile's `calibration.json`, keyed by serial. External cameras' poses come from DROID's
own calibration: copy each `<serial>_left` pose in DROID's `droid/calibration/calibration_info.json` into it
under the bare serial. Calibrate the wrist camera as TiPToP's
[guide](https://github.com/SamratSahoo/tiptop/blob/682047493b88e5301c6b2b49da914ea4f173e5d9/docs/getting-started.md)
describes, skipping its Bamboo controller step (the shim replaces it). Its scripts reach the robot at
`172.16.0.2` whatever the profile says ([another address](docs/CONFIGURATION.md#cameras-and-calibration)):

```bash
export TIPTOP_CALIBRATION="$(tandem profile path default)/calibration.json"
export TIPTOP_HAND_CAMERA_ID=<wrist serial> TIPTOP_EXTERNAL_CAMERA_ID=<external serial>
tandem runtime run calibrate-wrist-cam
tandem runtime run viz-calibration                        # checks the wrist camera
tandem runtime run -- viz-calibration --camera external   # checks an external one
```

### 6. Teleop

Human phases run TANDEM's teleop driver in the DROID fork's environment, through the DROID server from step 3.
The fork's `droid/misc/parameters.py` must name your NUC (`nuc_ip`). `tandem init` offers to set this up, or:

```bash
tandem config set teleop.enabled true
tandem config set teleop.droid_dir /path/to/droid
tandem config set teleop.python /path/to/droid/env/bin/python
tandem config set teleop.device spacemouse   # default vr; teleop.controller left|right picks the VR hand
tandem executors list                        # teleop should say `ready`
```

### 7. Check the setup

```bash
tandem doctor   # every check and what to do about it; --no-hardware skips the robot, camera and M2T2 probes
```

Common problems are in [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Usage

### 1. Create a profile

A profile is one collection setup plus every trajectory collected with it, in `~/tandem-data/profiles/<name>/`
([every setting](docs/CONFIGURATION.md)).

```bash
tandem profile create bread-box --from default --preset paper --prompt "place the bread inside the box" --use
```

`--preset paper` turns on phase planning (off in the template) and applies the paper's
[phase-planning](docs/CONFIGURATION.md#phase-planning-hitl) and TiPToP settings, keeping your
robot, cameras and perception. Planned motions then run at the paper's pace, not slowed by
`robot.time_dilation_factor`, so keep a hand on the E-stop.

If you have the paper's hitl-tamp-vla repository, you can instead import one of its task configs, with its
robot, camera serials, extrinsics, TAMP settings and phase planning:

```bash
tandem profile create bread-box-v3 --use --import-from ~/hitl-tamp-vla \
    --tamp-config ~/hitl-tamp-vla/data-collection/cfg/tamp/4_bread_box_v3.yml
```

### 2. Check the plan

```bash
tandem plan "place the bread inside the box" --image workspace.png   # -o LABEL (repeatable) pins object labels
```

This needs only the photo and the Gemini key. It prints the phases and who does each, each human phase's magic
operator, the invented predicates, and any part of the instruction the plan cannot express. That is usually an
object that was not detected: put it on the table, or reword the task.

### 3. Collect

```bash
tandem collect   # in the terminal; --episodes N, --task "...", --no-execute (plan only), --no-record
tandem ui        # or in the browser, at http://127.0.0.1:8787
```

The session warms the planner once, then waits for you; the footer shows the keys each state accepts:

- Task prompt: Enter runs the task, `n` types a new one. `q` ends the session from any state and parks the arm.
- Planning or executing: `p` preempts (the current motion segment still finishes; the E-stop is the only hard
  stop). `t` lends you the arm at the next plan-step boundary; the planner then replans from where you leave it.
- Human phase: `t` takes the arm to teleoperate, `a` gives up on the task, and `d` marks the step done by hand
  (refused while recording, by default).
- While you have the arm: `r` gives it back.
- Label prompt: `s` success, `f` failure.

At a human phase the screen says what to do. Once you give the arm back, a fresh camera image must show the step
done; if not, you get one more try, and a trial that still fails is saved as excluded and never exported. A
robot phase the planner cannot plan ends the trial. More in [docs/USAGE.md](docs/USAGE.md).

### 4. Review and export

```bash
tandem traj list                   # newest first; --status eval|success|failure
tandem traj relabel <id> failure   # move one between success, failure and eval
tandem export lerobot --repo <hf-user>/bread-box          # LeRobot v3.0 from success/, in ~/tandem-data/exports/
tandem export lerobot --repo <hf-user>/bread-box --push   # and upload (tandem config set-hf-token, or HF_TOKEN)
```

Each trial becomes one episode, with its robot and human legs merged
([on-disk format](docs/DATA.md)).

## Adding a planner

A planner perceives the scene, plans one goal in it, and executes and records that plan; the rest stays in TANDEM.

```bash
tandem planners new myplanner       # scaffold ./tandem-myplanner (--sidecar: it runs in its own environment)
# then run the two commands it prints: install the package where tandem runs, and run the conformance kit
tandem planners list                # myplanner is now listed
tandem planners install myplanner   # build its runtime, if it declares one
tandem planners use myplanner       # the active profile now plans with it
```

Subclass `tandem.planners.Planner`, register it under the `tandem.planners` entry point, and pass the
conformance kit (`tandem.planners.testing.PlannerConformance`); see [ADDING_A_PLANNER.md](docs/ADDING_A_PLANNER.md).
Human phases run the `teleop` executor unless the profile names another (`tandem executors use NAME`), which a
package can register ([ADDING_A_HUMAN_EXECUTOR.md](docs/ADDING_A_HUMAN_EXECUTOR.md)).

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

TANDEM is released under the [MIT License](LICENSE). cuRobo and cuTAMP, which the TiPToP planner runs on and
`tandem planners install` fetches, are under NVIDIA's license, which limits their use to research and
evaluation ([NOTICE](NOTICE)).

TANDEM builds on [TiPToP](https://github.com/SamratSahoo/tiptop) (MIT; William Shen, Nishanth Kumar and
contributors), [cuTAMP](https://github.com/SamratSahoo/cuTAMP) and [cuRobo](https://github.com/NVlabs/curobo)
(NVIDIA Seattle Robotics Lab), and [DATAFARM](https://github.com/SamratSahoo/DATAFARM)'s checkpoints. The
phase planner's design and prompts began as `tiptop.hitl` on the `feat/hitl-phase-planning` branch of
[LJ1356/tiptop](https://github.com/LJ1356/tiptop/tree/feat/hitl-phase-planning).
