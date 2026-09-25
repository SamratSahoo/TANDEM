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

Install the ZED SDK first, so the runtime build adds its Python API. Without it the runtime still builds, but
ZED cameras won't open until you install the SDK and run `tandem planners install tiptop` again.

```bash
tandem init   # builds TiPToP's runtime (5–20 min), asks for the Gemini key, the robot's address (the NUC),
              # the arm and the camera serials, adds the paper's five tasks as profiles and offers teleop
              # (step 6); safe to re-run
# to redo one step: tandem planners install tiptop, tandem rig set KEY VALUE, or tandem config set-gemini-key
```

Without a terminal: `tandem init -y --robot-host 172.16.0.2 --camera hand=SERIAL --camera external=SERIAL`.

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

TiPToP expects an M2T2 grasp server at `http://localhost:8123` and a FoundationStereo depth server at
`http://localhost:1234`; `tandem doctor` checks both. On another machine:
`tandem rig set planners.tiptop.perception.m2t2.url http://HOST:8123` (or `...foundation_stereo.url`).
Install each as TiPToP's
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

The robot and cameras are this machine's [rig](docs/CONFIGURATION.md#the-rig), shared by every profile.
`tandem init` asked for them; change one setting at a time:

```bash
tandem rig show                                # the robot, cameras and which have extrinsics
tandem rig set robot.host 172.16.0.5           # the NUC
tandem rig set robot.type panda_robotiq        # a Panda
tandem rig set cameras.external_2.serial SERIAL   # roles: hand, external, external_2
```

Extrinsics go in the rig's `calibration.json` (`tandem rig path --calibration`), keyed by serial. External
cameras' poses come from DROID's own calibration: copy each `<serial>_left` pose in DROID's
`droid/calibration/calibration_info.json` into it under the bare serial. Calibrate the wrist camera as
TiPToP's
[guide](https://github.com/SamratSahoo/tiptop/blob/682047493b88e5301c6b2b49da914ea4f173e5d9/docs/getting-started.md)
describes, skipping its Bamboo controller step (the shim replaces it). `tandem runtime run` points its
scripts at the rig: your NUC, your cameras, and this calibration file.

```bash
tandem runtime run calibrate-wrist-cam                 # writes the wrist camera's extrinsics
tandem runtime run viz-calibration                     # checks the wrist camera
tandem runtime run viz-calibration --camera external   # checks an external one
```

### 6. Teleop

Human phases run TANDEM's teleop driver in the DROID fork's environment, through the DROID server from step 3.
The fork's `droid/misc/parameters.py` must name your NUC (`nuc_ip`, the rig's `robot.host`). `tandem init`
offers to set this up, or:

```bash
tandem config set teleop.enabled true
tandem config set teleop.droid_dir /path/to/droid
tandem config set teleop.python /path/to/droid/env/bin/python
tandem config set teleop.device spacemouse   # default vr; teleop.controller left|right picks the VR hand
tandem executors list                        # teleop should say `ready`
```

### 7. Check the setup

```bash
tandem doctor   # every check and what to do about it; --no-hardware skips the robot, camera and server probes
```

Common problems are in [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Usage

### 1. Choose a profile

A profile is one task: its prompt, [phase planning](docs/CONFIGURATION.md#phase-planning-hitl) and TAMP
settings, in one YAML file, `~/tandem-data/profiles/<name>.yml` ([every setting](docs/CONFIGURATION.md#profiles)).
`tandem init` adds the paper's five tasks, each with the settings the paper collected it with:

```bash
tandem profile list                               # cover-bread-rolls, solve-constrained-puzzle, sort-and-cover-snacks,
                                                  # open-obstructed-book, store-bread-in-closed-box
tandem profile use store-bread-in-closed-box
```

Your own task starts from the paper's settings:

```bash
tandem profile create my-task --prompt "put the cup on the plate" --use   # --from PROFILE copies one instead
tandem profile edit my-task                                               # opens my-task.yml; validated on save
```

The paper's settings run planned motions at their own pace, not slowed by the rig's `time_dilation_factor`:
keep a hand on the E-stop.

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
tandem export lerobot --repo <hf-user>/my-task            # LeRobot v3.0 from success/, in ~/tandem-data/exports/
tandem export lerobot --repo <hf-user>/my-task --push     # and upload (tandem config set-hf-token, or HF_TOKEN)
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
