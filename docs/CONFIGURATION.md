# Configuration

Installing tandem and its planner, and every setting a profile and the machine hold. The `hitl:` keys are in
[METHOD.md §5](METHOD.md#5-every-hitl-setting-and-what-it-changes); the commands are in [USAGE.md](USAGE.md), and
common problems in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

- [Requirements](#requirements)
- [Installing](#installing)
- [Profiles](#profiles)
- [Importing a hitl-tamp-vla setup](#importing-a-hitl-tamp-vla-setup)
- [The planner's settings (`planner:`)](#the-planners-settings-planner)
- [Cameras and calibration](#cameras-and-calibration)
- [Surface-fitted placement (`placement_*`)](#surface-fitted-placement-placement_)
- [Machine settings and credentials](#machine-settings-and-credentials)

---

## Requirements

**To collect with TiPToP** (`tandem planners info tiptop` lists the planner's part of this):

- Linux, an NVIDIA GPU with CUDA 12 or newer, a recent driver.
- pixi (`tandem init` installs it if you agree) and about 25 GB of free disk for the runtime.
- An arm TiPToP drives, named by `planner.options.robot.type`: a Franka FR3 with a Robotiq 2F-85
  (`fr3_robotiq`) or a Franka Panda with a Robotiq 2F-85 (`panda_robotiq`), over the bamboo-polymetis shim;
  or a UR5 with a Robotiq gripper (`ur5`) over ur_rtde, which is tiptop's `ur5` extra and is not installed by
  the runtime. The options also accept a Panda with the Franka Hand (`panda`), but the shim drives only a
  Robotiq: it refuses the Franka Hand's gripper commands.
- 2–3 ZED cameras and the [ZED SDK](https://www.stereolabs.com/developers/release).
- An M2T2 grasp server, and a FoundationStereo server at `http://localhost:1234` on the workstation: TiPToP
  estimates the ZEDs' depth with it. Its address is not a setting, and `tandem doctor` does not check it.
- A [Gemini API key](https://aistudio.google.com/apikey). Phase planning uses it, and so does TiPToP's
  perception (its object detector is a Gemini model).
- ffmpeg, to join the legs' videos.
- For human phases: a checkout of the [DROID fork](https://github.com/SamratSahoo/droid) and its environment
  (the teleop driver imports its `droid.stable_camera_env`, which upstream DROID does not have), with `nuc_ip`
  in its `droid/misc/parameters.py` naming your NUC; DROID's NUC server (`scripts/server/run_server.py`)
  running, started before the shim since it restarts polymetis's servers; and a VR headset and controller or
  a SpaceMouse.

**To plan from a photo** (`tandem plan`): Python 3.10+ and a Gemini key. **To review trajectories**:
Python 3.10+.

`tandem doctor` checks what it can of these without moving anything.

---

## Installing

tandem is not on PyPI, so it installs from git. pipx or `uv tool` gives it a private environment; plain
`pip install` works inside a virtualenv, but not against Debian's or Ubuntu's system Python
(`externally-managed-environment`).

```bash
pipx install git+https://github.com/SamratSahoo/tandem.git
uv tool install git+https://github.com/SamratSahoo/tandem.git          # the same, with uv
```

The LeRobot export needs the `export` extra (av, pyarrow, huggingface_hub):

```bash
pipx inject tandem-tamp av pyarrow huggingface_hub
uv tool install --reinstall git+https://github.com/SamratSahoo/tandem.git --with av --with pyarrow --with huggingface_hub
```

Working on tandem itself:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[export,dev]'
pytest -q
```

### The planner runtime

tandem is pure Python. The planner's stack (torch, cuRobo's CUDA kernels, cuTAMP, tiptop) is built on the
machine into a **runtime** by `tandem planners install tiptop`, which `tandem init` runs. It fetches the
commits this version of tandem pins (`src/tandem/planners/tiptop/recipe.py`):

| source | branch | pinned commit |
|---|---|---|
| [SamratSahoo/tiptop](https://github.com/SamratSahoo/tiptop/tree/TANDEM) | `TANDEM` | `6820474` |
| [SamratSahoo/cuTAMP](https://github.com/SamratSahoo/cuTAMP/tree/TANDEM) | `TANDEM` | `fc8f233` |
| [SamratSahoo/curobo](https://github.com/SamratSahoo/curobo) | `main` | `3a90ff4` |

The `TANDEM` branches are each fork's `main` plus LJ1356's surface-fitted placement
([below](#surface-fitted-placement-placement_)). All of it is off unless a profile turns it on.

The install fetches each source with `git fetch --depth 1` and exports it with `git archive` (GitHub's
archive of the commit when there is no git), checks the commit, trims what the recipe lists, and applies one
patch (it lets a profile own its calibration file). It then solves tiptop's own pixi environment and
compiles cuRobo's kernels: 5–20 minutes the first time. What it installed is recorded in
`<runtime>/.tandem-runtime.json`. The pixi environment lives beside the sources (`<runtime>/env`), so a moved
pin replaces one tree without solving the environment again. When a tandem upgrade moves a pin,
`tandem planners list` shows the runtime as `outdated`, and `tandem planners install tiptop` updates it.
`--force` rebuilds from scratch.

**A workstation that cannot reach GitHub** installs from a bundle made by the same version of tandem on a
machine that can:

```bash
tandem planners bundle tiptop --out /media/usb/planner-sources      # on a machine with network (--archive: also a .tar.gz)
tandem planners install tiptop --sources /media/usb/planner-sources # on the workstation; or set TANDEM_PLANNER_SOURCES
```

Each export is checked against the pinned commit, and its files against the digest taken when it was
bundled. The bundle holds the sources only. `pixi install` still downloads the environment from conda-forge
and PyPI (and builds SAM-2 from GitHub), and the first warm-up downloads the SAM-2 checkpoint (about 0.9 GB).

The runtime does not install the ZED Python API (`pyzed`), which TiPToP opens the cameras with. tiptop's
own task installs it from the ZED SDK (`/usr/local/zed/get_python_api.py`):

```bash
tandem runtime run install-zed
```

`tandem runtime status | path | shell | run | python | clean` work on the active profile's planner's runtime
(`--planner NAME` for another). `tandem runtime run CMD` runs `CMD` inside it; put `--` before a command
that takes options (`tandem runtime run -- CMD --flag`).

---

## Profiles

A profile is one collection setup and everything collected under it:

```
~/tandem-data/profiles/bread-box/
├── profile.yml                    the task, cameras, planner and its settings, hitl:, recording, export
├── calibration.json               camera extrinsics, keyed by serial
├── planner-options.<planner>.yml  a previous planner's settings, kept by `tandem planners use`
└── trajectories/
    ├── eval/                      recorded, not yet labeled
    ├── success/
    └── failure/                   failures, and excluded trials
```

```bash
tandem profile create bread-box --prompt "place the bread inside the box"   # from the annotated template
tandem profile create bread-paper --from bread-box --preset paper           # clone, then lay a preset over it
tandem profile use bread-box
tandem profile edit bread-box      # $EDITOR, validated on save
tandem profile show bread-box      # the whole profile; --planner prints only what the planner receives
tandem profile presets             # the presets --preset can lay over a new profile
tandem profile list | path | delete
```

Switching profiles re-points collection, inspection and export at once. The template
(`src/tandem/resources/profile_template.yml`) documents every key in place.

**Unknown keys are refused when the profile loads**, with the nearest valid name. A setting that is silently
ignored would look exactly like one that worked.

**`--preset paper`** sets phase planning on with the paper's settings (METHOD.md §5's defaults) and
TiPToP's TAMP and DATAFARM settings from the paper's runs. It prints every setting it changes. It keeps the
rig's robot, cameras and perception, and sets `tamp.time_dilation_factor_literal: 1.0`, so planned motions
run at the pace blending gives them rather than slowed by `robot.time_dilation_factor`.

**Profile versions.** Profiles written before version 2 had `robot:`, `perception:` and `tamp:` at the top
level. They still load, with a notice, and are written in the new layout the next time they are saved.
`tandem profile migrate` rewrites them all at once, keeping each old file as `profile.yml.v1.bak`. A
version-1 profile's `on_robot_phase_failure: teleop` (version 1's default) is kept and named in the notice,
since the default is now `abort`. An older tandem cannot read a version-2 profile.

---

## Importing a hitl-tamp-vla setup

A `hitl-tamp-vla` checkout's setup can be imported instead of typed in:

```bash
tandem init --import-from ~/hitl-tamp-vla
tandem profile create bread-box --import-from ~/hitl-tamp-vla \
    --tamp-config ~/hitl-tamp-vla/data-collection/cfg/tamp/4_bread_box_v3.yml
```

What comes across:

- the robot config and the camera serials (from the checkout's `tiptop.yml`), and the extrinsics from
  `calibration_info.json`, with its per-workspace `calibration_info_<workspace>.json` layers;
- the task config's prompt and TAMP settings, under the same names, `placement_*` included;
- its `hitl:` block (`HITL_KEYS` in `src/tandem/planners/tiptop/importers.py` says what each key becomes).

What tandem cannot do is refused, not replaced with something it can:

- A config whose human phases a learned policy carries out (`policy_type: diffusion` or `act`, the HITL-TAMP
  baseline) is not imported with phase planning on; the refusal names the same task's config for a person.
- Settings only tiptop's own rollout loop reads (`auto_mode`, `reset_placement_region`,
  `clear_goal_surfaces`) are left out, with a warning.

The configs were tuned on LJ1356's tiptop, which always did three things the pinned TiPToP does only when
asked. The import sets all three (a key the config states keeps its value), and says so:
`table_plane_support_vote`, `disjoint_object_masks` and `blend_stretch_to_caps`
([below](#the-three-switches)). `--preset paper` sets them too.

---

## The planner's settings (`planner:`)

```yaml
planner:
  backend: tiptop          # `tandem planners use NAME` switches it
  options:                 # the planner's own settings, checked by the planner when the profile loads
    robot: ...
    perception: ...
    tamp: ...
```

`tandem planners info NAME` lists what a planner reads, and `tandem profile show NAME --planner` prints
exactly what it will receive. `tandem planners use NAME --option KEY=VALUE` passes a setting a new planner
requires. Switching planner moves the old planner's `options` to `planner-options.<planner>.yml`, and
switching back restores them.

TiPToP reads three blocks (`src/tandem/planners/tiptop/options.py`):

- **`robot`**: `type` (`fr3_robotiq`, `panda_robotiq`, `panda`, `ur5`; default `fr3_robotiq`), `dof` (7),
  `host` (`172.16.0.2`, the NUC running the shim), `port` (5555), `gripper_port` (5559), `state_port` (5557,
  where encoders are read while the arm moves), `time_dilation_factor` (0.2, in (0, 1]), `q_home`,
  `q_capture`.
- **`perception`**: `m2t2.url` (`http://localhost:8123`), `m2t2.apply_bounds`, `sam_mode` (`local`, or
  `remote` with `sam_url` naming a SAM-2 server), `depth_smoothing_frames` (5; 1 disables it),
  `robot_mask_margin_m`, `depth_trunc_m`, `voxel_downsample_size`, `contact_threshold_m`,
  `mask_erosion_pixels`. `gemini.model` and `gemini.temperature` only record which detector ran: the pinned
  tiptop always runs `gemini-robotics-er-2-preview`, and `tandem doctor` warns when they say otherwise.
  FoundationStereo, which gives the ZEDs' depth, is always asked at `http://localhost:1234`.
- **`tamp`**: cuTAMP and cuRobo overrides, under **tiptop's own key names**, so a `cfg/tamp/*.yml` pastes in
  unchanged. The accepted keys are in `src/tandem/planners/tiptop/tamp_keys.py`. `tandem doctor` warns about
  a key that is set but does nothing without another (`blend_ops` without `blend_trajectory`, say).

```yaml
planner:
  backend: tiptop
  options:
    robot: {host: 172.16.0.2, time_dilation_factor: 0.2}
    tamp:
      num_particles: 256              # cuTAMP coverage per skeleton
      opt_steps_per_skeleton: 250
      traj_length_norm: inf           # charge moves by the infinity norm, not Euclidean
      grasp_pose_change_weight: 0.1   # prefer grasps that reorient the wrist less
      vae_manifold_weight: 25000      # pull trajectories toward the DROID motion manifold
      blend_trajectory: true          # one continuous stroke per operation
      blend_ops: [Pick, Place, GoToInitial]
```

---

## Cameras and calibration

```yaml
cameras:
  perception: external   # the camera perception reads: external (arm stays at q_home) or hand (arm drives to q_capture)
  hand:     {serial: '14846828', type: zed, resolution: HD720, fps: 15}
  external: {serial: '32439448', type: zed, resolution: HD720, fps: 15}
  # external_2: {...}    recorded as DROID exterior_2; if listed and it fails to open, collection stops
```

Every leg is recorded from these cameras, the planner's and a person's. `fps` is 15 because three ZEDs at
HD720@30 exceed the USB bandwidth budget; the export resamples to 15 Hz anyway. The pinned tiptop opens
`hand` and `external` at every warm-up, so both are required to collect.

`calibration.json` holds one entry per serial, in TiPToP's `calibration_info.json` format:

```json
{"<serial>": {"pose": [x, y, z, roll, pitch, yaw]}}
```

The wrist camera's pose is `ee_from_cam` (the camera relative to the end effector); a fixed camera's is
`world_from_cam`. The rotation is `xyz` Euler angles in radians. A configured serial with no entry stops a
session before it warms up; `tandem doctor` reports it first.

TiPToP's own scripts calibrate the wrist camera and check any of them. Run inside the runtime with
`$TIPTOP_CALIBRATION` set, they read and write the profile's file (the patch tandem applies):

```bash
export TIPTOP_CALIBRATION="$(tandem profile path)/calibration.json"
export TIPTOP_HAND_CAMERA_ID=<wrist serial> TIPTOP_EXTERNAL_CAMERA_ID=<external serial>
tandem runtime run calibrate-wrist-cam                        # writes the wrist camera's entry
tandem runtime run -- viz-calibration --camera external       # checks one; the default is --camera hand
```

These scripts take the robot's address and type from the runtime's own
`$(tandem runtime path)/tiptop/tiptop/config/tiptop.yml` (`172.16.0.2`, `fr3_robotiq`), not from the profile.
For another rig, copy that file, change `robot.host` and `robot.type`, and point `$TIPTOP_CONFIG` at the copy.
The procedure (the ChArUco board, the Franka Desk steps) is in TiPToP's
[getting-started guide](https://github.com/SamratSahoo/tiptop/blob/682047493b88e5301c6b2b49da914ea4f173e5d9/docs/getting-started.md).

A fixed camera's `world_from_cam` comes from DROID's own camera calibration, which uses the same convention:
copy each `<serial>_left` pose from DROID's `droid/calibration/calibration_info.json` into `calibration.json`
under the bare serial. An import copies every entry from the checkout instead.

`cameras.perception: hand` also masks the gripper's fingers out of the wrist camera's point cloud, with the
runtime's `tiptop/tiptop/config/assets/gripper_mask.png`. `tandem runtime run compute-gripper-mask` (or
`paint-gripper-mask`) makes one for your camera. `external` needs no mask.

---

## Surface-fitted placement (`placement_*`)

By default cuTAMP may put an object down anywhere in a surface's bounding box, with the object's bottom at
the height of the surface's highest point. That suits a slab and fails for anything with structure: for an
open box it is the top of the folded-back lid, for a plate its rim. `placement_support: true` fits the region
to what the camera saw of the surface: the level patches that would hold the object's footprint, at that
patch's own height. These are `tamp:` keys.

| key | default | what it does |
|---|---|---|
| `placement_support` | `false` | Turns it on. The six keys below are read only when it is on; `tandem doctor` warns about one set without it. |
| `placement_support_margin` | `0.01` | Surface the object must keep around its footprint, in metres. `>= 0`. |
| `placement_flatness_tol` | `0.008` | How much the surface under a footprint may vary and still count as one level patch, in metres. `> 0`. Raise it for a noisy reconstruction; it is also how much slope a placement may sit on. |
| `placement_support_required` | `true` | When no patch of a goal surface would hold the object, the plan fails with that reason. `false` falls back to the bounding box. |
| `placement_into_surface` | `true` | A placed object may overlap the surface it was placed on in the collision check. Placing *into* a container needs it, since perception reconstructs one as a filled hull. |
| `placement_fill_occluded` | `false` | Unobserved cells inside a surface's outline count as floor, for a camera that cannot see a box's floor. The one setting that places onto surface nobody saw. |
| `placement_min_seen_frac` | `0.25` | The fraction of every footprint that must really have been observed: the guard on `placement_fill_occluded`. In `[0, 1]`. |

Two of the paper's tasks use it. **Solve Constrained Puzzle** (`1_toy_puzzle_v3.yml`) sets
`placement_support`, `placement_support_required` and `placement_into_surface`. **Store Bread in Closed Box**
(`4_bread_box.yml`, `4_bread_box_v3.yml`) sets all seven: margin `0.005`, flatness `0.012`,
`placement_fill_occluded: true` with `placement_min_seen_frac: 0.25`. Importing either config with
`--tamp-config` brings them across as they are.

When no surface can hold the object, the leg is an ordinary plan failure, and
`hitl.on_robot_phase_failure` decides what follows (`abort`, the default, ends the trial).

### The three switches

| key | what it switches on |
|---|---|
| `table_plane_support_vote` | Pick the table among RANSAC's planes by the objects resting on each one, not by any object within 3 cm of it on either side. |
| `disjoint_object_masks` | Build object meshes and point clouds from disjoint masks, every pixel two masks claim going to the smaller object, so a container's hull stops at what rests on it. (The placement fit always uses disjoint masks.) |
| `blend_stretch_to_caps` | With `blend_trajectory` on, slow a stroke that cannot be re-timed inside the velocity and acceleration caps until it fits, instead of running it at the plan's own timing. It can make a stroke many times slower. |

---

## Machine settings and credentials

`~/.config/tandem/config.toml`, edited with `tandem config list | get | set | edit | path`:

| key | default | |
|---|---|---|
| `active_profile` | `default` | Set by `tandem profile use`. |
| `data_root` | `~/tandem-data` | Where profiles and trajectories live. `$TANDEM_DATA_ROOT` wins. |
| `runtime_dir` | `~/.local/share/tandem/runtime` | TiPToP's runtime (about 25 GB). `$TANDEM_RUNTIME_DIR` wins. Other planners' are in `~/.local/share/tandem/runtimes/NAME` (`$TANDEM_RUNTIMES_DIR`). |
| `default_planner` | `tiptop` | The planner new profiles get (`tandem planners default NAME`). |
| `hf_org` | | The owner `tandem export lerobot` uses for a `--repo` with none. |
| `teleop.enabled`, `teleop.droid_dir`, `teleop.python` | off | The teleop driver: a DROID checkout and its environment's interpreter. |
| `teleop.device`, `teleop.controller` | `vr`, `right` | `vr` or `spacemouse`; `right` or `left` (the VR hand). |
| `ui.host`, `ui.port`, `ui.open_browser` | `127.0.0.1`, `8787`, `true` | `tandem ui`. A busy port steps to the next free one. |

Credentials go in `credentials.toml` beside it: `tandem config set-gemini-key` (`--stdin` keeps the key out
of shell history) and `tandem config set-hf-token`. `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the environment
wins over a stored key. The Hugging Face token is read from the stored one, then `HF_TOKEN` or
`HUGGING_FACE_HUB_TOKEN`, then `~/.cache/huggingface/token` (`huggingface-cli login`).

Every directory can be moved with an environment variable: `$TANDEM_CONFIG_DIR` (`~/.config/tandem`),
`$TANDEM_STATE_DIR` (`~/.local/state/tandem`: logs and session scratch), `$TANDEM_SHARE_DIR`
(`~/.local/share/tandem`), `$TANDEM_DATA_ROOT`, `$TANDEM_RUNTIME_DIR`, `$TANDEM_RUNTIMES_DIR`, and
`$TANDEM_PLANNER_SOURCES` for an offline bundle.
