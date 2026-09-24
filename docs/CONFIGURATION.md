# Configuration

Every setting tandem reads, and where it lives: a profile's `profile.yml` (including phase planning), camera
calibration, the planner's options, machine settings and credentials, and the planner runtime. Read it when
you change a setting. Setup steps are in the [README](../README.md#setup), commands in [USAGE.md](USAGE.md).

On this page: [Profiles](#profiles) · [Phase planning](#phase-planning-hitl) · [Cameras](#cameras-and-calibration) ·
[Planner settings](#planner-settings) · [Machine settings](#machine-settings-and-credentials) ·
[Runtime](#the-planner-runtime) · [Installing](#installing)

## Profiles

A [profile](README.md#terms) is one directory under `~/tandem-data/profiles/`:

```
~/tandem-data/profiles/bread-box/
├── profile.yml                    task, cameras, hitl, planner, recording, export
├── calibration.json               camera extrinsics, keyed by serial
├── planner-options.<planner>.yml  a previous planner's settings, kept by `tandem planners use`
└── trajectories/
    ├── eval/                      recorded, not yet labeled
    ├── success/
    └── failure/                   failures, including settled trials (no label)
```

```bash
tandem profile create bread-box --prompt "place the bread inside the box"   # from the template
tandem profile create bread-paper --from bread-box --preset paper           # clone, then apply a preset
tandem profile use bread-box       # collect, traj and export now work on it
tandem profile edit bread-box      # opens $EDITOR, validates on save
tandem profile show bread-box      # the whole profile; --planner: only what the planner receives
```

`src/tandem/resources/profile_template.yml` documents every key in place. An unknown key is refused when the
profile loads, naming the key (and, for a `tamp` key, the nearest valid one).

### Profile keys

| key | default | what it does |
|---|---|---|
| `version` | `2` | The layout version ([older profiles](#older-profiles)). |
| `name`, `description` | | The profile's name (lowercase letters, digits, `-`, `_`) and free text. |
| `task.prompt` | the template's | The language label stored with every episode. `profile create --prompt` sets it. |
| `task.goal` | `null` | The goal handed to the planner, when it must differ from the label. `null` uses `task.prompt`. |
| `task.target_episodes` | `20` | The target shown by `profile show`, the UI and the session. Must be > 0. |
| `cameras`, `hitl`, `planner` | | [Cameras](#cameras-and-calibration), [phase planning](#phase-planning-hitl), [planner settings](#planner-settings). |
| `recording.enabled` | `true` | Record camera video during trials. `tandem collect --no-record` overrides it. |
| `recording.fps` | `15` | Not read: each camera's `fps` sets the recorded rate. |
| `export.hf_repo` | `''` | The repository `tandem export lerobot` uses without `--repo`. |
| `export.private` | `false` | Upload the dataset as private. `--private` or `--public` overrides it. |

### Presets

`tandem profile create NAME --preset paper` (or `tandem init --preset paper`) applies the paper's collection
settings and prints every setting it changes:

- It turns phase planning on with the paper's `hitl` values (the [defaults](#phase-planning-hitl)).
- It replaces `planner.options.tamp` whole with the paper's TAMP and [DATAFARM](README.md#terms) settings,
  including [the three switches](#the-three-switches).
- It sets no robot, cameras or perception, so a profile made `--from` or `--import-from` keeps its rig's.
- It sets `tamp.time_dilation_factor_literal: 1.0`, so planned motions run at their planned speed, not slowed
  by `robot.time_dilation_factor` (homing and capture moves still are). Keep a hand on the E-stop.

Because it replaces the whole `tamp` block, `--preset paper` drops a task config's `placement_*` keys. To keep
them, import the task's config with `--tamp-config` and leave out `--preset`. `tandem profile presets` lists
presets; the file format is in [ADDING_A_PLANNER.md](ADDING_A_PLANNER.md#options-presets-and-doctor-rows).

### Importing a hitl-tamp-vla setup

```bash
tandem init --import-from ~/hitl-tamp-vla
tandem profile create bread-box --import-from ~/hitl-tamp-vla \
    --tamp-config ~/hitl-tamp-vla/data-collection/cfg/tamp/4_bread_box_v3.yml
```

- **Comes across:** the robot and camera serials from the checkout's `tiptop.yml`; the extrinsics from
  `calibration_info.json` and every `calibration_info_<workspace>.json`; from `--tamp-config`, the prompt,
  episode target, Hugging Face repo, TAMP settings (`placement_*` included) and the `hitl:` block
  (`HITL_KEYS` in `src/tandem/planners/tiptop/importers.py` says what each key becomes).
- **Refused, with phase planning on:** a config whose `robot_planner` isn't `cutamp`, or whose human phases run
  a learned policy (`policy_type: diffusion` or `act`). The error names the same task's config for a person.
- **Left out, with a warning:** `auto_mode`, `reset_placement_region` and `clear_goal_surfaces`, which only
  tiptop's own collection loop reads.
- **Turned on:** `table_plane_support_vote`, `disjoint_object_masks` and, with `blend_trajectory` on,
  `blend_stretch_to_caps` ([the three switches](#the-three-switches)), which those configs were tuned with.
  A key the config sets keeps its value.

### Older profiles

Version-1 profiles (with `robot:`, `perception:` and `tamp:` at the top level) still load with a notice and are
rewritten under `planner.options` at the next save; `tandem profile migrate [NAME]` does it now and keeps
`profile.yml.v1.bak`. Their `hitl.on_robot_phase_failure: teleop` (version 1's default) is kept and named in the
notice: set it to `abort` unless teleop was chosen on purpose. An older tandem can't read a version-2 profile.

## Phase planning (hitl)

The `hitl:` block turns [phase planning](README.md#terms) on and sets how each trial is checked. It is off in the
template, and `--preset paper` turns it on. Every other default is the paper's value.

| key | default | what it does |
|---|---|---|
| `enabled` | `false` | Phase planning. Off: each attempt is one leg toward the planner's own goal; no proposal, phases or camera checks. |
| `proposal_model` | `gemini-2.5-pro` | Splits the task into phases and invents predicates and operators. |
| `vlm_model` | `gemini-2.5-flash` | Answers each camera check, and names objects for `tandem plan`. |
| `max_attempts` | `3` | Model answers per proposal (each rejection fed back) and per check, and the most re-plans a trial gets under `replan`. ≥ 1. |
| `classify_initial` | `false` | Classify every invented atom on the first image, then (with `check_plan_effects` on) re-run the contract check against it; recorded, not refused. One model call per atom. |
| `verify_retries` | `1` | Extra tries at a human phase whose check failed. The operator is told what is missing. ≥ 0. |
| `verify_enforced` | `true` | Off: a failed check is recorded and the trial carries on. |
| `on_verification_failure` | `exclude` | When a check still fails after the retries. `exclude`: end the trial, file it under `failure/` with `excluded: true`, no label. `label`: end it as a failure and let the label decide. |
| `verify_final_phase` | `true` | Off: a final human phase isn't checked; the label covers it. |
| `check_human_effects` | `true` | Check a human phase's add effects (must hold) and delete effects (must not). |
| `check_human_preconditions` | `false` | Check a human phase's preconditions once, before the hand-off. |
| `check_tamp_preconditions` | `false` | Before a robot leg, check what earlier phases should have left true, on that perception pass's image. |
| `check_tamp_effects` | `false` | After a robot leg, check its add effects. Recorded only; never stops a trial. |
| `precondition_enforced` | `false` | An unmet precondition (either kind) ends the trial at `verification` instead of being recorded. |
| `check_plan_effects` | `true` | Run the [contract check](METHOD.md#the-contract-check) while repairing a proposal. Symbolic: no model call. |
| `save_vlm_io` | `true` | Keep every image sent to a model and its reply, rejected attempts included, as [`vlm/`](DATA.md#vlm). |
| `cache_path` | `null` | SQLite cache of proposal responses (never checks), keyed on model, prompt and image. Relative paths are beside the profile. A re-plan bypasses it. |
| `on_robot_phase_failure` | `abort` | When the planner can't plan a robot phase. `abort`: fail at `tamp_planning`. `teleop`: a person does it, checked like a human phase. `replan`: propose again with the failure fed back. A leg that fails to *execute* always ends the trial at `tamp_execution`. |
| `conjoin_robot_phases` | `true` | Plan consecutive robot phases as one goal where sound ([conditions](ADDING_A_PLANNER.md#capabilities)). Off: one leg and perception pass per robot phase. |
| `human_executor` | `teleop` | What carries out human phases, by registered name ([choosing one](ADDING_A_HUMAN_EXECUTOR.md#choosing-one)). |
| `human_executor_options` | `{}` | Each executor's settings, keyed by its name, passed as `ExecutorContext.options` and checked when the profile loads. |
| `allow_unrecorded_human_phase` | `false` | While recording, accept a human phase with no recording: `d` (done by hand), or an executor leg with no frames. Without it both are refused and the operator is asked again. With `--no-record` both are always accepted. |
| `verification_camera` | `external` | Where the check image comes from: `external`, `hand` or `perception` (the camera perception reads). |

Each camera check costs one model call per checkable atom with the arm parked, so only `check_human_effects`
is on by default. Checkable atoms are every invented predicate plus the planner's `checkable_predicates`
(TiPToP: `On`; see [capabilities](ADDING_A_PLANNER.md#capabilities)). How the keys act on a trial:
[METHOD.md](METHOD.md#the-trial-loop).

## Cameras and calibration

```yaml
cameras:
  perception: external
  hand:     {serial: '14846828', type: zed, resolution: HD720, fps: 15}
  external: {serial: '32439448', type: zed, resolution: HD720, fps: 15}
  # external_2: {serial: '...', type: zed, resolution: HD720, fps: 15}
```

| key | default | what it does |
|---|---|---|
| `perception` | `external` | The camera perception reads. `external`: the arm stays at `q_home`. `hand`: the arm drives to `q_capture` first. |
| `hand`, `external` | | The wrist and third-person cameras. TiPToP opens both at warm-up, so a missing one stops the session before it starts. |
| `external_2` | | Optional second third-person camera, recorded as DROID `exterior_2`. If listed and it fails to open, collection stops. |
| `<camera>.serial` | | The ZED serial, quoted (`'14846828'`): an unquoted number is refused. |
| `<camera>.type`, `.resolution` | `zed`, `HD720` | |
| `<camera>.fps` | `15` | Keep 15: three ZEDs at HD720@30 exceed USB bandwidth, and the export resamples to 15 Hz anyway. |

Every leg, the planner's and a person's, is recorded from these cameras.

`calibration.json` holds one pose per serial, in TiPToP's `calibration_info.json` format:

```json
{"<serial>": {"pose": [x, y, z, roll, pitch, yaw]}}
```

The wrist camera's pose is `ee_from_cam` (relative to the end effector); a fixed camera's is `world_from_cam`.
Rotations are `xyz` Euler angles in radians. A configured serial with no entry stops the session before
warm-up, and `tandem doctor` reports it first. To fill it in:

1. **Wrist camera.** Run TiPToP's scripts in the runtime. With `$TIPTOP_CALIBRATION` set, they read and write
   the profile's file (tandem patches TiPToP for this). The board and Franka Desk steps are in TiPToP's
   [getting-started guide](https://github.com/SamratSahoo/tiptop/blob/682047493b88e5301c6b2b49da914ea4f173e5d9/docs/getting-started.md).

   ```bash
   export TIPTOP_CALIBRATION="$(tandem profile path)/calibration.json"
   export TIPTOP_HAND_CAMERA_ID=<wrist serial> TIPTOP_EXTERNAL_CAMERA_ID=<external serial>
   tandem runtime run calibrate-wrist-cam                    # writes the wrist camera's entry
   tandem runtime run -- viz-calibration --camera external   # checks one camera (default: hand)
   ```

2. **Another robot address or type.** The scripts read the robot from
   `$(tandem runtime path)/tiptop/tiptop/config/tiptop.yml` (`172.16.0.2`, `fr3_robotiq`), not the profile.
   Copy that file, change `robot.host` and `robot.type`, and set `$TIPTOP_CONFIG` to the copy's absolute path.
3. **Fixed cameras.** Copy each `<serial>_left` pose from DROID's `droid/calibration/calibration_info.json`
   into `calibration.json` under the bare serial (same convention). An [import](#importing-a-hitl-tamp-vla-setup)
   copies every entry.
4. **Gripper mask**, for `cameras.perception: hand` only. The gripper's fingers are masked out of the wrist
   camera's point cloud with the runtime's `tiptop/tiptop/config/assets/gripper_mask.png`. Make one for your
   camera with `tandem runtime run compute-gripper-mask` (or `paint-gripper-mask`).

## Planner settings

```yaml
planner:
  backend: tiptop     # which planner
  options:            # its own settings, checked by the planner when the profile loads
    robot: ...
    perception: ...
    tamp: ...
```

`tandem planners use NAME` switches `planner.backend`:

- The old planner's `options` move to `planner-options.<planner>.yml`, and switching back restores them.
- `--option KEY=VALUE` (repeatable) gives the new planner a setting it requires.
- It warns, but doesn't refuse, when the planner's runtime isn't installed.
- It repairs a profile naming a planner this machine lacks.

**Did my setting apply?** `tandem profile show NAME --planner` prints exactly what the planner receives; a
key missing from it never applied. `tandem planners info NAME` lists what a planner reads.

### TiPToP options

Defaults are in `src/tandem/planners/tiptop/options.py`.

**`robot`**

| key | default | what it does |
|---|---|---|
| `type` | `fr3_robotiq` | The arm. `fr3_robotiq` (Franka FR3) or `panda_robotiq` (Franka Panda), each with a Robotiq 2F-85 over the bamboo-polymetis shim; or `ur5` (a UR5 with a Robotiq gripper) over ur_rtde, tiptop's `ur5` extra, which the runtime doesn't install. `panda` (the Franka Hand) loads, but the shim drives only a Robotiq and refuses the Hand's gripper commands. |
| `dof` | `7` | Joint count; `q_home` and `q_capture` must match it. |
| `host` | `172.16.0.2` | The NUC running the bamboo-polymetis shim. |
| `port`, `gripper_port`, `state_port` | `5555`, `5559`, `5557` | The shim's control, gripper and state ports. Encoders are read on the state port while the arm moves. |
| `time_dilation_factor` | `0.2` | Arm speed, in (0, 1]. `0.2` is 20%. |
| `q_home`, `q_capture` | see the template | The home pose, and the pose the arm drives to for `cameras.perception: hand`. |

**`perception`**

| key | default | what it does |
|---|---|---|
| `m2t2.url`, `m2t2.apply_bounds` | `http://localhost:8123`, `true` | The M2T2 grasp server, and a flag sent with each request. |
| `sam_mode` | `local` | `local` runs SAM-2 in the runtime. `remote` uses the server at `sam_url`, which is then required. |
| `depth_smoothing_frames` | `5` | Depth frames median-fused at the capture pose. `1` disables it. |
| `robot_mask_margin_m` | `0.02` | Padding on the arm's collision spheres when they are cut out of a third-person cloud. Raise it if a rim of arm survives. |
| `depth_trunc_m`, `voxel_downsample_size`, `contact_threshold_m`, `mask_erosion_pixels` | `5.0`, `0.0075`, `0.01`, `3` | tiptop's perception settings of the same names. |
| `gemini.model`, `gemini.temperature` | `gemini-robotics-er-2-preview`, `null` | Recorded only: the pinned tiptop always runs that model at its own temperature. `tandem doctor` warns when these differ. |

FoundationStereo, which gives the ZEDs' depth, is always reached at `http://localhost:1234`. It is not a
setting, and `tandem doctor` doesn't check it.

**`tamp`**

cuTAMP and cuRobo overrides under tiptop's own key names, so a `cfg/tamp/*.yml` pastes in unchanged. The
accepted keys, with what each does, are in `src/tandem/planners/tiptop/tamp_keys.py`. Where `tamp` and
`perception` both set a key (`contact_threshold_m`, `voxel_downsample_size`), `tamp` wins. `tandem doctor`
warns about a key that does nothing without another (for example `blend_ops` without `blend_trajectory`).

```yaml
tamp:                           # under planner.options
  num_particles: 256            # cuTAMP coverage per skeleton
  opt_steps_per_skeleton: 250
  traj_length_norm: inf         # charge moves by the infinity norm, not Euclidean
  grasp_pose_change_weight: 0.1 # prefer grasps that reorient the wrist less
  vae_manifold_weight: 25000    # pull trajectories toward the DROID motion manifold
  blend_trajectory: true        # one continuous stroke per operation
  blend_ops: [Pick, Place, GoToInitial]
```

### Surface-fitted placement

By default cuTAMP may put an object anywhere in a surface's bounding box, with its bottom at the surface's
highest point: wrong for an open box (the top of its lid) or a plate (its rim). `placement_support: true`
fits the region to the observed level patches that would hold the object's footprint, at each patch's own
height. These are `tamp:` keys.

| key | default | what it does |
|---|---|---|
| `placement_support` | `false` | Turns it on. The six keys below are read only when it is on; `tandem doctor` warns about one set without it. |
| `placement_support_margin` | `0.01` | Surface the object must keep around its footprint, in metres. ≥ 0. |
| `placement_flatness_tol` | `0.008` | How much the surface under a footprint may vary and still count as one level patch, in metres. > 0. Raise it for a noisy reconstruction; it is also how much slope a placement may sit on. |
| `placement_support_required` | `true` | When no patch of a goal surface would hold the object, the plan fails with that reason. `false` falls back to the bounding box. |
| `placement_into_surface` | `true` | A placed object may overlap its surface in the collision check. Placing into a container needs it, since perception reconstructs a container as a filled hull. |
| `placement_fill_occluded` | `false` | Unobserved cells inside a surface's outline count as floor, for a camera that can't see a box's floor. The one setting that places onto surface nobody saw. |
| `placement_min_seen_frac` | `0.25` | The fraction of every footprint that must really have been observed; the guard on `placement_fill_occluded`. In [0, 1]. |

The paper's **Solve Constrained Puzzle** (`1_toy_puzzle_v3.yml`) sets `placement_support`,
`placement_support_required` and `placement_into_surface`. **Store Bread in Closed Box** (`4_bread_box*.yml`)
sets all seven, with margin `0.005`, flatness `0.012`, `placement_fill_occluded: true` and
`placement_min_seen_frac: 0.25`. `--tamp-config` imports them unchanged.

When no surface can hold the object, the leg is an ordinary plan failure, and `hitl.on_robot_phase_failure`
decides what follows.

### The three switches

These `tamp:` keys are off by default. `--preset paper` and an [import](#importing-a-hitl-tamp-vla-setup)
turn them on.

| key | what it switches on |
|---|---|
| `table_plane_support_vote` | Pick the table among RANSAC's planes by the objects resting on each one, not by any object within 3 cm of it on either side. |
| `disjoint_object_masks` | Build object meshes and point clouds from disjoint masks: a pixel two masks claim goes to the smaller object, so a container's hull stops at what rests on it. (The placement fit always uses disjoint masks.) |
| `blend_stretch_to_caps` | With `blend_trajectory` on, slow a stroke that can't be re-timed within the velocity and acceleration caps until it fits, instead of running it at the plan's own timing. It can make a stroke many times slower. |

## Machine settings and credentials

`~/.config/tandem/config.toml`, edited with `tandem config list | get | set | edit | path`:

| key | default | what it does |
|---|---|---|
| `active_profile` | `default` | The profile commands act on. Set by `tandem profile use`. |
| `data_root` | `~/tandem-data` | Where profiles and trajectories live. `$TANDEM_DATA_ROOT` wins. |
| `runtime_dir` | `~/.local/share/tandem/runtime` | TiPToP's runtime (about 25 GB). `$TANDEM_RUNTIME_DIR` wins. |
| `default_planner` | `tiptop` | The planner new profiles get (`tandem planners default NAME`). |
| `hf_org` | | The owner `tandem export lerobot` uses for a `--repo` with none. |
| `teleop.enabled`, `teleop.droid_dir`, `teleop.python` | off | The teleop driver: a DROID checkout and its environment's Python ([teleop executor](ADDING_A_HUMAN_EXECUTOR.md#the-teleop-executor)). |
| `teleop.device`, `teleop.controller` | `vr`, `right` | `vr` or `spacemouse`; `right` or `left` picks the VR hand. |
| `ui.host`, `ui.port`, `ui.open_browser` | `127.0.0.1`, `8787`, `true` | `tandem ui`. A busy port steps to the next free one. |

Credentials go in `credentials.toml` beside it, readable only by you:

- **Gemini key:** `tandem config set-gemini-key`. Phase planning uses it, and so does TiPToP's perception (its
  object detector is a Gemini model). `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the environment wins over a
  stored key.
- **Hugging Face token:** `tandem config set-hf-token`. Read from the stored token, then `HF_TOKEN` or
  `HUGGING_FACE_HUB_TOKEN`, then `~/.cache/huggingface/token` (`huggingface-cli login`).

Each directory moves with an environment variable. The `~/.config` and `~/.local` defaults are Linux's; they
come from platformdirs and differ on other systems.

| directory | default | env var |
|---|---|---|
| config: `config.toml`, `credentials.toml` | `~/.config/tandem` | `$TANDEM_CONFIG_DIR` |
| state: logs and session scratch ([DATA.md](DATA.md#logs-and-session-files)) | `~/.local/state/tandem` | `$TANDEM_STATE_DIR` |
| share: holds the runtimes | `~/.local/share/tandem` | `$TANDEM_SHARE_DIR` |
| data: profiles and trajectories | `~/tandem-data` | `$TANDEM_DATA_ROOT` |
| TiPToP's runtime | `<share>/runtime` | `$TANDEM_RUNTIME_DIR` |
| other planners' runtimes | `<share>/runtimes/NAME` | `$TANDEM_RUNTIMES_DIR` |
| planner sources for an [offline install](#offline-install) | none | `$TANDEM_PLANNER_SOURCES` |

## The planner runtime

`tandem planners install tiptop` (which `tandem init` runs) builds TiPToP's stack into a
[runtime](README.md#terms): a pixi environment with torch, cuRobo's CUDA kernels, cuTAMP and tiptop. It needs
about 25 GB and takes 5–20 minutes the first time. It asks before installing pixi (`--yes` accepts).

It fetches the commits pinned in `src/tandem/planners/tiptop/recipe.py`; `tandem planners info tiptop` shows
them against what is installed. The `TANDEM` branches add [surface-fitted placement](#surface-fitted-placement)
and [the three switches](#the-three-switches), all off unless a profile turns them on.

| source | branch | pinned commit |
|---|---|---|
| [SamratSahoo/tiptop](https://github.com/SamratSahoo/tiptop/tree/TANDEM) | `TANDEM` | `6820474` |
| [SamratSahoo/cuTAMP](https://github.com/SamratSahoo/cuTAMP/tree/TANDEM) | `TANDEM` | `fc8f233` |
| [SamratSahoo/curobo](https://github.com/SamratSahoo/curobo) | `main` | `3a90ff4` |

The install checks each commit, applies one patch (so a profile owns its calibration file), places the
DATAFARM checkpoints tandem ships, and records what it installed in `<runtime>/.tandem-runtime.json`. Fetch
details: [ADDING_A_PLANNER.md](ADDING_A_PLANNER.md#a-runtime-recipe). Build log:
[DATA.md](DATA.md#logs-and-session-files).

- **Updating.** When a tandem upgrade moves a pin, `tandem planners list` shows `outdated`, and
  `tandem planners install tiptop` replaces only the moved source trees, reusing the environment. `--force`
  fetches everything again and rebuilds.
- **ZED.** The runtime lacks the ZED Python API (`pyzed`), which TiPToP opens the cameras with.
  `tandem runtime run install-zed` installs it from the ZED SDK (`/usr/local/zed/get_python_api.py`).

### Offline install

A workstation that can't reach GitHub installs from a bundle made on a machine that can:

```bash
tandem planners bundle tiptop --out /media/usb/planner-sources      # on a machine with network
tandem planners install tiptop --sources /media/usb/planner-sources # on the workstation
```

- Make the bundle with the same tandem version: the install checks each source's commit against its pins,
  and its files against the digest taken when it was bundled.
- `$TANDEM_PLANNER_SOURCES` works in place of `--sources`. The directory holds one checkout or export per
  source, named as the recipe names it (`tiptop/`, `cuTAMP/`, `curobo/`). While it is in force nothing is
  fetched, so a missing source is an error.
- Only sources are bundled. `pixi install` still downloads from conda-forge and PyPI (and builds SAM-2 from
  GitHub), and the first warm-up downloads the SAM-2 checkpoint (about 0.9 GB).

## Installing

tandem is not on PyPI; it installs from git with pipx or `uv tool` ([README](../README.md#1-install)). Plain
`pip install git+https://github.com/SamratSahoo/tandem.git` works inside a virtualenv, but not against
Debian's or Ubuntu's system Python (`externally-managed-environment`).

The `export` extra (`av`, `pyarrow`, `huggingface_hub`) is needed only for `tandem export lerobot`. To add it
to an existing install:

```bash
pipx inject tandem-tamp av pyarrow huggingface_hub
uv tool install --reinstall git+https://github.com/SamratSahoo/tandem.git --with av --with pyarrow --with huggingface_hub
```

To work on tandem itself:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[export,dev]'
pytest -q
```
