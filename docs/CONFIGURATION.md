# Configuration

This page lists every setting TANDEM reads. For setup, see the [README](../README.md#setup); for commands, [USAGE.md](USAGE.md).

## Profiles

A [profile](README.md#terms) is one task in one YAML file:

```yaml
# ~/tandem-data/profiles/my-task.yml
version: 3
description: 'Cup onto plate'
task:
  prompt: put the cup on the plate       # the episode's language label
  target_episodes: 30
hitl:                                    # phase planning (see below)
  enabled: true
  on_verification_failure: label
planner:
  backend: tiptop
  options:
    tamp:                                # TiPToP's task settings
      num_particles: 512
recording:
  enabled: true
export:
  hf_repo: my-org/cup-on-plate
```

Keys you leave out take their defaults. Usually you start from the paper's settings or a copy:

```bash
tandem profile create my-task --prompt "put the cup on the plate"   # the paper's settings, your task
tandem profile create my-box --from store-bread-in-closed-box      # a copy of any profile
tandem profile edit my-task                                        # opens $EDITOR; validated on save
```

Robot, camera and calibration settings aren't in a profile; they're in [the rig](#the-rig). Every key is documented
in `src/tandem/resources/profile_template.yml`. An unknown key fails at load, and a misspelled `tamp` key gets
a suggestion. Profiles and their trajectories live under `~/tandem-data/`:

```
~/tandem-data/
├── profiles/
│   ├── my-task.yml             one file per profile
│   └── .planner-options/       a previous planner's options (tandem planners use)
└── trajectories/
    └── my-task/
        ├── eval/               not yet labeled
        ├── success/
        └── failure/            failures, and settled trials (no label)
```

### Profile keys

| key | default | what it does |
|---|---|---|
| `version` | `3` | Layout version. See [older profiles](#older-profiles). |
| `description` | | Free text. The profile's name is its file name: lowercase letters, digits, `-` and `_`, not starting with `-` or `_`. |
| `task.prompt` | | The episode's language label, and what phase planning splits into steps. |
| `task.goal` | `null` | The planner's goal, when it must differ from `task.prompt`. |
| `task.target_episodes` | `20` | Target shown by `profile show`, the UI and the session. > 0. |
| `hitl` | | [Phase planning](#phase-planning-hitl). |
| `planner` | | [The planner and its task settings](#planner-settings). |
| `recording.enabled` | `true` | Record camera video. `tandem collect --no-record` overrides it. |
| `export.hf_repo` | `''` | Repository for `tandem export lerobot` without `--repo`. |
| `export.private` | `false` | Upload as private. `--private` or `--public` overrides it. |

### The paper's five

`tandem init` adds the paper's five tasks, each with the settings the paper collected it with, plus
[the three switches](#the-three-switches).

| profile | paper task (Fig. 3) | its own settings |
|---|---|---|
| `cover-bread-rolls` | Cover Bread Rolls | A grasp-centre cost and threshold, and finer voxels. |
| `solve-constrained-puzzle` | Solve Constrained Puzzle | [Surface-fitted placement](#surface-fitted-placement). |
| `sort-and-cover-snacks` | Sort & Cover Snacks | |
| `open-obstructed-book` | Open Obstructed Book | |
| `store-bread-in-closed-box` | Store Bread in Closed Box | Surface-fitted placement into the box, and finer voxels. |

- They're yours to edit. `tandem init` never overwrites one, and restores one you deleted.
- `profile create --prompt` gives a new profile what the five share: phase planning, the TAMP and
  [DATAFARM](README.md#terms) settings, and the three switches.
- **Safety:** they set `tamp.time_dilation_factor_literal: 1.0`, so planned motions ignore the rig's
  `time_dilation_factor` (homing and capture moves don't). Keep a hand on the E-stop.

### Older profiles

Before version 3, a profile was a directory (`profiles/<name>/profile.yml`) with its own robot, cameras and
`calibration.json`. They don't show up until you move them:

```bash
tandem profile migrate   # tandem init also does this
```

Each profile's task goes to `profiles/<name>.yml` and its trajectories to `trajectories/<name>/`. The robot
and cameras seed [the rig](#the-rig) (from the active profile) if `rig.yml` doesn't exist yet. Extrinsics are
added to the rig's `calibration.json` only for cameras without an entry.

- Old directories aren't deleted. Each is archived in `profiles/.migrated/<name>/` with a `migration.json`.
- If the rig can't be set up from them, no profile is moved. A profile that fails to move is left as it was, and
  running the command again finishes a partial move.
- `tandem rig set` and `rig edit` wait until old profiles are moved. Older TANDEM can't read version 3.
- Version-1 profiles keep `hitl.on_robot_phase_failure: teleop`, the old default. Set `abort` unless you
  chose teleop.

## Phase planning (hitl)

Phase planning is on in the paper's profiles and in every new profile, and off in a profile with no `hitl:`
block. The defaults match the paper. Some common changes:

```yaml
# Label failed human steps yourself, instead of excluding the trial from the dataset
hitl:
  enabled: true
  on_verification_failure: label
```

```yaml
# When the planner can't plan a robot step, let a person do it (checked like a human step)
hitl:
  enabled: true
  on_robot_phase_failure: teleop    # or replan: ask the model for a new plan
```

```yaml
# Turn phase planning off: the planner does the whole task, with no human steps or checks
hitl:
  enabled: false
```

Each check costs one model call per checkable atom, with the arm parked. Checkable atoms are invented
predicates plus the planner's `checkable_predicates` (TiPToP: `On`; [capabilities](ADDING_A_PLANNER.md#capabilities)).

### Planning

| key | default | what it does |
|---|---|---|
| `enabled` | `false` | When off, each attempt is one leg to the planner's goal, with no phases or checks. |
| `proposal_model` | `gemini-2.5-pro` | Splits the task into phases and invents predicates and operators. |
| `vlm_model` | `gemini-2.5-flash` | Answers checks, and names objects for `tandem plan`. |
| `max_attempts` | `3` | Tries per proposal and per check, rejections fed back. Also the re-plan limit under `replan`. ≥ 1. |
| `check_plan_effects` | `true` | Run the contract check when repairing a proposal (no model call). |
| `conjoin_robot_phases` | `true` | Plan consecutive robot phases as one goal where sound ([conditions](ADDING_A_PLANNER.md#capabilities)). When off, each robot phase gets its own leg and perception pass. |
| `cache_path` | `null` | SQLite cache of proposals (keyed by model, prompt, image), relative to `profiles/`. Re-plans skip it. |
| `save_vlm_io` | `true` | Save every model image and reply, rejected ones too, in [`vlm/`](DATA.md#vlm). |

### Checks

| key | default | what it does |
|---|---|---|
| `verification_camera` | `external` | Which camera's image the checks use: `external`, `hand` or `perception` (whichever camera perception uses). |
| `check_human_effects` | `true` | After a human phase, check its add effects hold and delete effects don't. |
| `check_human_preconditions` | `false` | Check a human phase's preconditions once, before the hand-off. |
| `check_tamp_preconditions` | `false` | Before a robot leg, check what earlier phases should have made true. |
| `check_tamp_effects` | `false` | After a robot leg, check its add effects. Recorded only. |
| `classify_initial` | `false` | Classify each invented atom on the first image (a call each); with `check_plan_effects`, re-run the contract check. Recorded only. |
| `verify_enforced` | `true` | When off, failed checks are only recorded. |
| `precondition_enforced` | `false` | When on, an unmet precondition ends the trial at `verification`. When off, it's only recorded. |
| `verify_final_phase` | `true` | When off, a final human phase isn't checked and your label decides. |

### Failures

| key | default | what it does |
|---|---|---|
| `verify_retries` | `1` | Extra tries for a human phase that fails its check; the operator is told what's missing. ≥ 0. |
| `on_verification_failure` | `exclude` | What happens when a check still fails after retries. `exclude` files the trial in `failure/` as excluded, with no label. `label` lets your label decide. |
| `on_robot_phase_failure` | `abort` | What happens when a robot phase can't be planned. `abort` fails the trial at `tamp_planning`. `teleop` has a person do it, checked like a human phase. `replan` asks the model for a new plan that accounts for the failure. Execution failures always end at `tamp_execution`. |

### Human executor

| key | default | what it does |
|---|---|---|
| `human_executor` | `teleop` | Who does human phases ([choosing one](ADDING_A_HUMAN_EXECUTOR.md#choosing-one)). |
| `human_executor_options` | `{}` | Each executor's settings, keyed by its name. Checked at load. |
| `allow_unrecorded_human_phase` | `false` | While recording, accept a human phase with no recording (`d`, or a leg with no frames). `--no-record` always accepts. |

## The rig

The rig holds this workstation's robot, camera and calibration settings. All profiles use it. `tandem init`
writes it to `~/.config/tandem/rig.yml`:

```yaml
version: 1
robot:
  type: fr3_robotiq
  host: 172.16.0.2          # the NUC
cameras:
  perception: external
  hand:     {serial: '14846828'}
  external: {serial: '32439448'}
calibration: calibration.json
planners: {}                # a planner's machine settings, where they differ from its defaults
```

To view or change it:

```bash
tandem rig show                                   # also --json; the web UI's Settings page shows it too
tandem rig set robot.host NUC_ADDRESS             # `null` removes a setting
tandem rig set planners.tiptop.perception.m2t2.url http://HOST:8123
tandem rig edit                                   # opens $EDITOR; validated on save
tandem rig path --calibration                     # where the extrinsics are
```


| key | default | what it does |
|---|---|---|
| `robot.type` | `fr3_robotiq` | The arm. TiPToP: `fr3_robotiq` (FR3) or `panda_robotiq` (Panda), with a [Robotiq 2F-85](https://robotiq.com/products/adaptive-grippers) via the shim. `panda` (Franka Hand): the shim refuses its gripper commands. `ur5`: needs tiptop's `ur5` extra (ur_rtde), not in the runtime. |
| `robot.host` | `172.16.0.2` | The NUC's hostname or IP address. The planner sets the ports. |
| `cameras.perception` | `external` | The camera perception uses. With `external` the arm stays at `q_home`; with `hand` it moves to `q_capture` first. |
| `cameras.hand`, `cameras.external` | | Wrist and third-person cameras. Every leg is recorded from them; both must open at warm-up. |
| `cameras.external_2` | | Optional second third-person camera ([DROID](https://droid-dataset.github.io/) `exterior_2`). If listed, it must open. |
| `<camera>.serial` | | ZED serial, quoted (`'14846828'`). One role per serial. |
| `<camera>.type`, `.resolution` | `zed`, `HD720` | |
| `<camera>.fps` | `15` | Keep 15: three ZEDs at HD720@30 exceed USB bandwidth, and export resamples to 15 Hz. |
| `calibration` | `calibration.json` | The extrinsics file, relative to `rig.yml`, or absolute. |
| `planners.<name>` | | Each planner's machine settings ([TiPToP's](#tiptop-options)). |

### Calibration file

`calibration.json` holds one pose per camera, keyed by serial:

```json
{
  "14846828": {"pose": [0.0266, 0.0705, -0.1392, -0.4509, -0.0045, -1.5620]},
  "32439448": {"pose": [0.1532, -0.5827, 0.4410, -2.0843, 0.0126, 0.1964]}
}
```

- `pose` is `[x, y, z, roll, pitch, yaw]`, in meters and radians (`xyz` Euler angles).
- The wrist camera's pose is relative to the end effector (`ee_from_cam`).
- An external camera's pose is relative to the robot's base (`world_from_cam`).
- Other keys in an entry, such as `timestamp`, are ignored.

A session won't start if a camera has no entry. To get the poses, see
[README](../README.md#5-cameras-and-calibration).

### Related settings outside the rig

- **TiPToP's scripts** (`calibrate-wrist-cam`, `viz-calibration`, `cutamp-demo`, …) run through
  `tandem runtime run` or `shell` use your rig settings, through `$TIPTOP_CONFIG` and `$TIPTOP_CALIBRATION`.
  `--raw` uses tiptop's stock config instead.
- **Teleop** uses `robot.host` too; TANDEM passes it to the teleop driver.
- **Gripper mask** (`perception: hand` only): make yours with `tandem runtime run compute-gripper-mask` (or
  `paint-gripper-mask`). It writes the runtime's `tiptop/tiptop/config/assets/gripper_mask.png`.

## Planner settings

A profile names its planner and that planner's settings for the task:

```yaml
planner:
  backend: tiptop     # which planner
  options:            # its task settings, checked by the planner at load
    tamp:
      num_particles: 512
```

Task settings go in the profile's `planner.options`. Machine settings (a server's address, a robot's ports) go
in [the rig](#the-rig) under `planners.<name>`. If you put a key in the wrong file, TANDEM refuses it and says
where it goes. `tandem planners info NAME` lists both ([how planners declare them](ADDING_A_PLANNER.md#options-and-doctor-rows)).

**Did my setting apply?** `tandem profile show NAME --planner` prints exactly what the planner receives.

### Switching planners

```bash
tandem planners use myplanner                  # the active profile now plans with it
tandem planners use myplanner -o KEY=VALUE     # also set an option it requires (repeatable)
tandem planners default myplanner              # the planner new profiles get
```

`use` sets `planner.backend` and stashes the old planner's `options` in
`profiles/.planner-options/<profile>.<planner>.yml`, restoring them when you switch back. It warns if the
runtime isn't installed, and it repairs a profile naming a planner this machine lacks.

### TiPToP options

**Machine settings** live in `rig.yml` under `planners.tiptop`:

```bash
tandem rig set planners.tiptop.robot.time_dilation_factor 0.3          # arm at 30% speed
tandem rig set planners.tiptop.perception.m2t2.url http://HOST:8123
tandem rig set planners.tiptop.perception.depth_smoothing_frames 1
```

The robot's address and type come from `robot.host` and `robot.type`.

`robot`:

| key | default | what it does |
|---|---|---|
| `dof` | `7` | Joint count. It must match `q_home` and `q_capture`. |
| `port`, `gripper_port`, `state_port` | `5555`, `5559`, `5557` | Shim control, gripper and state ports. |
| `time_dilation_factor` | `0.2` | Arm speed, in (0, 1]. `0.2` is 20%. |
| `q_home`, `q_capture` | TiPToP's | Home pose; capture pose for `cameras.perception: hand`. |

`perception`:

| key | default | what it does |
|---|---|---|
| `m2t2.url`, `m2t2.apply_bounds` | `http://localhost:8123`, `true` | [M2T2](https://github.com/SamratSahoo/M2T2/tree/TANDEM) grasp server, and a flag sent with each request. If it runs on this workstation, a session starts it when it's down ([servers](USAGE.md#servers)). `tandem doctor` probes it. |
| `foundation_stereo.url` | `http://localhost:1234` | [FoundationStereo](https://github.com/SamratSahoo/FoundationStereo/tree/TANDEM) depth server, used every rollout. If it runs on this workstation, a session starts it when it's down. `tandem doctor` probes it. |
| `sam_mode` | `local` | `local`: [SAM-2](https://github.com/facebookresearch/sam2) in the runtime. `remote`: the server at `sam_url` (then required). |
| `depth_smoothing_frames` | `5` | Depth frames median-fused at capture. `1` disables it. |
| `robot_mask_margin_m` | `0.02` | Padding when cutting the arm out of a third-person cloud. Raise it if arm points remain. |
| `depth_trunc_m`, `voxel_downsample_size`, `contact_threshold_m`, `mask_erosion_pixels` | `5.0`, `0.0075`, `0.01`, `3` | Passed to tiptop's settings of the same names. |
| `gemini.model`, `gemini.temperature` | `gemini-robotics-er-2-preview`, `null` | Only recorded; the pinned tiptop always uses this model at its own temperature. `tandem doctor` warns on a mismatch. |

**Task settings** live in the profile's `planner.options.tamp`:

```yaml
planner:
  backend: tiptop
  options:
    tamp:
      num_particles: 512
      voxel_downsample_size: 0.005   # finer than the rig's perception setting
      placement_support: true        # see Surface-fitted placement
```

- These are cuTAMP and cuRobo overrides with tiptop's key names, so a `cfg/tamp/*.yml`'s `tamp_overrides`
  paste in as is. Accepted keys: `src/tandem/planners/tiptop/tamp_keys.py`.
- For `contact_threshold_m` and `voxel_downsample_size`, the `tamp` value overrides the rig's.
- `tandem doctor` warns about a key that needs another (`blend_ops` without `blend_trajectory`).
- Relative checkpoint paths (`vae_path`, `blend_model_path`, `blend_stats_path`, `posture_ref`) resolve beside
  the profile, then in the runtime, where the install puts the DATAFARM checkpoints
  (`vae/checkpoints/vae_full_v2.pt`, `rnd/checkpoints/rnd_droid.pt`).

### Surface-fitted placement

By default, cuTAMP can place an object anywhere in a surface's bounding box, at the top of it, such as on an
open box's rim or a plate's edge. `placement_support: true` places only on observed level patches that fit the footprint:

```yaml
tamp:
  placement_support: true
  placement_fill_occluded: true    # e.g. a box whose floor the camera can't see
```

| key | default | what it does |
|---|---|---|
| `placement_support` | `false` | Turns it on. The other keys are read only then. |
| `placement_support_margin` | `0.01` | Surface kept around the footprint, in metres. ≥ 0. |
| `placement_flatness_tol` | `0.008` | Height variation (and slope) that still counts as level, in metres. > 0. Raise it for noisy reconstructions. |
| `placement_support_required` | `true` | If no patch fits, the plan fails. With `false`, it falls back to the bounding box. |
| `placement_into_surface` | `true` | Let the object overlap its surface in collision checks. Containers need it (they reconstruct as filled hulls). |
| `placement_fill_occluded` | `false` | Count unseen cells inside a surface's outline as floor, as in a box the camera can't see into. |
| `placement_min_seen_frac` | `0.25` | Observed fraction each footprint needs; guards `placement_fill_occluded`. In [0, 1]. |

In the paper's profiles, `solve-constrained-puzzle` sets `placement_support`, `placement_support_required` and
`placement_into_surface`. `store-bread-in-closed-box` sets all seven, with margin `0.005`, flatness `0.012` and
`placement_fill_occluded: true`.

### The three switches

These `tamp:` keys are off unless set. [The paper's five](#the-papers-five) and every new profile turn them on,
because every run in the paper used them.

```yaml
tamp:
  table_plane_support_vote: true
  disjoint_object_masks: true
  blend_stretch_to_caps: true
```

| key | what it switches on |
|---|---|
| `table_plane_support_vote` | Pick the table among RANSAC's planes by the objects resting on each, instead of by every object within 3 cm above or below. |
| `disjoint_object_masks` | Disjoint object masks (a shared pixel goes to the smaller object), so a container's hull stops at what rests on it. Surface-fitted placement always uses them. |
| `blend_stretch_to_caps` | With `blend_trajectory`, slow a stroke that can't be re-timed within the velocity and acceleration caps until it fits. It can get many times slower. |

## TANDEM settings and credentials

TANDEM's own settings live in `~/.config/tandem/config.toml`:

```bash
tandem config list
tandem config set ui.port 8800
tandem config set data_root /mnt/data/tandem
tandem config edit                 # also: get KEY, path
```

| key | default | what it does |
|---|---|---|
| `active_profile` | none | The profile commands act on (`tandem profile use`). `tandem init` sets `cover-bread-rolls`, and `tandem profile create` makes its profile active when none is. |
| `data_root` | `~/tandem-data` | Profiles and trajectories. `$TANDEM_DATA_ROOT` overrides it. |
| `runtime_dir` | `~/.local/share/tandem/runtime` | TiPToP's runtime (about 25 GB). `$TANDEM_RUNTIME_DIR` overrides it. |
| `default_planner` | `tiptop` | The planner new profiles get (`tandem planners default NAME`). |
| `hf_org` | | Owner for an export repo named without one. |
| `teleop.enabled` | `false` | Human phases can use teleop. `tandem executors install teleop` turns it on ([teleop](ADDING_A_HUMAN_EXECUTOR.md#the-teleop-executor)). |
| `teleop.droid_dir`, `teleop.python` | blank | Blank: the driver runs in the teleop runtime. Set both to run a DROID checkout and environment of your own. |
| `teleop.controller` | `right` | Which VR controller drives the arm: `right` or `left`. |
| `ui.host`, `ui.port`, `ui.open_browser` | `127.0.0.1`, `8787`, `true` | `tandem ui`. A busy port steps to the next free one. |

### Credentials

Credentials are stored in `credentials.toml` next to it, readable only by you:

```bash
tandem config set-gemini-key    # for phase planning and TiPToP's perception
tandem config set-hf-token      # for tandem export lerobot --push
```

- `GEMINI_API_KEY` or `GOOGLE_API_KEY` overrides the stored [Gemini key](https://aistudio.google.com/apikey).
- The Hugging Face token is looked up in order: stored, `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`, then
  `~/.cache/huggingface/token` (`huggingface-cli login`).

### Directories

These are the Linux defaults; other systems differ.

| directory | default | env var |
|---|---|---|
| config: `config.toml`, `credentials.toml`, `rig.yml`, `calibration.json` | `~/.config/tandem` | `$TANDEM_CONFIG_DIR` |
| state: logs, session scratch ([DATA.md](DATA.md#logs-and-session-files)) | `~/.local/state/tandem` | `$TANDEM_STATE_DIR` |
| share: runtimes | `~/.local/share/tandem` | `$TANDEM_SHARE_DIR` |
| data: profiles, trajectories | `~/tandem-data` | `$TANDEM_DATA_ROOT` |
| TiPToP's runtime | `<share>/runtime` | `$TANDEM_RUNTIME_DIR` |
| other planners' runtimes, teleop's and the servers' | `<share>/runtimes/NAME` (`teleop`, `m2t2`, `foundation_stereo`) | `$TANDEM_RUNTIMES_DIR` |
| [offline install](#offline-install) sources | none | `$TANDEM_PLANNER_SOURCES` |

## The planner runtime

```bash
tandem planners install tiptop    # build or update TiPToP's runtime (tandem init runs this)
tandem planners info tiptop       # pinned vs installed commits
tandem planners install tiptop --force   # refetch and rebuild everything
```

TiPToP's [runtime](README.md#terms) is a [pixi](https://pixi.sh) environment (torch, cuRobo's CUDA kernels, cuTAMP, tiptop).
The first build takes about 25 GB and 5–20 minutes, and asks before installing pixi (`--yes` accepts). Its
pinned sources are below; the `TANDEM` branches add [surface-fitted placement](#surface-fitted-placement) and
[the three switches](#the-three-switches).

| source | branch | pinned commit |
|---|---|---|
| [SamratSahoo/tiptop](https://github.com/SamratSahoo/tiptop/tree/TANDEM) | `TANDEM` | `6820474` |
| [SamratSahoo/cuTAMP](https://github.com/SamratSahoo/cuTAMP/tree/TANDEM) | `TANDEM` | `fc8f233` |
| [SamratSahoo/curobo](https://github.com/SamratSahoo/curobo) | `main` | `3a90ff4` |

The install applies one patch (so `$TIPTOP_CONFIG` and `$TIPTOP_CALIBRATION` can point TiPToP at the rig)
and places the DATAFARM checkpoints. How sources are fetched is in [ADDING_A_PLANNER.md](ADDING_A_PLANNER.md#a-runtime-recipe), and where the
build log goes is in [DATA.md](DATA.md#logs-and-session-files).

- **Updating.** A moved pin shows `outdated` in `tandem planners list`. Reinstalling replaces only the moved
  sources.
- **ZED.** With the [ZED SDK](https://www.stereolabs.com/developers/release) installed (`/usr/local/zed/get_python_api.py`), the install adds `pyzed`. Without
  it the install succeeds but ZED cameras won't open, and `tandem doctor` says so. Install the SDK, then
  reinstall.

### Offline install

If the workstation can't reach GitHub, bundle the sources on a machine that can:

```bash
tandem planners bundle tiptop --out /media/usb/planner-sources      # on a machine with network
tandem planners install tiptop --sources /media/usb/planner-sources # on the workstation
```

- Use the same TANDEM version on both: commits are checked against their pins and files against the digest.
- `$TANDEM_PLANNER_SOURCES` can replace `--sources` (one directory per source: `tiptop/`, `cuTAMP/`,
  `curobo/`). While set, nothing is fetched.
- Only sources are bundled. `pixi install` still needs conda-forge, PyPI and GitHub (to build SAM-2), and the
  first warm-up downloads the SAM-2 checkpoint (about 0.9 GB).

## Installing

TANDEM isn't on PyPI; install it from git ([README](../README.md#1-install)). Plain `pip install git+…` needs
a virtualenv on Debian and Ubuntu. Every install includes what `tandem export lerobot` needs (`av`, `pyarrow`
and `huggingface_hub`).

To develop TANDEM: `pip install -e '.[dev]'` in a virtualenv, then `pytest -q`.
