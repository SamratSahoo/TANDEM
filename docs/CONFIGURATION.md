# Configuration

Every setting tandem reads. Setup: [README](../README.md#setup). Commands: [USAGE.md](USAGE.md).

## Profiles

A [profile](README.md#terms) is a directory under `~/tandem-data/profiles/`:

```
~/tandem-data/profiles/bread-box/
├── profile.yml                    task, cameras, hitl, planner, recording, export
├── calibration.json               camera extrinsics, by serial
├── planner-options.<planner>.yml  a previous planner's options
└── trajectories/
    ├── eval/                      not yet labeled
    ├── success/
    └── failure/                   failures; settled trials (no label)
```

`src/tandem/resources/profile_template.yml` documents every key. Unknown keys fail at load, by name (a `tamp`
key also gets the nearest valid one).

### Profile keys

| key | default | what it does |
|---|---|---|
| `version` | `2` | Layout version ([older profiles](#older-profiles)). |
| `name`, `description` | | Name: lowercase letters, digits, `-`, `_`; not starting with `-` or `_`. Free-text description. |
| `task.prompt` | the template's | Language label stored with each episode. |
| `task.goal` | `null` | Planner goal, if not `task.prompt`. |
| `task.target_episodes` | `20` | Target shown by `tandem profile show`, the UI and the session. > 0. |
| `recording.enabled` | `true` | Record camera video. `tandem collect --no-record` overrides it. |
| `recording.fps` | `15` | Unused: each camera's `fps` sets the rate. |
| `export.hf_repo` | `''` | Repository for `tandem export lerobot` without `--repo`. |
| `export.private` | `false` | Upload as private. `--private` or `--public` overrides it. |

### Presets

`--preset paper` (on `tandem profile create` or `tandem init`) applies the paper's settings and prints each
change:

- `hitl`: on, with the paper's values (the [defaults](#phase-planning-hitl)).
- `planner.options.tamp`: replaced whole by the paper's TAMP and [DATAFARM](README.md#terms) settings,
  [the three switches](#the-three-switches) included. A task config's `placement_*` keys are lost; to keep
  them, use `--tamp-config` without `--preset`.
- Robot, cameras, perception: untouched, so `--from` or `--import-from` keeps the rig's.
- `tamp.time_dilation_factor_literal: 1.0`: planned motions ignore `robot.time_dilation_factor` (homing and
  capture moves don't). Keep a hand on the E-stop.

List: `tandem profile presets`. Format: [ADDING_A_PLANNER.md](ADDING_A_PLANNER.md#options-presets-and-doctor-rows).

### Importing a hitl-tamp-vla setup

```bash
tandem init --import-from ~/hitl-tamp-vla
tandem profile create bread-box --import-from ~/hitl-tamp-vla \
    --tamp-config ~/hitl-tamp-vla/data-collection/cfg/tamp/4_bread_box_v3.yml
```

- **Imported:** robot and camera serials (`tiptop.yml`); extrinsics (`calibration_info.json`,
  `calibration_info_<workspace>.json`); from `--tamp-config`: prompt, episode target, Hugging Face repo, TAMP
  settings (with `placement_*`), `hitl:`.
- **Refused** with phase planning on: a `robot_planner` other than `cutamp`; human phases run by a learned
  policy (`policy_type: diffusion` or `act`). The error names the task's config for a person.
- **Dropped with a warning:** `auto_mode`, `reset_placement_region`, `clear_goal_surfaces`.
- **Turned on** unless the config sets them: [the three switches](#the-three-switches) (`blend_stretch_to_caps`
  only with `blend_trajectory` on).

### Older profiles

Version-1 profiles (top-level `robot:`, `perception:`, `tamp:`) load with a notice and move under
`planner.options` on the next save. `tandem profile migrate [NAME]` (default: all) does it now, keeping
`profile.yml.v1.bak`. Their `hitl.on_robot_phase_failure: teleop` (the old default) is kept: set `abort` unless
you chose teleop. Older tandem can't read version 2.

## Phase planning (hitl)

[Phase planning](README.md#terms) is off in the template; `--preset paper` turns it on. Other defaults are the
paper's.

| key | default | what it does |
|---|---|---|
| `enabled` | `false` | Off: one leg per attempt, to the planner's goal; no proposal, phases or checks. |
| `proposal_model` | `gemini-2.5-pro` | Splits the task into phases; invents predicates and operators. |
| `vlm_model` | `gemini-2.5-flash` | Answers checks; names objects for `tandem plan`. |
| `max_attempts` | `3` | Tries per proposal and check (rejections fed back); max re-plans per trial under `replan`. ≥ 1. |
| `classify_initial` | `false` | Classify each invented atom on the first image (a call each); with `check_plan_effects`, re-run the contract check. Recorded only. |
| `verify_retries` | `1` | Retries for a human phase that fails its check; the operator is told what's missing. ≥ 0. |
| `verify_enforced` | `true` | Off: failed checks are recorded only. |
| `on_verification_failure` | `exclude` | Still failing after retries. `exclude`: file in `failure/` with `excluded: true`, no label. `label`: a failure; the label decides. |
| `verify_final_phase` | `true` | Off: a human phase that is the plan's last isn't checked; the label decides. |
| `check_human_effects` | `true` | Check a human phase's add effects hold and delete effects don't. |
| `check_human_preconditions` | `false` | Check a human phase's preconditions once, before hand-off. |
| `check_tamp_preconditions` | `false` | Before a robot leg, check what earlier phases should have made true, on its perception image. |
| `check_tamp_effects` | `false` | After a robot leg, check its add effects. Recorded only. |
| `precondition_enforced` | `false` | An unmet precondition (either kind) ends the trial at `verification`. Off: recorded only. |
| `check_plan_effects` | `true` | Run the [contract check](METHOD.md#the-contract-check) when repairing a proposal. No model call. |
| `save_vlm_io` | `true` | Save every model image and reply, rejected ones too, in [`vlm/`](DATA.md#vlm). |
| `cache_path` | `null` | Proposal-only SQLite cache (key: model, prompt, image), relative to the profile. Re-plans skip it. |
| `on_robot_phase_failure` | `abort` | A robot phase can't be planned. `abort`: fail at `tamp_planning`. `teleop`: a person does it, checked like a human phase. `replan`: propose again with the failure. An *execution* failure always ends at `tamp_execution`. |
| `conjoin_robot_phases` | `true` | Plan consecutive robot phases as one goal where sound ([conditions](ADDING_A_PLANNER.md#capabilities)). Off: one leg and perception pass each. |
| `human_executor` | `teleop` | Who does human phases, by registered name ([choosing one](ADDING_A_HUMAN_EXECUTOR.md#choosing-one)). |
| `human_executor_options` | `{}` | Each executor's settings, keyed by its name. Checked at load. |
| `allow_unrecorded_human_phase` | `false` | While recording, accept a human phase with no recording (`d`, done by hand, or an executor leg with no frames) instead of asking again. `--no-record` always accepts. |
| `verification_camera` | `external` | Check image: `external`, `hand` or `perception` (perception's camera). |

A check costs one model call per checkable atom, with the arm parked. Checkable atoms: invented predicates plus
the planner's `checkable_predicates` (TiPToP: `On`; [capabilities](ADDING_A_PLANNER.md#capabilities)). Trial
flow: [METHOD.md](METHOD.md#the-trial-loop).

## Cameras and calibration

```yaml
cameras:
  perception: external
  hand:     {serial: '14846828', type: zed, resolution: HD720, fps: 15}
  external: {serial: '32439448', type: zed, resolution: HD720, fps: 15}
  # external_2: {serial: '...', type: zed, resolution: HD720, fps: 15}
```

Every leg, robot or human, is recorded from these cameras.

| key | default | what it does |
|---|---|---|
| `perception` | `external` | Camera perception reads. `external`: the arm stays at `q_home`. `hand`: it moves to `q_capture` first. |
| `hand`, `external` | | Wrist and third-person cameras. Both must open at warm-up or the session stops. |
| `external_2` | | Optional second third-person camera (DROID `exterior_2`). If listed, it must open or collection stops. |
| `<camera>.serial` | | ZED serial, quoted (`'14846828'`). Unquoted is refused. |
| `<camera>.type`, `.resolution` | `zed`, `HD720` | |
| `<camera>.fps` | `15` | Keep 15: three ZEDs at HD720@30 exceed USB bandwidth, and the export resamples to 15 Hz. |

`calibration.json` holds one pose per serial (TiPToP's `calibration_info.json` format). Wrist: `ee_from_cam`.
Fixed: `world_from_cam`. Rotations: `xyz` Euler, radians.

```json
{"<serial>": {"pose": [x, y, z, roll, pitch, yaw]}}
```

A configured serial with no entry stops the session before warm-up; `tandem doctor` reports it. The
[README](../README.md#5-cameras-and-calibration) has the steps to fill it in. Also:

- **Another robot address or type.** TiPToP's calibration scripts read
  `$(tandem runtime path)/tiptop/tiptop/config/tiptop.yml` (`172.16.0.2`, `fr3_robotiq`), not the profile.
  Copy it, edit `robot.host` and `robot.type`, and set `$TIPTOP_CONFIG` to the copy's absolute path.
- **Gripper mask** (`perception: hand` only). The runtime's `tiptop/tiptop/config/assets/gripper_mask.png`
  masks the fingers out of the wrist cloud. Make yours: `tandem runtime run compute-gripper-mask` (or
  `paint-gripper-mask`).

## Planner settings

```yaml
planner:
  backend: tiptop     # which planner
  options:            # its settings, checked by the planner at load
    robot: ...
    perception: ...
    tamp: ...
```

`tandem planners use NAME` sets `planner.backend`. It:

- stashes the old planner's `options` in `planner-options.<planner>.yml`, restored on switching back;
- takes `--option KEY=VALUE` (repeatable) for options the new planner requires;
- warns, but doesn't refuse, if its runtime isn't installed;
- repairs a profile naming a planner this machine lacks.

**Did my setting apply?** `tandem profile show NAME --planner` prints what the planner gets; a key missing there
didn't apply. `tandem planners info NAME` lists what a planner reads.

### TiPToP options

**`robot`**

| key | default | what it does |
|---|---|---|
| `type` | `fr3_robotiq` | `fr3_robotiq` (FR3) or `panda_robotiq` (Panda): Robotiq 2F-85, via the bamboo-polymetis shim. `panda` (Franka Hand): the shim refuses its gripper commands. `ur5` (UR5, Robotiq gripper): needs tiptop's `ur5` extra (ur_rtde), not in the runtime. |
| `dof` | `7` | Joint count; must match `q_home` and `q_capture`. |
| `host` | `172.16.0.2` | The NUC running the shim. |
| `port`, `gripper_port`, `state_port` | `5555`, `5559`, `5557` | Shim control, gripper and state (encoder) ports. |
| `time_dilation_factor` | `0.2` | Arm speed, in (0, 1]; `0.2` is 20%. |
| `q_home`, `q_capture` | the template's | Home pose; capture pose for `cameras.perception: hand`. |

**`perception`**

| key | default | what it does |
|---|---|---|
| `m2t2.url`, `m2t2.apply_bounds` | `http://localhost:8123`, `true` | M2T2 grasp server; a flag sent with each request. |
| `sam_mode` | `local` | `local`: SAM-2 in the runtime. `remote`: the server at `sam_url` (then required). |
| `depth_smoothing_frames` | `5` | Depth frames median-fused at capture. `1` disables it. |
| `robot_mask_margin_m` | `0.02` | Arm collision-sphere padding when cutting the arm from a third-person cloud. Raise it if arm remains. |
| `depth_trunc_m`, `voxel_downsample_size`, `contact_threshold_m`, `mask_erosion_pixels` | `5.0`, `0.0075`, `0.01`, `3` | tiptop's settings of those names. |
| `gemini.model`, `gemini.temperature` | `gemini-robotics-er-2-preview`, `null` | Recorded only: the pinned tiptop always runs this model at its own temperature. `tandem doctor` warns on a mismatch. |

**`tamp`**

cuTAMP and cuRobo overrides with tiptop's key names, so a `cfg/tamp/*.yml` pastes in as is. Accepted keys:
`src/tandem/planners/tiptop/tamp_keys.py`. `tamp` beats `perception` for `contact_threshold_m` and
`voxel_downsample_size`. `tandem doctor` warns about a key that needs another (`blend_ops` without
`blend_trajectory`).

### Surface-fitted placement

By default cuTAMP places an object anywhere in a surface's bounding box, at its top: an open box's lid, a
plate's rim. `placement_support: true` uses the observed level patches that fit the footprint, each at its own
height. All are `tamp:` keys.

| key | default | what it does |
|---|---|---|
| `placement_support` | `false` | Turns it on. The rest are read only then; `tandem doctor` warns otherwise. |
| `placement_support_margin` | `0.01` | Surface kept around the footprint, in metres. ≥ 0. |
| `placement_flatness_tol` | `0.008` | Height variation that still counts as level, in metres; also the allowed slope. > 0. Raise it for a noisy reconstruction. |
| `placement_support_required` | `true` | No patch fits: the plan fails (see `hitl.on_robot_phase_failure`). `false`: use the bounding box. |
| `placement_into_surface` | `true` | The object may overlap its surface in the collision check. Containers need it: they reconstruct as filled hulls. |
| `placement_fill_occluded` | `false` | Unseen cells inside a surface's outline count as floor, as in a box the camera can't see into. The only setting that places on unseen surface. |
| `placement_min_seen_frac` | `0.25` | Observed fraction each footprint needs; guards `placement_fill_occluded`. In [0, 1]. |

The paper's configs: `1_toy_puzzle_v3.yml` sets `placement_support`, `placement_support_required` and
`placement_into_surface`; `4_bread_box*.yml` sets all seven (margin `0.005`, flatness `0.012`,
`placement_fill_occluded: true`). `--tamp-config` imports them unchanged.

### The three switches

`tamp:` keys, off by default. `--preset paper` and an [import](#importing-a-hitl-tamp-vla-setup) turn them on.

| key | what it switches on |
|---|---|
| `table_plane_support_vote` | Pick the table among RANSAC's planes by the objects resting on each, not every object within 3 cm above or below. |
| `disjoint_object_masks` | Meshes and clouds from disjoint masks (a pixel two masks claim goes to the smaller object), so a container's hull stops at what rests on it. Placement fitting always uses them. |
| `blend_stretch_to_caps` | With `blend_trajectory` on, slow a stroke that can't be re-timed within the velocity and acceleration caps until it fits, instead of keeping the plan's timing. A stroke can get many times slower. |

## Machine settings and credentials

`~/.config/tandem/config.toml`. Edit with `tandem config` (`list`, `get`, `set`, `edit`, `path`):

| key | default | what it does |
|---|---|---|
| `active_profile` | `default` | The profile commands act on (`tandem profile use`). |
| `data_root` | `~/tandem-data` | Profiles and trajectories. `$TANDEM_DATA_ROOT` wins. |
| `runtime_dir` | `~/.local/share/tandem/runtime` | TiPToP's runtime (about 25 GB). `$TANDEM_RUNTIME_DIR` wins. |
| `default_planner` | `tiptop` | The planner new profiles get (`tandem planners default NAME`). |
| `hf_org` | | Owner for an export repository named without one (`--repo NAME` or `export.hf_repo`). |
| `teleop.enabled`, `teleop.droid_dir`, `teleop.python` | off | The teleop driver: a DROID checkout and its Python ([teleop executor](ADDING_A_HUMAN_EXECUTOR.md#the-teleop-executor)). |
| `teleop.device`, `teleop.controller` | `vr`, `right` | `vr` or `spacemouse`; `right` or `left` VR hand. |
| `ui.host`, `ui.port`, `ui.open_browser` | `127.0.0.1`, `8787`, `true` | `tandem ui`. A busy port steps to the next free one. |

Credentials go in `credentials.toml` beside it, readable only by you:

- **Gemini key:** `tandem config set-gemini-key`. For phase planning and TiPToP's perception.
  `GEMINI_API_KEY` or `GOOGLE_API_KEY` overrides it.
- **Hugging Face token:** `tandem config set-hf-token`. Lookup order: stored token, `HF_TOKEN` or
  `HUGGING_FACE_HUB_TOKEN`, `~/.cache/huggingface/token` (`huggingface-cli login`).

Directories (Linux defaults; other systems differ):

| directory | default | env var |
|---|---|---|
| config: `config.toml`, `credentials.toml` | `~/.config/tandem` | `$TANDEM_CONFIG_DIR` |
| state: logs, session scratch ([DATA.md](DATA.md#logs-and-session-files)) | `~/.local/state/tandem` | `$TANDEM_STATE_DIR` |
| share: runtimes | `~/.local/share/tandem` | `$TANDEM_SHARE_DIR` |
| data: profiles, trajectories | `~/tandem-data` | `$TANDEM_DATA_ROOT` |
| TiPToP's runtime | `<share>/runtime` | `$TANDEM_RUNTIME_DIR` |
| other planners' runtimes | `<share>/runtimes/NAME` | `$TANDEM_RUNTIMES_DIR` |
| [offline install](#offline-install) sources | none | `$TANDEM_PLANNER_SOURCES` |

## The planner runtime

`tandem planners install tiptop` (run by `tandem init`) builds TiPToP's [runtime](README.md#terms), a pixi
environment (torch, cuRobo's CUDA kernels, cuTAMP, tiptop). First build: about 25 GB, 5–20 minutes. It asks
before installing pixi (`--yes` accepts).

Pinned sources (`tandem planners info tiptop` compares them with what's installed). The `TANDEM` branches add
[surface-fitted placement](#surface-fitted-placement) and [the three switches](#the-three-switches), off unless a
profile turns them on.

| source | branch | pinned commit |
|---|---|---|
| [SamratSahoo/tiptop](https://github.com/SamratSahoo/tiptop/tree/TANDEM) | `TANDEM` | `6820474` |
| [SamratSahoo/cuTAMP](https://github.com/SamratSahoo/cuTAMP/tree/TANDEM) | `TANDEM` | `fc8f233` |
| [SamratSahoo/curobo](https://github.com/SamratSahoo/curobo) | `main` | `3a90ff4` |

The install applies one patch (so `$TIPTOP_CALIBRATION` can point TiPToP at the profile's file) and places the
DATAFARM checkpoints tandem ships. Fetching:
[ADDING_A_PLANNER.md](ADDING_A_PLANNER.md#a-runtime-recipe). Build log: [DATA.md](DATA.md#logs-and-session-files).

- **Updating.** A moved pin shows `outdated` in `tandem planners list`; `tandem planners install tiptop`
  replaces only moved sources, reusing the environment. `--force` refetches and rebuilds all.
- **ZED.** `tandem runtime run install-zed` adds the ZED Python API (`pyzed`) from the ZED SDK
  (`/usr/local/zed/get_python_api.py`).

### Offline install

No GitHub on the workstation? Bundle the sources on a machine that has it:

```bash
tandem planners bundle tiptop --out /media/usb/planner-sources      # on a machine with network
tandem planners install tiptop --sources /media/usb/planner-sources # on the workstation
```

- Use the same tandem version on both: the install checks each commit against its pin and the files against
  the bundle's digest.
- `$TANDEM_PLANNER_SOURCES` can replace `--sources` (one checkout or export per source: `tiptop/`, `cuTAMP/`,
  `curobo/`). While set, nothing is fetched; a missing source is an error.
- Only sources are bundled: `pixi install` still needs conda-forge, PyPI and GitHub (to build SAM-2), and the
  first warm-up downloads the SAM-2 checkpoint (about 0.9 GB).

## Installing

tandem isn't on PyPI: install it from git ([README](../README.md#1-install)). Plain
`pip install git+https://github.com/SamratSahoo/tandem.git` needs a virtualenv on Debian and Ubuntu
(`externally-managed-environment`).

`tandem export lerobot` needs the `export` extra (`av`, `pyarrow`, `huggingface_hub`). To add it:

```bash
pipx inject tandem-tamp av pyarrow huggingface_hub
uv tool install --reinstall git+https://github.com/SamratSahoo/tandem.git --with av --with pyarrow --with huggingface_hub
```

To develop tandem: `pip install -e '.[export,dev]'` in a virtualenv, then `pytest -q`.
