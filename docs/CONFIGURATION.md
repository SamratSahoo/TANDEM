# Configuration

Every setting tandem reads. Setup: [README](../README.md#setup). Commands: [USAGE.md](USAGE.md).

## Profiles

A [profile](README.md#terms) is one task, in one YAML file under `~/tandem-data/profiles/`:

```
~/tandem-data/
├── profiles/
│   ├── cover-bread-rolls.yml   one file per profile: the task's settings
│   ├── my-task.yml
│   └── .planner-options/       a previous planner's options (tandem planners use)
└── trajectories/
    └── my-task/
        ├── eval/               not yet labeled
        ├── success/
        └── failure/            failures; settled trials (no label)
```

The robot, cameras and calibration are not a profile's: they are [the rig](#the-rig), which every profile
shares. `src/tandem/resources/profile_template.yml` documents every key. Unknown keys fail at load, by name (a
`tamp` key also gets the nearest valid one).

```bash
tandem profile create my-task --prompt "put the cup on the plate"   # the paper's settings, your task
tandem profile create my-box --from store-bread-in-closed-box      # a copy of any profile
tandem profile edit my-task                                        # $EDITOR; validated on save
```

### Profile keys

| key | default | what it does |
|---|---|---|
| `version` | `3` | Layout version ([older profiles](#older-profiles)). |
| `description` | | Free text. The name is the file's: lowercase letters, digits, `-`, `_`; not starting with `-` or `_`. |
| `task.prompt` | | Language label stored with each episode, and what phase planning splits into steps. |
| `task.goal` | `null` | Planner goal, if not `task.prompt`. |
| `task.target_episodes` | `20` | Target shown by `tandem profile show`, the UI and the session. > 0. |
| `hitl` | | [Phase planning](#phase-planning-hitl). |
| `planner` | | [The planner and its task settings](#planner-settings). |
| `recording.enabled` | `true` | Record camera video. `tandem collect --no-record` overrides it. |
| `export.hf_repo` | `''` | Repository for `tandem export lerobot` without `--repo`. |
| `export.private` | `false` | Upload as private. `--private` or `--public` overrides it. |

### The paper's five

`tandem init` adds the paper's five tasks as profiles, each with the settings the paper collected it with: the
prompt, `hitl:` block and `tamp_overrides` of its config in hitl-tamp-vla, plus
[the three switches](#the-three-switches). Nothing of the paper's rig comes with them.

| profile | paper task (Fig. 3) | from | its own settings |
|---|---|---|---|
| `cover-bread-rolls` | Cover Bread Rolls | `8c_pp_3bread_cloth_08272026_v3.yml` | a grasp-centre cost and threshold, finer voxels |
| `solve-constrained-puzzle` | Solve Constrained Puzzle | `1_toy_puzzle_v3.yml` | [surface-fitted placement](#surface-fitted-placement) |
| `sort-and-cover-snacks` | Sort & Cover Snacks | `2_bread_fruit_bowl_cloth_v3.yml` | |
| `open-obstructed-book` | Open Obstructed Book | `3_pen_open_book_v3.yml` | |
| `store-bread-in-closed-box` | Store Bread in Closed Box | `4_bread_box_v3.yml` | surface-fitted placement into the box, finer voxels |

- Once copied they're yours to edit. `tandem init` never overwrites one, and puts back one you deleted.
- A new profile (`--prompt`) is what the five share: phase planning on, the same TAMP and
  [DATAFARM](README.md#terms) settings, and the three switches.
- `tamp.time_dilation_factor_literal: 1.0`: planned motions ignore the rig's `time_dilation_factor` (homing and
  capture moves don't). Keep a hand on the E-stop.

### Older profiles

Before version 3 a profile was a directory (`profiles/<name>/profile.yml`) with its own cameras, robot and
`calibration.json`. Until moved they aren't listed; `tandem init`, or `tandem profile migrate`, moves each:

- the cameras and robot to [the rig](#the-rig), if `rig.yml` doesn't exist yet (from the active profile);
  otherwise a difference is noted. Every profile's extrinsics go into the rig's `calibration.json` for the cameras
  it has none for, never over one it has;
- the task to `profiles/<name>.yml`, the trajectories to `trajectories/<name>/` (a symlink stays a symlink);
- the old directory, whole, to `profiles/.migrated/<name>/`, with a `migration.json` of what was done.

Nothing is deleted. If the rig can't be set up from them (a bad setting in the active profile, named with its
file), nothing moves. A profile that can't be moved is left as it was and the others still move; one moved
part way says what was done, and running it again finishes it. Until the old profiles are moved, `tandem rig
set` and `rig edit` wait for them. Version-1 profiles' `hitl.on_robot_phase_failure: teleop` (the old default)
is kept: set `abort` unless you chose teleop.
Older tandem can't read version 3.

## Phase planning (hitl)

[Phase planning](README.md#terms) is on in the paper's profiles and in every new one; a profile with no `hitl:`
block has it off. The other defaults are the paper's.

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
| `cache_path` | `null` | Proposal-only SQLite cache (key: model, prompt, image), relative to `profiles/`. Re-plans skip it. |
| `on_robot_phase_failure` | `abort` | A robot phase can't be planned. `abort`: fail at `tamp_planning`. `teleop`: a person does it, checked like a human phase. `replan`: propose again with the failure. An *execution* failure always ends at `tamp_execution`. |
| `conjoin_robot_phases` | `true` | Plan consecutive robot phases as one goal where sound ([conditions](ADDING_A_PLANNER.md#capabilities)). Off: one leg and perception pass each. |
| `human_executor` | `teleop` | Who does human phases, by registered name ([choosing one](ADDING_A_HUMAN_EXECUTOR.md#choosing-one)). |
| `human_executor_options` | `{}` | Each executor's settings, keyed by its name. Checked at load. |
| `allow_unrecorded_human_phase` | `false` | While recording, accept a human phase with no recording (`d`, done by hand, or an executor leg with no frames) instead of asking again. `--no-record` always accepts. |
| `verification_camera` | `external` | Check image: `external`, `hand` or `perception` (perception's camera). |

A check costs one model call per checkable atom, with the arm parked. Checkable atoms: invented predicates plus
the planner's `checkable_predicates` (TiPToP: `On`; [capabilities](ADDING_A_PLANNER.md#capabilities)). Trial
flow: [METHOD.md](METHOD.md#the-trial-loop).

## The rig

This machine's robot, cameras and calibration, shared by every profile: `~/.config/tandem/rig.yml`.

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

`tandem init` writes it: the robot and the cameras. A planner's own machine settings keep their defaults until you
set one (`tandem rig show` lists them all, with their defaults). Then:

```bash
tandem rig show                                   # also --json; the web's Settings page shows it too
tandem rig set robot.host 172.16.0.5              # one setting; `null` removes one
tandem rig set planners.tiptop.perception.m2t2.url http://gpu-box:8123
tandem rig edit                                   # $EDITOR; validated on save
tandem rig path --calibration                     # where the extrinsics are
```

| key | default | what it does |
|---|---|---|
| `robot.type` | `fr3_robotiq` | The arm, named as the planner names it. TiPToP: `fr3_robotiq` (FR3) or `panda_robotiq` (Panda), with a Robotiq 2F-85 via the bamboo-polymetis shim; `panda` (Franka Hand): the shim refuses its gripper commands; `ur5` (UR5, Robotiq gripper): needs tiptop's `ur5` extra (ur_rtde), not in the runtime. |
| `robot.host` | `172.16.0.2` | The robot computer (the NUC). A hostname or IP address; ports are the planner's. |
| `cameras.perception` | `external` | Camera perception reads. `external`: the arm stays at `q_home`. `hand`: it moves to `q_capture` first. |
| `cameras.hand`, `cameras.external` | | Wrist and third-person cameras. Every leg, robot or human, is recorded from them; both must open at warm-up or the session stops. |
| `cameras.external_2` | | Optional second third-person camera (DROID `exterior_2`). If listed, it must open or collection stops. |
| `<camera>.serial` | | ZED serial, quoted (`'14846828'`); `tandem rig set` keeps it text. One camera per serial. |
| `<camera>.type`, `.resolution` | `zed`, `HD720` | |
| `<camera>.fps` | `15` | Keep 15: three ZEDs at HD720@30 exceed USB bandwidth, and the export resamples to 15 Hz. |
| `calibration` | `calibration.json` | The extrinsics file, relative to `rig.yml`, or absolute. |
| `planners.<name>` | | Each planner's machine settings ([TiPToP's](#tiptop-options)); `tandem planners info NAME` lists them. |

`calibration.json` holds one pose per serial (TiPToP's `calibration_info.json` format). Wrist: `ee_from_cam`.
Fixed: `world_from_cam`. Rotations: `xyz` Euler, radians.

```json
{"<serial>": {"pose": [x, y, z, roll, pitch, yaw]}}
```

A configured serial with no entry stops the session before warm-up; `tandem doctor` reports it. The
[README](../README.md#5-cameras-and-calibration) has the steps to fill it in. Also:

- **TiPToP's own scripts** (`calibrate-wrist-cam`, `viz-calibration`, `cutamp-demo`, ...). `tandem runtime run`
  and `tandem runtime shell` give them a `tiptop.yml` written from the rig (`$TIPTOP_CONFIG`: the robot's
  address and type, the cameras, TiPToP's machine settings) and the rig's calibration file
  (`$TIPTOP_CALIBRATION`), so they reach your NUC and write your extrinsics. `--raw` runs them on tiptop's stock
  config instead, as they run on a machine with no rig.yml yet (and say so).
- **Teleop** reaches the NUC through DROID's own `droid/misc/parameters.py` (`nuc_ip`): keep it the same as
  `robot.host`.
- **Gripper mask** (`perception: hand` only). The runtime's `tiptop/tiptop/config/assets/gripper_mask.png`
  masks the fingers out of the wrist cloud. Make yours: `tandem runtime run compute-gripper-mask` (or
  `paint-gripper-mask`).

## Planner settings

A planner's settings are of two kinds: the task's, in each profile's `planner.options`, and this machine's, in
[the rig](#the-rig) under `planners.<name>`. The planner declares which is which
([how](ADDING_A_PLANNER.md#options-and-doctor-rows)): a server's address or a robot's ports are the machine's,
and a key put in the wrong file is refused with where it belongs. `tandem planners info NAME` lists both.

```yaml
planner:
  backend: tiptop     # which planner
  options:            # its task settings, checked by the planner at load
    tamp: ...
```

`tandem planners use NAME` sets `planner.backend`. It:

- stashes the old planner's `options` in `profiles/.planner-options/<profile>.<planner>.yml`, restored on
  switching back;
- takes `--option KEY=VALUE` (repeatable) for options the new planner requires;
- warns, but doesn't refuse, if its runtime isn't installed;
- repairs a profile naming a planner this machine lacks.

New profiles plan with the machine's default planner (`tandem planners default NAME`).

**Did my setting apply?** `tandem profile show NAME --planner` prints what the planner gets; a key missing there
didn't apply. `tandem planners info NAME` lists what a planner reads.

### TiPToP options

**Machine settings**, in `rig.yml` under `planners.tiptop` (`tandem rig set planners.tiptop.KEY VALUE`). The
robot's address and type are the rig's own `robot.host` and `robot.type`.

`robot`:

| key | default | what it does |
|---|---|---|
| `dof` | `7` | Joint count; must match `q_home` and `q_capture`. |
| `port`, `gripper_port`, `state_port` | `5555`, `5559`, `5557` | Shim control, gripper and state (encoder) ports. |
| `time_dilation_factor` | `0.2` | Arm speed, in (0, 1]; `0.2` is 20%. |
| `q_home`, `q_capture` | TiPToP's | Home pose; capture pose for `cameras.perception: hand`. |

`perception`:

| key | default | what it does |
|---|---|---|
| `m2t2.url`, `m2t2.apply_bounds` | `http://localhost:8123`, `true` | M2T2 grasp server; a flag sent with each request. `tandem doctor` probes it. |
| `foundation_stereo.url` | `http://localhost:1234` | FoundationStereo depth server, for the ZEDs' depth every rollout. `tandem doctor` probes it. |
| `sam_mode` | `local` | `local`: SAM-2 in the runtime. `remote`: the server at `sam_url` (then required). |
| `depth_smoothing_frames` | `5` | Depth frames median-fused at capture. `1` disables it. |
| `robot_mask_margin_m` | `0.02` | Arm collision-sphere padding when cutting the arm from a third-person cloud. Raise it if arm remains. |
| `depth_trunc_m`, `voxel_downsample_size`, `contact_threshold_m`, `mask_erosion_pixels` | `5.0`, `0.0075`, `0.01`, `3` | tiptop's settings of those names. |
| `gemini.model`, `gemini.temperature` | `gemini-robotics-er-2-preview`, `null` | Recorded only: the pinned tiptop always runs this model at its own temperature. `tandem doctor` warns on a mismatch. |

**Task settings**, in the profile's `planner.options.tamp`: cuTAMP and cuRobo overrides with tiptop's key
names, so a `cfg/tamp/*.yml` config's `tamp_overrides` paste in as is. Accepted keys:
`src/tandem/planners/tiptop/tamp_keys.py`. `tamp` beats the rig's `perception` for `contact_threshold_m` and
`voxel_downsample_size`. `tandem doctor` warns about a key that needs another (`blend_ops` without
`blend_trajectory`). A relative checkpoint path (`vae_path`, `blend_model_path`, `blend_stats_path`,
`posture_ref`) is looked up beside the profile's file, then in the runtime, where the install puts the DATAFARM
checkpoints (`vae/checkpoints/vae_full_v2.pt`, `rnd/checkpoints/rnd_droid.pt`).

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

The paper's: `solve-constrained-puzzle` sets `placement_support`, `placement_support_required` and
`placement_into_surface`; `store-bread-in-closed-box` sets all seven (margin `0.005`, flatness `0.012`,
`placement_fill_occluded: true`).

### The three switches

`tamp:` keys, off unless set. On in [the paper's five](#the-papers-five) and every new profile: every run in the
paper had them, because LJ1356's tiptop always did them.

| key | what it switches on |
|---|---|
| `table_plane_support_vote` | Pick the table among RANSAC's planes by the objects resting on each, not every object within 3 cm above or below. |
| `disjoint_object_masks` | Meshes and clouds from disjoint masks (a pixel two masks claim goes to the smaller object), so a container's hull stops at what rests on it. Placement fitting always uses them. |
| `blend_stretch_to_caps` | With `blend_trajectory` on, slow a stroke that can't be re-timed within the velocity and acceleration caps until it fits, instead of keeping the plan's timing. A stroke can get many times slower. |

## tandem settings and credentials

tandem's own settings, in `~/.config/tandem/config.toml` beside [the rig](#the-rig). Edit with `tandem config`
(`list`, `get`, `set`, `edit`, `path`):

| key | default | what it does |
|---|---|---|
| `active_profile` | none | The profile commands act on (`tandem profile use`); `tandem init` sets `cover-bread-rolls`, and `tandem profile create` makes its profile active when none is. |
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
| config: `config.toml`, `credentials.toml`, `rig.yml`, `calibration.json` | `~/.config/tandem` | `$TANDEM_CONFIG_DIR` |
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

The install applies one patch (so `$TIPTOP_CONFIG` and `$TIPTOP_CALIBRATION` can point TiPToP at the rig's
config and calibration file) and places the DATAFARM checkpoints tandem ships. Fetching:
[ADDING_A_PLANNER.md](ADDING_A_PLANNER.md#a-runtime-recipe). Build log: [DATA.md](DATA.md#logs-and-session-files).

- **Updating.** A moved pin shows `outdated` in `tandem planners list`; `tandem planners install tiptop`
  replaces only moved sources, reusing the environment. `--force` refetches and rebuilds all.
- **ZED.** With the ZED SDK installed (`/usr/local/zed/get_python_api.py`), the install adds its Python API
  (`pyzed`) to the runtime. Without it the install still succeeds, and it, `tandem planners info tiptop` and
  `tandem doctor` say ZED cameras won't open: install the SDK, then `tandem planners install tiptop` again.

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
