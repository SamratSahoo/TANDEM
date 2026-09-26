# Using TANDEM

## A typical session

```bash
tandem profile use store-bread-in-closed-box                  # pick a task
tandem plan "place the bread inside the box" -i workspace.png # preview its phases from a photo
tandem collect                                                # collect trials in the terminal
tandem traj list --status success                             # review what was filed
tandem export lerobot --repo <hf-user>/bread-box --push       # build and upload a dataset
```

Each step has its own section below, and every command is listed at the end in [Commands](#commands).

## Collecting

```bash
tandem collect                            # the active profile, in the terminal
tandem collect my-task -n 20              # another profile; stop after 20 trials
tandem collect --task "stack the cups"    # a different task for this session
tandem collect --no-execute               # plan only; the arm never moves
tandem collect --web                      # drive the session from the browser
```

The session starts the planner once, then repeats: task prompt, trial, label prompt. The footer shows which
keys work at each step.

| state | keys |
|---|---|
| task prompt | `↵` repeat the task, `n` type a new one |
| planning or executing | `p` preempt, `t` take the arm (if teleop is ready) |
| you have the arm | `r` return control |
| human phase | `t` take the arm or run the executor (if ready), `d` I did it (if allowed), `a` give up (aborts) |
| label prompt | `s` or `y` success, `f` or `n` failure |
| any | `q` or Ctrl-C finish |

> **Preempt does not stop motion.** `p` stops further plan steps, but the motion segment already sent still
> finishes ([TiPToP](https://github.com/SamratSahoo/tiptop/tree/TANDEM) has no cooperative stop). Only the E-stop stops the arm at once.

**Preempt** (`p`) files the attempt's legs as aborted. The session stays warm for the next trial.

**Hand-off** (`t` outside a human phase) gives you the arm through teleop at the next plan-step boundary. When
you press `r`, TANDEM perceives again and replans the phase from where you left the arm, without homing. The
legs merge into one episode.

**Human phase.** The screen says what to do and what a fresh camera image will then check.

- If the check fails, the screen lists what is missing and you get `hitl.verify_retries` more tries.
- `d` is refused while recording, unless
  [`hitl.allow_unrecorded_human_phase`](CONFIGURATION.md#phase-planning-hitl) is true.
- If you don't return control within an hour, TANDEM ends the leg and takes the arm back.

**Label.** You are asked for one only if the plan ran to the end, or if a check failed under
`hitl.on_verification_failure: label`. [Settled](README.md#terms) trials are filed without a label, with the
reason. `--episodes N` counts labeled trials and part-way failures, but not excluded or aborted ones.

**Finish** (`q`) stops at the next step boundary and parks the arm without opening the gripper.

- A trial still running is filed as aborted.
- A trial waiting at the label prompt stays unmerged in `eval/`. The session prints how to file it:
  `tandem traj merge <trajectory id> --status success`, or `tandem traj relabel` for a one-leg trial.

## Planning from a photo

```bash
tandem plan "place the bread inside the box" --image workspace.png -o bread -o box -o plate
```

This previews a task's phases before you collect. It prints who does each phase, each phase's goal or magic
operator, and the invented predicates. It needs only TANDEM (Python 3.10+) and a
[Gemini key](CONFIGURATION.md#tandem-settings-and-credentials): no runtime, GPU or robot. Answers vary between
runs.

**It tells you if part of the instruction can't be planned.** Usually an object wasn't detected; put it on
the table or reword the task.

| flag | what it does |
|---|---|
| `-i`, `--image PHOTO` | The workspace photo. Required. |
| `-o`, `--object LABEL` | Pin an object label. Repeatable. Without it, a vision model names the objects. A session's labels reproduce its plan. |
| `-p`, `--profile P` | Take the planning settings and planner from this profile. |
| `-b`, `--planner NAME` | Plan in this planner's goal language. The default is the profile's planner, else the machine's. `--backend` is an older alias. |
| `--table NAME` | What the planner calls the table. The default is `table`. |
| `--json` | Print the record that `hitl.json` is written from. |
| `--save-vlm-io DIR` | Keep every image sent to the model, and its reply, in `DIR`. |

### From Python

```python
import tandem

plan = tandem.plan_task("place the bread inside the box", "workspace.png", objects=["bread", "box", "plate"])
for phase in plan.phases:
    print(phase.executor, phase.description, phase.atoms)
```

- `plan_task` returns a `PhasePlan` with `.phases`, `.spec` and `.to_json()`. Each phase has `.executor`,
  `.description` and `.atoms`.
- The image can be a path, a PIL image or an RGB `uint8` array.
- The keywords match the flags (`planner`, `profile`, `table`, `save_vlm_io`), plus `config` (a
  `PlanningConfig`).
- Inside a running event loop, use `await tandem.plan_task_async(...)`.

`import tandem` also exports `Planner`, `SidecarPlanner`, `Capabilities`, `PlannerInfo`, `Predicate`,
`Parameter`, `RuntimeRecipe`, `register_backend` (alias `register_planner`), `register_human_executor` and
`TandemError`.

## Reviewing and exporting

```bash
tandem traj list -s success -n 0        # every success, newest first
tandem traj open <id>                   # replay one in its planner's viewer (TiPToP: Rerun)
tandem traj relabel <id> failure        # move it to success, failure or eval
```

`<id>` is a trajectory's directory name (a timestamp) or a unique prefix of it. Every `traj` command except
`list` takes `-p/--profile`.

| command | what it does |
|---|---|
| `traj list [profile]` | Newest first. `-s/--status eval\|success\|failure`; `-n N` shows N (default 30, `0` for all). |
| `traj show <id>` | One trajectory in detail. |
| `traj open <id>` | Replay in its planner's viewer. |
| `traj relabel <id> <status>` | Move it to `success`, `failure` or `eval`. |
| `traj merge [<trajectory id>]` | Re-join a trial's legs after a failed merge. `--status STATUS` files the result. |
| `traj copy <id> <profile>` | Copy it into another profile. |
| `traj rm <id>` | Delete it. `-y` skips the confirmation. |
| `traj path <id>` | Print its directory. |

**Relabeling a settled trial.** `relabel ... success` refuses a trial TANDEM settled. `--force` overrules it
(the web UI asks you to confirm), and `hitl.json` records it as `overruled`. This is the only way to export an
excluded trial.

**Merging** takes the trial's `trajectory_id` from `_meta.json`, not the timestamp. With no id, it merges every
trial that has unmerged legs.

### Exporting

```bash
tandem export lerobot --repo <owner>/<name>          # writes ~/tandem-data/exports/<owner>/<name>
tandem export lerobot --repo <owner>/<name> --push   # and uploads it
tandem export manifest --out index.json              # a JSON index of the trajectories (default: stdout)
```

`export lerobot` writes a [LeRobot](https://github.com/huggingface/lerobot) v3.0 dataset in [`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1)'s schema, for [π₀.₅-DROID](https://github.com/Physical-Intelligence/openpi) fine-tuning.
Each run is logged to `export.log` ([logs](DATA.md#logs-and-session-files)).

**Only `success/` is exported.** These are skipped, each with its reason:

- settled trials, unless a forced relabel overruled them;
- episodes whose `cmd_gripper` isn't binary;
- episodes missing an exterior or wrist video;
- episodes whose state arrays don't fit [DROID](https://droid-dataset.github.io/)'s schema.

What goes into each episode:

- **Action:** `cmd_joint_velocity`, clipped to [-1, 1] and never rescaled, plus `cmd_gripper`.
  `action_joint_velocity` (in `robot_state.npz`) isn't exported.
- **Task:** the episode's `instruction` from `_meta.json`, else the profile's `task.prompt`.
- **Cameras:** without `external_2`, the first exterior video fills both exterior slots.

| flag | what it does |
|---|---|
| `--repo OWNER/NAME` | The dataset. The default is the profile's `export.hf_repo`. `hf_org` fills in a missing owner. |
| `--out DIR` | Write to `DIR/<owner>/<name>` instead. |
| `-n`, `--max-episodes N` | Export only the first N. |
| `--push` | Upload after building. Needs a Hugging Face token ([where TANDEM looks](CONFIGURATION.md#tandem-settings-and-credentials)). |
| `--private`, `--public` | Visibility when pushing. The default is the profile's `export.private`. |
| `--force` | Replace whatever is at the destination. Without it, a rebuild replaces only a dataset TANDEM built, and only once the new one is complete. |

## The web UI

```bash
tandem ui                    # http://127.0.0.1:8787, or the next free port
tandem ui --port 9000 --no-open
```

The UI needs no Node, CDN, GPU or robot. `ui.host`, `ui.port` and `ui.open_browser` set its defaults. Ctrl-C
lets running sessions park the arm and finish merging. A second Ctrl-C quits at once and leaves the arm where
it is.

- **Trajectories:** click a plot to seek every video. A merged trajectory's ribbon shows who drove each
  stretch. Shaded bands mark the frames π₀.₅-DROID's training drops as idle.
- **Collect:** the same controls as the terminal. Anything the plan leaves out is shown before the arm moves,
  and you can review the trial right at the label prompt.
- **Profiles:** the paper's five and your own. You can create one from a task or as a copy, edit it, and see
  what its planner receives.
- **Settings:** the rig (robot, cameras, calibration, each planner's machine settings), credentials, paths,
  catalogs, runtime status and `tandem doctor`.

Flags: `-p/--port`, `--host`, `--no-open`, and `--profile` (open on a profile and make it active).

### HTTP API

```bash
curl http://127.0.0.1:8787/api/rig
curl -X PATCH http://127.0.0.1:8787/api/rig -H 'Content-Type: application/json' \
     -d '{"robot.host": "NUC_ADDRESS", "cameras.external_2": null}'
```

| route | returns or does |
|---|---|
| `GET /api/planners[/{name}]`, `GET /api/executors` | Same as `planners list\|info --json` and `executors list --json`. Add `?profile=P` for another profile. |
| `POST /api/planners/{name}/use\|default`, `POST /api/executors/{name}/use` | Same as `planners use` (optional body: `profile`, `options`), `planners default` and `executors use` (optional body: `profile`). |
| `GET /api/profiles/{name}` | The profile, plus `planner_view.receives` (as `profile show --planner`). Each card in `GET /api/profiles` has a `planner_summary` line. |
| `POST /api/profiles` | Same as `profile create`. Body `{name, prompt}` uses the paper's settings; `{name, from}` copies a profile (`prompt` optional). `POST /api/profiles/builtin` adds the paper's five, as `init` does. |
| `GET /api/rig`, `PATCH /api/rig` | Same as `rig show --json`, and `rig set` for several keys at once. |
| `GET /api/sessions/{id}` | The [session summary](DATA.md#session-summary) and its last 500 log lines. |
| `GET /api/media/{profile}/{id}/{file}` | A trajectory video, served with HTTP Range. A proxy in front must pass `Range` headers. |

The API can't install a planner. Each catalog row includes the `install_command` to run instead.

## Commands

`tandem <command> --help` lists every flag. `[profile]` defaults to the active profile. Global options go
first: `tandem --debug collect` shows full tracebacks, and `--no-color` (or `NO_COLOR`) turns off color.

`--json` works with `doctor`, `plan`, `profile list|show`, `rig show`, `traj list|show`,
`planners list|info|use|default`, `executors list|use`, `runtime status` and `config list`.

### Setup

| command | what it does |
|---|---|
| `tandem init` | Checks, data directory, planner runtime and its [servers](#servers), [Gemini key](https://aistudio.google.com/apikey), [the rig](CONFIGURATION.md#the-rig), [the paper's five](CONFIGURATION.md#the-papers-five) and teleop. Moves [older profiles](CONFIGURATION.md#older-profiles). Skips steps already done. |
| `tandem doctor` | Runs every check and says how to fix what fails. Changes nothing. |

| flag | what it does |
|---|---|
| `init --viz-only` | Set up a laptop for reviewing only: no runtime or robot. |
| `init -y`, `--yes` | Accept every prompt. |
| `init --repair` | Redo the runtime, key, rig and teleop. Never touches a profile. |
| `init --robot-host`, `--robot-type`, `--camera ROLE=SERIAL` | The robot and cameras. `--camera` is repeatable. |
| `init --planner NAME`, `--profile NAME` | The planner, and the profile to make active. |
| `doctor -p`, `--profile` | Check another profile. |
| `doctor --no-hardware` | Skip the robot, camera and perception-server probes. |

Sessions: [`collect`](#collecting), [`plan`](#planning-from-a-photo) and [`ui`](#the-web-ui) have their flags in
their sections. So do [`traj`](#reviewing-and-exporting) and [`export`](#exporting).

### Profiles

`tandem profile list|show|use|edit|path|delete|create|migrate` manages [profiles](CONFIGURATION.md#profiles),
one YAML file each.

| flag | what it does |
|---|---|
| `create NAME --prompt "..."` | A new profile: the paper's settings with your task. |
| `create NAME --from PROFILE` | A copy of another profile. |
| `create --use`, `--force` | Make it active; replace a profile of that name (its trajectories are kept). |
| `show --planner` | Show only what the planner receives. |
| `delete --purge`, `-y` | Also delete its trajectories; skip the confirmation. |

`edit` opens the file in `$EDITOR` and validates it on save. `migrate` moves
[older profiles](CONFIGURATION.md#older-profiles).

### Rig

`tandem rig show|set|edit|path` manages [the robot, camera and calibration settings](CONFIGURATION.md#the-rig).
`set KEY VALUE` takes one dotted key, such as `robot.host`, and `null` removes it. `path --calibration` prints
where the extrinsics file is.

### Planners and executors

| command | what it does and its flags |
|---|---|
| `planners list`, `info NAME` | Every planner and its state; one planner's needs, pinned vs installed commits, goal language and settings. `-p/--profile`. |
| `planners install NAME` | Build or update its [runtime](CONFIGURATION.md#the-planner-runtime). `--sources DIR` ([offline](CONFIGURATION.md#offline-install)), `--force` (refetch and rebuild), `-y/--yes`. |
| `planners remove NAME` | Delete its runtime only. |
| `planners use NAME` | Make a profile plan with it ([settings](CONFIGURATION.md#planner-settings)). `-p/--profile`, `-o/--option KEY=VALUE` (repeatable), `--default` (new profiles too). |
| `planners default NAME` | Set only the planner new profiles get. |
| `planners bundle NAME --out DIR` | Save its pinned sources for an [offline install](CONFIGURATION.md#offline-install). `--only SOURCE` and `--from SOURCE=PATH` (a local checkout), both repeatable; `--archive` (also `DIR.tar.gz`). |
| `planners new NAME` | Scaffold [your own planner](ADDING_A_PLANNER.md#quick-start). `--sidecar`, `--dir DIR`. |
| `executors list`, `use NAME` | Human executors and readiness; set a profile's ([choosing one](ADDING_A_HUMAN_EXECUTOR.md#choosing-one)). `-p/--profile`. |
| `executors install teleop`, `remove teleop` | Build (or delete) the teleop driver's runtime; install also turns teleop on. `install --force` (refetch, rebuild), `--sources DIR`, `-y/--yes`; `remove -y`. |

Planner states are `installed`, `not installed`, `outdated` (built at commits other than the pinned ones),
`no runtime needed` and `broken`. Executor states are `ready`, `needs setup` and `broken`. Each says why when
it isn't ready, and `●` marks the profile's choice.

### Settings and runtime

`tandem config list|get|set|edit|path|set-gemini-key|set-hf-token` manages
[TANDEM's settings](CONFIGURATION.md#tandem-settings-and-credentials). `set KEY VALUE` takes one dotted key,
such as `ui.port`. The `set-*` commands store a credential typed at a prompt, or read it with `--stdin`.
`set-gemini-key --key KEY` also works, but leaves the key in your shell history.

`tandem runtime status|build|shell|python|run|clean|path` works on the active profile's planner runtime.
`--planner NAME` or `-p/--profile` picks another.

```bash
tandem runtime run viz-calibration --camera external   # a planner script, run with your rig settings
```

- `run` and `shell` run the planner's own scripts with your [rig](CONFIGURATION.md#the-rig) settings. `--raw`
  uses the planner's stock config instead.
- TANDEM's options (`--planner`, `-p`, `--raw`) go before the script's name, and the script's own after it.
  `--` still works.
- `build` builds or repairs the runtime (`--force` refetches, plus `--env-only` and `--sources DIR`).
  `python` prints the interpreter, and `clean` deletes the runtime (`-y` skips the confirmation).

### Servers

`tandem servers install|status|start|stop` manages the helper servers the planner calls: TiPToP's [M2T2](https://github.com/SamratSahoo/M2T2/tree/TANDEM) (grasps)
and [FoundationStereo](https://github.com/SamratSahoo/FoundationStereo/tree/TANDEM) (depth), at the URLs in [the rig](CONFIGURATION.md#tiptop-options). `tandem init` builds
them, and `tandem collect` starts any that are down and stops the ones it started when it ends.

| command | what it does |
|---|---|
| `servers install` | Build their runtimes. `--force` (refetch, rebuild), `--sources DIR`, `-y/--yes`. |
| `servers status` | Whether each is installed and answering, and whether TANDEM started it. `--json`. |
| `servers start [NAME]` | Start those that are down, wait for each to load, and leave them running. |
| `servers stop [NAME]` | Stop the ones TANDEM started, such as after a crashed session. |

`NAME` is `m2t2` or `foundation_stereo`. TANDEM never starts a server whose URL points at another machine.
Each server's log is `server-<name>.log` ([logs](DATA.md#logs-and-session-files)).
