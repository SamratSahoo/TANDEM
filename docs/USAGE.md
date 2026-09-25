# Using tandem

## Commands

`tandem <command> --help` lists every flag. `[profile]` defaults to the active profile.

| command | does | notable flags |
|---|---|---|
| `tandem init` | Setup: checks, data directory, planner runtime, Gemini key, [the rig](CONFIGURATION.md#the-rig), [the paper's five](CONFIGURATION.md#the-papers-five), teleop. Moves [older profiles](CONFIGURATION.md#older-profiles). Skips done steps. | `--viz-only` (laptop: no runtime or robot), `-y/--yes`, `--repair` (redo runtime, key, rig, teleop; never touches a profile), `--robot-host`, `--robot-type`, `--camera ROLE=SERIAL` (repeatable), `--planner`, `--profile` (make it active) |
| `tandem doctor` | Every check, with fixes. Changes nothing. | `-p/--profile`, `--no-hardware` (skip robot, camera, perception-server probes) |
| `tandem collect [profile]` | A [session](#collecting) in the terminal. | `-t/--task`, `-n/--episodes`, `--no-execute` (plan, never move), `--no-record`, `--web` (in the browser) |
| `tandem plan "<task>" -i PHOTO` | [Phases from a photo](#planning-from-a-photo). | |
| `tandem ui` | The [web UI](#the-web-ui). | `-p/--port`, `--host`, `--no-open`, `--profile` (also makes it active) |
| `tandem profile list\|show\|use\|edit\|path\|delete\|create\|migrate` | Manage [profiles](CONFIGURATION.md#profiles), one YAML file each. `create NAME --prompt "..."`: the paper's settings with your task. `edit` validates on save; `migrate` moves [older ones](CONFIGURATION.md#older-profiles). | `create --prompt --from PROFILE --use --force`; `show --planner` (only what the planner receives); `delete --purge` (and its trajectories), `-y` |
| `tandem rig show\|set\|edit\|path` | [This machine's robot, cameras and calibration](CONFIGURATION.md#the-rig), shared by every profile. | `set KEY VALUE` (dotted, e.g. `robot.host`; `null` removes one); `path --calibration` |
| `tandem planners list\|info NAME` | Every planner and its state; one's needs, pinned vs installed commits, goal language, task and machine settings. | `-p/--profile` |
| `tandem planners install\|remove NAME` | Builds or updates its [runtime](CONFIGURATION.md#the-planner-runtime) from pinned sources. `remove` deletes only the runtime. | `--sources` ([offline](CONFIGURATION.md#offline-install)), `--force` (refetch, rebuild), `-y/--yes` |
| `tandem planners use\|default NAME` | Makes a profile plan with it ([settings](CONFIGURATION.md#planner-settings)); `default` sets only what new profiles get. | `use -p/--profile -o/--option KEY=VALUE --default` (new profiles too) |
| `tandem planners bundle NAME --out DIR` | Its pinned sources, for an [offline install](CONFIGURATION.md#offline-install). | `--only SOURCE`, `--from SOURCE=PATH` (local checkout), repeatable; `--archive` (also `DIR.tar.gz`) |
| `tandem planners new NAME` | Scaffolds [your own planner](ADDING_A_PLANNER.md#quick-start). | `--sidecar`, `--dir` |
| `tandem executors list\|use NAME` | Human executors and readiness; `use` sets a profile's ([choosing one](ADDING_A_HUMAN_EXECUTOR.md#choosing-one)). | `-p/--profile` |
| `tandem traj …`, `tandem export lerobot\|manifest [profile]` | [Review and file trajectories](#reviewing-and-exporting); [export](#exporting) a LeRobot dataset or JSON index. | |
| `tandem config list\|get\|set\|edit\|path\|set-gemini-key\|set-hf-token` | [tandem's settings](CONFIGURATION.md#tandem-settings-and-credentials) (config.toml). `set-*` store a credential typed at a prompt. | `set KEY VALUE` (dotted, e.g. `ui.port`); `--stdin`; `set-gemini-key --key KEY` (lands in shell history) |
| `tandem runtime status\|build\|shell\|python\|run\|clean\|path` | The active profile's planner runtime. `run` and `shell` point the planner's own scripts at [the rig](CONFIGURATION.md#the-rig); `build` also repairs; `python` prints the interpreter; `clean` deletes it. | `--planner` or `-p/--profile` for another; `run`/`shell --raw` (the planner's stock config); `build --force` (refetch), `--env-only`, `--sources`; `clean -y` |

- `--json`: `doctor`, `plan`, `profile list|show`, `rig show`, `traj list|show`, `planners list|info|use|default`,
  `executors list|use`, `runtime status`, `config list`.
- `--debug` (full tracebacks) and `--no-color` (or `NO_COLOR`) go first: `tandem --debug collect`.
- `tandem runtime run`: tandem's own options (`--planner`, `-p`, `--raw`) go before the command; everything
  after it is the command's: `tandem runtime run viz-calibration --camera external` (`--` still works).
- Planner states: `installed`, `not installed`, `outdated` (built at commits other than the pinned ones),
  `no runtime needed`, `broken` (with why). Executor states: `ready`, `needs setup` (with what is missing),
  `broken`. `●` marks the profile's choice.

## Collecting

`tandem collect` warms the planner once, then loops: task prompt, trial, label prompt.

| state | keys |
|---|---|
| task prompt | `↵` repeat task, `n` new task |
| planning or executing | `p` preempt, `t` take the arm (if teleop is ready) |
| you have the arm | `r` return control |
| human phase | `t` take the arm or run the executor (if ready), `d` I did it (if allowed), `a` give up (aborts) |
| label prompt | `s`/`y` success, `f`/`n` failure |
| any | `q` or Ctrl-C finish |

- **Finish** stops at the next step boundary and parks the arm without opening the gripper. A trial in flight is
  filed as aborted. One at the label prompt stays unmerged in `eval/`; the session prints how to file it:
  `tandem traj merge <trajectory id> --status success`, or `traj relabel` for one leg.
- **Preempt** files the attempt's legs as aborted; the session stays warm. **It does not stop motion:** the
  segment finishes unless the planner supports a cooperative stop (TiPToP doesn't). Only the E-stop stops at once.
- **Hand-off** (`t` outside a human phase) gives you the arm via teleop at the next plan-step boundary. After
  `r`, tandem re-perceives and replans the phase from where you left the arm, without homing; the legs merge
  into one episode.
- **Human phase**: the screen shows what to do and what a fresh camera image will then check. A failed check
  lists what is missing and allows `hitl.verify_retries` more tries. `d` is refused while recording unless
  [`hitl.allow_unrecorded_human_phase`](CONFIGURATION.md#phase-planning-hitl) is true.
- **Label** is asked only if the plan ran to the end, or a check failed under `hitl.on_verification_failure: label`.
  [Settled](README.md#terms) trials are filed without one, with the reason
  ([how a trial ends](METHOD.md#how-a-trial-ends)).
- **`--episodes N`** counts labeled trials and part-way failures, not excluded or aborted ones.
- Control not returned within an hour: tandem ends the leg and takes the arm back.

## Planning from a photo

`tandem plan` previews a task's phases before you collect: who does each, its goal or magic operator, and the
invented predicates. It needs only tandem (Python 3.10+) and a
[Gemini key](CONFIGURATION.md#tandem-settings-and-credentials), no runtime, GPU or robot. Answers vary by run.

```bash
tandem plan "place the bread inside the box" --image workspace.png -o bread -o box -o plate
```

**It flags any clause the plan can't express**, usually an undetected object: put it on the table or reword
the task.

- `-o/--object LABEL` (repeatable) pins object labels; a session's labels reproduce its plan. Otherwise a vision
  model names them.
- `-p/--profile P` takes its planning settings and planner. `-b/--planner NAME` (alias `--backend`) sets the goal
  language (default: the profile's planner, else the machine's).
- `--table NAME` names the table (default `table`). `--json` prints the record `hitl.json` is written from.
  `--save-vlm-io DIR` keeps each model image and reply.

### From Python

```python
import tandem
plan = tandem.plan_task("place the bread inside the box", "workspace.png", objects=["bread", "box", "plate"])
```

- `plan_task` returns a `PhasePlan`: `.phases` (each with `.executor`, `.description`, `.atoms`), `.spec`,
  `.to_json()`.
- `image`: a path, PIL image or RGB `uint8` array. Keywords match the flags (`planner`, `profile`, `table`,
  `save_vlm_io`) plus `config` (a `PlanningConfig`).
- In a running event loop: `await tandem.plan_task_async(...)`.
- `import tandem` also exports `Planner`, `SidecarPlanner`, `Capabilities`, `PlannerInfo`, `Predicate`, `Parameter`,
  `RuntimeRecipe`, `register_backend` (alias `register_planner`), `register_human_executor`, `TandemError`.

## The web UI

`tandem ui` serves `http://127.0.0.1:8787` or the next free port (`ui.host`, `ui.port`, `ui.open_browser`). It
needs no Node, CDN, GPU or robot. Ctrl-C lets sessions park the arm and finish merging; a second Ctrl-C quits at
once, leaving the arm where it is.

- **Trajectories**: click a plot to seek every video. A merged trajectory's ribbon shows who drove each stretch.
  Shaded bands mark frames π₀.₅-DROID's training drops as idle.
- **Collect**: the terminal's controls. Clauses the plan leaves out show before the arm moves; the label prompt
  has an inline review.
- **Profiles**: the paper's five and yours. Create one from a task or as a copy, edit one and see what its
  planner receives. **Settings**: the rig (robot, cameras, calibration, each planner's machine settings),
  credentials, paths, catalogs, runtime status, `tandem doctor`.

### HTTP API

| route | returns or does |
|---|---|
| `GET /api/planners[/{name}]`, `GET /api/executors` | `planners list\|info --json`, `executors list --json`. `?profile=P` for another profile. |
| `POST /api/planners/{name}/use\|default`, `POST /api/executors/{name}/use` | `planners use` (optional body: `profile`, `options`), `planners default`, `executors use` (optional body: `profile`). |
| `GET /api/profiles/{name}` | The profile and `planner_view.receives` (`profile show --planner`). Cards in `GET /api/profiles` have a `planner_summary` line. |
| `POST /api/profiles` | `profile create`: body `{name, prompt}` (the paper's settings) or `{name, from}` (a copy; `prompt` optional). `POST /api/profiles/builtin` adds the paper's five, as `init` does. |
| `GET /api/rig`, `PATCH /api/rig` | `rig show --json`; `rig set` for several keys at once (body `{"robot.host": "172.16.0.5", "cameras.external_2": null}`). |
| `GET /api/sessions/{id}` | The [session summary](DATA.md#session-summary) and its last 500 log lines. |
| `GET /api/media/{profile}/{id}/{file}` | A trajectory video, served with HTTP Range; a proxy must pass `Range` headers. |

No endpoint installs a planner; catalog rows carry the `install_command` to run.

## Reviewing and exporting

```bash
tandem traj list [profile]           # newest first; -s/--status eval|success|failure, -n N (default 30, 0 = all)
tandem traj show <id>
tandem traj open <id>                # replay in its planner's viewer (TiPToP: Rerun)
tandem traj relabel <id> <status>    # success, failure or eval
tandem traj merge [<trajectory id>]  # re-join a trial's legs after a failed merge; --status STATUS
tandem traj copy <id> <profile>      # also: traj rm <id> [-y], traj path <id>
```

`<id>`: a trajectory's directory name (a timestamp) or unique prefix. All but `list` take `-p/--profile`.

- **Relabel** to `success` refuses a settled trial. `--force` (a confirm in the web UI) overrules it, recorded as
  `overruled` in `hitl.json`; only this exports an excluded trial.
- **Merge** takes the trial's `trajectory_id` (in `_meta.json`), not the timestamp. With no id, it merges every
  trial with unmerged legs.

### Exporting

```bash
tandem export lerobot --repo <owner>/<name>          # writes ~/tandem-data/exports/<owner>/<name>
tandem export lerobot --repo <owner>/<name> --push   # and uploads it; --private or --public
tandem export manifest [--out FILE]                  # JSON index (default: stdout)
```

`export lerobot` writes a LeRobot v3.0 dataset in `lerobot/droid_1.0.1`'s schema, for π₀.₅-DROID fine-tuning. It
needs av, pyarrow and huggingface_hub ([installing](CONFIGURATION.md#installing)), and logs each run to `export.log`
([logs](DATA.md#logs-and-session-files)).

- **Only `success/` is exported.** Skipped, each with its reason: settled trials no forced relabel overruled,
  and episodes with a non-binary `cmd_gripper`, a missing exterior or wrist video, or state arrays that don't
  fit DROID's schema.
- **Action**: `cmd_joint_velocity` clipped to [-1, 1], never rescaled, plus `cmd_gripper`. `action_joint_velocity`
  (in `robot_state.npz`) isn't exported.
- **Task**: the episode's `_meta.json` `instruction`, else the profile's `task.prompt`. Without `external_2`, the
  first exterior video fills both exterior slots.
- **Destination**: `--repo` defaults to `export.hf_repo`; `hf_org` fills a missing owner.
  `--out DIR` writes `DIR/<owner>/<name>`. `-n/--max-episodes N` exports the first N.
- **Rebuilds** replace only a dataset tandem built, once the new one is complete. `--force` replaces anything.
- **`--push`** needs a Hugging Face token ([where tandem looks](CONFIGURATION.md#tandem-settings-and-credentials)).
  Visibility defaults to `export.private`.
