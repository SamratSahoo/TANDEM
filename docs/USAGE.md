# Using tandem

How to run tandem: every command and its main flags, a collection session, `tandem plan`, the web UI, and
reviewing and exporting data. Settings are in [CONFIGURATION.md](CONFIGURATION.md), files on disk in
[DATA.md](DATA.md), fixes in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Commands

`tandem <command> --help` lists every flag. A `[profile]` argument defaults to the active profile.

| command | what it does | notable flags |
|---|---|---|
| `tandem init` | Sets up this machine: checks, data directory, the planner and its runtime, the Gemini key, a profile, teleop. Skips finished steps, so it is safe to re-run. | `--viz-only` (a laptop: no runtime, no robot), `-y/--yes`, `--repair` (redo the runtime, key and teleop steps; never touches an existing profile), `--profile NAME` (default `default`), `--import-from DIR`, `--planner NAME`, `--preset NAME` |
| `tandem doctor` | Runs every check and says what to do about each problem. Changes nothing. | `-p/--profile`, `--no-hardware` (skip the robot, camera and grasp-server probes) |
| `tandem collect [profile]` | Runs a session in the terminal ([Collecting](#collecting)). | `-t/--task TEXT`, `-n/--episodes N`, `--no-execute` (perceive and plan, never move; for tuning TAMP settings), `--no-record`, `--web` (drive it from the browser) |
| `tandem plan "<task>" -i PHOTO` | Splits a task into phases from a photo ([below](#planning-from-a-photo)). | `-o`, `-b`, `-p`, `--table`, `--json`, `--save-vlm-io` |
| `tandem ui` | Serves the web UI ([below](#the-web-ui)). | `-p/--port`, `--host`, `--no-open`, `--profile NAME` (also makes it the active profile) |
| `tandem profile list\|show\|use\|edit\|path\|delete` | Lists, prints, activates, edits (validated on save) and deletes profiles ([profiles](CONFIGURATION.md#profiles)). | `show --planner` (only what the planner receives); `delete --purge` (also delete its trajectories, kept otherwise), `-y` |
| `tandem profile create\|presets\|migrate` | Creates a profile, lists presets, rewrites [older profiles](CONFIGURATION.md#older-profiles) (all by default). | `create NAME --from P --preset NAME --prompt TEXT --use --force`, `--import-from DIR --tamp-config FILE --planner NAME`; `presets --planner NAME` |
| `tandem planners list` | Every planner tandem can see, built in or from an installed package, with its [state](#planner-and-executor-states). | `-p/--profile` |
| `tandem planners info NAME` | What it is and needs, its pinned commits against what is installed, its goal language, the `planner.options` it reads, its presets. | `-p/--profile` |
| `tandem planners install NAME` | Fetches its pinned sources and builds its runtime, or updates an outdated one ([runtime](CONFIGURATION.md#the-planner-runtime)). | `--sources DIR` ([offline](CONFIGURATION.md#offline-install)), `--force` (fetch again and rebuild), `-y/--yes` |
| `tandem planners use NAME` | Makes a profile plan with it ([planner settings](CONFIGURATION.md#planner-settings)). | `-p/--profile`, `-o/--option KEY=VALUE`, `--default` (new profiles too) |
| `tandem planners default NAME` | Sets the planner new profiles get. Changes no profile. | |
| `tandem planners bundle NAME --out DIR` | Writes its pinned sources to DIR for an [offline install](CONFIGURATION.md#offline-install). | `--only SOURCE` (repeatable), `--from SOURCE=PATH` (export from a local checkout), `--archive` (also write `DIR.tar.gz`) |
| `tandem planners remove NAME` | Deletes its runtime. It stays listed and can be installed again. | `-y` |
| `tandem planners new NAME` | Scaffolds a package for your own planner ([quick start](ADDING_A_PLANNER.md#quick-start)). | `--sidecar`, `--dir` |
| `tandem executors list`, `use NAME` | Lists human executors and whether each is ready here; `use` picks one for a profile ([choosing one](ADDING_A_HUMAN_EXECUTOR.md#choosing-one)). | `-p/--profile` |
| `tandem traj …` | Inspects and files trajectories ([below](#reviewing-and-exporting)). | |
| `tandem export lerobot\|manifest [profile]` | Builds a LeRobot dataset, or a JSON index ([below](#exporting)). | |
| `tandem config list\|get\|set\|edit\|path` | Machine settings ([keys](CONFIGURATION.md#machine-settings-and-credentials)). | `get KEY`, `set KEY VALUE` (dotted, for example `ui.port`) |
| `tandem config set-gemini-key\|set-hf-token` | Stores a credential; asks for it at a prompt. | `--stdin` (read it from stdin); `set-gemini-key --key KEY` (lands in shell history) |
| `tandem runtime status\|build\|shell\|python\|run\|clean\|path` | Works on the active profile's planner runtime: what is built, build or repair it, open a shell in it, print its interpreter, `run CMD` in it, delete it, print its directory. | `--planner NAME` or `-p/--profile P` for another; `build --force` (fetch again), `--env-only`, `--sources DIR`; `clean -y` |

`--json` works on `doctor`, `plan`, `profile list|show|presets`, `traj list|show`,
`planners list|info|use|default`, `executors list|use`, `runtime status` and `config list`.

`tandem --debug <command>` shows full tracebacks, and `tandem --no-color <command>` (or `NO_COLOR`) turns colour
off. Both go before the command.

Put `--` before a runtime command that takes options: `tandem runtime run -- viz-calibration --camera external`.

### Planner and executor states

- Planners: `installed`, `not installed`, `outdated` (built at other commits than tandem now pins),
  `no runtime needed`, `broken` (with why).
- Executors: `ready`, `needs setup` (with what is missing), `broken`.
- In both listings, `●` marks the one the profile uses.

## Collecting

`tandem collect` warms the planner once, then loops: task prompt, trial, label prompt. The footer shows only
the keys the current state accepts.

| state | keys |
|---|---|
| task prompt | `↵` run the task again · `n` type a new task |
| planning or executing | `p` preempt · `t` take the arm (when teleop is ready) |
| you have the arm | `r` return control |
| human phase | `t` take the arm, or run the profile's executor (shown when it is ready) · `d` I did it (shown only when allowed) · `a` give up (the trial is aborted) |
| label prompt | `s` or `y` success · `f` or `n` failure |
| any state | `q` or Ctrl-C finish the session |

- **Finish (`q`)** stops at the next step boundary and parks the arm without opening the gripper. A trial in
  flight is filed as aborted. A trial waiting at the label prompt stays unmerged in `eval/`; the session prints
  the command that files it (`tandem traj merge <trajectory id> --status success`, or `traj relabel` for a
  one-leg trial).
- **Preempt (`p`)** abandons the attempt and files its legs as aborted; the session stays warm. **It does not
  stop motion:** the current motion segment finishes unless the planner declares a cooperative stop, which
  TiPToP doesn't. The E-stop is the only instant stop.
- **Hand-off (`t` outside a human phase)** gives you the arm through teleop at the next plan-step boundary.
  After `r`, the same phase is perceived and planned again from where you left the arm, with no homing.
  All the trial's legs merge into one episode.
- **Human phase.** The screen shows what to do and what will be checked. `t` runs the profile's human
  executor and `r` hands the arm back. A fresh camera image is then checked; a failed check lists what is
  missing and allows `hitl.verify_retries` more tries. `d` while recording is refused unless
  [`hitl.allow_unrecorded_human_phase`](CONFIGURATION.md#phase-planning-hitl) is true.
- **Label** is asked only when the plan ran to the end, or a check failed under
  `hitl.on_verification_failure: label`. A [settled](README.md#terms) trial is filed without one, and the prompt
  says why ([how a trial ends](METHOD.md#how-a-trial-ends)).
- **`--episodes N`** counts labeled trials and trials that failed part-way, not excluded or aborted ones.
- If nobody returns control within an hour, tandem ends the leg and takes the arm back.

## Planning from a photo

`tandem plan` shows how a task splits into phases before you collect. It needs only tandem (Python 3.10+)
and a [Gemini key](CONFIGURATION.md#machine-settings-and-credentials): no runtime, no GPU, no robot. The
model's answer varies from run to run:

```console
$ tandem plan "place the bread inside the box" --image workspace.png -o bread -o box -o plate
  · objects  box, bread, plate

◆ place the bread inside the box  3 phase(s) · planner: tiptop
  0  robot  move the bread off the box's lid onto the plate
        · bread is resting on top of plate
        goal: [{"predicate": "on", "args": ["bread", "plate"]}]
  1  human  lift the box's lid open
        · the lid of box is lifted open, so its inside is visible
        Lift the lid of the box until it stays open.
        operator LiftOpen(x0: surface)  as LiftOpen(box)
          preconditions  HandEmpty()
          add effects    IsOpen(box)
          delete effects none
  2  robot  put the bread inside the box
        · bread is resting on top of box
        goal: [{"predicate": "on", "args": ["bread", "box"]}]

  ✔ the phases' contracts hang together  checked in the repair loop

invented predicates ─────────────────────────────────────────────────────────────
  IsOpen(surface)
      the lid of {0} is lifted open, so its inside is visible
```

**It also flags any clause of the instruction the plan can't express.** That is usually an object the
instruction names that wasn't detected. Put it on the table or reword the task before collecting.

- `-o/--object LABEL` (repeatable) fixes the object labels. Pass a session's labels to reproduce its plan.
  Without it, a vision model names the objects in the photo first.
- `-p/--profile P` takes the planning settings, and the planner, from a profile.
- `-b/--planner NAME` (alias `--backend`) plans in that planner's goal language. Default: the profile's
  planner, else the machine's default.
- `--table NAME` is what the planner calls the table (default `table`).
- `--json` prints the plan record that `hitl.json` is written from.
- `--save-vlm-io DIR` keeps every image sent to the model and its reply.

### From Python

```python
import tandem

plan = tandem.plan_task("place the bread inside the box", "workspace.png", planner="tiptop",
                        objects=["bread", "box", "plate"])
for i, phase in enumerate(plan.phases):
    print(i, phase.executor, phase.description, sorted(map(str, phase.atoms)))
# 0 robot move the bread off the box's lid onto the plate ['On(bread, plate)']
# 1 human lift the box's lid open ['IsOpen(box)']
# 2 robot put the bread inside the box ['On(bread, box)']
```

- `plan_task` runs what `tandem plan` runs and returns the `PhasePlan` a session would walk (`.phases`,
  `.spec`, `.to_json()`).
- `image` is a path, a PIL image or an RGB `uint8` array. The keywords mirror the flags (`planner`,
  `profile`, `objects`, `table`, `save_vlm_io`), plus `config` for a `PlanningConfig`.
- Inside a running event loop, use `await tandem.plan_task_async(...)`.
- `import tandem` loads nothing heavy; its names resolve on first use. They are `plan_task`,
  `plan_task_async`, `PhasePlan`, `PlanningConfig`, `Planner`, `SidecarPlanner`, `Capabilities`,
  `PlannerInfo`, `Predicate`, `Parameter`, `RuntimeRecipe`, `register_backend` (alias `register_planner`),
  `register_human_executor` and `TandemError`.

## The web UI

`tandem ui` serves `http://127.0.0.1:8787` (settings `ui.host`, `ui.port`, `ui.open_browser`), moving to the
next free port if that one is busy. It needs no Node, build step or CDN, and runs on a laptop with no GPU or
robot. Ctrl-C lets running sessions park the arm and finish their merges; a second Ctrl-C quits at once and
leaves the arm where it is.

- **Trajectories**: camera videos and per-frame plots; click a plot to seek every video to that frame. A
  merged trajectory has a ribbon showing which stretches the planner drove and which a person did; click one
  to jump to it. Shaded bands mark the frames π₀.₅-DROID's training drops as idle (`src/tandem/core/nonidle.py`).
- **Collect**: a session with the terminal's controls. A clause the plan leaves out is shown before the arm
  moves, and the label prompt has an inline review.
- **Profiles**: edit a profile and see what its planner will receive.
- **Settings**: credentials, paths, the planner and executor catalogs (with a switch and each install
  command), runtime status, and the full `tandem doctor` report.

### HTTP API

The page talks to routes under `/api`. These are the ones useful from a script:

| route | returns or does |
|---|---|
| `GET /api/planners`, `GET /api/planners/{name}` | What `planners list --json` and `planners info --json` print. `?profile=P` for another profile. |
| `POST /api/planners/{name}/use` | `planners use`. Optional body: `profile`, `options`. |
| `POST /api/planners/{name}/default` | `planners default`. |
| `GET /api/executors`, `POST /api/executors/{name}/use` | `executors list` and `executors use` (optional body: `profile`). |
| `GET /api/profiles/{name}` | The profile, with `planner_view`, whose `receives` is what `profile show --planner` prints. Each card from `GET /api/profiles` carries a one-line `planner_summary`. |
| `GET /api/sessions/{id}` | The [session summary](DATA.md#session-summary) and its last 500 log lines. |
| `GET /api/media/{profile}/{id}/{file}` | A trajectory's video, served with HTTP Range so the browser can scrub. A proxy in front must pass `Range` headers through. |

Installing a planner is not an endpoint: a catalog row that needs it carries the `install_command` to run.

## Reviewing and exporting

```bash
tandem traj list [profile]            # newest first; -s/--status eval|success|failure, -n N (default 30, 0 = all)
tandem traj show <id>                 # frames, cameras, outcome, hand-off legs, motion stats
tandem traj open <id>                 # replay it in the viewer of the planner that recorded it (TiPToP: Rerun, in its runtime)
tandem traj relabel <id> <status>     # move it between success, failure and eval
tandem traj merge [<trajectory id>]   # re-join a trial's legs if the automatic merge failed; --status STATUS
tandem traj copy <id> <profile>       # copy it into another profile; also traj rm <id> [-y], traj path <id>
```

`<id>` is a trajectory's directory name (a timestamp), or a unique prefix of it. Every command but `list` takes
`-p/--profile`.

- **Relabel.** `tandem traj relabel <id> success` refuses a settled trial. `--force` (a confirm in the web UI)
  overrules that, and the trial's `hitl.json` records it under `overruled`. It is the only way an excluded
  trial reaches an export.
- **Merge.** `traj merge` takes the trial's `trajectory_id` (in `_meta.json`), not the timestamp. It never
  leaves the legs half-written. With no id, it merges every trial that has unmerged legs.

### Exporting

```bash
tandem export lerobot --repo <owner>/<name>          # writes ~/tandem-data/exports/<owner>/<name>
tandem export lerobot --repo <owner>/<name> --push   # and uploads it; --private or --public
tandem export manifest [--out FILE]                  # a JSON index of the profile's trajectories (default: stdout)
```

`tandem export lerobot` writes a LeRobot v3.0 dataset in `lerobot/droid_1.0.1`'s schema, for a π₀.₅-DROID
fine-tune. It needs av, pyarrow and huggingface_hub ([installing them](CONFIGURATION.md#installing)).

- **Only `success/` is exported**, and a settled trial filed there is skipped unless a forced relabel
  overruled it. An episode is also skipped when its `cmd_gripper` isn't binary, its exterior or wrist video
  is missing, or its state arrays don't fit DROID's schema. Every skip is listed with its reason.
- **The action is `cmd_joint_velocity`**, clipped to [-1, 1] and never rescaled, plus `cmd_gripper`.
  `action_joint_velocity` stays in `robot_state.npz` but is not exported.
- **Each episode's task** is its `_meta.json` `instruction`, else the profile's `task.prompt`. A rig with no
  `external_2` camera gets its first exterior video in both exterior slots.
- **Destination.** `--repo` defaults to the profile's `export.hf_repo`, and a name with no owner takes
  `hf_org`. `--out DIR` writes `DIR/<owner>/<name>`. `-n/--max-episodes N` exports the first N.
- **Rebuilds** replace the old dataset only once the new one is complete, and only if tandem built the old
  one. `--force` replaces anything at the destination.
- **`--push`** needs a Hugging Face token ([where tandem looks](CONFIGURATION.md#machine-settings-and-credentials)).
  Visibility defaults to the profile's `export.private`.
- Each run appends to `export.log` ([logs](DATA.md#logs-and-session-files)).
