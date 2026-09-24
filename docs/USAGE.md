# Using tandem

Every command, the keys of a collection session, `tandem plan`, the web UI, and reviewing and exporting what
was collected. Settings are in [CONFIGURATION.md](CONFIGURATION.md); what a trial leaves on disk is in
[METHOD.md §6](METHOD.md#6-what-a-trial-leaves-on-disk); common problems are in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

- [Commands](#commands)
- [Collecting](#collecting)
- [Planning from a photo](#planning-from-a-photo)
- [The web UI](#the-web-ui)
- [Reviewing and exporting](#reviewing-and-exporting)

---

## Commands

| | |
|---|---|
| `tandem init` | Set up this machine: checks, the planner (`--planner NAME`) and its runtime, the Gemini key, a profile (`--profile NAME`, `--import-from DIR`, `--preset NAME`), teleop. `--viz-only` for a laptop, `--yes` to ask nothing. Idempotent; `--repair` redoes the runtime, key and teleop steps and never touches an existing profile. |
| `tandem doctor` | Every check, what it found, and what to do about it. `--profile P`; `--no-hardware` skips the robot, camera and grasp-server probes. |
| `tandem collect [profile]` | Run a session in the terminal. `--task`, `--episodes N`, `--no-execute` (perceive and plan, never move), `--no-record`, `--web` (drive it from the browser). |
| `tandem plan "<task>" --image photo.png` | Decompose a task into phases from a photo, with no robot and no GPU ([below](#planning-from-a-photo)). |
| `tandem ui` | Serve the browser UI. `--port`, `--host`, `--no-open`, `--profile`. |
| `tandem profile list \| show \| create \| presets \| migrate \| use \| edit \| delete \| path` | Manage profiles. `show --planner` prints what the planner receives. |
| `tandem planners list \| info \| install \| use \| default \| remove \| bundle \| new` | The planner catalog ([CONFIGURATION.md](CONFIGURATION.md#the-planner-runtime), [ADDING_A_PLANNER.md](ADDING_A_PLANNER.md)). |
| `tandem executors list \| use` | Who carries out a human phase ([ADDING_A_HUMAN_EXECUTOR.md](ADDING_A_HUMAN_EXECUTOR.md)). |
| `tandem traj list \| show \| open \| relabel \| rm \| merge \| copy \| path` | Inspect trajectories ([below](#reviewing-and-exporting)). |
| `tandem export lerobot \| manifest` | Build a LeRobot v3.0 dataset from `success/`, or write a JSON index. |
| `tandem config list \| get \| set \| set-gemini-key \| set-hf-token \| path \| edit` | Machine settings and credentials. |
| `tandem runtime status \| build \| shell \| python \| run \| clean \| path` | The runtime of the active profile's planner (`--planner NAME` for another). |

`doctor`, `plan`, `profile list | show | presets`, `traj list | show`, `planners list | info | use | default`,
`executors list | use`, `runtime status` and `config list` take `--json`. `--debug` shows full tracebacks.
`NO_COLOR` (or `--no-color`) is honoured.

The planner catalog and the human executors:

| | |
|---|---|
| `tandem planners list` | Every planner tandem can see, built in or registered by an installed package, with its status: `installed`, `not installed`, `outdated`, `no runtime needed` or `broken` (with why). `●` marks the one the active profile uses. |
| `tandem planners info NAME` | What it is and needs, the commits it pins (and each one's branch) against what is installed, its goal language as the phase planner sees it, what it supports, the `planner.options` it reads, and its presets. |
| `tandem planners install NAME` | Fetch its pinned sources and build its runtime. A current runtime returns at once; an outdated one is updated. `--sources DIR` takes the sources from DIR, `--force` rebuilds, `--yes` asks nothing. |
| `tandem planners use NAME` | Make a profile plan with it (`--profile P`, `--option KEY=VALUE`, `--default` for new profiles too). The old planner's `planner.options` move to `planner-options.<planner>.yml`, and switching back restores them. It also repairs a profile naming a planner this machine no longer has. |
| `tandem planners default NAME` | The planner new profiles get. Changes no profile. |
| `tandem planners bundle NAME --out DIR` | Its pinned sources in DIR, for `install --sources DIR` on a machine that cannot fetch them (`--archive` also writes a `.tar.gz`). |
| `tandem planners remove NAME` | Delete its runtime. It stays listed and can be installed again. |
| `tandem planners new NAME` | Scaffold a package for a planner of your own (`--sidecar`, `--dir`). It passes the conformance kit as generated. |
| `tandem executors list` | Who can carry out a human phase (`hitl.human_executor`), whether each is ready here, and what one still needs. |
| `tandem executors use NAME` | Make a profile's human phases run with it (`--profile P`). |

---

## Collecting

`tandem collect` warms the planner once, then loops: task prompt, trial, label prompt. The footer shows the
keys the current state accepts:

| state | keys |
|---|---|
| the task prompt | `↵` run the task · `n` type a new task |
| planning or executing | `p` preempt · `t` hand the arm to a person (when teleop is set up) |
| a human phase | `t` take the arm (or run the configured executor) · `d` I did it (only when allowed) · `a` give up on this task |
| an operator hand-off | `r` return control |
| the label prompt | `s` success · `f` failure |

`q` (or Ctrl-C) finishes the session from any state, parking the arm first. It is a step boundary like a
preempt: nothing more is perceived, planned or executed, and the trial in flight is filed as aborted. A trial
waiting at the label prompt is left unlabeled in `eval/` with its `hitl.json`;
`tandem traj merge <trajectory id> --status success` files it later.

- **Preempt** abandons the attempt in flight. The session stays warm and returns to the task prompt, and
  what was recorded is filed as aborted. It does not stop the arm mid-motion: unless the planner declares a
  cooperative stop, the current motion segment runs to its end. The E-stop is the only instant stop.
- **Hand-off** (`t` outside a human phase) lends you the arm at the next plan-step boundary. The planner lets
  go of the robot and cameras; when you press `r`, the same phase is perceived and planned again from
  wherever you left the arm, with no homing. Every leg merges into one trajectory.
- **A human phase** shows its instructions and what will be checked. `t` runs `hitl.human_executor`
  (teleop by default); with teleop, `r` hands the arm back when the step is done, and the check runs. While
  recording, `d` is refused unless `hitl.allow_unrecorded_human_phase: true`, since that phase would have no
  demonstration. After the step a fresh image is checked; a failed check says what is still missing and gets
  `hitl.verify_retries` more tries.
- **Label** only a trial whose plan ran to the end (and a failed check, with
  `hitl.on_verification_failure: label`). One the loop ended itself (excluded, failed at a TAMP or
  human-policy stage, or aborted) is filed under `failure/` without a label, and the prompt says why.

How each trial ends, stage by stage, is [METHOD.md §3](METHOD.md#3-the-trial-loop).

---

## Planning from a photo

The phase planner is tandem's own code, so it runs without a planner, a GPU or a robot: only the planner's
declared goal language and a Gemini key. An example (the model's answer will vary):

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

It prints the ordered phases, who does each and the goal the planner would be handed; each human phase's
operator; whether the operators hang together; the invented predicates and their classifier sentences; and,
loudly, any clause of the instruction the model could not express. That is usually an object the
instruction names that was not detected, and the remedy (put it on the table, or reword the task) is only
available before collecting.

- `--object` / `-o LABEL`, repeated, pins the object labels. Without it, a vision model names the objects
  in the photo first.
- `--profile P` takes the planning settings, and the planner, from a profile.
- `--planner NAME` plans in another planner's goal language (`--backend` is its older name).
- `--table NAME` is what the planner calls the table (default `table`).
- `--json` prints the plan record, the structure `hitl.json` has.
- `--save-vlm-io DIR` keeps every image sent to the model and its reply.

The same from Python:

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

`plan_task` runs exactly what `tandem plan` runs and returns the `PhasePlan` a session would walk (phases,
operators, invented predicates, `to_json()`). `plan_task_async` is the same inside an event loop.
`import tandem` imports nothing heavy; its names (`plan_task`, `PhasePlan`, `PlanningConfig`, `Planner`,
`SidecarPlanner`, `Capabilities`, `PlannerInfo`, `Predicate`, `Parameter`, `RuntimeRecipe`,
`register_backend` or its alias `register_planner`, `register_human_executor`, `TandemError`) are resolved
on first use.

---

## The web UI

`tandem ui` (default `http://127.0.0.1:8787`) serves a browser view of a profile. The package serves it
itself: no Node, no build step, no CDN.

- **Trajectories**: every rollout, with its camera videos and per-frame plots. Clicking a chart seeks every
  video to that frame. A merged trajectory has a ribbon showing which stretch the planner drove and which a
  person did; click a stretch to jump to it. The shaded bands in the plots are the frames π₀.₅-DROID's
  non-idle filter drops at training time, computed at full resolution on the clipped action the dataset
  stores (`src/tandem/core/nonidle.py`).
- **Collect**: a session from the browser, with the terminal's controls. A human phase shows its
  instructions and what will be checked, a clause the plan leaves out is shown before the arm moves, and the
  label prompt has an inline review. An excluded trial says so, and is never offered for a label.
- **Profiles**: edit a profile and see what its planner will receive.
- **Settings**: credentials and paths; the planners and human executors (which is installed, which the
  profile uses, a button to switch, and the command that installs one); runtime status; the full
  `tandem doctor` report.

---

## Reviewing and exporting

```bash
tandem traj list [profile]            # newest first; --status eval|success|failure, --limit N (0: all)
tandem traj show <id>                 # one trajectory in detail (an id is a timestamp or a unique prefix)
tandem traj open <id>                 # replay it in its planner's own viewer (TiPToP's: Rerun, in the runtime)
tandem traj relabel <id> success      # move it between success, failure and eval
tandem traj merge [<trajectory id>]   # re-join a trial's legs if the automatic merge failed; --status
tandem traj copy <id> <profile> | rm <id> | path <id>
```

`tandem traj relabel <id> success` refuses a trial the method settled itself (excluded, aborted, or failed
part-way). `--force` (a confirm in the web UI) overrules it, and its `hitl.json` then records that under
`overruled`.

```bash
tandem export lerobot --repo <owner>/<name>          # to ~/tandem-data/exports/<owner>/<name> (--out DIR)
tandem export lerobot --repo <owner>/<name> --push   # and upload; --private/--public
tandem export manifest [--out FILE]                  # a JSON index of the profile's trajectories
```

`tandem export lerobot` writes a LeRobot v3.0 dataset in `lerobot/droid_1.0.1`'s schema, for a π₀.₅-DROID
fine-tune:

- It exports `success/` only, so failed and excluded trials never reach a dataset. A trial under `success/`
  whose `hitl.json` says the method settled it (excluded, aborted, or failed part-way) is skipped too, unless
  a forced relabel overruled it.
- An episode whose `cmd_gripper` is not binary is skipped, loudly; so is one missing its exterior or wrist
  video. Every skip is listed with its reason.
- `--repo` defaults to the profile's `export.hf_repo`; a name without an owner takes `hf_org`.
  `--max-episodes N` exports the first N.
- A rebuild replaces the previous dataset only once the new one is complete, and only one tandem built;
  `--force` replaces anything else at the destination.
- `--push` needs a Hugging Face token (`tandem config set-hf-token`, `HF_TOKEN`, or `huggingface-cli login`).
