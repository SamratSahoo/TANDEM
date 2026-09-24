# Data on disk

What a trial leaves on disk, and where tandem writes everything else: for reading, processing or debugging
collected data. Paths are the Linux defaults ([moving them](CONFIGURATION.md#machine-settings-and-credentials)),
and terms such as leg, episode and settled are defined in [README.md](README.md#terms).

On this page: [Episode layout](#episode-layout) · [robot_state.npz](#robot_statenpz) · [_meta.json](#_metajson) ·
[hitl.json](#hitljson) · [vlm/](#vlm) · [The events file](#the-events-file) · [Logs and session files](#logs-and-session-files)

## Episode layout

Each trial is filed as one episode directory under `success/` or `failure/`
([status directories](CONFIGURATION.md#profiles)):

```
~/tandem-data/profiles/<profile>/trajectories/<status>/<YYYY-MM-DD_HH-MM-SS>/
├── external_cam.mp4  external_cam_2.mp4  hand_cam.mp4   exterior 1, exterior 2 (optional), wrist: every leg joined
├── robot_state.npz        per-frame arrays, every leg joined
├── _meta.json             lineage, timing, and which frames came from which leg and phase
├── hitl.json              the phase plan and everything that happened to it (phase planning on)
├── vlm/                   every image sent to a model, and its answer (phase planning on)
├── tiptop_plan.json, …    the primary leg's other files
└── segments/NN_<source>_<leg dir>/   each raw leg as recorded, in recording order
```

- The episode takes the [primary leg](README.md#terms)'s directory name, `_meta.json` and other files (for
  TiPToP: `tiptop_plan.json`, `metadata.json`, `rgb.png`, `perception/`). `*.log` files stay in `segments/`.
- A trial with a single leg is not merged: that leg's directory is the episode.
- The format is hitl-tamp-vla's, so data moves between the two in either direction.

`tandem export manifest` writes a JSON index of a profile's trajectories. Which id each `tandem traj` command
takes, and relabeling, re-merging and exporting, are in [USAGE.md](USAGE.md#reviewing-and-exporting).

## `robot_state.npz`

One row per frame, every leg's rows joined in order. The per-leg arrays, their shapes, and which are measured or
commanded are in [the recording contract](ADDING_A_PLANNER.md#the-recording-contract). An episode may also have:

| array | shape | what it is |
|---|---|---|
| `video_time` | `[F]` float64 | Merged episodes only: where each frame sits in the joined clips, in seconds. |
| `action_joint_velocity` | `[F,7]` | The DROID joint-velocity action, when some leg recorded it (TiPToP legs do). A merge fills it in for the other legs: a teleop or policy leg takes its `cmd_joint_velocity`, and a planner leg gets `5 × (cmd_joint_position − joint_position)`. |

To match frames to video in a merged episode, use `video_time`, not `frame_time`: `frame_time` keeps the
wall-clock gaps between legs, and the joined clips don't. In a single leg, frame *i* sits at
`(frame_time[i] − record_start) / (record_stop − record_start)` of the way through each clip. Which arrays the
export uses: [USAGE.md](USAGE.md#exporting).

## `_meta.json`

A merged episode's `_meta.json` is the primary leg's, with these keys set. A single-leg episode keeps the leg's
own ([per-leg keys](ADDING_A_PLANNER.md#the-recording-contract)).

| key | what it is |
|---|---|
| `trajectory_id` | The id every leg of the trial carries. |
| `planner`, `instruction` | Which planner recorded the trial, and the task (the language label). Where the planner didn't write them, the session stamps them into the primary leg (`planner` only on a planner's leg). `tandem traj open` picks its viewer by `planner`. |
| `source`, `segment_source` | `"trajectory"` and `null`. |
| `video_aligned` | `true`: align on `video_time`. |
| `n_frames`, `fps`, `total_video_frames` | Rows in `robot_state.npz`, the frame rate, and frames in each joined clip. |
| `record_start`, `record_stop` | Epoch seconds: the first leg's start and the last leg's stop. |
| `cameras` | Dataset key → clip: `exterior_image_1_left` → `external_cam.mp4`, `exterior_image_2_left` → `external_cam_2.mp4`, `wrist_image_left` → `hand_cam.mp4`. |
| `n_phases` | Kept only when every leg that states it agrees and all of them carried out one plan. |
| `segments[]` | One per leg, in order (below). |
| `cameras_dropped` | Clips only some legs had, so not joined. |
| `frames_trimmed` | Leg → camera → trailing frames cut so that leg's cameras match in length. |
| `proportional_fallback_legs` | Legs with no usable recording window, whose frames were spread evenly over their clips. |
| `legs_skipped` | Legs with no `robot_state.npz`, left out of the merge. |
| `action_convention`, `action_notes` | Only with `action_joint_velocity`: `"droid_joint_velocity"`, and leg → a note for each leg whose action was recomputed or could not be. |

Each `segments[k]` has `source` (`tamp`, `teleop` or `policy`), `timestamp` (the leg's directory name),
`n_frames`, `n_video_frames`, `video_start` / `video_stop` (seconds into the joined clips) and `record_start` /
`record_stop`. When the leg stated them, it also has `phase_index` and `phase_description` (found only here, never
at the top level), `n_phases`, `plan_generation` after a `replan` (0 for the first plan; absent means one plan),
and `config_id` (teleop legs: `teleop/teleop`).

A conjoined robot leg is stamped with the first phase it covers, and its `phase_description` joins every covered
phase's description with `; `. `hitl.json` lists the phases under `phases[k].covers_phases`.

## `hitl.json`

The phase record (`PhasePlan.to_json` in `src/tandem/planning/plan.py`), written only when phase planning is on
and the trial has a plan. Filing writes it into the primary leg and again beside the merged episode. A trial
stopped at the label prompt keeps it in its primary leg in `eval/`.

| key | what it is |
|---|---|
| `instruction`, `trajectory_id`, `planner`, `human_executor` | The task as planned, the trial's id, the planner's name, and `hitl.human_executor`. |
| `outcome`, `excluded` | `success`, `failure`, `excluded` or `aborted` ([how a trial ends](METHOD.md#how-a-trial-ends)). `excluded` is `true` exactly when `outcome` is `excluded`. |
| `failure_stage` | `invention`, `tamp_planning`, `tamp_execution`, `verification`, `human_policy`, or `null`. Always the loop's, whatever the label said. |
| `filed_under` | `success` or `failure`. `null` when the trial was never labeled or was moved to `eval/`. A relabel rewrites it. |
| `outcome_reason` | Only when the loop ended the trial: what was wrong, in words. |
| `overruled` | Only after `tandem traj relabel --force` filed a settled trial under `success/`: `{outcome, excluded, failure_stage, by}` as the loop had them. `outcome` then reads `success` ([relabel rules](USAGE.md#reviewing-and-exporting)). |
| `plan_generation` | Which plan this record is: 0, plus one per `replan`. |
| `leg_plan_generations` | After a replan only: leg directory → the plan it ran under. |
| `superseded_plans[]` | Every plan a `replan` replaced, oldest first: its record as it then stood (no filing keys), plus `plan_generation` and `superseded_because`. `[]` without a replan. |
| `initially_true` | Invented atoms seen true on the first image (`hitl.classify_initial`), else `[]`. |
| `provenance` | Who produced each part, in words: `phases_and_their_order`, `phase_sub_goals`, `invented_predicates` and `human_instructions` (the model), `robot_phases` (the planner), `who_does_what` (tandem), `human_steps` (the executor meant to carry steps out; `phases[k].carried_out` says what did). `human_operators` and `robot_operators` are `{by, signatures}`, each signature written `Name(param: type)`. |
| `handed_over_phases` | Robot phases given to a person because the planner couldn't plan them (`on_robot_phase_failure: teleop`). |
| `phase_index` | How far the trial got. It equals the number of phases when the plan finished. |
| `specification`, `checks`, `phases[]`, `verifications[]` | What the proposal made of the instruction, which checks ran, one record per phase, and every camera verdict kept (failing ones included). Each is described below. |

**`specification`**

- `invented_predicates[]`: `{name, types, instructions}`. `instructions` is the classifier sentence, with `{0}`,
  `{1}`, … for the arguments.
- `human_operators[]`: `{phase, name, args, signature, instance, preconditions, add_effects, delete_effects}`, one
  per human phase with an operator. `tandem.planning.structs.HumanOperator.from_json` reads one back.
- `surfaces[]`, `movables[]`: the object types, fixed for the task.
- `unrepresented[]`: `{clause, reason}` for each part of the instruction the plan leaves out. `coverage[]`:
  `{clause, phase}` for every clause, with `phase: -1` for an unrepresented one.

**`checks`**

- `human_preconditions`, `human_effects`, `tamp_preconditions`, `tamp_effects`, `plan_effects`,
  `verify_enforced`, `precondition_enforced`, `verify_final_phase`: the `hitl` switches as configured.
- `initial_state_classified`, `plan_effects_rechecked`, `plan_effects_warning`: whether the first image was
  classified, whether the [contract check](METHOD.md#the-contract-check) then ran again against it, and what it
  found (or `null`).
- `unchecked_phases[]`: phases accepted without a verdict. `unrecorded_human_phases[]`: human phases with no leg
  on disk in any attempt, so the episode has no segment for them.

**`phases[k]`** has `index`, `executor`, `description`, `atoms` and `planned_by`, plus:

- **Robot phase:** `goal` (the atoms the planner was handed, in its spelling) and `goal_description`. Once
  planned: `planning_seconds`, `plan_reused`, `task_plan` (the planner's operator sequence, when it reports one)
  and `covers_phases` (when one plan covered several phases).
- **Human phase:** `instructions` and `operator`. Once run: `carried_out[]`, one
  `{attempt, carried_out_by, status, leg_recorded, n_frames}` per attempt. `carried_out_by` is the executor's name,
  or `by_hand` with `status: null` when "done" was answered with no executor
  ([status values](ADDING_A_HUMAN_EXECUTOR.md#humanphaseresult)).
- **Robot phase handed to a person:** recorded as the human phase it became, with no `operator`, plus
  `proposed_executor: "robot"`, `handed_over_because` and `instructions_by: "tandem"`.
- **Any phase:** `unchecked`, with the reason, when its check could not run or had nothing a camera can judge.

**`verifications[k]`** is `{atom, statement, holds, expected, satisfied, role, reason, phase}`.

- `holds` is what the model saw and `expected` is what the plan wanted. **Read `satisfied`** (`holds ==
  expected`): for a delete effect, `holds: true` is the failure.
- `role` is `effect`, `effect (deleted)` or `precondition`.
- Only the attempt that settled a phase is kept, so an excluded trial carries the verdicts it was excluded on.
  Earlier attempts are in the events file (`human_phase_verified`).

A trimmed example, from the test suite's toy planner (items dropped into bins):

```json
{
  "instruction": "put the duck in the red bin, close its lid, ...", "trajectory_id": "7777777777777777",
  "planner": "toy", "outcome": "success", "failure_stage": null, "excluded": false, "filed_under": "success",
  "specification": {
    "invented_predicates": [{"name": "LidClosed", "types": ["container"], "instructions": "the lid of {0} is shut, covering its opening"}],
    "human_operators": [{"phase": 1, "instance": "Close(red_bin)", "add_effects": ["LidClosed(red_bin)"], "delete_effects": ["LidOpen(red_bin)"], "...": "..."}]
  },
  "phases": [
    {"index": 1, "executor": "human", "atoms": ["LidClosed(red_bin)"],
     "carried_out": [{"attempt": 1, "carried_out_by": "teleop", "status": "done", "leg_recorded": true, "n_frames": 12}]}
  ],
  "verifications": [{"atom": "LidOpen(red_bin)", "holds": false, "expected": false, "satisfied": true, "role": "effect (deleted)", "phase": 1}]
}
```

## `vlm/`

Every model query of the trial, rejected and cached answers included. It is written when phase planning is on
and `hitl.save_vlm_io` is true (the default), first into the session directory, then copied into the episode
when the trial is filed. Each query leaves:

- `NNN_<label>_input.png`: the image exactly as sent, for example `002_classify-LidClosed-red_bin_input.png`.
- `NNN_<label>_output.png`: that image above the model's answer, marked REJECTED when the answer was refused and
  CACHED when it was replayed from `hitl.cache_path`.
- A line in `index.jsonl`: `seq`, `label`, `attempt`, `model`, `input_image`, `output_image`, `rejected` (why, or
  `null`), `cached`, `prompt`, `response`.

Labels are `task plan` (the proposal), `classify <atom>` (each camera check) and `detect objects` (`tandem plan`
without `--object`). A retry adds `_attempt<N>` to the name. `NNN` counts on from what the directory already holds,
so a retry, or a second `tandem plan --save-vlm-io DIR` into the same directory, never overwrites a file.

## The events file

A session appends one JSON line per event to `~/.local/state/tandem/sessions/<profile>/<session id>/events.jsonl`,
for example `{"event": "rollout_saved", "ts": 1790270037.40, "dir": "…", "n_frames": 12}`. A sidecar planner's
own events go to the same file. The web UI receives the same records on `/api/sessions/{id}/stream`, as
`{"type": "event", "event": …, "ts": …, …, "at": <ts>}`.

From the phase loop:

| event | payload |
|---|---|
| `rollout_start` | `dir`, `phase_index`, `n_phases`, plus `movables` and `return_home` when passed to the planner |
| `rollout_saved` | `dir`, `n_frames` |
| `phase_complete` | `phase_index` (the next phase), `n_phases`: a phase finished and more remain |
| `phase_plan_failed` | `reason`, `policy` (`on_robot_phase_failure`), `phase_index` (`null` with phase planning off) |
| `instruction_not_fully_represented` | `unrepresented`: `[{clause, reason}]` |
| `awaiting_human_phase` | `description`, `instructions`, `expected` (in words), `expected_atoms` (must hold after), `expected_deleted_atoms` (must no longer hold), `phase_index`, `n_phases`, `is_last_phase`, `operator` |
| `human_phase_refused` | `phase_index`, `executor`, `reason`: a phase with no recorded leg, while recording |
| `human_phase_by_hand` | `phase_index`, `attempt`: "done" with no executor was accepted, so the phase has no leg |
| `human_leg_ended` | `executor`, `status`, `n_frames`, `dir`, `phase_index` |
| `human_phase_verified` | `phase_index`, `attempt`, `ok` (`true`, `false`, or `null` when not judged), `verdicts`, plus `skipped` or `unchecked` saying why `ok` is `null` |
| `phase_preconditions_checked` | `phase_index`, `description`, `what` (`human phase` or `robot leg`), `ok`, `enforced`, `verdicts`, plus `unchecked` when the check could not run |
| `phase_effects_checked` | The same, for a robot leg's effects (`enforced` is always `false`) |
| `teleop_handoff_warning` | `message`: the planner could not release the robot |
| `trial_outcome` | `trajectory_id`, `outcome`, `failure_stage`, `excluded`, `reason`, `phase_index`: only when the loop ended the trial itself |

From the session. The *trial summary* is `trajectory_id`, `outcome`, `excluded`, `filed_under`, `failure_stage`,
`reason` and `labeled`.

| event | payload |
|---|---|
| `session_start`, `session_end`, `awaiting_task` | None |
| `awaiting_label` | `dir` |
| `labeled` | `dir`, `success`, and the trial summary |
| `trial_excluded` | `dir` and the trial summary |
| `trial_filed` | `dir` and the trial summary: failed part-way or aborted, filed without a label |
| `trial_unlabeled` | `dir` and the trial summary: the session stopped at the label prompt |
| `rollout_discarded` | The trial summary: nothing was recorded |
| `rollout_aborted` | Nothing after a preempt; `reason` when the session is stopping; `error` when the attempt failed |
| `teleop_switch_pending` | None: the operator asked for the arm, handed over at the next plan-step boundary |
| `teleop_handoff_start`, `awaiting_teleop_resume` (`trajectory_id`), `teleop_handoff_done` | A hand-off, from release to return |

### Session summary

A session's live state, from `Session.summary()` in `src/tandem/core/session.py` (which lists every field). The
web UI shows it, `GET /api/sessions/{id}` returns it, and `session-<id>.json` saves it. The fields a script
usually reads:

| field | what it is |
|---|---|
| `id`, `profile`, `task` | The session. |
| `state`, `error`, `end_reason` | Where it is (`warming`, `rolling`, `awaiting_task`, `awaiting_human_phase`, `teleop_handoff`, `awaiting_label`, `stopped`, `failed`, …), and how it ended. |
| `labeled`, `success`, `excluded`, `aborted`, `target` | Trial counts, and the target (`--episodes`, else `task.target_episodes`). |
| `last_trial`, `rollouts[]` | The last attempt's trial summary, and every trial so far (`{dir, id, started_at, n_frames, status, success}`). |
| `human_phase`, `phase_progress` | The phase on screen (its `description`, `instructions`, `expected`, `attempt`, `missing`, …), and `[phase index, n_phases]`. |

## Logs and session files

| path | what it holds |
|---|---|
| `~/.local/state/tandem/logs/session-<id>.json` | Written when `tandem collect` ends: `summary` (the [session summary](#session-summary)) and `logs` (the last 4,000 log lines). A session started from the web UI doesn't write one. |
| `~/.local/state/tandem/sessions/<profile>/<id>/events.jsonl` | [The events file](#the-events-file). |
| `…/<id>/vlm/<trajectory id>/` | Each trial's [model I/O](#vlm). The only copy for a trial that recorded nothing. |
| `…/<id>/perception/<leg dir>/` | Each perception pass that recorded nothing, moved out of `eval/`. For TiPToP it holds `metadata.json`, with the planning result and failure reason. |
| `…/<id>/teleop/leg-*/teleop-events.jsonl` | The teleop driver's own events, one directory per hand-off. |
| `~/.local/state/tandem/logs/export.log` | What `tandem export lerobot` logged, appended per run. |
| `~/.local/state/tandem/logs/runtime-build-<YYYYMMDD-HHMMSS>.log` | One per runtime build (`tandem planners install`, `tandem runtime build`, `tandem init`). |
| `~/tandem-data/exports/<owner>/<name>/` | Exported datasets. `--out DIR` puts them under `DIR` instead ([exporting](USAGE.md#exporting)). |
| `<profile>/trajectories/.merge-<trajectory id>/` | A merge in progress. One left behind holds the legs of a merge that did not finish, and the next `tandem traj merge` says how to move them back. |

A trial that failed at `invention` usually has no legs (a re-proposal after a `replan` can fail with legs
already recorded). Then its only traces are the events file, and its `vlm/` and `perception/` directories in the
session directory.
