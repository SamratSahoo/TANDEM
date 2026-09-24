# Data on disk

What a trial leaves on disk, and tandem's logs. Paths are Linux defaults
([moving them](CONFIGURATION.md#machine-settings-and-credentials)). Terms: [README.md](README.md#terms).

## Episode layout

One directory per trial, under `success/` or `failure/` ([status directories](CONFIGURATION.md#profiles)):

```
~/tandem-data/profiles/<profile>/trajectories/<status>/<YYYY-MM-DD_HH-MM-SS>/
├── external_cam.mp4  external_cam_2.mp4  hand_cam.mp4   exterior 1, exterior 2 (optional), wrist
├── robot_state.npz        per-frame arrays
├── _meta.json             lineage, timing, frame → leg and phase
├── hitl.json              phase record
├── vlm/                   model queries
├── tiptop_plan.json, …    primary leg's other files
└── segments/NN_<source>_<leg dir>/   raw legs, in order
```

- Clips and arrays join every leg in order.
- The episode takes the [primary leg](README.md#terms)'s directory name, `_meta.json` and other files except
  `*.log` (TiPToP: `tiptop_plan.json`, `metadata.json`, `rgb.png`, `perception/`).
- A single-leg trial is not merged: the leg is the episode.
- Same format as hitl-tamp-vla.
- Reviewing and exporting: [USAGE.md](USAGE.md#reviewing-and-exporting).

## `robot_state.npz`

One row per frame. Per-leg arrays: [the recording contract](ADDING_A_PLANNER.md#the-recording-contract). An
episode may also have:

| array | shape | what it is |
|---|---|---|
| `video_time` | `[F]` float64 | Merged only: frame time in the joined clips, in seconds. |
| `action_joint_velocity` | `[F,7]` | The DROID joint-velocity action, if any leg recorded it (TiPToP does). Legs without it: teleop and policy copy `cmd_joint_velocity`; planner legs get `5 × (cmd_joint_position − joint_position)`. |

To match frames to video, use `video_time` in a merged episode (`frame_time` keeps the gaps between legs). In a
single leg, frame *i* is `(frame_time[i] − record_start) / (record_stop − record_start)` of the way through each
clip. Arrays the export uses: [USAGE.md](USAGE.md#exporting).

## `_meta.json`

A merged episode's is the primary leg's, with these keys set. A single-leg episode keeps its own
([per-leg keys](ADDING_A_PLANNER.md#the-recording-contract)).

| key | what it is |
|---|---|
| `trajectory_id` | Shared by every leg of the trial. |
| `planner`, `instruction` | Planner and task (the language label), added by the session if missing (`planner` only on a planner's leg). `tandem traj open` picks its viewer by `planner`. |
| `source`, `segment_source`, `video_aligned` | `"trajectory"`, `null`, `true`. |
| `n_frames`, `fps`, `total_video_frames` | Rows in `robot_state.npz`, frame rate, frames per joined clip. |
| `record_start`, `record_stop` | Epoch seconds: first leg's start, last leg's stop. |
| `cameras` | Dataset key → clip: `exterior_image_1_left` → `external_cam.mp4`, `exterior_image_2_left` → `external_cam_2.mp4`, `wrist_image_left` → `hand_cam.mp4`. |
| `n_phases` | Only if every leg that states it agrees and all ran one plan. |
| `segments[]` | One per leg, in order (below). |
| `cameras_dropped`, `legs_skipped` | Left out: clips only some legs had; legs without `robot_state.npz`. |
| `frames_trimmed` | Leg → camera → trailing frames cut to even out that leg's clips. |
| `proportional_fallback_legs` | Legs with no usable recording window, their frames spread evenly over their clips. |
| `action_convention`, `action_notes` | With `action_joint_velocity`: `"droid_joint_velocity"`; leg → note for each leg whose action was recomputed or could not be. |

`segments[k]` has `source` (`tamp`, `teleop` or `policy`), `timestamp` (leg directory name), `n_frames`,
`n_video_frames`, `video_start` / `video_stop` (seconds into the joined clips) and `record_start` / `record_stop`.
If the leg stated them: `phase_index` and `phase_description` (both never top-level), `n_phases`, `config_id`
(teleop: `teleop/teleop`). After a `replan`: `plan_generation` (0 is the first plan).

A conjoined robot leg gets the first phase it covers; its `phase_description` joins the covered phases' with `; `.

## `hitl.json`

Written when phase planning is on and the trial has a plan: into the primary leg, then beside the merged episode.
A trial stopped at the label prompt keeps it in its primary leg in `eval/`.

| key | what it is |
|---|---|
| `instruction`, `trajectory_id`, `planner`, `human_executor` | Task as planned, trial id, planner, `hitl.human_executor`. |
| `outcome`, `excluded` | `success`, `failure`, `excluded` or `aborted` ([how a trial ends](METHOD.md#how-a-trial-ends)); `excluded` is `outcome == "excluded"`. |
| `failure_stage` | `invention`, `tamp_planning`, `tamp_execution`, `verification`, `human_policy` or `null`. Always the loop's, whatever the label. |
| `filed_under` | `success`, `failure`, or `null` while in `eval/`. A relabel rewrites it. |
| `outcome_reason` | What was wrong, when the loop ended the trial. |
| `overruled` | After `tandem traj relabel --force` files a settled trial under `success/` (`outcome` then reads `success`): the loop's `{outcome, excluded, failure_stage, by}` ([rules](USAGE.md#reviewing-and-exporting)). |
| `plan_generation`, `leg_plan_generations` | 0, plus one per `replan`. After a replan: leg directory → its plan. |
| `superseded_plans[]` | Plans a `replan` replaced, oldest first, as they stood (no filing keys), plus `plan_generation` and `superseded_because`; `[]` if none. |
| `initially_true` | Invented atoms true on the first image (`hitl.classify_initial`), else `[]`. |
| `provenance` | Who produced each part, in words: `phases_and_their_order`, `phase_sub_goals`, `invented_predicates`, `human_instructions` (the model), `robot_phases` (the planner), `who_does_what` (tandem), `human_steps` (the intended executor; `phases[k].carried_out`: who did). `human_operators`, `robot_operators`: `{by, signatures}`, as `Name(param: type)`. |
| `handed_over_phases` | Robot phases the planner couldn't plan, given to a person (`on_robot_phase_failure: teleop`). |
| `phase_index` | How far the trial got (the phase count if the plan finished). |

**`specification`**, the proposal's reading of the instruction:

- `invented_predicates[]`: `{name, types, instructions}`; `instructions` is the classifier sentence, `{0}`, `{1}`,
  … for arguments.
- `human_operators[]`: `{phase, name, args, signature, instance, preconditions, add_effects, delete_effects}` per
  human phase with an operator (load with `tandem.planning.structs.HumanOperator.from_json`).
- `surfaces[]`, `movables[]`: the task's object types.
- `unrepresented[]`: `{clause, reason}` per clause the plan leaves out. `coverage[]`: `{clause, phase}` per clause
  (`phase: -1` if left out).

**`checks`**:

- `human_preconditions`, `human_effects`, `tamp_preconditions`, `tamp_effects`, `plan_effects`,
  `verify_enforced`, `precondition_enforced`, `verify_final_phase`: the `hitl` switches as configured.
- `initial_state_classified`, `plan_effects_rechecked`, `plan_effects_warning`: whether the first image was
  classified, whether the [contract check](METHOD.md#the-contract-check) re-ran on it, and its finding (or `null`).
- `unchecked_phases[]`: accepted without a verdict. `unrecorded_human_phases[]`: human phases with no leg, so no
  segment.

**`phases[k]`**: `index`, `executor`, `description`, `atoms`, `planned_by`, plus:

- **Robot:** `goal` (atoms given to the planner, in its spelling), `goal_description`; once planned,
  `planning_seconds`, `plan_reused`, `task_plan` (operator sequence, if reported), `covers_phases` (if one plan
  covered several).
- **Human:** `instructions`, `operator`; once run, `carried_out[]`:
  `{attempt, carried_out_by, status, leg_recorded, n_frames}` per attempt. `carried_out_by` is the executor, or
  `by_hand` (`status: null`) if "done" came with no executor
  ([status values](ADDING_A_HUMAN_EXECUTOR.md#humanphaseresult)).
- **Handed to a person:** a human phase without `operator`, plus `proposed_executor: "robot"`,
  `handed_over_because`, `instructions_by: "tandem"`.
- **Any:** `unchecked` (why) if its check could not run or had nothing a camera can judge.

**`verifications[k]`**: `{atom, statement, holds, expected, satisfied, role, reason, phase}` per camera verdict,
failing ones included. `role` is `effect`, `effect (deleted)` or `precondition`.

- `holds` is what the model saw, `expected` what the plan wanted. **Read `satisfied`** (`holds == expected`): for
  a delete effect, `holds: true` is the failure.
- Only the attempt that settled a phase is kept (for an excluded trial, the verdicts it was excluded on). Earlier
  attempts: `human_phase_verified` events.

## `vlm/`

Every model query of the trial. Written when phase planning is on and `hitl.save_vlm_io` is true (default): into
the session directory, then copied into the episode on filing. Per query:

- `NNN_<label>_input.png`: the image sent, e.g. `002_classify-LidClosed-red_bin_input.png`.
- `NNN_<label>_output.png`: that image above the answer, marked REJECTED if tandem rejected it or CACHED if
  replayed from `hitl.cache_path`.
- An `index.jsonl` line: `seq`, `label`, `attempt`, `model`, `input_image`, `output_image`, `rejected` (why, or
  `null`), `cached`, `prompt`, `response`.

Labels: `task plan` (the proposal), `classify <atom>` (a camera check), `detect objects` (`tandem plan` without
`--object`). A retry adds `_attempt<N>`. `NNN` continues from the directory's highest, so nothing is overwritten.

## The events file

A session appends one JSON line per event to `~/.local/state/tandem/sessions/<profile>/<session id>/events.jsonl`,
e.g. `{"event": "rollout_saved", "ts": 1790270037.40, "dir": "…", "n_frames": 12}`. A sidecar planner writes to
it too. The web UI streams each on `/api/sessions/{id}/stream`, adding `"type": "event"` and `"at": <ts>`.

From the phase loop:

| event | payload |
|---|---|
| `rollout_start` | `dir`, `phase_index`, `n_phases`; `movables`, `return_home` if passed to the planner |
| `rollout_saved` | `dir`, `n_frames` |
| `phase_complete` | `phase_index` (the next), `n_phases`: more phases remain |
| `phase_plan_failed` | `reason`, `policy` (`on_robot_phase_failure`), `phase_index` (`null` with phase planning off) |
| `instruction_not_fully_represented` | `unrepresented`: `[{clause, reason}]` |
| `awaiting_human_phase` | `description`, `instructions`, `expected` (in words), `expected_atoms` / `expected_deleted_atoms` (must / must no longer hold after), `phase_index`, `n_phases`, `is_last_phase`, `operator` |
| `human_phase_refused` | `phase_index`, `executor`, `reason`: while recording, an answer with no leg was turned down |
| `human_phase_by_hand` | `phase_index`, `attempt`: "done" with no executor was accepted; no leg |
| `human_leg_ended` | `executor`, `status`, `n_frames`, `dir`, `phase_index` |
| `human_phase_verified` | `phase_index`, `attempt`, `ok` (`true`, `false`, or `null` if not judged, with `skipped` or `unchecked` saying why), `verdicts` |
| `phase_preconditions_checked` | `phase_index`, `description`, `what` (`human phase` or `robot leg`), `ok`, `enforced`, `verdicts`; `unchecked` if the check could not run |
| `phase_effects_checked` | The same, for a robot leg's effects (`enforced` always `false`) |
| `teleop_handoff_warning` | `message`: the planner could not release the robot |
| `trial_outcome` | `trajectory_id`, `outcome`, `failure_stage`, `excluded`, `reason`, `phase_index`: only when the loop ended the trial |

From the session. *Trial summary* is `trajectory_id`, `outcome`, `excluded`, `filed_under`, `failure_stage`,
`reason`, `labeled`.

| event | payload |
|---|---|
| `session_start`, `session_end`, `awaiting_task` | None |
| `awaiting_label` | `dir` |
| `labeled`, `trial_excluded` | `dir`, trial summary; `labeled` adds `success` |
| `trial_filed` | `dir`, trial summary: failed part-way or aborted, filed unlabeled |
| `trial_unlabeled` | `dir`, trial summary: the session stopped at the label prompt |
| `rollout_discarded` | Trial summary: nothing was recorded |
| `rollout_aborted` | Nothing after a preempt; `reason` if the session is stopping; `error` if the attempt failed |
| `teleop_switch_pending` | None: the operator asked for the arm, handed over at the next plan-step boundary |
| `teleop_handoff_start`, `awaiting_teleop_resume` (`trajectory_id`), `teleop_handoff_done` | A hand-off, from release to return |

### Session summary

A session's live state: in the web UI, from `GET /api/sessions/{id}`, and in `session-<id>.json`. Common fields
(there are more):

| field | what it is |
|---|---|
| `id`, `profile`, `task` | The session. |
| `state`, `error`, `end_reason` | Where it is (`warming`, `rolling`, `awaiting_task`, `awaiting_human_phase`, `teleop_handoff`, `awaiting_label`, `stopped`, `failed`, …), and how it ended. |
| `labeled`, `success`, `excluded`, `aborted`, `target` | Trial counts, and the target (`--episodes`, else `task.target_episodes`). |
| `last_trial`, `rollouts[]` | The last attempt's trial summary; every trial so far (`{dir, id, started_at, n_frames, status, success}`). |
| `human_phase`, `phase_progress` | The phase on screen (`description`, `instructions`, `expected`, `attempt`, `missing`, …); `[phase index, n_phases]`. |

## Logs and session files

| path | what it holds |
|---|---|
| `~/.local/state/tandem/logs/session-<id>.json` | Written when `tandem collect` ends (not web UI sessions): `summary` (the [session summary](#session-summary)), `logs` (the last 4,000 log lines). |
| `~/.local/state/tandem/sessions/<profile>/<id>/events.jsonl` | [The events file](#the-events-file). |
| `…/<id>/vlm/<trajectory id>/` | Each trial's [model I/O](#vlm); the only copy if the trial recorded nothing. |
| `…/<id>/perception/<leg dir>/` | Perception passes that recorded nothing, moved out of `eval/`. TiPToP's `metadata.json` there has the planning result and failure reason. |
| `…/<id>/teleop/leg-*/teleop-events.jsonl` | Teleop driver events, one directory per hand-off. |
| `~/.local/state/tandem/logs/export.log` | `tandem export lerobot`'s log, appended per run. |
| `~/.local/state/tandem/logs/runtime-build-<YYYYMMDD-HHMMSS>.log` | One per runtime build (`tandem planners install`, `tandem runtime build`, `tandem init`). |
| `~/tandem-data/exports/<owner>/<name>/` | Exported datasets (or `--out DIR`; [exporting](USAGE.md#exporting)). |
| `<profile>/trajectories/.merge-<trajectory id>/` | A merge in progress. If one is left behind, the next `tandem traj merge` says how to move its legs back. |

A trial that failed at `invention` usually has no legs, so only its events file and session `vlm/` and
`perception/` directories remain.
