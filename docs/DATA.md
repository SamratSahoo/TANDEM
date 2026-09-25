# Data on disk

This page describes what a trial leaves on disk, and where tandem writes its logs. Paths are the Linux defaults
([moving them](CONFIGURATION.md#tandem-settings-and-credentials)). For the terms used here, see
[README.md](README.md#terms).

## Reading an episode

```python
import json
from pathlib import Path

import numpy as np

ep = Path("~/tandem-data/trajectories/my-task/success/2026-09-25_14-03-12").expanduser()
meta = json.loads((ep / "_meta.json").read_text())
state = np.load(ep / "robot_state.npz")
print(meta["instruction"], meta["n_frames"], "frames at", meta["fps"], "fps")
print(state["joint_position"].shape)          # (F, 7)

# Who did each phase, and did its camera checks pass?
hitl = json.loads((ep / "hitl.json").read_text())
for phase in hitl["phases"]:
    verdicts = [v["satisfied"] for v in hitl["verifications"] if v.get("phase") == phase["index"]]
    print(phase["index"], phase["executor"], phase["description"], verdicts)

# The frame shown 4.2 s into the videos (merged episodes have video_time).
frame = int(np.searchsorted(state["video_time"], 4.2))
```

## Episode layout

Each trial is one directory, under `success/` or `failure/`:

```
~/tandem-data/trajectories/<profile>/<status>/<YYYY-MM-DD_HH-MM-SS>/
├── external_cam.mp4  external_cam_2.mp4  hand_cam.mp4   exterior 1, exterior 2 (optional), wrist
├── robot_state.npz        per-frame arrays
├── _meta.json             lineage, timing, and which leg and phase each frame belongs to
├── hitl.json              the phase record
├── vlm/                   model queries
├── tiptop_plan.json, …    the primary leg's other files
└── segments/NN_<source>_<leg dir>/   the raw legs, in order
```

The videos and arrays join every leg of the trial in order. The episode takes the
[primary leg](README.md#terms)'s directory name, its `_meta.json`, and its other files except `*.log`. For TiPToP
those are `tiptop_plan.json`, `metadata.json`, `rgb.png` and `perception/`.

A trial with a single leg is not merged: the leg is the episode. The format is the same as hitl-tamp-vla's. To
review and export episodes, see [USAGE.md](USAGE.md#reviewing-and-exporting).

## `robot_state.npz`

The file has one row per frame. The per-leg arrays are listed in
[the recording contract](ADDING_A_PLANNER.md#the-recording-contract). An episode may also have:

| array | shape | what it is |
|---|---|---|
| `video_time` | `[F]` float64 | Merged episodes only: the frame's time in the joined videos, in seconds. |
| `action_joint_velocity` | `[F,7]` | The DROID joint-velocity action, if any leg recorded it (TiPToP does). For legs without it, teleop and policy legs copy `cmd_joint_velocity`, and planner legs get `5 × (cmd_joint_position − joint_position)`. |

To match frames to video in a merged episode, use `video_time`. Don't use `frame_time`, which keeps the gaps
between legs. In a single leg, frame *i* sits `(frame_time[i] − record_start) / (record_stop − record_start)` of
the way through each video.

The arrays the export uses are listed in [USAGE.md](USAGE.md#exporting).

## `_meta.json`

```json
{
  "trajectory_id": "3f9c0a1b7d2e4c55",
  "planner": "tiptop",
  "instruction": "place the bread inside the box",
  "n_frames": 912, "fps": 15,
  "cameras": {"exterior_image_1_left": "external_cam.mp4", "wrist_image_left": "hand_cam.mp4"},
  "segments": [
    {"source": "tamp",   "timestamp": "2026-09-25_14-03-12", "phase_index": 0, "video_start": 0.0,  "video_stop": 31.4},
    {"source": "teleop", "timestamp": "2026-09-25_14-03-50", "phase_index": 1, "video_start": 31.4, "video_stop": 60.8}
  ]
}
```

A merged episode's `_meta.json` is the primary leg's, with the keys below set. A single-leg episode keeps its
own ([per-leg keys](ADDING_A_PLANNER.md#the-recording-contract)).

| key | what it is |
|---|---|
| `trajectory_id` | Shared by every leg of the trial. |
| `planner`, `instruction` | The planner and the task (the language label). The session adds them if missing, and adds `planner` only on a planner's leg. `tandem traj open` picks its viewer by `planner`. |
| `source`, `segment_source`, `video_aligned` | `"trajectory"`, `null`, `true`. |
| `n_frames`, `fps`, `total_video_frames` | Rows in `robot_state.npz`, the frame rate, and the frames in each joined video. |
| `record_start`, `record_stop` | Epoch seconds: the first leg's start and the last leg's stop. |
| `cameras` | Dataset key → video: `exterior_image_1_left` → `external_cam.mp4`, `exterior_image_2_left` → `external_cam_2.mp4`, `wrist_image_left` → `hand_cam.mp4`. |
| `n_phases` | Set only if every leg that states it agrees and all ran one plan. |
| `segments[]` | One entry per leg, in order (below). |
| `cameras_dropped`, `legs_skipped` | What was left out: videos only some legs had, and legs without `robot_state.npz`. |
| `frames_trimmed` | Leg → camera → trailing frames cut to even out that leg's videos. |
| `proportional_fallback_legs` | Legs with no usable recording window. Their frames are spread evenly over their videos. |
| `action_convention`, `action_notes` | Set with `action_joint_velocity`: `"droid_joint_velocity"`, and leg → a note for each leg whose action was recomputed or could not be. |

Each `segments[k]` has:

- `source` (`tamp`, `teleop` or `policy`), `timestamp` (the leg's directory name), `n_frames` and `n_video_frames`.
- `video_start` and `video_stop` (seconds into the joined videos), and `record_start` and `record_stop`.
- If the leg stated them: `phase_index`, `phase_description`, `n_phases` and `config_id` (teleop: `teleop/teleop`).
  `phase_index` and `phase_description` never appear at the top level.
- After a `replan`: `plan_generation` (0 is the first plan).

A conjoined robot leg gets the first phase it covers. Its `phase_description` joins the covered phases'
descriptions with `; `.

## `hitl.json`

This file is written when phase planning is on and the trial has a plan. It goes into the primary leg, then
beside the merged episode. A trial stopped at the label prompt keeps it in its primary leg in `eval/`.

A trimmed example, for a trial where the robot moves the bread and a person closes the lid:

```json
{
  "instruction": "put the bread in the box and close it",
  "trajectory_id": "3f9c0a1b7d2e4c55",
  "planner": "tiptop",
  "human_executor": "teleop",
  "outcome": "success", "excluded": false, "failure_stage": null, "filed_under": "success",
  "specification": {
    "invented_predicates": [{"name": "IsClosed", "types": ["surface"], "instructions": "Is the lid of {0} closed?"}],
    "unrepresented": [],
    "coverage": [{"clause": "put the bread in the box", "phase": 0}, {"clause": "close it", "phase": 1}]
  },
  "phases": [
    {"index": 0, "executor": "robot", "description": "Put the bread in the box",
     "atoms": ["On(bread, box)"], "goal": [{"predicate": "on", "args": ["bread", "box"]}],
     "planning_seconds": 41.2},
    {"index": 1, "executor": "human", "description": "Close the box",
     "atoms": ["IsClosed(box)"], "instructions": "Close the box's lid.",
     "carried_out": [{"attempt": 1, "carried_out_by": "teleop", "status": "done", "leg_recorded": true, "n_frames": 440}]}
  ],
  "verifications": [
    {"atom": "IsClosed(box)", "statement": "Is the lid of box closed?", "holds": true, "expected": true,
     "satisfied": true, "role": "effect", "reason": "The lid is down.", "phase": 1}
  ],
  "phase_index": 2
}
```

### Top-level keys

| key | what it is |
|---|---|
| `instruction`, `trajectory_id`, `planner`, `human_executor` | The task as planned, the trial id, the planner, and `hitl.human_executor`. |
| `outcome`, `excluded` | `success`, `failure`, `excluded` or `aborted`. `excluded` is `outcome == "excluded"`. |
| `failure_stage` | `invention`, `tamp_planning`, `tamp_execution`, `verification`, `human_policy` or `null`. It is always the loop's, whatever the label. |
| `filed_under` | `success`, `failure`, or `null` while in `eval/`. A relabel rewrites it. |
| `outcome_reason` | What was wrong, when the loop ended the trial. |
| `overruled` | Set after `tandem traj relabel --force` files a settled trial under `success/` (`outcome` then reads `success`). It holds the loop's `{outcome, excluded, failure_stage, by}` ([rules](USAGE.md#reviewing-and-exporting)). |
| `plan_generation`, `leg_plan_generations` | 0, plus one per `replan`. After a replan, leg directory → its plan. |
| `superseded_plans[]` | Plans a `replan` replaced, oldest first, as they stood (without filing keys), plus `plan_generation` and `superseded_because`. `[]` if none. |
| `initially_true` | Invented atoms true on the first image (`hitl.classify_initial`), else `[]`. |
| `provenance` | Who produced each part, in words (below). |
| `handed_over_phases` | Robot phases the planner couldn't plan, given to a person (`on_robot_phase_failure: teleop`). |
| `phase_index` | How far the trial got. It equals the phase count if the plan finished. |

`provenance` has `phases_and_their_order`, `phase_sub_goals`, `invented_predicates` and `human_instructions` (the
model), `robot_phases` (the planner), `who_does_what` (tandem), and `human_steps` (the intended executor; each
phase's `carried_out` says who actually did it). `human_operators` and `robot_operators` are `{by, signatures}`,
with signatures written `Name(param: type)`.

### `specification`

The proposal's reading of the instruction.

| key | what it is |
|---|---|
| `invented_predicates[]` | `{name, types, instructions}`. `instructions` is the classifier sentence, with `{0}`, `{1}`, … for arguments. |
| `human_operators[]` | `{phase, name, args, signature, instance, preconditions, add_effects, delete_effects}` for each human phase with an operator. Load one with `tandem.planning.structs.HumanOperator.from_json`. |
| `surfaces[]`, `movables[]` | The task's object types. |
| `unrepresented[]` | `{clause, reason}` for each clause the plan leaves out. |
| `coverage[]` | `{clause, phase}` for each clause. `phase` is `-1` if left out. |

### `checks`

| key | what it is |
|---|---|
| `human_preconditions`, `human_effects`, `tamp_preconditions`, `tamp_effects`, `plan_effects`, `verify_enforced`, `precondition_enforced`, `verify_final_phase` | The `hitl` switches as configured. |
| `initial_state_classified` | Whether the first image was classified. |
| `plan_effects_rechecked`, `plan_effects_warning` | Whether the contract check re-ran on the first image, and its finding (or `null`). |
| `unchecked_phases[]` | Phases accepted without a verdict. |
| `unrecorded_human_phases[]` | Human phases with no leg, and so no segment. |

### `phases[k]`

Every phase has `index`, `executor`, `description`, `atoms` and `planned_by`. The rest depends on the kind:

| kind | extra keys |
|---|---|
| Robot | `goal` (the atoms given to the planner, in its spelling) and `goal_description`. Once planned: `planning_seconds`, `plan_reused`, `task_plan` (the operator sequence, if reported), and `covers_phases` (if one plan covered several). |
| Human | `instructions` and `operator`. Once run, `carried_out[]` has `{attempt, carried_out_by, status, leg_recorded, n_frames}` per attempt. `carried_out_by` is the executor, or `by_hand` (with `status: null`) if "done" came with no executor ([status values](ADDING_A_HUMAN_EXECUTOR.md#humanphaseresult)). |
| Handed to a person | A human phase without `operator`, plus `proposed_executor: "robot"`, `handed_over_because` and `instructions_by: "tandem"`. |
| Any | `unchecked` (why), if its check could not run or had nothing a camera can judge. |

### `verifications[k]`

One entry per camera verdict, failing ones included: `{atom, statement, holds, expected, satisfied, role, reason,
phase}`. `role` is `effect`, `effect (deleted)` or `precondition`.

**Read `satisfied`.** `holds` is what the model saw and `expected` is what the plan wanted, so
`satisfied == (holds == expected)`. For a delete effect, `holds: true` is the failure.

Only the attempt that settled a phase is kept. For an excluded trial, those are the verdicts it was excluded on.
Earlier attempts are in the `human_phase_verified` events.

## `vlm/`

```
vlm/
├── 001_task-plan_input.png
├── 001_task-plan_output.png
├── 002_classify-LidClosed-red_bin_input.png
├── 002_classify-LidClosed-red_bin_output.png
└── index.jsonl
```

This directory holds every model query of the trial. It is written when phase planning is on and
`hitl.save_vlm_io` is true (the default): first into the session directory, then copied into the episode when it is
filed. Each query leaves:

- `NNN_<label>_input.png`: the image sent.
- `NNN_<label>_output.png`: that image above the answer. It is marked REJECTED if tandem rejected the answer, or
  CACHED if it was replayed from `hitl.cache_path`.
- An `index.jsonl` line with `seq`, `label`, `attempt`, `model`, `input_image`, `output_image`, `rejected` (why, or
  `null`), `cached`, `prompt` and `response`.

The labels are `task plan` (the proposal), `classify <atom>` (a camera check), and `detect objects` (`tandem plan`
without `--object`). A retry adds `_attempt<N>`. `NNN` continues from the directory's highest number, so nothing
is overwritten.

## The events file

```json
{"event": "rollout_saved", "ts": 1790270037.40, "dir": "…/eval/2026-09-25_14-03-12", "n_frames": 12}
{"event": "human_phase_verified", "ts": 1790270101.12, "phase_index": 1, "attempt": 1, "ok": true, "verdicts": […]}
{"event": "trial_outcome", "ts": 1790270110.87, "outcome": "excluded", "failure_stage": "verification", "reason": "…"}
```

A session appends one JSON line per event to
`~/.local/state/tandem/sessions/<profile>/<session id>/events.jsonl`. A sidecar planner writes to it too. The web
UI streams each event on `/api/sessions/{id}/stream`, adding `"type": "event"` and `"at": <ts>`.

### From the phase loop

| event | payload |
|---|---|
| `rollout_start` | `dir`, `phase_index`, `n_phases`; also `movables` and `return_home` if they were passed to the planner |
| `rollout_saved` | `dir`, `n_frames` |
| `phase_complete` | `phase_index` (the next one), `n_phases`: more phases remain |
| `phase_plan_failed` | `reason`, `policy` (`on_robot_phase_failure`), `phase_index` (`null` with phase planning off) |
| `instruction_not_fully_represented` | `unrepresented`: `[{clause, reason}]` |
| `awaiting_human_phase` | `description`, `instructions`, `expected` (in words), `expected_atoms` and `expected_deleted_atoms` (must, or must no longer, hold after), `phase_index`, `n_phases`, `is_last_phase`, `operator` |
| `human_phase_refused` | `phase_index`, `executor`, `reason`: while recording, an answer with no leg was turned down |
| `human_phase_by_hand` | `phase_index`, `attempt`: "done" with no executor was accepted, with no leg |
| `human_leg_ended` | `executor`, `status`, `n_frames`, `dir`, `phase_index` |
| `human_phase_verified` | `phase_index`, `attempt`, `ok` (`true`, `false`, or `null` if not judged, with `skipped` or `unchecked` saying why), `verdicts` |
| `phase_preconditions_checked` | `phase_index`, `description`, `what` (`human phase` or `robot leg`), `ok`, `enforced`, `verdicts`; `unchecked` if the check could not run |
| `phase_effects_checked` | The same, for a robot leg's effects (`enforced` is always `false`) |
| `teleop_handoff_warning` | `message`: the planner could not release the robot |
| `trial_outcome` | `trajectory_id`, `outcome`, `failure_stage`, `excluded`, `reason`, `phase_index`: only when the loop ended the trial |

### From the session

Below, *trial summary* means `trajectory_id`, `outcome`, `excluded`, `filed_under`, `failure_stage`, `reason` and
`labeled`.

| event | payload |
|---|---|
| `session_start`, `session_end`, `awaiting_task` | None |
| `awaiting_label` | `dir` |
| `labeled`, `trial_excluded` | `dir` and the trial summary; `labeled` adds `success` |
| `trial_filed` | `dir` and the trial summary: the trial failed part-way or was aborted, and was filed unlabeled |
| `trial_unlabeled` | `dir` and the trial summary: the session stopped at the label prompt |
| `rollout_discarded` | The trial summary: nothing was recorded |
| `rollout_aborted` | Nothing after a preempt; `reason` if the session is stopping; `error` if the attempt failed |
| `teleop_switch_pending` | None: the operator asked for the arm, which is handed over at the next plan-step boundary |
| `teleop_handoff_start`, `awaiting_teleop_resume` (`trajectory_id`), `teleop_handoff_done` | A hand-off, from release to return |

### Session summary

A session's live state. The web UI gets it from `GET /api/sessions/{id}`, and `tandem collect` saves it in
`session-<id>.json`. The common fields are below; there are more.

| field | what it is |
|---|---|
| `id`, `profile`, `task` | The session. |
| `state`, `error`, `end_reason` | Where it is (`warming`, `rolling`, `awaiting_task`, `awaiting_human_phase`, `teleop_handoff`, `awaiting_label`, `stopped`, `failed`, …), and how it ended. |
| `labeled`, `success`, `excluded`, `aborted`, `target` | Trial counts, and the target (`--episodes`, else `task.target_episodes`). |
| `last_trial`, `rollouts[]` | The last attempt's trial summary, and every trial so far (`{dir, id, started_at, n_frames, status, success}`). |
| `human_phase`, `phase_progress` | The phase on screen (`description`, `instructions`, `expected`, `attempt`, `missing`, …), and `[phase index, n_phases]`. |

## Logs and session files

| path | what it holds |
|---|---|
| `~/.local/state/tandem/logs/session-<id>.json` | Written when `tandem collect` ends (not for web UI sessions): `summary` (the [session summary](#session-summary)) and `logs` (the last 4,000 log lines). |
| `~/.local/state/tandem/sessions/<profile>/<id>/events.jsonl` | [The events file](#the-events-file). |
| `…/<id>/vlm/<trajectory id>/` | Each trial's [model queries](#vlm). This is the only copy if the trial recorded nothing. |
| `…/<id>/perception/<leg dir>/` | Perception passes that recorded nothing, moved out of `eval/`. TiPToP's `metadata.json` there has the planning result and failure reason. |
| `…/<id>/teleop/leg-*/teleop-events.jsonl` | Teleop driver events, one directory per hand-off. |
| `~/.local/state/tandem/logs/export.log` | `tandem export lerobot`'s log, appended on each run. |
| `~/.local/state/tandem/logs/runtime-build-<YYYYMMDD-HHMMSS>.log` | One per runtime build (`tandem planners install`, `tandem runtime build`, `tandem init`). |
| `~/tandem-data/exports/<owner>/<name>/` | Exported datasets, or `--out DIR` ([exporting](USAGE.md#exporting)). |
| `~/tandem-data/trajectories/<profile>/.merge-<trajectory id>/` | A merge in progress. If one is left behind, the next `tandem traj merge` says how to move its legs back. |

A trial that failed at `invention` usually has no legs. Only its events file and its session `vlm/` and
`perception/` directories remain.
