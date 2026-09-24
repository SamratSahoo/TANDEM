# The method, as implemented

This is the TANDEM paper's method (*TANDEM: Task and Motion Planning with As-Needed Demonstrations
for Efficient Vision-Language-Action Model Fine-tuning*, Sec. IV) mapped onto this package's code.
It covers where each part of the method lives, the trial loop as the code runs it, what every
`hitl:` setting changes, and what a trial leaves on disk. It also lists every place the code departs
from the paper on purpose, and why.

The short version: **a model decides what each phase must achieve and in what order. The planner
decides how the robot's phases are carried out. tandem decides who does what, hands the arm over,
checks the person's work, and records the result.** Nothing about phases, invented predicates or
hand-offs lives inside a planner. That is what lets another task and motion planner be swapped in
without changing any of it ([ADDING_A_PLANNER.md](ADDING_A_PLANNER.md)).

---

## 1. Where each part of the paper lives

| Paper | Code | Notes |
|---|---|---|
| Base domain M₀ = ⟨Ψ₀, Ω₀⟩ | `Capabilities` (`planners/base.py`), declared by each planner. TiPToP's is `planners/tiptop/capabilities.py`. | Ψ₀ is `goal_predicates` (TiPToP: `On`, `Holding`, `HandEmpty`). Ω₀ is `robot_operators` (TiPToP: `Pick`, `Place`), which is provenance only: nothing searches over it. |
| Symbolic state s_t = abs(o₀:t; Ψ) | The planner's own `perceive()` → `SceneView` for Ψ₀; the VLM classifier g_ψ for invented predicates. | tandem never builds a full symbolic state. It reasons only over what the plan establishes (§4). |
| Phase φ_k = (γ_k, e_k) | `planning/structs.py` `Phase(executor, description, atoms, instructions, operator)` | `atoms` is γ_k; `executor` is `"robot"` or `"human"`. |
| Predicate invention Ψ_Δ, and g_ψ(o, b) | `structs.VLMPredicate`, whose `instructions` sentence is the whole definition. It is judged by `grounding.classify` with the Appendix-B classifier prompt (`prompts.classifier_prompt`). | The same sentence is the classifier and what the operator is told to bring about. |
| Magic operators Ω_Δ (name, args, pre, add, del) | `structs.HumanOperator`, built by `proposal._build_operator`. | Grounded, not lifted: the proposal has already fixed which objects it acts on. Required on every human phase. |
| The VLM as task planner over M_H | `prompts.plan_prompt` (Appendix B, with planner-specific slots from `Capabilities.prompt_fragments`), then `proposal.propose_plan` | A golden test pins TiPToP's rendered prompt byte for byte (`tests/golden/plan_prompt_tiptop.txt`). |
| Repair of a rejected proposal, at most 3 attempts | `llm.query_json`: the rejection is fed back, up to `hitl.max_attempts` (3) answers in all. | A transient API failure (429, 5xx, timeout) is retried underneath with backoff and never uses up an attempt. |
| Operators checked against each other | `contracts.check_plan_effects`, run inside the repair loop by `proposal.check_plan` | Together with `feasibility.check_robot_phases`: every robot phase asks only for something some robot operator can achieve. |
| Autonomous execution | `TampBackend.plan` / `execute`, called from `core/phase_loop.py` `_run_robot_phase` | "An interface for specifying subgoals and executing the resulting plans", and nothing more. |
| Human execution, π_ωΔ | `executors/` (`TeleopExecutor` ships), called from `phase_loop._lend_arm` | The loop releases the robot and cameras before the executor runs and takes them back after. |
| Re-perception after every phase | `phase_loop._perceive` before every robot leg; a fresh `capture_frame` for every check | See §7 for consecutive robot phases. |
| Verification of a human phase | `grounding.verify_effects`: add effects must hold, delete effects must not | Preconditions can be checked first (`check_human_preconditions`). |
| A failed trial is terminated and excluded | `phase_loop._verification_failed` → `session._file_without_label` → `episodes.write_phase_record` | Filed under `failure/` with `excluded: true`. No label prompt. |
| τ = ((τ₁, φ₁), …, (τ_N, φ_N)) | `core/merge.py` joins the legs; each leg's `phase_index` becomes `segments[k].phase_index`. `hitl.json` holds the φ_k. | One episode per trial. |
| DATAFARM alignment | TiPToP's `planner.options.tamp` (the VAE manifold cost, `blend_mode: vae`), set by the `paper` preset | A planner option, not part of the method's core. |
| Failure taxonomy (Fig. 4) | `plan.OUTCOMES`, `plan.FAILURE_STAGES` | Stages: `invention`, `tamp_planning`, `tamp_execution`, `human_policy`, plus `verification`. |

---

## 2. A proposal, from instruction to plan

`planning/plan.py` `build_plan(image, instruction, object_names, table_name, cfg, caps)` returns the
`PhasePlan`, or raises `ProposalError` when no proposal validates within `max_attempts`:

1. **Prompt.** `prompts.plan_prompt` renders the Appendix-B prompt. The planner's goal language is
   rendered from `Capabilities.goal_predicates`, and the robot is described by its one-sentence
   `robot_description`. The planner-specific paragraphs come from `prompt_fragments` (the slots are
   `prompts.PROMPT_SLOTS`), so the template itself names no planner predicate.
2. **Answer.** The model returns JSON (`PLAN_SCHEMA`): `new_predicates`, an ordered list of
   `phases` (each human phase with an `operator`), `coverage` (clause → phase index) and
   `unrepresented` (clause, reason).
3. **Parse and validate** (`proposal.parse_plan_response`). Every atom is grounded against the
   detected objects, and each object's type is fixed once for the whole task (`SceneTypes`). A robot
   phase may use only the planner's goal predicates. An invented predicate must be used, with the
   same argument types everywhere, and may not reuse a name the planner reserves. Its `instructions`
   may use only `{0}`, `{1}`, … as placeholders, within its arity; anything else in braces (`{box}`,
   `{0.x}`, a stray `{`) is refused, since `describe` could not render it (`symbols.validate_template`).
   Each human phase's operator
   must add something, may not add and delete the same atom, must not delete its own phase's
   `atoms`, and must add every one of them.
4. **Check the plan as a whole** (`proposal.check_plan`). Every robot phase must be achievable
   (`feasibility.check_robot_phases`). Every robot leg, cut the way `robot_run` will cut it, must
   give the planner a goal: a leg of nothing but planner-supplied atoms (TiPToP's `HandEmpty`) is
   refused (`feasibility.robot_leg_without_a_goal`). No phase may add two atoms that claim one
   exclusive slot, such as `On(toy, box)` and `On(toy, shelf)` (`contracts.exclusive_conflicts`). With
   `check_plan_effects` on, no phase may require an atom an earlier phase made false
   (`contracts.check_plan_effects`), by a delete effect or by putting the same object somewhere else.
5. **Repair.** Any rejection in steps 3–4 is a `ProposalError` written for the model. It goes back
   with the model's own answer, and the model tries again, up to `max_attempts` answers. The last
   rejection ends the trial at stage `invention`.
6. **Measure the start** (only with `classify_initial`). Every invented atom the plan mentions is
   classified on the first image (`initially_true`), and the contract check runs again against it.
   What that second check finds is recorded (`checks.plan_effects_warning`), not refused, because
   the model is out of the loop by then.

The accepted plan is a `PhasePlan`: the phases, a cursor (`index`), and everything `hitl.json` will
say about the trial, filled in as it runs.

`tandem plan` and `tandem.plan_task()` run exactly steps 1–6 on a photo, with no planner built and
no robot. With no object labels given, a vision model names the objects first
(`planning/objects.py`).

---

## 3. The trial loop

`core/phase_loop.py` `PhaseLoop.run(task, instruction, trajectory_id)`. The session mints one
`trajectory_id` per attempt, and every leg of that attempt is stamped with it.

```
plan = None;  first = True
until the trial is over:
    check for a preempt, or the session stopping     # either ends the attempt here, as aborted
    if plan is finished: stop

    if plan and the next phase is a person's:
        take_turn()                                    # no perception before a person's step
    else:
        leg_dir = a new directory under <profile>/trajectories/eval/
        scene = backend.perceive(task_hint=task, save_dir=leg_dir,
                                 reset_arm=first,       # park the arm only before the first leg
                                 open_gripper=True if a person had the arm last)
        # perceive raised (a sidecar that crashed, or was stopped for not answering) → failure at
        # "tamp_planning"; the session warms the planner again before the next task
        if plan is None:
            if hitl.enabled is false:  goal = scene.detected_goal       # the planner's own goal
            else:  plan = build_plan(scene.rgb_path, task, scene.object_labels, ...)   # §2
                   # no image, or no usable proposal → failure at "invention"
        else:
            rebind object labels that drifted since the plan was made
            # a label the plan needs that cannot be matched → failure at "tamp_planning"
        take_turn(scene, leg_dir)
    first = False

take_turn:
    the current phase is a person's → human_phase(phase)   # a pending hand-off request is its answer
    the operator asked for the arm → lend it to teleop with no phase attached; nothing advances
    otherwise                     → robot_leg(scene)
```

A planner verb (`perceive`, `plan`, `execute`) or a human executor that raises ends the trial at its
stage (`tamp_planning`, `tamp_execution`, `human_policy`) before the error goes on up, so the record
never reads as a trial that ran.

**A robot leg** (`_run_robot_phase`):

```
run  = plan.robot_run()           # this phase, or it and the consecutive robot phases after it (§7)
goal = to_goal_atoms(the union of their atoms)     # in the planner's wire spelling; a leg whose atoms
                                                   # are all planner-supplied (HandEmpty) is refused at
                                                   # proposal (§2); if one gets here → "invention"
[check_tamp_preconditions] put plan.expected_now() to the camera on this pass's image;
                           unmet and precondition_enforced → end at "verification"
result = backend.plan(scene.scene_id, goal, surfaces=plan.surfaces(), save_dir=leg_dir,
                      movables=plan.robot_movables()   if caps.supports_movable_restriction,
                      return_home=plan.is_last_leg()   if caps.supports_return_home)
if plan raised:    end: failure at "tamp_planning"     # not a goal it could not plan: no teleop, no replan
if not result.ok:  on_robot_phase_failure
                     abort  → end: failure at "tamp_planning"
                     teleop → this phase becomes a person's (atoms only, no operator), recorded as
                              handed over with the planner's reason; run it now
                     replan → drop the plan, remember why; the next pass proposes again with every
                              failure so far fed back (at most max_attempts re-plans, then as abort)
record result.task_plan against every phase in run
execution = backend.execute(result.plan_handle,
                            LegSpec(trajectory_id, instruction, phase_index, n_phases,
                                    phase_description, record), save_dir=leg_dir,
                            should_stop=preempt or stop requested   if caps.supports_cooperative_stop)
if execute raised:  end: failure at "tamp_execution"      # never advances
if execution.stopped_early:  end: aborted (the operator's stop)     # never advances
if not execution.ok:  end: failure at "tamp_execution"      # never advances
[check_tamp_effects] put the run's add effects to the camera; recorded, never enforced
plan.advance()   # past every phase in run
```

**A human phase** (`_run_human_phase`):

```
[check_human_preconditions] once, on a fresh frame: put the operator's preconditions to the camera;
                            unmet and precondition_enforced → end at "verification"
for attempt in 1 .. 1 + verify_retries:
    show the phase: its instructions, and what will be checked afterwards
    wait for the operator's answer:
        abort  → end: aborted
        teleop → release the arm; human_executor.run(request, leg); count the leg; take the arm back
                   (a forced stop before the leg starts → it never starts, status "aborted")
                   status "aborted"                        → end: failure at "human_policy"
                   the session is stopping                 → end: aborted, unchecked
                   nothing recorded, while recording        → refused unless allow_unrecorded_human_phase;
                                                              ask again (no retry spent)
        done   → the step was done by hand; while recording, refused the same way
    record how the attempt was carried out (phases[k].carried_out): the executor or by hand, and
    whether it left a leg
    status "ended_by_operator"                             → accept, recorded as unchecked
    check_human_effects off, or the last phase with verify_final_phase off
                                                           → accept, recorded as not checked
    no effect a camera can settle (HandEmpty, Holding)     → accept, no frame, recorded as unchecked
    check = verify_effects(a fresh frame from verification_camera)
        the check could not run (camera or model error)    → accept, recorded as unchecked
        the planner has no capture_frame at all            → end at "verification" (the session
                                                              refuses such a planner at start)
        passed, or verify_enforced off                     → record the verdicts; advance
        failed with a retry left                           → tell the operator what is still missing
        failed with none left                              → record the failing verdicts;
              on_verification_failure: exclude → end: excluded at "verification"
                                       label   → end: failure at "verification" (the label decides)
```

**After the loop** (`core/session.py` `_run_task`), on every way out, a preempt included:

- **Legs on disk, and the loop ended the trial itself:** filed under `failure/` with no label prompt,
  since there is nothing left for a label to decide. Excluded (`trial_excluded`), and aborted --
  the operator gave up, preempted, or stopped the session (`trial_filed`) -- are not counted towards
  `--episodes`; a failure at `tamp_planning`, `tamp_execution` or `human_policy` (`trial_filed`) is,
  as a failure label was. The one exception is a check left to the label on purpose
  (`on_verification_failure: label`), which is labeled.
- **Legs on disk, otherwise** (the plan ran to the end): the operator labels it success or failure
  (`awaiting_label`, then `labeled`). A session stopped at that prompt leaves the legs unmerged in
  `eval/`, with `hitl.json` written into the primary leg (`trial_unlabeled`); `tandem traj merge
  <trajectory id> --status success|failure` files it later.
- **Nothing recorded:** nothing is filed (`rollout_discarded`). A trial that failed at `invention`
  has no legs, so its only trace is the events file, plus the model's inputs and replies in the
  session's scratch directory.

Filing writes `hitl.json` and `vlm/` into the leg the trial is filed under, then merges the legs into
one episode and writes them again beside it (`episodes.merge_trajectory`). The merge runs in the
background, because a merge of several GB of video must not hold up the next task; a session that
is ending waits for it (bounded), and the record written first survives a merge that never finishes.

### How a trial ends

`TrialOutcome.outcome` is set only when the loop ends a trial itself. `episodes.trial_outcome`
resolves what `hitl.json` says:

| `outcome` | set by | label asked |
|---|---|---|
| `excluded` | the loop: a check that stops the trial failed, with `on_verification_failure: exclude` | **no**; filed under `failure/` |
| `aborted` | the loop: the operator abandoned a human phase, preempted the attempt, or stopped the session, with the plan unfinished | **no**; filed under `failure/` |
| `failure` at `tamp_planning`, `tamp_execution`, `human_policy`, `invention` | the loop: the plan did not finish (Fig. 4) | **no**; filed under `failure/` |
| `failure` at `verification` | the loop, with `on_verification_failure: label` | yes: the label decides, and may overrule the check |
| `success` / `failure` | the operator's label, for a trial whose plan ran to the end | yes |

An unfinished plan is never a success. `failure_stage` is always the loop's. A trial the loop
settled keeps its outcome whatever directory it is later moved to: `tandem traj relabel <id>
success` refuses it without `--force`, a forced one is recorded under `overruled`, and
`tandem export lerobot` skips a settled record it finds under `success/`.

After a `replan`, `hitl.json` is written from the last plan, and every plan it replaced is kept as
`superseded_plans` (its phases, planner records and verdicts, `plan_generation`, and
`superseded_because`). The merged `segments[]` then carry `plan_generation` beside `phase_index`,
and `leg_plan_generations` maps each leg directory to its plan, so every recorded leg's phase and
verdicts can still be found.

---

## 4. The plan-time contract check

`planning/contracts.py`. The check walks the phase list symbolically. Each phase leaves
`after = (before − displaced − delete_effects) ∪ add_effects`. Its unmet preconditions are
`preconditions − before`.

- **A robot phase has no operator.** Its add effects are its `atoms`. It has no preconditions and
  no declared delete effects. What the planner needs first is the planner's business, and it
  re-perceives before every leg.
- **Displacement.** A placement ends the previous placement of the same object even though no
  delete effect says so. `Capabilities.exclusive_arguments` declares which argument is exclusive:
  TiPToP's `{"On": 0}` means `On(toy, box)` retracts `On(toy, table)`.
- **Sound, not complete.** The starting workspace is unknown unless `classify_initial` measured it.
  So a plan is refused only where the plan itself is the reason: a precondition an earlier phase
  deleted and nothing restored. With the start measured, a precondition over an invented predicate
  that no phase establishes and the scene does not show is also provably unmet. Everything else
  passes.
- **Wasted robot moves** (`wasted_robot_move`) are logged, never refused. Two consecutive robot
  phases that move the same object usually mean a step the robot cannot do was written as a
  pick-and-place. The same shape is also a supported continuation, so refusing it would make that
  continuation unreachable.

`expected_before` (what earlier phases should have left true) is the robot-side precondition set
that `check_tamp_preconditions` puts to the camera.

---

## 5. Every `hitl:` setting, and what it changes

These live in a profile's `hitl:` block (`core/profiles.py` `HitlSpec`), which is resolved to
`planning/config.py` `PlanningConfig`. `tandem profile create --preset paper` sets all of them to the
paper's values. With phase planning on, those values are also the defaults, apart from `enabled`
(`tests/test_presets.py` pins this).

| key | default | effect on the loop |
|---|---|---|
| `enabled` | `false` | Off: no proposal, no phases; each attempt is one leg toward the planner's own goal (`SceneView.detected_goal`), and nothing in `tandem.planning` runs. On: §2–§3. |
| `proposal_model` | `gemini-2.5-pro` | The model that segments the task and invents predicates and operators. |
| `vlm_model` | `gemini-2.5-flash` | The model that answers each per-atom check (g_ψ), and names objects for `tandem plan`. |
| `max_attempts` | `3` | Answers the model may give per proposal (the rejection is fed back each time), per classifier question, and the most re-plans one trial gets under `replan`. Must be ≥ 1. |
| `classify_initial` | `false` | Classify every invented atom the plan mentions on the first image, then re-run the contract check against it (recorded, not refused). One model call per atom. |
| `verify_retries` | `1` | Extra goes at a human phase whose effect check failed. The operator is told what is still missing. |
| `verify_enforced` | `true` | Off: a failed effect check is recorded and the trial carries on. |
| `on_verification_failure` | `exclude` | `exclude`: a trial whose check still fails is ended, filed under `failure/` with `excluded: true`, and not labeled. `label`: it ends as a failure and the operator's label decides. |
| `verify_final_phase` | `true` | Off: the last phase, when it is a person's, is not put to the camera; the label covers it. |
| `check_human_effects` | `true` | Put a human phase's add effects (must hold) and delete effects (must not) to the camera. |
| `check_human_preconditions` | `false` | Put a human phase's preconditions to the camera once, before the hand-off. |
| `check_tamp_preconditions` | `false` | Before a robot leg, put what earlier phases should have left true (`expected_now`) to the camera, on that pass's perception image. |
| `check_tamp_effects` | `false` | After a robot leg, put its add effects to the camera. Observational only: never stops a trial. |
| `precondition_enforced` | `false` | An unmet precondition (either kind) ends the trial at `verification` instead of being recorded and waved through. |
| `check_plan_effects` | `true` | Run the contract check (§4) inside the repair loop. Symbolic: no model call. |
| `save_vlm_io` | `true` | Keep every image sent to a model and its reply, rejected attempts included, and file them as `vlm/` beside the episode. |
| `cache_path` | `null` | A SQLite cache of *proposal* responses only, keyed on model, prompt and a noise-robust image hash. A relative path is beside the profile. Never applied to a check, and bypassed on a re-plan. |
| `on_robot_phase_failure` | `abort` | What happens when the planner cannot plan a robot phase: `abort` (failure at `tamp_planning`), `teleop` (a person does it, checked against its atoms), or `replan` (propose again with the failure fed back). A leg that was planned but failed to *execute* always ends the trial at `tamp_execution`. |
| `conjoin_robot_phases` | `true` | Hand consecutive robot phases to the planner as one goal where that is sound (§7). Off: each robot phase is its own leg with its own perception pass. |
| `human_executor` | `teleop` | Who carries out a human phase, by registered name ([ADDING_A_HUMAN_EXECUTOR.md](ADDING_A_HUMAN_EXECUTOR.md)). Checked against the executor registry when the profile loads for collection; a profile naming one this machine lacks can still be browsed, exported and repaired. |
| `human_executor_options` | `{}` | Each executor's own settings, keyed by its name; the executor built for a phase receives its own block as `ExecutorContext.options`, checked by its `validate_options` when it is installed. |
| `allow_unrecorded_human_phase` | `false` | While recording, accept "done" for a human phase that was not carried out through the executor, and accept an executor leg that recorded nothing. Without it that step would be missing from a demonstration that looks complete. With recording off this is always allowed. |
| `verification_camera` | `external` | Which camera the verification frame comes from: `external`, `hand` or `perception`. |

Only `check_human_effects` among the camera checks is on by default. Each check costs one model call
per checkable atom, with the arm parked. A human phase's effects are the only evidence the step
happened at all. Its preconditions were already proved symbolically by the contract check (and the
paper's experiments ran with them off). A robot leg's effects are something the arm reports better
than a third-person camera sees.

Which atoms a camera can settle: every invented predicate, and only those of the planner's own
predicates listed in `Capabilities.checkable_predicates`. For TiPToP that is `On`; `Holding` and
`HandEmpty` are left out because the gripper is usually out of a third-person shot.

---

## 6. What a trial leaves on disk

After the merge, one episode directory, under `success/` or `failure/`:

```
<profile>/trajectories/<status>/<timestamp>/
├── external_cam.mp4  external_cam_2.mp4  hand_cam.mp4   every leg's clips, joined
├── robot_state.npz        every leg's per-frame arrays, joined
├── _meta.json             lineage, timing, and segments[]: which frames were which leg and phase
├── hitl.json              the phase plan and everything that happened to it
├── vlm/                   every image sent to a model, what it said, and index.jsonl
└── segments/NN_<source>_<timestamp>/    each raw leg, as it was recorded
```

A trial with a single leg is not merged. Its leg directory is the episode.

### `_meta.json` of a merged episode

The primary leg's `_meta.json` (the first planner leg) with these keys set:

- `trajectory_id`: the id every leg carries.
- `segment_source`: `null`.
- `source`: `"trajectory"`.
- `video_aligned`: `true`.
- `n_frames`, `fps`, `total_video_frames`.
- `record_start` / `record_stop`: from the first and last leg.
- `cameras`: dataset key → clip file name.
- `n_phases`: kept only when every leg that states it agrees, and all of them carried out phases of
  one plan.
- `segments[]`: one per leg, with `source` (`tamp`, `teleop` or `policy`), `timestamp`,
  `n_frames`, `n_video_frames`, `video_start` / `video_stop` (seconds into the merged clip) and
  `record_start` / `record_stop`. A leg that recorded its phase also has `phase_index`, `n_phases`
  and `phase_description`, and after a `replan`, `plan_generation`: which plan that phase index
  belongs to (0 for the first; absent means the trial had one plan). The phase keys are per segment:
  they are removed from the top level because they describe one leg.
- `cameras_dropped`, `proportional_fallback_legs`, `frames_trimmed`, `legs_skipped`: what the
  merge had to work around.
- `action_convention` and `action_notes`: only when some leg carried `action_joint_velocity`, which
  the merged npz then carries for every frame.

A conjoined robot leg is stamped with the first phase it covers, and its `phase_description` joins
the descriptions of every phase it covers. `hitl.json` says which phases it covered
(`phases[k].covers_phases`).

### `hitl.json`

Written by `PhasePlan.to_json()` plus `episodes.write_phase_record`. Only when phase planning is on
and there is a plan.

| key | what it is |
|---|---|
| `instruction` | The task, as it was planned. |
| `trajectory_id` | The lineage id. |
| `planner` | The planner's name (`Capabilities.name`). |
| `human_executor` | `hitl.human_executor`. |
| `outcome` | `success`, `failure`, `excluded` or `aborted` (§3). |
| `failure_stage` | `invention`, `tamp_planning`, `tamp_execution`, `verification`, `human_policy` or `null`. Always the loop's word, whatever the label said. |
| `excluded` | `true` exactly when `outcome` is `excluded`. |
| `filed_under` | `success` or `failure`: where the episode was filed. `null` for a trial the session stopped before it was labeled. A relabel rewrites it. |
| `outcome_reason` | Present when the loop ended the trial: what was wrong, in words. |
| `overruled` | Present when `tandem traj relabel --force` filed a trial the loop settled under `success/`: `{outcome, excluded, failure_stage, by}` as the loop had them. |
| `plan_generation` | Which plan this record is: 0, or one more for each `replan`. |
| `superseded_plans[]` | Every plan a `replan` replaced, oldest first: each a record of this same shape as it stood when it was given up, with `plan_generation` and `superseded_because`. |
| `leg_plan_generations` | After a `replan`: leg directory name → the `plan_generation` it carried out a phase of. |
| `specification` | What the proposal made of the instruction (below). |
| `initially_true` | Invented atoms measured true on the first image (`classify_initial`); `[]` otherwise. |
| `provenance` | Who produced what: `phases_and_their_order`, `phase_sub_goals`, `invented_predicates` and `human_instructions` (the VLM); `human_operators` and `robot_operators`, each `{by, signatures}` in one spelling, `Name(param: type)` (a declared signature is read and re-rendered, whatever its spacing); `robot_phases` (the planner); `who_does_what` (tandem); `human_steps` (the executor that would carry a step out; whether it did is `phases[k].carried_out`). |
| `checks` | Which checks ran (below). |
| `phases[]` | One record per phase (below). |
| `handed_over_phases` | Robot phases handed to a person because the planner could not plan them (`on_robot_phase_failure: teleop`); `[]` otherwise. |
| `phase_index` | How far through the plan the trial got. Equal to the number of phases when it finished. |
| `verifications[]` | Every verdict recorded, failing ones included (below). |

`specification`:

- `invented_predicates[]`: `{name, types, instructions}`. `instructions` is the classifier sentence,
  with `{0}`, `{1}`, … for the arguments.
- `human_operators[]`: `{phase, name, args, signature, instance, preconditions, add_effects,
  delete_effects}`, one per human phase that has an operator. `HumanOperator.from_json` reads one
  back.
- `surfaces[]`, `movables[]`: the object types, fixed once for the task.
- `unrepresented[]`: `{clause, reason}` for each part of the instruction the plan knowingly leaves
  out.
- `coverage[]`: `{clause, phase}`, the model's clause-by-clause account. `phase` is `-1` for a
  clause that is also in `unrepresented`.

`checks`:

- `human_preconditions`, `human_effects`, `tamp_preconditions`, `tamp_effects`, `plan_effects`,
  `verify_enforced`, `precondition_enforced`, `verify_final_phase`: the switches, as configured.
- `initial_state_classified`: whether `classify_initial` measured the start.
- `plan_effects_rechecked`: whether the contract check then ran again against that measurement.
- `plan_effects_warning`: what that re-check found, or `null`.
- `unchecked_phases[]`: phases accepted without a verdict, because the check could not run or because
  nothing it was to check is something a camera can settle (a human phase that adds only
  `HandEmpty()`, say).
- `unrecorded_human_phases[]`: human phases carried out with no leg on disk in any attempt (staged by
  hand, or an executor that recorded nothing), so the merged episode has no segment for them.

`phases[k]`: `index`, `executor`, `description`, `atoms` and `planned_by`, plus:

- **a robot phase:** `goal` (the literal atoms the planner was handed, in its wire spelling),
  `goal_description`, and once it has been planned, `planning_seconds`, `plan_reused`, `task_plan`
  (the planner's operator sequence, e.g. `["Pick(bread)", "Place(bread, plate)"]`) and
  `covers_phases` (when one plan covered several phases).
- **a human phase:** `instructions` and `operator` (absent for a robot phase handed to a person
  under `on_robot_phase_failure: teleop`), and once it has run, `carried_out[]`: one
  `{attempt, carried_out_by, status, leg_recorded, n_frames}` per attempt, where `carried_out_by` is
  the executor's name or `by_hand` (answered "done" with no executor; `status` is then `null`).
- **a robot phase handed to a person** (`on_robot_phase_failure: teleop`): recorded as the human
  phase it became, plus `proposed_executor: "robot"`, `handed_over_because` (the planner's failure
  reason) and `instructions_by: "tandem"`; its `planned_by` says tandem handed it over.
- **either:** `unchecked`, when a check of that phase could not run, or had nothing a camera can
  settle, and why.

`verifications[k]`: `{atom, statement, holds, expected, satisfied, role, reason, phase}`.

- `holds` is what the model saw. `expected` is what the plan wanted.
- `satisfied` is `holds == expected`, and it is the field to read: for a delete effect, `holds:
  true` is the failure.
- `role` is `effect`, `effect (deleted)` or `precondition`.
- Only the attempt that settled a phase is recorded. An excluded trial's record therefore carries
  the verdicts it was excluded on.

An example, from a trial on the test suite's toy planner (items dropped into bins), trimmed:

```json
{
  "instruction": "put the duck in the red bin, close its lid, put the ball in the blue bin, then put the cube in the blue bin",
  "trajectory_id": "7777777777777777",
  "planner": "toy",
  "human_executor": "teleop",
  "outcome": "success", "failure_stage": null, "excluded": false, "filed_under": "success",
  "specification": {
    "invented_predicates": [
      {"name": "LidClosed", "types": ["container"], "instructions": "the lid of {0} is shut, covering its opening"},
      {"name": "LidOpen", "types": ["container"], "instructions": "the lid of {0} is open, so the inside of {0} is visible"}
    ],
    "human_operators": [
      {"phase": 1, "name": "Close", "args": ["red_bin"], "signature": "Close(x0: container)", "instance": "Close(red_bin)",
       "preconditions": ["InBin(duck, red_bin)", "LidOpen(red_bin)"],
       "add_effects": ["LidClosed(red_bin)"], "delete_effects": ["LidOpen(red_bin)"]}
    ],
    "surfaces": ["blue_bin", "floor", "red_bin"], "movables": ["ball", "cube", "duck", "sponge"],
    "unrepresented": [],
    "coverage": [{"clause": "put the duck in the red bin", "phase": 0}, "..."]
  },
  "phases": [
    {"index": 0, "executor": "robot", "description": "put the duck in the red bin", "atoms": ["InBin(duck, red_bin)"],
     "planned_by": "vlm (order and sub-goal); toy (how)",
     "goal": [{"predicate": "in_bin", "args": ["duck", "red_bin"]}], "goal_description": ["duck is inside red_bin"],
     "planning_seconds": 0.01, "plan_reused": false, "task_plan": ["Drop(duck, red_bin)"]},
    {"index": 1, "executor": "human", "description": "close the red bin's lid", "atoms": ["LidClosed(red_bin)"],
     "instructions": "Close the lid of the red_bin, with the duck inside it.", "operator": {"...": "as above"},
     "planned_by": "vlm"},
    "..."
  ],
  "phase_index": 4,
  "verifications": [
    {"atom": "LidClosed(red_bin)", "statement": "the lid of red_bin is shut, covering its opening",
     "holds": true, "expected": true, "satisfied": true, "role": "effect", "reason": "...", "phase": 1},
    {"atom": "LidOpen(red_bin)", "statement": "the lid of red_bin is open, so the inside of red_bin is visible",
     "holds": false, "expected": false, "satisfied": true, "role": "effect (deleted)", "reason": "...", "phase": 1}
  ]
}
```

### `vlm/`

Copied from the session's scratch directory
(`~/.local/state/tandem/sessions/<profile>/<session>/vlm/<trajectory_id>/`) when the episode is
filed. Each model query is three things:

- `NNN_<label>_input.png`: the image exactly as it was sent;
- `NNN_<label>_output.png`: that image above what the model answered, marked REJECTED when the
  answer was refused and CACHED when it was replayed from `cache_path` rather than asked for;
- a line in `index.jsonl`: `seq`, `label`, `attempt`, `model`, `input_image`, `output_image`,
  `rejected`, `cached`, `prompt`, `response`.

The labels are `task plan` for the proposal, `classify <atom>` for each check, and
`detect objects` for `tandem plan` without `--object`. `NNN` counts on from whatever the directory
already holds, so a retried check, a second proposal, or a second `--save-vlm-io` run into the same
directory never overwrites an earlier image.

### The events file

A session appends one JSON line per event to `~/.local/state/tandem/sessions/<profile>/<session>/events.jsonl`:
`{"event": <name>, "ts": <epoch seconds>, ...payload}`. The web UI receives the same records over its
event stream, as `{"type": "event", ..., "at": <ts>}`. A sidecar planner's own events are appended
to the same file.

From the phase loop:

| event | payload |
|---|---|
| `rollout_start` | `dir`, `phase_index`, `n_phases`, plus `movables` and `return_home` when passed to the planner |
| `rollout_saved` | `dir`, `n_frames` |
| `phase_complete` | `phase_index`, `n_phases`: a step finished and more remain |
| `phase_plan_failed` | `reason`, `policy` (`on_robot_phase_failure`), `phase_index` (`null` with phase planning off) |
| `instruction_not_fully_represented` | `unrepresented`: `[{clause, reason}]` |
| `awaiting_human_phase` | `description`, `instructions`, `expected` (in words), `expected_atoms` (must hold after), `expected_deleted_atoms` (must no longer hold), `phase_index`, `n_phases`, `is_last_phase`, `operator` |
| `human_phase_refused` | `phase_index`, `executor`, `reason`: a step with no recorded leg, while recording |
| `human_phase_by_hand` | `phase_index`, `attempt`: a step answered "done" with no executor was accepted, so it has no leg |
| `human_leg_ended` | `executor`, `status`, `n_frames`, `dir`, `phase_index` |
| `human_phase_verified` | `phase_index`, `attempt`, `ok` (`true`, `false`, or `null` when not checked), `verdicts`, plus `skipped` or `unchecked` saying why when `ok` is `null` |
| `phase_preconditions_checked` | `phase_index`, `description`, `what` (`human phase` or `robot leg`), `ok`, `enforced`, `verdicts`, plus `unchecked` |
| `phase_effects_checked` | the same, for a robot leg's effects (`enforced` is always `false`) |
| `teleop_handoff_warning` | `message`: the planner could not release the robot |
| `trial_outcome` | `trajectory_id`, `outcome`, `failure_stage`, `excluded`, `reason`, `phase_index`: only when the loop ended the trial itself |

From the session:

| event | payload |
|---|---|
| `session_start`, `session_end` | |
| `awaiting_task` | |
| `awaiting_label` | `dir` |
| `labeled` | `dir`, `success`, and the trial summary: `trajectory_id`, `outcome`, `excluded`, `filed_under`, `failure_stage`, `reason`, `labeled` |
| `trial_excluded` | `dir` and the trial summary |
| `trial_filed` | `dir` and the trial summary: failed part-way or aborted, filed without a label |
| `trial_unlabeled` | `dir` and the trial summary: the session stopped at the label prompt |
| `rollout_discarded` | the trial summary: nothing was recorded |
| `rollout_aborted` | `error` when the attempt failed rather than being preempted |
| `teleop_switch_pending` | the operator asked for the arm; honoured at the next phase boundary |
| `teleop_handoff_start`, `awaiting_teleop_resume` (`trajectory_id`), `teleop_handoff_done` | a hand-off, from release to return |

`tandem.core.session.Session.summary()` (the web UI's state, and `GET /api/sessions/{id}`) carries
the live view:

- `state`, `labeled`, `success`, `excluded`, `aborted`, `target`;
- `last_trial`: how the last attempt ended;
- `human_phase`: the step being shown, with `executor` and `by_hand`, which says whether "done"
  would be accepted;
- `phase_progress`, `unrepresented`;
- `human_executor` (name, readiness, what is unmet), and `teleop_available` (whether the teleop
  executor is ready here).

---

## 7. Where the code departs from the paper, and why

**Consecutive robot phases are planned as one goal (`conjoin_robot_phases: true`).** The paper
re-perceives after every phase. Planning two robot phases together gives one plan, one continuous
motion, and no perception pass in the middle for object labels to drift across. It is only done
where it is sound (`feasibility.conjoinable_run`):

- the planner must declare `initial_state_is_clean`, meaning every goal is planned from the same
  clean state, so nothing symbolic orders two robot phases;
- the run stops at a phase that moves an object an earlier phase in the run already moved, when the
  planner declares `one_pick_per_object`. cuTAMP picks each object at most once per plan, so
  `On(toy, table)` and `On(toy, shelf)` together would be unsatisfiable.

Set it to `false` for the paper's rule read strictly. The reference implementation (LJ1356/tiptop
cf75a68) conjoined too.

**The last human phase is verified (`verify_final_phase: true`).** The reference implementation
left the task's last phase to the operator's label. The paper verifies every human phase, and with
`on_verification_failure: exclude` the label no longer settles that question, because a trial that
fails a check is never offered for labeling.

**Only a human phase's effects go to a camera by default.** The paper checks a human phase's effects
after execution, and says preconditions *can* be checked before. The defaults match the paper's
experiments:

- human preconditions off;
- robot-leg checks off (the paper does not describe them);
- the contract check on (it is symbolic and free).

Every check is a switch.

**A planning failure ends the trial (`on_robot_phase_failure: abort`).** This is how the paper
counts it (Fig. 4, "Planning Failure"). `teleop` turns a robot phase into a human one, so a dataset
collected that way understates the human effort it cost. `replan` feeds the planner's failure back
to the proposer. **An execution failure always ends the trial.** The arm is somewhere no plan put
it, and every later phase was planned against a scene that no longer exists.

**Human operators are grounded, and the order is the phase list.** The paper describes the task
planner composing operators. Here the VLM is the task planner over the extended domain (as Sec. IV-D
says). It writes an ordered phase list, and each human phase's operator is its contract: the camera
checks it, and the contract check reasons over it. Nothing searches over it. Robot phases have no
operator of their own; the planner chooses Ω₀ operators itself.

**A check that cannot run does not fail the step.** A camera read or a model call that errors is
recorded as *unchecked* (`checks.unchecked_phases`, `phases[k].unchecked`), and the step goes ahead.
One unreachable service should not cost a demonstration, and a phase with no verdicts must not read
as one that passed. The same goes for a check with nothing a camera can settle (a step that only
empties the gripper): no frame is taken, and it is recorded as unchecked, not passed.

**Other additions the paper does not describe**, each gated on the planner declaring support:

- `movables`: only the objects some robot phase moves may be picked. A person's tool stays in the
  scene as an obstacle.
- `return_home`: only the task's last leg drives the arm home.
- `open_gripper`: the gripper is opened before the first perception pass after a human phase.

**Label drift.** Perception names objects afresh each pass. A plan whose objects were renamed is
rebound to the new labels when the match is unambiguous (`drift.match_drifted_names`). Otherwise the
trial ends at `tamp_planning`.

**Not implemented here:** the offline prompt evaluation, a learned-policy human executor (the
HITL-TAMP baseline; the executor registry is where it would plug in), and exporting the DROID
joint-velocity action (the merge carries `action_joint_velocity` when a leg recorded it; the export
still reads `cmd_joint_velocity`).

### Known limitations

- A trial that is re-planned (`replan`) keeps the legs it already recorded, stamped with the phase
  indices of the plan they were recorded under. Read them with `plan_generation` (in `segments[]`,
  or `leg_plan_generations` for legs never merged) against `superseded_plans`.
- `segments[]` stamps a conjoined robot leg with its first phase only. Read `hitl.json`
  `covers_phases` for the rest.
