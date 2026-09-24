# The method, as implemented

The TANDEM paper's method (*TANDEM: Task and Motion Planning with As-Needed Demonstrations for
Efficient Vision-Language-Action Model Fine-tuning*, Sec. IV) mapped onto this package's code. Read it
to find where a part of the method lives, or to know exactly what a trial does.

**A model decides what each phase must achieve and in what order. The planner decides how robot
phases are carried out. tandem decides who does what, hands the arm over, checks the person's work,
and records the result.** Nothing about phases lives inside a planner.

On this page: [Paper to code](#paper-to-code) · [From instruction to plan](#from-instruction-to-plan) ·
[The trial loop](#the-trial-loop) · [How a trial ends](#how-a-trial-ends) ·
[The contract check](#the-contract-check) · [Departures from the paper](#departures-from-the-paper) ·
[Package layout](#package-layout)

Code paths are relative to `src/tandem/`. Terms are defined in [the docs index](README.md#terms).

## Paper to code

| Paper | Code | Note |
|---|---|---|
| Base domain M₀ = ⟨Ψ₀, Ω₀⟩ | `Capabilities` (`planners/base.py`), declared by each planner. TiPToP's is `planners/tiptop/capabilities.py`. | Ψ₀ is `goal_predicates` (TiPToP: `On`, `Holding`, `HandEmpty`). Ω₀ is `robot_operators` (`Pick`, `Place`): recorded, never searched. |
| Symbolic state s_t = abs(o₀:t; Ψ) | The planner's `perceive()` → `SceneView` for Ψ₀; the classifier g_ψ for invented predicates | tandem tracks only what the plan establishes ([contract check](#the-contract-check)). |
| Phase φ_k = (γ_k, e_k) | `planning/structs.py` `Phase(executor, description, atoms, instructions, operator)` | `atoms` is γ_k. `executor` is `"robot"` or `"human"`. |
| Predicate invention Ψ_Δ, classifier g_ψ(o, b) | `structs.VLMPredicate`, judged by `grounding.classify` with the Appendix-B classifier prompt (`prompts.classifier_prompt`) | Its `instructions` sentence is both the classifier's question and what the person is told to bring about. |
| Magic operators Ω_Δ (name, args, pre, add, del) | `structs.HumanOperator`, built by `proposal._build_operator` | Grounded, not lifted. Required on every human phase. |
| The VLM as task planner over M_H | `prompts.plan_prompt` (Appendix B, with slots from `Capabilities.prompt_fragments`), then `proposal.propose_plan` | `tests/golden/plan_prompt_tiptop.txt` pins TiPToP's rendered prompt byte for byte. |
| Repair of a rejected proposal, at most 3 attempts | `llm.query_json` feeds each rejection back, up to `hitl.max_attempts` (3) answers | A transient API failure (timeout, rate limit, server error) is retried with backoff and doesn't use an attempt. |
| Operators checked against each other | `contracts.check_plan_effects`, run inside the repair loop by `proposal.check_plan` | Alongside `feasibility.check_robot_phases`: some robot operator must be able to achieve each robot phase. |
| Autonomous execution | `TampBackend.plan` / `execute`, called from `core/phase_loop.py` `_run_robot_phase` | [The protocol](ADDING_A_PLANNER.md#the-protocol). |
| Human execution π_ωΔ | `executors/` (`TeleopExecutor` ships), called from `phase_loop._lend_arm` | The loop releases the robot and cameras before the executor runs and takes them back after ([custody](ADDING_A_HUMAN_EXECUTOR.md#custody)). |
| Re-perception after every phase | `phase_loop._perceive` before every robot leg; a fresh `capture_frame` for each human-phase check | Consecutive robot phases can share one pass ([departures](#departures-from-the-paper)). |
| Verification of a human phase | `grounding.verify_effects`: add effects must hold, delete effects must not | Preconditions can be checked first (`check_human_preconditions`). |
| A failed trial is terminated and excluded | `phase_loop._verification_failed` → `session._file_without_label` → `episodes.write_phase_record` | Filed under `failure/` with `excluded: true`, with no label prompt. |
| τ = ((τ₁, φ₁), …, (τ_N, φ_N)) | `core/merge.py` joins the legs. Each leg's `phase_index` becomes `segments[k].phase_index`, and `hitl.json` holds the φ_k. | One episode per trial ([layout](DATA.md#episode-layout)). |
| [DATAFARM](README.md#terms) alignment | TiPToP's `planner.options.tamp` (the VAE manifold cost, `blend_mode: vae`), set by the [`paper` preset](CONFIGURATION.md#presets) | A planner option, not part of the method. |
| Failure taxonomy (Fig. 4) | `plan.OUTCOMES`, `plan.FAILURE_STAGES` | Stages `invention`, `tamp_planning`, `tamp_execution`, `human_policy`, plus `verification` ([How a trial ends](#how-a-trial-ends)). |

## From instruction to plan

`planning/plan.py` `build_plan` returns a `PhasePlan`, or raises `ProposalError` when no proposal
validates within `hitl.max_attempts`.

1. **Prompt.** `prompts.plan_prompt` renders the Appendix-B prompt. The goal language comes from
   `Capabilities.goal_predicates`, the robot from its one-sentence `robot_description`, and the
   planner-specific paragraphs from `prompt_fragments` (slots: `prompts.PROMPT_SLOTS`). The template
   itself names no planner predicate.
2. **Answer.** The model returns JSON (`PLAN_SCHEMA`): `new_predicates`, an ordered list of `phases`
   (each human phase with an `operator`), `coverage` (clause → phase index) and `unrepresented`
   (clause, reason).
3. **Parse and validate** (`proposal.parse_plan_response`):
   - Every phase has atoms. Every human phase has `instructions`. A robot phase declares no `operator`.
   - Every atom is grounded against the detected objects. Each object is a surface or a movable for
     the whole task (`SceneTypes`).
   - A robot phase uses only the planner's goal predicates.
   - An invented predicate must be used, with the same argument types everywhere, and may not reuse a
     name the planner reserves. Its `instructions` may contain only `{0}`, `{1}`, … within its arity
     (`symbols.validate_template`).
   - A human phase's operator must add something, may not add and delete the same atom, must not
     delete its phase's `atoms`, and must add every one of them.
4. **Check the plan as a whole** (`proposal.check_plan`):
   - Every robot phase is achievable (`feasibility.check_robot_phases`).
   - Every robot leg, cut the way the loop will cut it, gives the planner a goal. A leg of only
     planner-supplied atoms (TiPToP's `HandEmpty`) is refused (`feasibility.robot_leg_without_a_goal`).
   - No phase adds two atoms that claim one exclusive slot, such as `On(toy, box)` and
     `On(toy, shelf)` (`contracts.exclusive_conflicts`).
   - With `check_plan_effects` on, the [contract check](#the-contract-check) passes.
5. **Repair.** A rejection in step 3 or 4 is a `ProposalError` written for the model. It goes back with
   the model's own answer, up to `max_attempts` answers. The last rejection ends the trial at
   `invention`.
6. **Measure the start** (only with `classify_initial`). Every invented atom the plan mentions is
   classified on the first image (`initially_true`). With `check_plan_effects` on, the contract check
   then runs again against it, and what it finds is recorded (`checks.plan_effects_warning`), not refused.

The accepted `PhasePlan` holds the phases, a cursor (`index`), and the trial's record, which becomes
[`hitl.json`](DATA.md#hitljson).

`tandem plan` and `tandem.plan_task()` run these six steps on a photo, with no planner built and no
robot ([USAGE.md](USAGE.md#planning-from-a-photo)). Without `--object`, a vision model names the
objects first (`planning/objects.py`).

## The trial loop

`core/phase_loop.py` `PhaseLoop.run`. The session mints one trajectory id per trial; every leg carries it.

```
plan = None;  first = True
until the trial is over:
    check for a preempt or a session stop            → aborted
    if plan is finished: stop
    if plan and the next phase is a human phase:
        take_turn()                                  # no perception pass before a human phase
    else:
        leg_dir = a new directory under <profile>/trajectories/eval/
        scene = backend.perceive(task_hint=task, save_dir=leg_dir,
                                 reset_arm=first,     # park the arm only before the first pass
                                 open_gripper=True if a human phase ran last)
        perceive raised                              → failure at tamp_planning
        if plan is None:
            phase planning off: goal = scene.detected_goal
            phase planning on:  plan = build_plan(scene.rgb_path, task, scene.object_labels, …)
                                no image, or no valid proposal → failure at invention
        else:
            rebind object labels that drifted since the plan was made
            a label the next leg needs can't be matched      → failure at tamp_planning
        take_turn(scene, leg_dir)
    first = False

take_turn:
    the current phase is a human phase → human_phase(phase)  # `t`, or a hand-off already requested, runs the executor
    the operator asked for the arm     → lend it to teleop with no phase attached; nothing advances
    otherwise                          → robot_leg(scene)
```

**A robot leg** (`_run_robot_phase`):

```
run  = plan.robot_run()          # this phase, plus the robot phases conjoined with it
goal = to_goal_atoms(the union of their atoms)       # empty → failure at invention
[check_tamp_preconditions] put plan.expected_now() to the camera, on this pass's image;
                           unmet and precondition_enforced → verification failure (as below)
result = backend.plan(scene.scene_id, goal, surfaces=plan.surfaces(), save_dir=leg_dir,
                      movables=plan.robot_movables()  if caps.supports_movable_restriction,
                      return_home=plan.is_last_leg()  if caps.supports_return_home)
plan raised        → failure at tamp_planning        # no teleop, no replan
not result.ok      → on_robot_phase_failure:
                       abort  → failure at tamp_planning
                       teleop → this phase becomes a human phase (atoms only, no operator),
                                recorded as handed over with the planner's reason; run it now
                       replan → drop the plan; the next pass proposes again with every failure
                                so far fed back (at most max_attempts re-plans, then as abort)
record result.task_plan against every phase in run
execution = backend.execute(result.plan_handle, LegSpec(trajectory_id, instruction, phase_index, …),
                            save_dir=leg_dir, should_stop=…  if caps.supports_cooperative_stop)
execute raised                   → failure at tamp_execution
execution.stopped_early          → aborted if the operator asked (preempt or stop), else failure at tamp_execution
                                   (checked before ok, whatever ok says)
not execution.ok                 → failure at tamp_execution
[check_tamp_effects] put the run's add effects to the camera; recorded, never enforced
plan.advance()                   # past every phase in run
```

With phase planning off there is no plan: the goal is `scene.detected_goal` (empty → failure at
`tamp_planning`), neither `movables` nor `return_home` is passed, and the whole trial is one leg.

**A human phase** (`_run_human_phase`):

```
[check_human_preconditions] once, on a fresh frame: put the operator's preconditions to the camera;
                            unmet and precondition_enforced → verification failure
for attempt in 1 .. 1 + verify_retries:
    show the phase: its instructions, and what will be checked afterwards
    wait for the operator's answer:
        abort  → aborted
        teleop → release the arm; human_executor.run(request, leg); count the leg; take the arm back
                   status "aborted"            → failure at human_policy
                   the session is stopping     → aborted
                   no frames, while recording  → refused unless allow_unrecorded_human_phase;
                                                 ask again (no retry spent)
        done   → done by hand, with no leg; while recording, refused the same way
    record how the attempt was carried out (phases[k].carried_out)
    status "ended_by_operator"                          → accept, recorded as unchecked
    check_human_effects off, or the last phase with verify_final_phase off
                                                        → accept, recorded as not checked
    no effect a camera can settle (HandEmpty, Holding)  → accept, unchecked, no frame taken
    check = verify_effects(a fresh frame from verification_camera)
        the check could not run (camera or model error) → accept, recorded as unchecked
        the planner has no capture_frame                → verification failure (the session
                                                          refuses such a planner at start)
        passed, or verify_enforced off                  → record the verdicts; advance
        failed with a retry left                        → tell the operator what is missing
        failed with none left                           → record the failing verdicts; then
              on_verification_failure: exclude → excluded at verification
                                       label   → failure at verification (the label decides)
```

Every `hitl.*` key above is described in [CONFIGURATION.md](CONFIGURATION.md#phase-planning-hitl).

A planner verb (`perceive`, `plan`, `execute`) that raises ends the trial at its stage; the loop catches
it, and the session warms the planner again before the next task. A human executor that raises ends
the trial at `human_policy`; the error then reaches the session, which files the trial and carries on.

**After the loop** (`core/session.py` `_run_task`), on every way out, a preempt included:

- **Nothing recorded:** nothing is filed (`rollout_discarded`), as for most failures at `invention`
  ([what is left](DATA.md#logs-and-session-files)).
- **Settled by the loop** (excluded, aborted, or a failure at any stage but `verification`): filed under
  `failure/` with no label prompt (`trial_excluded` or `trial_filed`).
- **Otherwise** (the plan ran to the end, or a check was left to the label): the operator labels it
  (`awaiting_label`, then `labeled`). A session stopped at that prompt leaves the trial unmerged
  (`trial_unlabeled`; [filing it later](USAGE.md#collecting)).

Filing writes `hitl.json` and `vlm/` into the leg the trial is filed under, then merges the legs into
one episode and writes them again beside it (`episodes.merge_trajectory`). The merge runs in the
background. A session that is ending waits up to 5 minutes for it; if it doesn't finish, the legs are
intact and `tandem traj merge <trajectory id>` joins them.

### How a trial ends

The loop sets `TrialOutcome.outcome` only when it ends a trial itself. `episodes.trial_outcome`
resolves what `hitl.json` says:

| `outcome` | Set when | Label asked |
|---|---|---|
| `excluded` | a check that stops the trial failed, with `on_verification_failure: exclude` | no; filed under `failure/` |
| `aborted` | the operator abandoned a human phase, preempted, or stopped the session, with the plan unfinished | no; filed under `failure/` |
| `failure` at `invention`, `tamp_planning`, `tamp_execution` or `human_policy` | the plan didn't finish (Fig. 4) | no; filed under `failure/` |
| `failure` at `verification` | a check failed, with `on_verification_failure: label` | yes; the label decides, and may overrule the check |
| `success` / `failure` | the plan ran to the end; the operator's label | yes |

An unfinished plan is never a success, and `failure_stage` is always the loop's. A settled outcome
stays in `hitl.json` wherever the episode is later moved. Overruling it takes
`tandem traj relabel --force` ([USAGE.md](USAGE.md#reviewing-and-exporting)).

## The contract check

`planning/contracts.py` walks the phase list symbolically, with no image and no model call. Each phase
leaves `after = (before − displaced − delete_effects) ∪ add_effects`. Its unmet preconditions are
`preconditions − before`.

- **A robot phase has no operator.** Its add effects are its `atoms`. It has no preconditions and no
  delete effects, since the planner re-perceives before every leg.
- **Displacement.** A placement ends the object's previous placement without a delete effect saying so.
  `Capabilities.exclusive_arguments` says which argument is exclusive: TiPToP's `{"On": 0}` means
  `On(toy, box)` retracts `On(toy, table)`.
- **Sound, not complete.** The start is unknown unless `classify_initial` measured it. So a plan is
  refused only where the plan itself is the reason: a precondition an earlier phase deleted or
  displaced and nothing restored. With the start measured, a precondition over an invented predicate
  that no phase establishes and the scene doesn't show is also reported. Everything else passes.
- **Wasted robot moves** (`contracts.wasted_robot_move`) are logged, never refused: two consecutive
  robot phases that move the same object usually mean a human step was written as a pick-and-place,
  but can be a real continuation.

`contracts.expected_before` (what earlier phases should have left true) is the set that
`check_tamp_preconditions` puts to the camera.

## Departures from the paper

| Topic | What tandem does | Setting (default) |
|---|---|---|
| Consecutive robot phases | The paper re-perceives after every phase. tandem plans consecutive robot phases as one goal, with one perception pass, where that is sound (`feasibility.conjoinable_run`; [conditions](ADDING_A_PLANNER.md#capabilities)). | `conjoin_robot_phases: true`; `false` follows the paper strictly |
| A robot phase that can't be planned | Ends the trial, as Fig. 4 counts it. `teleop` hands it to a person instead (a dataset collected that way understates the human effort); `replan` feeds the failure back to the proposer. | `on_robot_phase_failure: abort` |
| A robot leg that fails to execute | Always ends the trial at `tamp_execution`. | none |
| Human operators | Grounded, and ordered by the phase list. The camera checks and the contract check read them; nothing searches over them. Robot phases have no operator; the planner picks its own Ω₀ operators. | none |
| A check that can't run | Recorded as unchecked (`checks.unchecked_phases`, `phases[k].unchecked`), not failed, and the step goes ahead. So is a check with nothing a camera can settle. | none |
| Additions the paper doesn't describe | `movables` (only objects some robot phase moves may be picked, so a person's tool stays an obstacle) and `return_home` (only the last leg drives home), each passed only when the planner declares support. `open_gripper` opens the gripper before the first perception pass after a human phase. Camera checks of a robot leg. | `check_tamp_preconditions`, `check_tamp_effects`: `false` |
| Label drift | Perception names objects afresh on each pass. A renamed object is rebound when exactly one new label contains all its words, or the reverse (`toy` → `blue_toy`; `drift.match_drifted_names`). Otherwise the trial ends at `tamp_planning`. | none |

**Not part of tandem:** the offline prompt evaluation; a learned-policy human executor (the paper's HITL-TAMP
baseline), which can be added as a [plugin](ADDING_A_HUMAN_EXECUTOR.md); and exporting the DROID joint-velocity
action ([what is exported](USAGE.md#exporting)).

### Known limitations

- A re-planned trial keeps the legs it already recorded, stamped with the phase indices of the plan they
  ran under. `hitl.json` is written from the last plan and keeps every plan it replaced in
  `superseded_plans`; match legs to them with `plan_generation` (in `segments[]`, or `leg_plan_generations`
  for legs never merged) ([DATA.md](DATA.md#hitljson)).
- A conjoined robot leg is stamped with its first phase only ([DATA.md](DATA.md#_metajson)).

## Package layout

```
src/tandem/
├── __init__.py, api.py   the library surface: tandem.plan_task and the SDK's names
├── planning/        the method: proposal, invented predicates, magic operators, the contract check,
│                    verification. Pure Python: no planner, no robot.
├── planners/        the planner protocol, and the kit for writing one
│   ├── base.py      what tandem needs from a planner
│   ├── sdk.py       Planner: the base class a new planner subclasses
│   ├── sidecar.py   SidecarPlanner: a planner in its own environment, over JSON lines
│   ├── sidecar_kit/ tandem_sidecar: the stdlib-only helper a sidecar script is written with
│   ├── runtime.py   builds a planner's runtime from its recipe
│   ├── bundle.py    offline bundles of a planner's pinned sources
│   ├── testing.py   the conformance kit a planner's tests subclass
│   ├── registry.py  planners by name: built in, registered, or a `tandem.planners` entry point
│   └── tiptop/      TiPToP: its declaration, recipe, options, presets, importer and sidecar
├── executors/       who carries out a human phase: the protocol, the registry, teleop
├── core/            the session and its trial loop (phase_loop.py), episodes, merge, profiles
├── cli/             the command tree (Typer and Rich)
├── server/          FastAPI and a no-build single-page app
├── export/          the LeRobot v3.0 writer
├── teleop/          the hand-off driver, run in a DROID environment
└── resources/       the annotated profile template, tandem's presets, the planner scaffold
```

- The **session** (`core/session.py`) owns the state machine, the prompts and the label; `core/phase_loop.py`
  runs the trial. One session backs both `tandem collect` and the web UI, and survives preempts,
  re-warms and hand-offs.
- A planner that needs torch, CUDA kernels, a camera SDK or a robot client runs as a **sidecar** in its
  own runtime ([ADDING_A_PLANNER.md](ADDING_A_PLANNER.md#sidecars)).
- TiPToP's sidecar (`planners/tiptop/sidecar.py`) is tandem's own file, run by the runtime's
  interpreter. It imports nothing from `tandem`, and hands goals to cuTAMP through `run_perception`'s
  `goal_builder` hook, so tiptop needs no change.
- A runtime is declared, not shipped ([ADDING_A_PLANNER.md](ADDING_A_PLANNER.md#a-runtime-recipe)).
  TiPToP's recipe (`planners/tiptop/recipe.py`) pins tiptop, cuTAMP and cuRobo, applies one patch, and
  keeps the monorepo layout (`vae/checkpoints/`, `rnd/checkpoints/`) so the cuRobo costs find the
  DATAFARM checkpoints.
