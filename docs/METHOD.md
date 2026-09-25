# The method, as implemented

Where each part of the TANDEM paper's method (Sec. IV) lives in the code, and what a trial does.

**A model decides what each phase must achieve, and in what order. The planner carries out robot phases.
tandem assigns phases, hands the arm over, checks the person's work and records the result.** Planners know
nothing about phases.

Code paths are relative to `src/tandem/`. Terms: [the docs index](README.md#terms).

## Paper to code

| Paper | Code |
|---|---|
| Base domain M₀ = ⟨Ψ₀, Ω₀⟩ | `Capabilities` (`planners/base.py`; TiPToP's in `planners/tiptop/capabilities.py`). Ψ₀ is `goal_predicates` (TiPToP: `On`, `Holding`, `HandEmpty`); Ω₀ is `robot_operators` (`Pick`, `Place`), recorded only; the planner picks its own. |
| Symbolic state s_t | `perceive()` → `SceneView` for Ψ₀; classifiers g_ψ for invented predicates. |
| Phase φ_k = (γ_k, e_k) | `Phase` (`planning/structs.py`): `atoms` is γ_k, `executor` (`robot` or `human`) is e_k. |
| Invented predicates Ψ_Δ, classifier g_ψ | `structs.VLMPredicate`, judged by `grounding.classify`. Its `instructions` is both the classifier's question and the person's instruction. |
| Magic operators Ω_Δ | `structs.HumanOperator`: grounded, not lifted, never searched (the phase list orders them), and required on every proposed human phase. |
| VLM task planning, repair | `proposal.propose_plan` ([instruction to plan](#from-instruction-to-plan)). |
| Operators checked against each other | `contracts.check_plan_effects` ([contract check](#the-contract-check)). |
| Autonomous execution | `TampBackend.plan` / `execute` ([protocol](ADDING_A_PLANNER.md#the-protocol)). |
| Human execution π_ωΔ | `executors/` (`TeleopExecutor` ships), lent the robot and cameras ([custody](ADDING_A_HUMAN_EXECUTOR.md#custody)). |
| Re-perception after every phase | Perception before every robot leg ([departures](#departures-from-the-paper)); a fresh `capture_frame` per human-phase check. |
| Human-phase verification | `grounding.verify_effects`: add effects must hold, delete effects must not. Preconditions optionally first. |
| τ = ((τ₁, φ₁), …, (τ_N, φ_N)) | `core/merge.py`: one episode per trial; `segments[k].phase_index` names each leg's phase ([layout](DATA.md#episode-layout)). |
| [DATAFARM](README.md#terms) alignment | TiPToP's `planner.options.tamp` (`blend_mode: vae`), set in [the paper's profiles](CONFIGURATION.md#the-papers-five). A planner option, not the method. |
| Failure taxonomy (Fig. 4); failed trials excluded | `plan.OUTCOMES`, `plan.FAILURE_STAGES`. `core/phase_loop.py` ends a trial, `core/session.py` files it ([outcomes](#how-a-trial-ends)). |

## From instruction to plan

`build_plan` (`planning/plan.py`) turns an instruction and an image into a `PhasePlan`, or raises
`ProposalError`:

1. **Prompt.** `prompts.plan_prompt` fills the Appendix-B template from `Capabilities` (`goal_predicates`,
   `robot_description`, `prompt_fragments`). The template names no planner predicate.
2. **Answer.** JSON: `new_predicates`, ordered `phases` (human ones with an `operator`), `coverage`
   (clause → phase) and `unrepresented` (clause, reason).
3. **Validate** (`proposal.parse_plan_response`, then `proposal.check_plan`):
   - Each phase has atoms over detected objects; each object stays a surface or a movable for the whole task.
   - A robot phase uses only goal predicates, has no `operator`, and some robot operator achieves it.
   - A human phase has `instructions` and an operator that adds all its `atoms`, deletes none of them, and never
     adds and deletes one atom.
   - An invented predicate is used, with consistent argument types and a name the planner doesn't reserve;
     its `instructions` use only `{0}`, `{1}`, … within its arity.
   - Each robot leg, as the loop will cut it, has a goal beyond planner-supplied atoms (TiPToP's
     `HandEmpty`).
   - No phase adds two atoms for one exclusive slot, like `On(toy, box)` and `On(toy, shelf)`.
   - With `check_plan_effects` on, the [contract check](#the-contract-check) passes.
4. **Repair.** Each rejection goes back to the model, up to `max_attempts` (3) answers; the last ends the
   trial at `invention`. Transient API errors are retried without costing an attempt.
5. **Measure the start** (`classify_initial` only). Invented atoms are classified on the first image and,
   with `check_plan_effects` on, the contract check re-runs against them. Findings are recorded
   (`checks.plan_effects_warning`), not refused.

The accepted plan becomes [`hitl.json`](DATA.md#hitljson). `tandem plan` and `tandem.plan_task()` run these
steps on a photo, with no planner or robot ([USAGE.md](USAGE.md#planning-from-a-photo)).

## The trial loop

`PhaseLoop.run` in `core/phase_loop.py`:

```
until the trial is over:
    preempt or session stop → aborted
    plan finished           → stop
    unless the next phase is human:
        perceive into a new leg in <profile>/trajectories/eval/ (reset_arm on the first pass only,
            open_gripper only after a human phase); raised → failure at tamp_planning
        no plan, planning off → goal = scene.detected_goal
        no plan, planning on  → build_plan(); no image or no valid proposal → failure at invention
        otherwise             → rebind drifted labels; one the leg needs unmatched → failure at tamp_planning
    current phase is human     → human phase  # `t`, or a pending hand-off request, runs the executor
    operator asked for the arm → lend it to teleop with no phase; nothing advances
    otherwise                  → robot leg
```

**A robot leg:**

```
goal = atoms of this phase and the robot phases conjoined with it   # empty → failure at invention
[check_tamp_preconditions] what earlier phases should have left true, on this pass's image;
                           unmet + precondition_enforced → check failure
plan(goal, surfaces, movables, return_home)   # extras only if declared; return_home true on the last leg
    raised  → failure at tamp_planning        # no teleop, no replan
    no plan → on_robot_phase_failure:
                abort  → failure at tamp_planning
                teleop → run it as a human phase (atoms, no operator), recorded as handed over
                replan → drop the plan; the next pass re-proposes, told every failure so far
                         (at most max_attempts times, then as abort)
record the planner's task_plan on each phase in the leg
execute(plan_handle, LegSpec, should_stop if declared)
    stopped_early    → (checked first) aborted if the operator asked, else failure at tamp_execution
    raised or not ok → failure at tamp_execution
[check_tamp_effects] the leg's add effects vs. the camera; recorded only
advance past every phase in the leg

phase planning off: one leg; an empty goal or any plan failure → failure at tamp_planning;
no movables or return_home
```

**A human phase:**

```
[check_human_preconditions] once, on a fresh frame; unmet + precondition_enforced → check failure
for attempt in 1 .. 1 + verify_retries:
    show the instructions and what will be checked
    abort  → aborted
    teleop → lend the arm to human_executor, then take it back
               "aborted", or raised       → failure at human_policy
               session stopping           → aborted
               no frames, while recording → ask again (no retry spent), unless allow_unrecorded_human_phase
    done   → by hand, no leg; while recording, refused the same way
    record how it was carried out (phases[k].carried_out)
    "ended_by_operator"                                             → accept, unchecked
    check_human_effects off, or last phase + verify_final_phase off → accept, not checked
    no effect a camera can settle (HandEmpty, Holding)              → accept, unchecked, no frame
    verify_effects on a fresh verification_camera frame:
        could not run (camera or model error) → accept, unchecked
        planner has no capture_frame          → check failure (refused at session start)
        passed, or verify_enforced off        → record the verdicts; advance
        failed, a retry left                  → tell the operator what is missing; retry
        failed, none left                     → record the verdicts; check failure
```

A **check failure** ends the trial as `on_verification_failure` says ([outcomes](#how-a-trial-ends)). The
`hitl.*` keys are in [CONFIGURATION.md](CONFIGURATION.md#phase-planning-hitl). After a planner verb raises,
the session re-warms the planner before the next task.

**Filing**, on every way out: a trial with nothing recorded (most `invention` failures) is not filed. A
settled one goes under `failure/`, unlabeled. Any other waits for the operator's label; a session stopped
there leaves it unmerged ([filing it later](USAGE.md#collecting)). Filing runs in the background
([what it writes](DATA.md#hitljson)); a stopping session waits up to 5 minutes for it. If it doesn't
finish, `tandem traj merge <trajectory id>` joins the intact legs.

### How a trial ends

The loop sets `outcome` only when it ends a trial itself; otherwise the label does. What `hitl.json` records:

| `outcome` | Set when | Label asked |
|---|---|---|
| `excluded` | a check failure, with `on_verification_failure: exclude` | no |
| `aborted` | the operator abandoned a human phase, preempted or stopped the session, with the plan unfinished | no |
| `failure` at `invention`, `tamp_planning`, `tamp_execution` or `human_policy` | the plan didn't finish (Fig. 4) | no |
| `failure` at `verification` | a check failure, with `on_verification_failure: label` | yes; the label decides |
| `success` / `failure` | the plan ran to the end; the operator's label | yes |

`failure_stage` is always the loop's. A settled outcome stays in `hitl.json` wherever the episode moves;
`tandem traj relabel --force` overrules it ([USAGE.md](USAGE.md#reviewing-and-exporting)).

## The contract check

`planning/contracts.py` walks the phases symbolically, with no image or model call. Each phase leaves
`after = (before − displaced − delete_effects) ∪ add_effects`; its unmet preconditions are
`preconditions − before`.

- **Robot phases** add their `atoms`, with no preconditions or delete effects.
- **Displacement.** A new placement ends the object's old one: with TiPToP's
  `Capabilities.exclusive_arguments` of `{"On": 0}`, `On(toy, box)` retracts `On(toy, table)`.
- **Sound, not complete.** Unless `classify_initial` measured the start, a plan is refused only for a
  precondition an earlier phase deleted or displaced and nothing restored. With the start measured, an
  invented-predicate precondition that no phase establishes and the scene lacks is also reported.
- **Wasted robot moves** (consecutive robot phases moving one object) are logged, not refused: usually a
  human step written as a pick-and-place.

## Departures from the paper

| Topic | What tandem does | Setting (default) |
|---|---|---|
| Consecutive robot phases | One goal and one perception pass, where sound ([conditions](ADDING_A_PLANNER.md#capabilities)). The paper re-perceives after each phase. | `conjoin_robot_phases: true` (`false` follows the paper) |
| An unplannable robot phase | Ends the trial, as in Fig. 4. `teleop` gives it to a person (the dataset then understates human effort); `replan` re-proposes with the failure. | `on_robot_phase_failure: abort` |
| A robot leg that fails to execute | Always ends the trial at `tamp_execution`. | none |
| A check that can't run | Recorded as unchecked (`checks.unchecked_phases`, `phases[k].unchecked`), not failed; the step goes ahead. Likewise when a camera can settle none of its atoms. | none |
| Additions | `movables` (only objects robot phases move may be picked, so a person's tool stays an obstacle) and `return_home` (last leg only), if the planner supports them; `open_gripper` after a human phase; camera checks of robot legs. | `check_tamp_preconditions`, `check_tamp_effects`: `false` |
| Label drift | A renamed object is rebound when exactly one new label contains all its words, or the reverse (`toy` → `blue_toy`); otherwise the trial ends at `tamp_planning`. | none |

**Not part of tandem:** the offline prompt evaluation; a learned-policy human executor (the paper's HITL-TAMP
baseline; add one as a [plugin](ADDING_A_HUMAN_EXECUTOR.md)); and exporting the DROID joint-velocity action
([what is exported](USAGE.md#exporting)).

### Known limitations

- A re-planned trial keeps its earlier legs, stamped with the old plan's phase indices. Match them to the
  plans in `superseded_plans` by `plan_generation` ([DATA.md](DATA.md#hitljson)).
- A conjoined robot leg is stamped with its first phase only ([DATA.md](DATA.md#_metajson)).

## Package layout

```
src/tandem/
├── __init__.py, api.py   library surface: tandem.plan_task and the SDK
├── planning/     the method (proposal, predicates, operators, checks); no planner or robot
├── planners/     protocol (base.py), SDK (sdk.py), sidecars, runtimes, registry, conformance kit
│                 (testing.py), and tiptop/ (declaration, recipe, options, sidecar)
├── executors/    human executors: protocol, registry, teleop
├── core/         session, trial loop, episodes, merge, profiles, the rig
├── cli/          commands (Typer, Rich)
├── server/       FastAPI and a no-build single-page app
├── export/       LeRobot v3.0 writer
├── teleop/       hand-off driver, run in a DROID environment
└── resources/    the paper's five profiles, the profile and rig templates, planner scaffold
```

- The **session** (`core/session.py`) owns the state machine, prompts and label, and survives preempts,
  re-warms and hand-offs. One session backs `tandem collect` and the web UI.
- A planner with heavy dependencies runs as a **sidecar** ([sidecars](ADDING_A_PLANNER.md#sidecars)) in a
  declared runtime ([recipes](ADDING_A_PLANNER.md#a-runtime-recipe);
  [TiPToP's](CONFIGURATION.md#the-planner-runtime)). TiPToP's sidecar ships with tandem, so tiptop needs no
  change.
