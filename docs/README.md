# TANDEM documentation

Setup and a first collection are in the [top-level README](../README.md). These pages are the reference.

| Doc | Read it when |
|---|---|
| [USAGE.md](USAGE.md) | You run commands, a collection session, `tandem plan` or the web UI, or review and export data. |
| [CONFIGURATION.md](CONFIGURATION.md) | You change a setting: a profile and its `hitl:` block, cameras, TiPToP options, machine settings, the planner runtime. |
| [DATA.md](DATA.md) | You read what a trial left on disk, or look for a log. |
| [METHOD.md](METHOD.md) | You want to know how the paper's method runs in code. |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Something went wrong. |
| [ADDING_A_PLANNER.md](ADDING_A_PLANNER.md) | You plug in another task and motion planner. |
| [ADDING_A_HUMAN_EXECUTOR.md](ADDING_A_HUMAN_EXECUTOR.md) | You want something other than teleop, such as a learned policy, to carry out human phases. |

## Terms

Every doc uses these words in these senses.

- **profile**: one collection setup (task, cameras, planner, settings) plus every trajectory collected with it, in `~/tandem-data/profiles/<name>/` by default.
- **phase planning**: splitting an instruction into an ordered list of robot and human phases. It is on when `hitl.enabled` is true.
- **atom**: a predicate applied to objects, such as `On(bread, plate)`.
- **phase**: one step of that list, with the atoms that must hold when it ends. A **robot phase** is handed to the planner as a goal. A **human phase** is carried out by a human executor.
- **proposal**: the model's phase list for an instruction. One that fails validation is sent back to the model with the reason, which is a **repair**.
- **invented predicate**: a predicate the model adds for something the planner's goal language cannot express, such as `IsOpen`. A camera check judges it.
- **magic operator**: a human phase's contract: its preconditions, add effects and delete effects.
- **camera check**: a vision-language model judging, from one image, whether an atom holds.
- **human executor**: what carries out a human phase. `teleop` (a person driving the arm) is the only one that ships.
- **hand-off**: the arm lent to a human executor and taken back afterwards.
- **perception pass**: one call to the planner's `perceive`, which looks at the scene and names the objects. One runs before every robot leg.
- **trial**: one attempt at the task. Its legs share one `trajectory_id` (16 hex characters).
- **leg**: one continuous recording in a trial: the planner's (one or more robot phases), or a human executor's.
- **conjoined**: consecutive robot phases planned as one goal and recorded as one leg (`hitl.conjoin_robot_phases`).
- **primary leg**: a trial's first planner leg, or its first leg if it has none. The episode takes its name and files.
- **episode**: a filed trial: its legs merged into one directory under `success/` or `failure/` (a one-leg trial's leg is its episode). Only `success/` is [exported](USAGE.md#exporting).
- **trajectory**: a directory under a profile's `trajectories/`: an episode, or a leg not yet merged. `tandem traj` lists them.
- **settled**: a trial tandem ended itself: excluded, aborted, or failed part-way ([outcomes](METHOD.md#how-a-trial-ends)). It is filed under `failure/` without a label.
- **excluded**: a settled trial that a failed camera check ended under `hitl.on_verification_failure: exclude` (the default), usually a human phase still failing after its retries.
- **runtime**: where a planner's heavy dependencies are installed: its pinned source trees and, usually, a pixi environment. `tandem planners install` builds it.
- **sidecar**: a planner script that tandem runs as a separate process (in the planner's runtime, when it has one) and drives over JSON lines.
- **DATAFARM**: motion costs (a VAE manifold and an RND novelty term) that make TiPToP's planned motions look like DROID teleoperation data. They are TiPToP `tamp` options.
