# TANDEM documentation

Setup and collecting: [repo README](../README.md).

| Doc | Covers |
|---|---|
| [USAGE.md](USAGE.md) | Every command; `collect`, `plan`, web UI, review, export |
| [CONFIGURATION.md](CONFIGURATION.md) | Profiles, the paper's five, the rig, `hitl:`, TiPToP, tandem's settings, planner runtime |
| [DATA.md](DATA.md) | Trial files, logs |
| [METHOD.md](METHOD.md) | Paper's method in code |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Fixes |
| [ADDING_A_PLANNER.md](ADDING_A_PLANNER.md) | Another task and motion planner |
| [ADDING_A_HUMAN_EXECUTOR.md](ADDING_A_HUMAN_EXECUTOR.md) | Non-teleop executors, e.g. a learned policy |

## Terms

- **profile**: a task (prompt, phase planning, planner settings) in one YAML file, `~/tandem-data/profiles/<name>.yml` by default, plus its trajectories in `~/tandem-data/trajectories/<name>/`. `tandem init` adds the paper's five.
- **rig**: this machine's robot, cameras and calibration, shared by every profile; `~/.config/tandem/rig.yml`.
- **phase planning**: splitting an instruction into ordered robot and human phases. Needs `hitl.enabled: true`.
- **phase**: a step and the **atoms** (predicates on objects, e.g. `On(bread, plate)`) that must hold after it. **Robot phase**: a planner goal. **Human phase**: for a human executor.
- **proposal**: the model's phase list. **Repair**: sending a failing one back with the reason.
- **invented predicate**: a model-added predicate outside the planner's goal language, e.g. `IsOpen`; camera-checked.
- **magic operator**: a human phase's preconditions, add and delete effects.
- **camera check**: a vision-language model judging from one image whether an atom holds.
- **human executor**: what does a human phase; only `teleop` (a person driving the arm) ships. A **hand-off** lends it the arm and takes it back.
- **perception pass**: one `perceive` call, naming scene objects, before each robot leg.
- **trial**: one task attempt; its legs share a 16-hex-character `trajectory_id`.
- **leg**: one continuous recording: planner (one or more robot phases) or human executor.
- **conjoined**: consecutive robot phases planned as one goal, recorded as one leg (`hitl.conjoin_robot_phases`).
- **primary leg**: a trial's first planner leg (else first leg); the episode takes its name and files.
- **episode**: a filed trial, legs merged into one directory in `success/` or `failure/` (or a one-leg trial's leg). Only `success/` is [exported](USAGE.md#exporting).
- **trajectory**: an episode or unmerged leg in a profile's `trajectories/`; `tandem traj` lists them.
- **settled**: ended by tandem (excluded, aborted, failed part-way; [outcomes](METHOD.md#how-a-trial-ends)); filed in `failure/` unlabelled.
- **excluded**: settled by a failed camera check under `hitl.on_verification_failure: exclude` (default), usually a human phase after retries.
- **runtime**: a planner's pinned sources and usually a pixi env, from `tandem planners install`.
- **sidecar**: a planner subprocess tandem drives over JSON lines (in its runtime, if any).
- **DATAFARM**: TiPToP `tamp` motion costs (VAE manifold, RND novelty) mimicking DROID teleop motion.
