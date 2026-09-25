# TANDEM documentation

To set up and start collecting, begin with the [repo README](../README.md).

| I want to… | Read |
|---|---|
| Look up a command, or run collection, `plan`, the web UI, review or export | [USAGE.md](USAGE.md) |
| Change a setting: profiles, the paper's five tasks, the rig, `hitl:`, TiPToP, credentials, the runtime | [CONFIGURATION.md](CONFIGURATION.md) |
| Understand the files a trial writes, or the logs | [DATA.md](DATA.md) |
| Fix something that went wrong | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Plug in another task and motion planner | [ADDING_A_PLANNER.md](ADDING_A_PLANNER.md) |
| Do human phases with something other than teleop, such as a learned policy | [ADDING_A_HUMAN_EXECUTOR.md](ADDING_A_HUMAN_EXECUTOR.md) |

## Terms

### Tasks and setup

- **profile**: one task, kept in one YAML file (`~/tandem-data/profiles/<name>.yml` by default). It holds the
  prompt, the phase-planning settings and the planner's settings. Its trajectories go in
  `~/tandem-data/trajectories/<name>/`. `tandem init` adds the paper's five tasks as profiles.
- **rig**: this machine's robot, cameras and calibration, in `~/.config/tandem/rig.yml`. Every profile shares it.

### Phase planning

- **phase planning**: splitting an instruction into an ordered list of robot and human phases. It is on when a
  profile sets `hitl.enabled: true`.
- **phase**: one step, plus the **atoms** that must hold after it. An atom is a predicate applied to objects,
  such as `On(bread, plate)`. A **robot phase** becomes a goal for the planner. A **human phase** goes to a
  human executor.
- **proposal**: the model's list of phases. **Repair** sends a proposal that fails a check back to the model,
  with the reason.
- **invented predicate**: a predicate the model adds because the planner's goal language lacks it, such as
  `IsOpen(box)`. A camera check judges it.
- **magic operator**: a human phase's preconditions, add effects and delete effects. For example, opening a box
  might need `Closed(box)` and add `IsOpen(box)`.
- **camera check**: a vision-language model judging, from one image, whether an atom holds.

### Trials and recordings

- **human executor**: whatever carries out a human phase. Only `teleop`, a person driving the arm, ships with
  tandem. A **hand-off** lends the executor the arm and then takes it back.
- **perception pass**: one `perceive` call, which names the objects in the scene. One runs before each robot leg.
- **trial**: one attempt at a task. All its legs share a 16-hex-character `trajectory_id`.
- **leg**: one continuous recording, made either by the planner (one or more robot phases) or by a human executor.
- **conjoined**: consecutive robot phases planned as one goal and recorded as one leg. See
  `hitl.conjoin_robot_phases`.
- **primary leg**: a trial's first planner leg, or its first leg if no planner leg exists. The episode takes its
  directory name and files.
- **episode**: a filed trial. Its legs are merged into one directory in `success/` or `failure/`. A trial with
  one leg is that leg. Only `success/` is [exported](USAGE.md#exporting).
- **trajectory**: an episode, or a leg not yet merged, in a profile's `trajectories/`. `tandem traj` lists them.
- **settled**: ended by tandem rather than labeled by you: excluded, aborted, or failed part-way. Settled trials
  are filed in `failure/` without a label.
- **excluded**: settled because a camera check failed under `hitl.on_verification_failure: exclude` (the
  default). This is usually a human phase that still failed after its retries.

### Planners

- **runtime**: a planner's pinned sources, and usually a pixi environment, built by `tandem planners install`.
- **sidecar**: a planner running as a subprocess that tandem drives over JSON lines, inside its runtime if it
  has one.
- **DATAFARM**: TiPToP `tamp` motion costs (a VAE manifold and RND novelty) that make planned motion resemble
  DROID teleop motion.
