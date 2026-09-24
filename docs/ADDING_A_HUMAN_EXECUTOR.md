# Adding a human executor

The paper gives every magic operator an executor, π_ωΔ: whatever brings about the effects the model
invented for a human phase. TANDEM's own is a person driving the arm through teleoperation, and it
is the only one that ships (`teleop`). It is not the only thing that could do the job. A phase is a
person's because the planner cannot express what it asks for, not because only a person can do it.
The HITL-TAMP baseline, for one, hands the same phase to a policy trained on earlier teleop legs.

So the phase loop asks for an executor by name (`hitl.human_executor`), and a package can add one.
The code is `src/tandem/executors/`: the protocol and registry in `base.py`, teleop in `teleop.py`.

## What an executor decides, and what it does not

- **It decides how a phase is carried out**, and records that as one leg of the trajectory, stamped
  with the `LegSpec` it is given, so the merge can join it to the robot's legs.
- **It does not decide the phase, its operator, or whether the phase happened.** The proposal fixed
  the first two before the arm moved. The loop's camera check decides the third, against the same
  effects whoever did the work. Swapping a person for a policy changes who drives the arm for one
  leg; the plan does not know the difference.
- **It does not own the hardware.** The planner holds the robot and every camera exclusively. The
  loop releases them before the executor runs and takes them back after it. An executor starts
  from an arm that is free, and must hand it back free.

## The protocol

```python
class HumanExecutor(Protocol):            # tandem.executors.HumanExecutor
    name: str                             # the registered name
    segment_source: str                   # "teleop" or "policy": what its legs are, in _meta.json
    display_name: str
    summary: str

    def run(self, request: HumanPhaseRequest | None, leg: LegSpec, *,
            save_root: Path, should_stop: Callable[[], bool]) -> HumanPhaseResult: ...
    def kill(self) -> None: ...
    # optional:
    def close(self) -> None: ...
```

**`run`** carries out one phase and records it. It returns once whatever drove the arm has let go
of the robot and the cameras.

- `request` is the phase (below). It is `None` only for a hand-off the operator asked for between
  phases. The loop always gives those to `teleop`, since only an executor a person drives can take
  an arm lent with no phase attached.
- `leg` is the recording contract, exactly as a robot leg gets it. Write the leg under
  `save_root/eval/<a new directory>/`, which is where the merge looks for it. Stamp its
  `_meta.json` with `leg.trajectory_id` and `leg.segment_source` (the loop builds `leg` with your
  `segment_source`), and with `phase_index`, `n_phases` and `phase_description` when
  `leg.phase_index` is set. The rest of the contract is the planner's: `robot_state.npz` with
  `tandem.core.merge.STATE_KEYS`, one row per frame, and the camera clips
  ([ADDING_A_PLANNER.md](ADDING_A_PLANNER.md#the-recording-contract)).
- `save_root` is the profile's trajectories directory.
- `should_stop` turns true when the leg must end. That happens when the operator returns control
  (`r` in `tandem collect`, "return control" in the UI), when the session is stopping, or after
  an hour with nobody there. It is the only way a leg ends besides the executor finishing, so never
  wait on anything without polling it. It is cheap and never blocks.

**`kill`** ends the leg in flight at once. It is called from another thread by a forced stop. `run`
then returns `aborted` as soon as it can. It is a no-op when nothing is running. A kill can land
just before `run` is entered (the arm takes seconds to release), so do not clear the kill flag when
`run` starts: clear it when `run` returns. A forced stop that arrives before the loop calls `run`
at all never starts the leg.

**`close`** is optional. Define it when building the executor starts something that outlives a leg
(a policy server, a device it opened). It is called exactly once, when the session ends, on every
way it ends (a failure and a forced stop included), after the arm is parked and the planner closed.
It must not raise; one that does is logged.

**Raise `CustodyError`** from `run` when the arm cannot be given back: a process that will not die
still holds the cameras. The session then ends and says so, rather than reaching for hardware
another process holds and reporting a failure that looks like broken hardware.

### What a phase is asked as: `HumanPhaseRequest`

| field | |
|---|---|
| `phase_index`, `n_phases` | 0-based, like every leg stamp. `n_phases` is 0 when there is no plan behind the phase (`stamped` is then false). |
| `description` | The phase in the model's words. |
| `instructions` | What a person is shown. |
| `operator` | The magic operator, as `HumanOperator.to_json()` writes it: `name`, `args`, `signature`, `instance`, `preconditions`, `add_effects`, `delete_effects`. `None` for a robot phase handed to a person under `on_robot_phase_failure: teleop`. |
| `expected` | What the camera will be asked afterwards, in words. |
| `attempt` | From 1. |
| `missing` | On a retry, what the last check said is still not true, phrased for a person. |

`request.leg_spec(trajectory_id=, instruction=, segment_source=, record=)` builds the matching
`LegSpec`. The loop already does this, so an executor only reads `leg`.

### How a leg ended: `HumanPhaseResult(status, n_frames=0, leg_dir=None)`

| `status` | meaning | what the loop does next |
|---|---|---|
| `done` | The executor carried the phase through and handed the arm back. | Checks the phase's effects on a fresh frame (unless `check_human_effects` is off, or it is the last phase and `verify_final_phase` is off). A failed check gets another attempt while `verify_retries` last, then the trial is excluded (or failed, under `on_verification_failure: label`). |
| `ended_by_operator` | The operator stopped the executor on purpose, to move on: the HITL-TAMP baseline's "continue" for a policy that runs until a step limit. | Accepts the phase **without** a camera check, recorded as unchecked in `hitl.json`. The leg is kept. |
| `aborted` | The leg was cut off, or the executor could not carry the phase out (a policy server that never came up). | Ends the trial as a failure at `human_policy`. Nothing the arm did counts as the phase being done. |

`n_frames` counts every frame the leg wrote, and `leg_dir` is where. A leg with `n_frames=0`,
whether `done` or `ended_by_operator`, recorded nothing. While recording, the loop refuses it and
asks the operator again without spending a retry, unless `hitl.allow_unrecorded_human_phase` is set.
An operator giving up is not `aborted`: that is the "give up" answer at the prompt, given before any
executor runs.

Each attempt starts when the operator hands the phase over: `t` in `tandem collect`, or the button
in the UI. On a retry the log tells them what is missing. For an executor other than `teleop` it
says `Running the <name> executor on this phase again (N attempt(s) left).`

### Custody, in order

`PhaseLoop._lend_arm`:

1. `backend.release_hardware()`. It blocks until the robot and every camera are free. A failure
   here is a warning, not the end of the leg: a person may already have their hands on the arm.
2. `executor.run(request, leg, save_root=..., should_stop=...)`.
3. The leg is counted as recorded, **before** the arm is taken back. Taking it back can fail and
   end the session, and a leg on disk must still be labeled and merged on the way out.
4. `backend.reacquire_hardware()`, on every way out except a `CustodyError`.

## Registering one

An executor is found through a factory, so that listing executors costs nothing and works on a
laptop. Building one may start a process or load a policy, so it happens only when a session first
needs it.

```python
# my_package/executor.py -- keep it light: listing executors imports this module.
from pathlib import Path

from tandem.executors import ExecutorFactory, HumanPhaseResult


class PolicyExecutor:
    """Runs a learned policy for a human phase, one leg per attempt."""

    name = "mypolicy"
    segment_source = "policy"
    display_name = "My policy"
    summary = "Runs a learned policy on the phase's instructions."

    def __init__(self, ctx):
        self.ctx = ctx               # ExecutorContext: profile, session_dir, settings, on_log, ...
        self._killed = False

    def run(self, request, leg, *, save_root, should_stop):
        if request is None:          # an arm lent with no phase: a person's to take, not a policy's
            return HumanPhaseResult("aborted")
        try:
            if self._killed:         # killed before the leg started: never start it
                return HumanPhaseResult("aborted")
            leg_dir, n_frames, stopped = self._run_policy(request, leg, Path(save_root), should_stop)
            if self._killed:
                return HumanPhaseResult("aborted", n_frames=n_frames, leg_dir=leg_dir)
            status = "ended_by_operator" if stopped else "done"
            return HumanPhaseResult(status, n_frames=n_frames, leg_dir=leg_dir)
        finally:
            self._killed = False     # cleared as the leg ends, never as it starts

    def kill(self):
        self._killed = True

    def close(self):
        """Stop the policy server this executor started, once, when the session ends."""

    def _run_policy(self, request, leg, save_root, should_stop):
        """Yours: open the robot and cameras; run the policy on request.instructions (and
        request.operator); record every frame into a new directory under save_root / "eval", with
        _meta.json stamped from `leg`; poll should_stop() and self._killed at every step; release
        the robot and cameras. Return (leg_dir, n_frames, whether should_stop ended it)."""
        raise NotImplementedError


def unmet(settings) -> list[str]:
    """What this machine lacks, in words. Cheap: read settings, stat a path. Start nothing."""
    return []


FACTORY = ExecutorFactory(
    create=PolicyExecutor,                       # called with an ExecutorContext
    display_name=PolicyExecutor.display_name,
    summary=PolicyExecutor.summary,
    segment_source="policy",                     # must equal what the built executor says
    requirements=("a policy checkpoint",),       # shown by `tandem executors list`
    check=unmet,
)
```

Then either register it in the process:

```python
import tandem
tandem.register_human_executor("mypolicy", FACTORY)     # or "my_package.executor:FACTORY"
```

or declare it in the package's `pyproject.toml`:

```toml
[project.entry-points."tandem.human_executors"]
mypolicy = "my_package.executor:FACTORY"
```

- **The factory** may be an `ExecutorFactory`, or a class taking an `ExecutorContext` whose metadata
  (`segment_source`, `display_name`, `summary`, `requirements`) are class attributes. Its summary
  defaults to the first line of its docstring. Only an `ExecutorFactory` can carry a readiness
  `check`.
- **`segment_source`** is `"teleop"` or `"policy"`. An executor may not claim `"tamp"`: the merge
  and the export treat those legs as the planner's.
- **The name rule** is the planner rule: lowercase letters, digits, `_` and `-`, starting with a
  letter.
- **Taken names.** Registering a taken name is refused unless `replace=True`. Two installed packages
  claiming one name is an error, not a guess, because a dataset recorded by one executor and
  attributed to another cannot be interpreted afterwards.
- **Broken plugins.** One that fails to import is listed with its error, and every other executor
  keeps working.

### What `create` is handed: `ExecutorContext`

| field | |
|---|---|
| `profile` | The validated profile (its cameras, for one). |
| `session_dir` | The session's scratch directory, for anything that is not a recording. |
| `settings` | The machine's settings. `None` re-reads them at every leg, so a `tandem config set` between two hand-offs takes effect at the next one. |
| `on_log(stream, text)` | A line in the session log. |
| `on_emit(payload)` | A message to every UI subscriber. |
| `on_problem(message)` | A problem the operator must see now, shown until the leg ends. |
| `options` | Executor-specific settings. **Nothing fills them yet:** a profile has no block for an executor's own settings, so read yours from your own configuration for now. |

## Choosing one

```bash
tandem executors list                 # every executor: ready, needs setup (and what), or broken
tandem executors use mypolicy         # the active profile's human phases run with it (--profile P)
```

`executors use` sets `hitl.human_executor` and warns, without refusing, when the executor is not
ready on this machine. `--json` on both prints what the web UI's `GET /api/executors` and
`POST /api/executors/{name}/use` return.

The name is checked when the profile loads: a profile naming an executor this machine does not have
does not load until the package providing it is installed. A session with phase planning on
describes the executor before it warms anything, so a plugin that will not import is found then,
not at the first human phase. `tandem doctor` has a `human executor` row: OK when ready, WARN with
what is missing, FAIL when it will not load.

**While recording, a human phase can only be completed through a ready executor.** On a machine
where it needs setup, the operator's only answer at a human step is to give up, unless the profile
sets `hitl.allow_unrecorded_human_phase: true`, or the session runs with `--no-record`.

## The one that ships: `teleop`

`TeleopExecutor` (`executors/teleop.py`) lends the arm to a person through the DROID teleop driver
(`tandem/teleop/driver.py`), one driver process per leg. It needs:

- `teleop.enabled: true`;
- `teleop.python`: the DROID environment's interpreter;
- `teleop.droid_dir`: a DROID checkout;
- a VR headset and controller, or a SpaceMouse (`teleop.device`: `vr` or `spacemouse`;
  `teleop.controller`: `right` or `left`).

`tandem init` asks for them, or set them with `tandem config set teleop.<key> <value>`. Two
behaviours are deliberate:

- **A driver that cannot start does not end the leg.** The arm is already released and may be in
  someone's hands, so the problem is shown and the leg waits for "return control" like any other.
- **The frame count is read after the driver exits.** It arrives with the driver's last event,
  after its videos are written.

## Known gaps

- `ExecutorContext.options` is always empty (above).
