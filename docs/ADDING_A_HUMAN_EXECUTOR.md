# Adding a human executor

A [human executor](README.md#terms) carries out a human phase and records it as one leg. `teleop` (a person
driving the arm) is the only one that ships. A package can add another, such as a learned policy, and a
profile picks it with `hitl.human_executor`. Code: `src/tandem/executors/` (`base.py`, `teleop.py`).

## What an executor does

- It decides **how** a phase is carried out, and records that as one leg stamped from the `LegSpec` it gets.
- It does **not** decide the phase, its magic operator (both fixed by the proposal), or whether the phase
  happened (the loop's camera check decides).
- It does **not** own the hardware. It starts from a free arm and must hand it back free ([custody](#custody)).

## The protocol

```python
class HumanExecutor(Protocol):            # tandem.executors.HumanExecutor
    name: str                             # the registered name
    segment_source: str                   # "teleop" or "policy": what its legs are, in _meta.json
    display_name: str                     # display_name and summary are shown when choosing one
    summary: str

    def run(self, request: HumanPhaseRequest | None, leg: LegSpec, *,
            save_root: Path, should_stop: Callable[[], bool]) -> HumanPhaseResult: ...
    def kill(self) -> None: ...
    def close(self) -> None: ...          # optional
```

| method | contract |
|---|---|
| `run` | Carry out one phase, record it, and return a [`HumanPhaseResult`](#humanphaseresult) only once whatever drove the arm has let go of the robot and cameras. |
| `kill` | End the leg in flight at once; `run` then returns `aborted`. Called from another thread; a no-op when nothing runs. Clear the kill flag when `run` returns, not when it starts: a kill can land while the arm is still being released. |
| `close` | Optional. Release what building the executor started (a policy server, a device). Called once when the session ends, however it ends, after the arm is parked. Must not raise. |

| `run` argument | meaning |
|---|---|
| `request` | The phase ([`HumanPhaseRequest`](#humanphaserequest)). `None` only for a hand-off the operator asked for between phases, which always goes to `teleop`. |
| `leg` | The `LegSpec` a robot leg gets. Write the leg in a new directory under `save_root/eval/` and stamp its `_meta.json` from `leg`: `trajectory_id`, `segment_source`, and `phase_index`/`n_phases`/`phase_description` when `leg.phase_index` is set. The rest is [the recording contract](ADDING_A_PLANNER.md#the-recording-contract). |
| `save_root` | The profile's `trajectories/` directory. |
| `should_stop` | Turns true when the operator returns control (`r`), the session stops, or an hour passes. It is the only other way a leg ends, so poll it and never block without it. It is cheap. |

Raise `tandem.executors.CustodyError` from `run` when the arm can't be given back, for example because a
process that won't die still holds the cameras. The session then ends and says why. `teleop` raises it
when its driver outlives a kill.

### `HumanPhaseRequest`

| field | meaning |
|---|---|
| `phase_index`, `n_phases` | 0-based. `n_phases` is 0 when no plan is behind the phase; the leg then carries no phase (`leg.phase_index` is `None`). |
| `description` | The phase in the model's words. |
| `instructions` | What a person is shown. |
| `operator` | The magic operator as `HumanOperator.to_json()` writes it (`name`, `args`, `signature`, `instance`, `preconditions`, `add_effects`, `delete_effects`). `None` for a robot phase handed over under `on_robot_phase_failure: teleop`. |
| `expected` | What the camera will be asked afterwards, in words. |
| `attempt` | Counts from 1. |
| `missing` | On a retry: what the last check found still missing, phrased for a person. |

### `HumanPhaseResult`

`HumanPhaseResult(status, n_frames=0, leg_dir=None, leg_dirs=())`

| `status` | meaning | the loop then |
|---|---|---|
| `done` | The phase was carried through. | Checks its effects on a fresh camera image, as the [`hitl` keys](CONFIGURATION.md#phase-planning-hitl) set. A failed check gets `verify_retries` more attempts. |
| `ended_by_operator` | The operator stopped the executor to move on, for example a policy that would run to a step limit. | Accepts the phase without a check, recorded as unchecked in [`hitl.json`](DATA.md#hitljson). Keeps the leg. |
| `aborted` | The leg was cut off, or the executor couldn't carry the phase out (a policy server that never came up). | Ends the trial as a failure at `human_policy` ([outcomes](METHOD.md#how-a-trial-ends)). |

`n_frames` counts every frame the hand-off wrote. `leg_dir` is the last recording directory, or `None` when
nothing was recorded. `leg_dirs` lists every recording directory in order, when the executor knows them;
empty means `leg_dir` is the only one.

- `run` is called once per attempt, when the operator hands the phase over (`t`, or the UI's button).
- A leg with `n_frames=0` is refused while recording, and the operator is asked again without spending a
  retry, unless [`hitl.allow_unrecorded_human_phase`](CONFIGURATION.md#phase-planning-hitl) is true.
- An operator giving up is not `aborted`: that is `a` at the prompt, before any executor runs.

### Custody

`PhaseLoop._lend_arm` runs every human leg in this order:

1. `backend.release_hardware()` blocks until the robot and cameras are free. A failure is only a warning,
   because a person may already have their hands on the arm.
2. `executor.run(...)`. If a forced stop arrived during step 1, `run` is skipped and the result is `aborted`.
3. The leg is counted as recorded, so it is still labeled and merged if step 4 fails.
4. `backend.reacquire_hardware()`, unless `run` raised `CustodyError`. If this fails, the session ends with
   a `CustodyError`.

Any other exception from `run` ends the trial at `human_policy`. The arm is still taken back, and the session
goes on.

## Registering one

Listing executors imports each one's module but builds nothing, so keep the module light. A session builds
the executor the first time it needs it.

```python
# my_package/executor.py
from tandem.executors import ExecutorFactory, HumanPhaseResult

class PolicyExecutor:
    name = "mypolicy"
    segment_source = "policy"
    display_name = "My policy"
    summary = "Runs a learned policy on the phase's instructions."

    def __init__(self, ctx):                     # an ExecutorContext
        self.ctx = ctx
        self._killed = False

    def run(self, request, leg, *, save_root, should_stop):
        try:
            if request is None or self._killed:  # a hand-off with no phase is a person's; a kill came first
                return HumanPhaseResult("aborted")
            leg_dir, n_frames, stopped = self._run_policy(request, leg, save_root, should_stop)
            status = "aborted" if self._killed else "ended_by_operator" if stopped else "done"
            return HumanPhaseResult(status, n_frames=n_frames, leg_dir=leg_dir)
        finally:
            self._killed = False                 # cleared as the leg ends, never as it starts

    def kill(self):
        self._killed = True

    def close(self):                             # optional: stop the policy server it started
        pass

    def _run_policy(self, request, leg, save_root, should_stop):
        # Yours: run the policy on request.instructions, recording every frame into a new directory under
        # save_root / "eval" stamped from `leg`; poll should_stop() and self._killed at every step; let go
        # of the robot and cameras. Return (leg_dir, n_frames, whether should_stop ended it).
        raise NotImplementedError

FACTORY = ExecutorFactory(
    create=PolicyExecutor,
    display_name=PolicyExecutor.display_name,
    summary=PolicyExecutor.summary,
    segment_source="policy",                     # must equal the built executor's
    requirements=("a policy checkpoint",),       # shown by `tandem executors list`
    check=lambda settings: [],                   # the unmet ones, in words; cheap, starts nothing
)
```

Register it in the process with `tandem.register_human_executor("mypolicy", FACTORY)` (or the path
`"my_package.executor:FACTORY"`), or with an entry point in the package's `pyproject.toml`:

```toml
[project.entry-points."tandem.human_executors"]
mypolicy = "my_package.executor:FACTORY"
```

- **The factory** is an `ExecutorFactory` (`create`, `display_name`, `summary`, `segment_source`,
  `requirements`, `check`, `validate_options`), or a class taking an `ExecutorContext`, with
  `segment_source`, `display_name`, `summary` and `requirements` as class attributes and no `check` (its
  summary defaults to its docstring's first line).
- **`check(settings)`** returns the unmet requirements in words; an empty list means ready.
- **The built executor** must have `name`, `segment_source`, `display_name`, `summary`, `run` and `kill`.
- **`segment_source`** is the same on the factory and the executor: `"teleop"` or `"policy"`, never `"tamp"`,
  which the merge and export treat as the planner's.
- **The name** follows the [planner name rule](ADDING_A_PLANNER.md#names). A taken name is refused unless
  you pass `replace=True`, and two installed packages claiming one name is an error.
- **A plugin that fails to import** is listed with its error, and the other executors keep working.

### `ExecutorContext`

What `create` is called with, once per session.

| field | meaning |
|---|---|
| `profile` | The validated profile (its cameras, for one). |
| `session_dir` | The session's scratch directory, for anything that is not a recording. |
| `settings` | The machine's settings. A session passes `None`: read them at each leg with `tandem.core.settings.load()`, so a `tandem config set` between hand-offs applies at the next one. |
| `on_log(stream, text)` | Adds a line to the session log. |
| `on_emit(payload)` | Sends a message to every UI subscriber. |
| `on_problem(message)` | Shows the operator a problem, until the leg ends. |
| `options` | This executor's `hitl.human_executor_options.<name>` block, as the factory's optional `validate_options(options)` returned it at profile load; `{}` when unset. `validate_options` returns plain data (it is written to `profile.yml`) or raises `TandemError` or `ValueError` naming the key. |

## Choosing one

```bash
tandem executors list           # ready, needs setup (and what), or broken; ● marks the profile's
tandem executors use mypolicy   # the active profile's human phases now run with it (-p PROFILE for another)
```

- `executors use` sets `hitl.human_executor`. It warns, without refusing, when the executor isn't ready
  here, and refuses a name that is unknown or won't load. Flags: [USAGE.md](USAGE.md#commands); HTTP:
  [HTTP API](USAGE.md#http-api).
- A profile naming an executor this machine lacks won't load for collection, but can still be browsed,
  exported and edited. `tandem executors use NAME` repairs it.
- With phase planning on, a session looks the executor up before warm-up, so a broken plugin is found
  then, and `tandem doctor` has a `human executor` row: OK, WARN (with what is missing) or FAIL (won't load).
- While recording, an executor that isn't ready leaves only `a` (give up) at a human phase (`d` needs
  [`hitl.allow_unrecorded_human_phase`](CONFIGURATION.md#phase-planning-hitl)).

## The teleop executor

`TeleopExecutor` (`executors/teleop.py`) runs the DROID teleop driver (`tandem/teleop/driver.py`), one
process per leg. It needs:

- the `teleop.*` settings ([keys](CONFIGURATION.md#machine-settings-and-credentials), [setup](../README.md#6-teleop));
- a checkout of the [DROID fork](https://github.com/SamratSahoo/droid): the driver imports its
  `droid.stable_camera_env`, which upstream DROID lacks;
- a VR headset and controller, or a SpaceMouse.

`tandem executors list` checks the settings (`teleop.enabled`, `teleop.python`, `teleop.droid_dir`), not the
device, so a missing headset shows up as the driver's error when the leg starts.

- **A driver that can't start doesn't end the leg.** The arm is already released and may be in someone's
  hands, so the problem is shown and the leg waits for "return control" like any other.
- **The frame count is read after the driver exits.** It arrives with the driver's last event, once its
  videos are written.
- **One hand-off can hold several recordings.** The driver starts a new one whenever one ends short of
  quitting, and each is listed in `leg_dirs`.

## Known gaps

- Only `hitl.human_executor_options` fills `ExecutorContext.options`. The
  [hitl-tamp-vla import](CONFIGURATION.md#importing-a-hitl-tamp-vla-setup) never maps a config's `policy_*`
  keys into it: it refuses a learned-policy config with phase planning on, and otherwise drops those keys.
