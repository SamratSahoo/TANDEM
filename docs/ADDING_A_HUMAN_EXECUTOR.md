# Adding a human executor

A [human executor](README.md#terms) carries out a human phase and records it as one leg. It only decides
**how** the phase is done: the plan decides which phase comes next, and the camera check decides whether it
worked.

TANDEM ships one executor, `teleop`. A package can add others, such as a learned policy. The code is in
`src/tandem/executors/`.

## Choosing one

```bash
tandem executors list           # each executor: ready, needs setup (and what), or broken
tandem executors use mypolicy   # the active profile now uses it; -p PROFILE for another
```

`use` sets the profile's `hitl.human_executor`. It warns if the executor isn't ready, and refuses an unknown or
broken one ([HTTP API](USAGE.md#http-api) too).

- A profile that names a missing executor can't collect until `use` fixes it. It can still be browsed,
  exported and edited.
- With phase planning on, a session looks the executor up before warm-up, so a broken plugin fails early.
  `tandem doctor`'s `human executor` row reports it.
- While recording, an executor that isn't ready leaves a human phase only `a` (give up). `d` also works if
  [`hitl.allow_unrecorded_human_phase`](CONFIGURATION.md#phase-planning-hitl) is true.

## Registering one

A minimal executor that runs a policy, and the factory that registers it:

```python
# my_package/executor.py
from tandem.executors import ExecutorFactory, HumanPhaseResult

class PolicyExecutor:
    name = "mypolicy"
    segment_source = "policy"
    display_name = "My policy"
    summary = "Runs a learned policy."

    def __init__(self, ctx):
        self.ctx, self._killed = ctx, False

    def run(self, request, leg, *, save_root, should_stop):
        try:
            # yours: run the policy, recording under save_root/"eval"; poll should_stop() and self._killed
            leg_dir, n_frames, stopped = self._run_policy(request, leg, save_root, should_stop)
            status = "aborted" if self._killed else "ended_by_operator" if stopped else "done"
            return HumanPhaseResult(status, n_frames=n_frames, leg_dir=leg_dir)
        finally:
            self._killed = False

    def kill(self):
        self._killed = True

FACTORY = ExecutorFactory(
    create=PolicyExecutor,
    display_name=PolicyExecutor.display_name,
    summary=PolicyExecutor.summary,
    segment_source=PolicyExecutor.segment_source,
    requirements=("a policy checkpoint",),
    check=lambda settings: [],
)
```

Register it with an entry point in `pyproject.toml`:

```toml
[project.entry-points."tandem.human_executors"]
mypolicy = "my_package.executor:FACTORY"
```

Or call `tandem.register_human_executor("mypolicy", FACTORY)` (or `"my_package.executor:FACTORY"`). Keep the
module light: listing executors imports it.

- **`check(settings)`** returns unmet requirements in words. An empty list means ready. Keep it cheap and start
  nothing.
- **A class can be the factory.** It needs `segment_source`, `display_name`, `summary` and `requirements` class
  attributes, and optionally `validate_options`. It can't have `check`.
- **Names** follow the [planner rule](ADDING_A_PLANNER.md#names). Reusing a name needs `replace=True`. Two
  installed packages claiming one name is an error.

### `ExecutorContext`

`create` receives this once per session.

| field | meaning |
|---|---|
| `profile`, `session_dir` | The validated profile; scratch space (not for recordings). |
| `rig` | The [rig](CONFIGURATION.md#the-rig): the cameras a leg records from and the robot's address. `None` in a context built by hand; then read `tandem.core.rig.load()`. |
| `settings` | TANDEM's settings. `None` in a session: call `tandem.core.settings.load()` each leg, so `tandem config set` applies at the next hand-off. |
| `on_log(stream, text)`, `on_emit(payload)`, `on_problem(message)` | Log a line; message every UI subscriber; show the operator a problem until the leg ends. |
| `options` | The profile's `hitl.human_executor_options.<name>`, or `{}` if unset. |

If the factory has `validate_options(options)`, it runs when the profile loads. It returns plain data, which is
saved to the profile's file, or raises `TandemError` or `ValueError` naming the bad key.

## The protocol

Implement `tandem.executors.HumanExecutor`. Every member is required except `close()`.

| member | contract |
|---|---|
| `name`, `display_name`, `summary` | The registered name, and what listings show. |
| `segment_source` | `"teleop"` or `"policy"`, matching the factory. It is written to each leg's `_meta.json`. |
| `run(request, leg, *, save_root, should_stop)` | Called once per attempt, at hand-over (`t` or the UI). Record the phase, then return a [`HumanPhaseResult`](#humanphaseresult) once the robot and cameras are free. |
| `kill()` | Ends the running leg, called from another thread. `run` then returns `aborted`. Safe when idle. Clear the kill flag when `run` returns, not when it starts. |
| `close()` | Frees what the executor started. Called once at any session end, after parking. Must not raise. |

The arguments to `run`:

| argument | meaning |
|---|---|
| `request` | A [`HumanPhaseRequest`](#humanphaserequest). It is `None` only when the operator takes the arm between phases, which always uses `teleop`. |
| `leg` | Stamps the leg per [the recording contract](ADDING_A_PLANNER.md#the-recording-contract). |
| `save_root` | The profile's `trajectories/<profile>/`. Record under a new directory in `save_root/eval/`. |
| `should_stop` | Turns true when the operator returns control (`r`), the session stops, or an hour passes. Poll it, and never block without it. |

If the arm can't be freed, raise `tandem.executors.CustodyError`, and the session ends. `teleop` does this when
its driver survives a kill.

### `HumanPhaseRequest`

| field | meaning |
|---|---|
| `phase_index`, `n_phases` | 0-based. `n_phases` is 0 when no plan is behind the phase (then `leg.phase_index` is `None`). |
| `description`, `instructions`, `expected` | The phase in the model's words; what the person sees; what the camera check looks for. |
| `operator` | The magic operator, as `HumanOperator.to_json()`. `None` for a robot phase handed to a person (`on_robot_phase_failure: teleop`). |
| `attempt`, `missing` | Counts from 1; on a retry, what the last check found missing. |

### `HumanPhaseResult`

```python
HumanPhaseResult(status, n_frames=0, leg_dir=None, leg_dirs=())
```

| `status` | meaning | what the loop does next |
|---|---|---|
| `done` | The phase was carried out. | Checks its effects ([`hitl` keys](CONFIGURATION.md#phase-planning-hitl)), with `verify_retries` more tries on failure. |
| `ended_by_operator` | The operator stopped it to move on, e.g. a step-limited policy. | Keeps the leg and accepts the phase unchecked ([`hitl.json`](DATA.md#hitljson)). |
| `aborted` | Cut off, or couldn't run (e.g. no policy server). | Fails the trial at `human_policy`. |

`n_frames` counts every frame the hand-off wrote. `leg_dir` is the last recording's directory (or `None`), and
`leg_dirs` lists every recording in order (empty means just `leg_dir`).

While recording, a result with `n_frames=0` is refused and the phase is asked again, without spending a retry.
[`hitl.allow_unrecorded_human_phase: true`](CONFIGURATION.md#phase-planning-hitl) accepts it instead. Giving up
with `a` runs no executor, so it is not `aborted`.

### Custody

Each human leg runs in four steps:

1. `backend.release_hardware()` frees the robot and cameras. A failure only warns.
2. `executor.run(...)` runs. It is skipped as `aborted` if a forced stop came during step 1.
3. The leg is counted. It is labeled and merged even if step 4 fails.
4. `backend.reacquire_hardware()` takes the robot back, unless `run` raised `CustodyError`. A failure ends the
   session.

Any other exception from `run` fails the trial at `human_policy`, and the session goes on.

## The teleop executor

`TeleopExecutor` runs the [DROID](https://github.com/SamratSahoo/droid/tree/TANDEM) teleop driver, one process per leg. It needs:

- the teleop runtime, from `tandem executors install teleop` ([setup](../README.md#6-teleop)). It holds the
  workstation side of [DROID's TANDEM branch](https://github.com/SamratSahoo/droid/tree/TANDEM) and
  [oculus_reader](https://github.com/rail-berkeley/oculus_reader), in their own environment;
- `teleop.enabled` ([keys](CONFIGURATION.md#tandem-settings-and-credentials)), which the install turns on;
- a VR headset and controller ([Meta Quest](https://www.meta.com/quest/)).

TANDEM sets `DROID_NUC_IP` and `TIPTOP_*_CAMERA_ID` for the driver from the rig, which DROID's
[`droid/misc/parameters.py`](https://github.com/SamratSahoo/droid/blob/TANDEM/droid/misc/parameters.py) reads. To run a DROID checkout and environment of your own instead, set both
`teleop.droid_dir` and `teleop.python`; the driver then runs there, and the runtime is ignored.

`tandem executors list` checks `teleop.enabled` and the runtime (or both overrides), but not the device. A
missing headset shows up as a driver error when the leg starts.

If the driver can't start, the leg stays open and shows the problem until you return control. One hand-off can
make several recordings, and all of them are in `leg_dirs`.
