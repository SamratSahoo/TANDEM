# Adding a human executor

A [human executor](README.md#terms) carries out a human phase as one recorded leg. It decides only **how**:
the proposal picks the phase, the camera check judges it. Only `teleop` ships; a package can add
others. `hitl.human_executor` picks one. Code: `src/tandem/executors/`.

## The protocol

Implement `tandem.executors.HumanExecutor`. Only `close()` is optional.

| member | contract |
|---|---|
| `name`, `display_name`, `summary` | The registered name; what listings show. |
| `segment_source` | `"teleop"` or `"policy"`, matching the factory; written to each leg's `_meta.json`. |
| `run(request, leg, *, save_root, should_stop)` | Once per attempt, at hand-over (`t` or UI): record the phase; return a [`HumanPhaseResult`](#humanphaseresult) once the robot and cameras are free. |
| `kill()` | Ends the running leg (from another thread); `run` returns `aborted`. Safe when idle. Clear the kill flag on return, not on start. |
| `close()` | Frees what it started, once, at any session end, after parking. Must not raise. |

| `run` argument | meaning |
|---|---|
| `request` | A [`HumanPhaseRequest`](#humanphaserequest); `None` only for an operator's hand-off between phases (always `teleop`). |
| `leg`, `save_root` | Record under a new `save_root/eval/` directory (`save_root`: the profile's `trajectories/`), stamped from `leg` per [the recording contract](ADDING_A_PLANNER.md#the-recording-contract). |
| `should_stop` | True once the operator returns control (`r`), the session stops, or an hour passes. Poll it; never block without it. |

Raise `tandem.executors.CustodyError` if the arm can't be freed; the session ends. `teleop` does when its
driver survives a kill.

### `HumanPhaseRequest`

| field | meaning |
|---|---|
| `phase_index`, `n_phases` | 0-based; `n_phases` is 0 with no plan behind the phase (then `leg.phase_index` is `None`). |
| `description`, `instructions`, `expected` | The phase in the model's words; what a person sees; what the camera check looks for. |
| `operator` | The magic operator, as `HumanOperator.to_json()`. `None` for a robot phase handed over (`on_robot_phase_failure: teleop`). |
| `attempt`, `missing` | Counts from 1; on a retry, what the last check found missing. |

### `HumanPhaseResult`

`HumanPhaseResult(status, n_frames=0, leg_dir=None, leg_dirs=())`

| `status` | meaning | the loop then |
|---|---|---|
| `done` | Carried out. | Checks its effects ([`hitl` keys](CONFIGURATION.md#phase-planning-hitl)), with `verify_retries` more tries on failure. |
| `ended_by_operator` | Stopped by the operator to move on (e.g. a step-limited policy). | Keeps the leg; accepts the phase unchecked ([`hitl.json`](DATA.md#hitljson)). |
| `aborted` | Cut off, or couldn't run (e.g. no policy server). | Fails the trial at `human_policy` ([outcomes](METHOD.md#how-a-trial-ends)). |

`n_frames` counts every frame the hand-off wrote. `leg_dir` is the last recording's directory (or `None`),
`leg_dirs` all of them in order (empty means just `leg_dir`).

While recording, a leg with `n_frames=0` is refused and re-asked (no retry spent), unless
[`hitl.allow_unrecorded_human_phase`](CONFIGURATION.md#phase-planning-hitl) is true. Giving up (`a`) runs
no executor; it is not `aborted`.

### Custody

Each human leg:

1. `backend.release_hardware()` frees the robot and cameras. A failure only warns.
2. `executor.run(...)`, skipped as `aborted` if a forced stop came during step 1.
3. The leg is counted (labeled and merged even if step 4 fails).
4. `backend.reacquire_hardware()`, unless `run` raised `CustodyError`. Failure ends the session.

Other exceptions from `run` fail the trial at `human_policy`; the session goes on.

## Registering one

Keep the module light: listing executors imports it.

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

Register it in-process with `tandem.register_human_executor("mypolicy", FACTORY)` (or
`"my_package.executor:FACTORY"`), or in `pyproject.toml`:

```toml
[project.entry-points."tandem.human_executors"]
mypolicy = "my_package.executor:FACTORY"
```

- **`check(settings)`** lists unmet requirements in words (empty means ready). Keep it cheap; start nothing.
- **A class** with `segment_source`, `display_name`, `summary`, `requirements` and optional
  `validate_options` attributes can be the factory, minus `check`.
- **Names** follow the [planner rule](ADDING_A_PLANNER.md#names). Reusing one needs `replace=True`; two
  installed packages claiming one is an error.

### `ExecutorContext`

Passed to `create`, once per session.

| field | meaning |
|---|---|
| `profile`, `session_dir` | The validated profile; scratch space (not for recordings). |
| `settings` | Machine settings; `None` in a session: call `tandem.core.settings.load()` each leg, so `tandem config set` applies next hand-off. |
| `on_log(stream, text)`, `on_emit(payload)`, `on_problem(message)` | Log a line; message every UI subscriber; show the operator a problem until the leg ends. |
| `options` | The profile's `hitl.human_executor_options.<name>` (`{}` if unset), after the factory's optional `validate_options(options)` at profile load. It returns plain data (saved to `profile.yml`) or raises `TandemError`/`ValueError` naming the key. |

## Choosing one

```bash
tandem executors list           # ready, needs setup (and what), or broken; ● marks the profile's
tandem executors use mypolicy   # for the active profile, or -p PROFILE
```

- `use` sets `hitl.human_executor`, warns if the executor isn't ready, and refuses an unknown or broken one
  ([HTTP](USAGE.md#http-api) too).
- A profile naming a missing executor can't collect until `use` fixes it, but can still be browsed, exported and edited.
- With phase planning on, a session looks the executor up before warm-up, so a broken plugin fails then;
  `tandem doctor`'s `human executor` row reports it.
- While recording, an unready executor leaves a human phase only `a` (give up), plus `d` with
  [`hitl.allow_unrecorded_human_phase`](CONFIGURATION.md#phase-planning-hitl).

## The teleop executor

`TeleopExecutor` runs the DROID teleop driver, one process per leg. It needs:

- the `teleop.*` settings ([keys](CONFIGURATION.md#machine-settings-and-credentials), [setup](../README.md#6-teleop));
- a [DROID fork](https://github.com/SamratSahoo/droid) checkout (upstream lacks `droid.stable_camera_env`);
- a VR headset and controller, or a SpaceMouse.

`tandem executors list` checks `teleop.enabled`, `teleop.python` and `teleop.droid_dir`, not the device: a
missing headset is a driver error at leg start. A driver
that can't start leaves the leg open, showing the problem until "return control". One hand-off can make
several recordings, all in `leg_dirs`.

## Known gaps

The [hitl-tamp-vla import](CONFIGURATION.md#importing-a-hitl-tamp-vla-setup) never fills
`hitl.human_executor_options` from learned-policy keys (`policy_*`, `open_loop_horizon`): it refuses them with
phase planning on and drops them otherwise.
