# Troubleshooting

Symptom, then fix. Run `tandem doctor` first: it checks everything tandem needs and says what to do about each
problem. Logs and session files are listed in [DATA.md](DATA.md#logs-and-session-files).

## A preempt didn't stop the arm

Expected: a preempt (`p`) stops further plan steps, not the motion segment already sent to the robot.
**The physical E-stop is the only instant stop** ([details](USAGE.md#collecting)).

## A camera won't open, or shows serial number 0

Another process still holds the camera. After a hand-off, TiPToP's save workers release the cameras within a
few seconds: wait, then retry. If it persists, look for a stray process: `ps aux | grep -E 'tandem|tiptop'`.

## "no camera extrinsics for serial(s) …"

A camera in the profile has no entry in its `calibration.json`, so the session stops before warm-up. Check the
serial, then add its entry ([format](CONFIGURATION.md#cameras-and-calibration)), or import a hitl-tamp-vla
checkout that has it: `tandem profile create <name> --import-from <path>`
([details](CONFIGURATION.md#importing-a-hitl-tamp-vla-setup)).

## A trial was excluded

Usually a human phase still failed its camera check after its retries. Its `hitl.json` lists the failed checks as
`verifications` with `satisfied: false`, each with the model's `reason`; the images judged are the
`*_classify-*` files in its `vlm/` ([format](DATA.md#hitljson)).

- If the check, not the person, was wrong: set `hitl.on_verification_failure: label` while you calibrate it,
  so you label each such trial yourself.
- To file one as a success anyway: `tandem traj relabel <id> success --force`
  ([details](USAGE.md#reviewing-and-exporting)).

## The plan leaves part of the instruction out

Usually an object the instruction names wasn't detected. `tandem plan`, the session log
(`NOT part of the plan — …`) and the web UI list each such clause with its reason. Put the object on the table,
or reword the task, before collecting ([details](USAGE.md#planning-from-a-photo)).

## "I did it" is refused at a human step

While recording, a human phase must be carried out through its executor. Either:

- take the arm with `t` (no `t` means the executor isn't ready: `tandem executors list` says what it needs);
- set [`hitl.allow_unrecorded_human_phase: true`](CONFIGURATION.md#phase-planning-hitl); or
- collect with `tandem collect --no-record`.

## The planner shows as outdated

Run `tandem planners install tiptop` ([why](CONFIGURATION.md#the-planner-runtime)).

## The runtime build failed

The build prints the path of its log (`runtime-build-<time>.log`) as it starts. Usual causes: a missing `nvcc`,
a torch/CUDA mismatch, or a full disk (`tandem doctor`'s `nvcc`, `cuda runtime` and `disk space` rows). Fix it
and run `tandem planners install tiptop` again: it skips every step already done.

## A leg fails with "No level patch of … is large enough"

With `placement_support: true`, no observed level patch of the goal surface holds the object's footprint plus
`placement_support_margin`. The full message is in the session log, and in that perception pass's
`metadata.json` under the session directory's `perception/` ([where](DATA.md#logs-and-session-files)). Under
`planner.options.tamp` ([details](CONFIGURATION.md#surface-fitted-placement)):

- A box's near wall or lid hides its floor: `placement_fill_occluded: true` counts the hidden floor.
- A noisy floor: raise `placement_flatness_tol`.
- Last resort: `placement_support_required: false` places on the bounding box.

## A TAMP setting seems to do nothing

`tandem profile show <name> --planner` prints exactly what the planner receives; a key missing from it never
applied. `tandem doctor`'s `tamp settings` row flags a key that does nothing without another (for example a
`placement_*` key without `placement_support: true`). A misspelled key can't be the cause: the profile refuses
to load and names the closest valid key ([details](CONFIGURATION.md#planner-settings)).

## The videos won't scrub in the browser

A proxy in front of `tandem ui` is probably dropping the `Range` header that `/api/media/` needs: pass it
through ([details](USAGE.md#http-api)).

## A profile names a planner or executor this machine doesn't have

`tandem collect` refuses it (`no planner named '…' is installed on this machine`, or
`Unknown human executor '…'`). Install the package that provides it, or switch with `tandem planners use NAME`
or `tandem executors use NAME` (`-p PROFILE` for one that isn't active). Meanwhile the profile can still be
browsed, exported and edited.

## A sidecar dies with "No module named 'tandem_sidecar'"

Your environment's activation replaced the sidecar's `PYTHONPATH`, where tandem puts the `tandem_sidecar` kit
(for example a pixi `[activation.env]` that sets it). Append to `PYTHONPATH` instead of setting it
([details](ADDING_A_PLANNER.md#sidecars)).
