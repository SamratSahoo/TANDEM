# Troubleshooting

Run `tandem doctor` first: it says how to fix what it finds. Logs and session files:
[DATA.md](DATA.md#logs-and-session-files).

## A preempt didn't stop the arm

`p` stops further plan steps, not the motion already sent. **Only the physical E-stop stops the arm at once**
([details](USAGE.md#collecting)).

## A camera won't open, or shows serial number 0

Another process holds it. After a hand-off, TiPToP's save workers free the cameras within seconds: wait and
retry. Else find the stray process: `ps aux | grep -E 'tandem|tiptop'`.

## "the ZED SDK is not installed …, so ZED cameras will not open"

The runtime was built without the ZED SDK, so it has no ZED Python API (`pyzed`). Install the
[ZED SDK](https://www.stereolabs.com/developers/release), then run `tandem planners install tiptop`: it adds
only what is missing ([details](CONFIGURATION.md#the-planner-runtime)).

## "no camera extrinsics for serial(s) …"

A rig camera has no entry in the rig's `calibration.json` (`tandem rig path --calibration`), so the session
won't start. Check the serial (`tandem rig show`), then add its entry ([format](CONFIGURATION.md#the-rig);
[steps](../README.md#5-cameras-and-calibration)).

## The robot doesn't answer ("robot control … TimeoutError")

tandem reaches the robot at the rig's `robot.host` (`tandem rig show`). `TimeoutError`: nothing answers at that
address; if it isn't the NUC's, `tandem rig set robot.host 172.16.0.5`. `ConnectionRefusedError`: the NUC
answers but the shim isn't running ([step 3](../README.md#3-robot)). Teleop reaches the NUC through DROID's
`nuc_ip` instead: keep the two equal.

## A perception server doesn't answer ("foundation stereo depth server …", "m2t2 grasp server …")

TiPToP asks both on every rollout. Start each with `pixi run server` in its checkout
([step 4](../README.md#4-perception-servers)). One on another machine goes in the rig:
`tandem rig set planners.tiptop.perception.foundation_stereo.url http://HOST:1234` (M2T2: `…m2t2.url`, port
8123). `gaierror`: the hostname doesn't resolve. `ConnectionRefusedError`: nothing listens on that port.

## A trial was excluded

Usually a human phase failed its camera check, retries included. `hitl.json`'s `verifications` show each
failed check (`satisfied: false`) and the model's `reason`; `vlm/*_classify-*` are the judged images
([format](DATA.md#hitljson)).

- Check wrong? Set `hitl.on_verification_failure: label` while calibrating, and label such trials yourself.
- File one as a success anyway: `tandem traj relabel <id> success --force` ([details](USAGE.md#reviewing-and-exporting)).

## The plan leaves part of the instruction out

Usually a named object wasn't detected. `tandem plan`, the session log (`NOT part of the plan — …`) and the
web UI list each dropped clause and why. Put the object on the table or reword the task
([details](USAGE.md#planning-from-a-photo)).

## "I did it" is refused at a human step

While recording, a human phase must run through its executor. Either:

- take the arm with `t` (no `t`? `tandem executors list` says why);
- set [`hitl.allow_unrecorded_human_phase: true`](CONFIGURATION.md#phase-planning-hitl); or
- run `tandem collect --no-record`.

## The planner shows as outdated

Run `tandem planners install tiptop` ([why](CONFIGURATION.md#the-planner-runtime)).

## The runtime build failed

It prints its log path (`runtime-build-<time>.log`) first. Usual causes (`tandem doctor` rows): missing
`nvcc`, torch/CUDA mismatch (`cuda runtime`), full disk (`disk space`). Fix, then rerun
`tandem planners install tiptop`; it skips finished steps.

## A leg fails with "No level patch of … is large enough"

With `placement_support: true`, no seen level patch of the goal surface fits the object's footprint plus
`placement_support_margin`. Full message: the session log, or that pass's `metadata.json` in the session's
`perception/` ([where](DATA.md#logs-and-session-files)). In `planner.options.tamp`
([details](CONFIGURATION.md#surface-fitted-placement)):

- Box wall or lid hides the floor: `placement_fill_occluded: true` counts it.
- Noisy floor: raise `placement_flatness_tol`.
- Last resort: `placement_support_required: false` uses the bounding box.

## "vae_path does not exist"

The paper's profiles name the DATAFARM checkpoint the runtime install places, `vae/checkpoints/vae_full_v2.pt`,
found beside the profile's file or in the runtime. `tandem doctor` warns about it until
`tandem planners install tiptop` has run.

## A TAMP setting seems to do nothing

`tandem profile show <name> --planner` prints what the planner receives; a key not there never applied.
`tandem doctor`'s `tamp settings` row flags a key that needs another (e.g. `placement_*` without
`placement_support: true`). Typos can't be it: the profile won't load and names the closest key
([details](CONFIGURATION.md#planner-settings)).

## The videos won't scrub in the browser

A proxy in front of `tandem ui` is likely dropping the `Range` header `/api/media/` needs; pass it through
([details](USAGE.md#http-api)).

## My profiles are gone after updating tandem

Profiles from before version 3 (a directory each, with its own cameras and robot) aren't listed until moved:
`tandem profile migrate`, or `tandem init`, moves them and sets up the rig from them. Nothing is deleted
([details](CONFIGURATION.md#older-profiles)).

## A profile's planner or executor isn't installed

`tandem collect` refuses it (`no planner named '…'` or `Unknown human executor '…'`). Install its package,
or switch with `tandem planners use NAME` / `tandem executors use NAME` (`-p PROFILE` for another profile).
Browsing, export and edits still work.

## A sidecar dies with "No module named 'tandem_sidecar'"

Your environment's activation (e.g. pixi `[activation.env]`) overwrote `PYTHONPATH`, dropping tandem's
`tandem_sidecar` kit. Append to it instead ([details](ADDING_A_PLANNER.md#sidecars)).
