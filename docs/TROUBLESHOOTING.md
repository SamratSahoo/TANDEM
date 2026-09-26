# Troubleshooting

Start with `tandem doctor`. It runs every check and says how to fix what it finds. Logs and session files are
described in [DATA.md](DATA.md#logs-and-session-files).

## A preempt didn't stop the arm

**Use the physical E-stop. It is the only thing that stops the arm at once.**

`p` stops the plan from sending further steps, but the motion already sent still finishes
([details](USAGE.md#collecting)).

## A camera won't open, or shows serial number 0

```bash
ps aux | grep -E 'tandem|tiptop'   # find the process holding the camera
```

Another process holds the camera. Right after a hand-off, [TiPToP](https://github.com/SamratSahoo/tiptop/tree/TANDEM)'s save workers free the cameras within a few
seconds, so wait and retry first.

## "the ZED SDK is not installed …, so ZED cameras will not open"

```bash
tandem planners install tiptop   # after installing the ZED SDK; adds only what is missing
```

The runtime was built without the ZED SDK, so it has no ZED Python API (`pyzed`). Install the
[ZED SDK](https://www.stereolabs.com/developers/release) first ([details](CONFIGURATION.md#the-planner-runtime)).

## "no camera extrinsics for serial(s) …"

```bash
tandem rig show                  # check the serial is right
tandem rig path --calibration    # the file to add its entry to
```

A camera in the rig has no entry in the rig's `calibration.json`, so the session won't start. Add the entry
([format and steps](../README.md#5-cameras-and-calibration)).

## The robot doesn't answer ("robot control … TimeoutError")

```bash
tandem rig show                          # the address tandem uses: robot.host
tandem rig set robot.host NUC_ADDRESS    # if that isn't the NUC's address
```

- `TimeoutError`: nothing answers at that address.
- `ConnectionRefusedError`: the NUC answers, but the shim isn't running. Start it ([step 3](../README.md#3-robot)).

Teleop uses the same `robot.host`, so fixing it fixes both.

## A perception server doesn't answer ("foundation stereo depth server …", "m2t2 grasp server …")

```bash
tandem servers status    # installed? answering?
tandem servers install   # if it says "not installed"
tandem servers start     # starts it and waits; a failure names its log
```

TiPToP asks both servers on every rollout ([step 4](../README.md#4-perception-servers)). Sessions start local
servers automatically, so this usually means the server isn't built, or it crashed while loading.
The log is `~/.local/state/tandem/logs/server-<name>.log`. Running out of GPU memory is the usual crash.

For a server on another machine, check its URL (`tandem rig show`):

- `gaierror`: the hostname doesn't resolve.
- `ConnectionRefusedError`: nothing listens on that port.

## A trial was excluded

```bash
tandem traj relabel <id> success --force   # file it as a success anyway
```

Usually a human phase failed its camera check, retries included. To see why, open the trial's `hitl.json`: each
failed check in `verifications` has `satisfied: false` and the model's `reason`. The judged images are
`vlm/*_classify-*` ([format](DATA.md#hitljson)).

If the checks themselves are wrong, set `hitl.on_verification_failure: label` while you tune them, and label
those trials yourself ([relabeling](USAGE.md#reviewing-and-exporting)).

## The plan leaves part of the instruction out

Put the missing object on the table, or reword the task.

Usually a named object wasn't detected. `tandem plan`, the web UI and the session log (`NOT part of the plan — …`)
list each dropped clause and why ([details](USAGE.md#planning-from-a-photo)).

## "I did it" is refused at a human step

While recording, a human phase must run through its executor. Do one of these:

- Take the arm with `t`. If `t` isn't offered, `tandem executors list` says why.
- Set [`hitl.allow_unrecorded_human_phase: true`](CONFIGURATION.md#phase-planning-hitl).
- Run `tandem collect --no-record`.

## Teleop says "the teleop runtime is not installed"

```bash
tandem executors install teleop
```

Teleop's driver runs in an environment TANDEM builds. This builds it (a few minutes) and turns teleop on. If it
says ZED cameras won't open, install the [ZED SDK](https://www.stereolabs.com/developers/release) and run it
again. It adds only what is missing.

## The teleop driver can't reach the VR headset

```bash
adb devices   # the headset should be listed as "device"
```

[oculus_reader](https://github.com/rail-berkeley/oculus_reader) talks to the headset over `adb` (`sudo apt install adb`). Connect the headset by USB, turn on
developer mode, and accept the prompt in the headset. A headset that has never run [DROID](https://github.com/SamratSahoo/droid/tree/TANDEM) teleop also needs the
app ([step 6](../README.md#6-teleop)).

## The planner shows as outdated

```bash
tandem planners install tiptop
```

Its pinned sources moved, so the runtime was built from other commits ([why](CONFIGURATION.md#the-planner-runtime)).

## The runtime build failed

```bash
tandem planners install tiptop   # after fixing the cause; finished steps are skipped
```

The build prints its log path (`runtime-build-<time>.log`) first. The usual causes each have a `tandem doctor`
row: missing `nvcc`, a torch/CUDA mismatch (`cuda runtime`), or a full disk (`disk space`).

## A leg fails with "No level patch of … is large enough"

```yaml
planner:
  options:
    tamp:
      placement_fill_occluded: true   # a box wall or lid hides the floor
      # placement_flatness_tol: 0.012 # a noisy floor: raise it (default 0.008)
      # placement_support_required: false   # last resort: use the bounding box
```

With `placement_support: true`, no visible level patch of the goal surface fits the object's footprint plus
`placement_support_margin` ([details](CONFIGURATION.md#surface-fitted-placement)). The full message is in the
session log, or in that pass's `metadata.json` under the session's `perception/`
([where](DATA.md#logs-and-session-files)).

## "encoder_path does not exist"

```bash
tandem planners install tiptop
```

The paper's profiles name the DATAFARM checkpoint `vae/checkpoints/vae_full_v2.pt`, which the runtime install
places. TANDEM looks for it beside the profile's file, then in the runtime. `tandem doctor` warns until the
install has run.

## A TAMP setting seems to do nothing

```bash
tandem profile show <name> --planner   # exactly what the planner receives
```

A key missing from that output never applied. `tandem doctor`'s `tamp settings` row flags a key that needs
another, such as a `placement_*` key without `placement_support: true`. It isn't a typo; TANDEM refuses to load a
profile with a misspelled key and suggests the closest one ([details](CONFIGURATION.md#planner-settings)).

## The videos won't scrub in the browser

A proxy in front of `tandem ui` is probably dropping the `Range` header that `/api/media/` needs. Configure it
to pass `Range` through ([details](USAGE.md#http-api)).

## My profiles are gone after updating TANDEM

```bash
tandem profile migrate   # or tandem init
```

Profiles from before version 3 (a directory each, with its own cameras and robot) aren't listed until they are
moved. Migrating moves them and sets up the rig from them, without deleting anything
([details](CONFIGURATION.md#older-profiles)).

## A profile's planner or executor isn't installed

```bash
tandem planners use NAME     # or install the missing planner's package
tandem executors use NAME    # add -p PROFILE for another profile
```

`tandem collect` refuses such a profile, with `no planner named '…'` or `Unknown human executor '…'`. You can
still browse, export and edit it.

## A sidecar dies with "No module named 'tandem_sidecar'"

Your environment's activation (for example [pixi](https://pixi.sh)'s `[activation.env]`) overwrote `PYTHONPATH`, dropping TANDEM's
`tandem_sidecar` kit. Make it append to `PYTHONPATH` instead ([details](ADDING_A_PLANNER.md#sidecars)).
