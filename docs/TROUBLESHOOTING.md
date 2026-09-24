# Troubleshooting

`tandem doctor` is the first thing to run: every check, what it found, and what to do about it.

- [A preempt didn't stop the arm](#a-preempt-didnt-stop-the-arm)
- [A camera won't open, or shows serial number 0](#a-camera-wont-open-or-shows-serial-number-0)
- ["no camera extrinsics for serial(s) …"](#no-camera-extrinsics-for-serials-)
- [A trial was excluded](#a-trial-was-excluded)
- [The plan leaves part of the instruction out](#the-plan-leaves-part-of-the-instruction-out)
- ["I did it" is refused at a human step](#i-did-it-is-refused-at-a-human-step)
- [The planner shows as outdated](#the-planner-shows-as-outdated)
- [The runtime build failed](#the-runtime-build-failed)
- [A leg fails with "No level patch of … is large enough"](#a-leg-fails-with-no-level-patch-of--is-large-enough)
- [A TAMP setting seems to do nothing](#a-tamp-setting-seems-to-do-nothing)
- [The videos won't scrub in the browser](#the-videos-wont-scrub-in-the-browser)
- [Where everything is after a session that went wrong](#where-everything-is-after-a-session-that-went-wrong)

## A preempt didn't stop the arm

No software button can. Unless the planner declares a cooperative stop, it is handed a whole trajectory
segment in one request and has no abort, so the motion runs to the end of that segment (one that does
declare it stops at its next step boundary). A preempt stops *further plan steps*. **The physical E-stop is
the only instant stop.**

## A camera won't open, or shows serial number 0

Serial `0` means another process still holds the camera. After a teleop hand-off the cameras take about
15 seconds to be released, because the save workers inherited the device descriptors and have to exit first.
Wait, then retry. If it persists, another tandem or tiptop process is still running.

## "no camera extrinsics for serial(s) …"

Extrinsics are keyed by camera serial, and a configured serial with no entry stops the session before it
warms up. Add it to the profile's `calibration.json`
([format](CONFIGURATION.md#cameras-and-calibration)), or import from a checkout that has it:
`tandem profile create <name> --import-from <path>`.

## A trial was excluded

A human phase still failed its check after its retries, so the trial was kept out of the dataset, as the
paper does. Open its `hitl.json`: the failing verdicts under `verifications` say which effect the camera did
not see, and why (`satisfied: false` is the field to read). `vlm/` holds the image each verdict was made on.
If the classifier was wrong rather than the person, set `hitl.on_verification_failure: label` while you
calibrate it, so every disagreement between you and the check becomes a labeled data point.
`tandem traj relabel <id> success` refuses an excluded trial; `--force` (a confirm in the web UI) overrules
the check on purpose, records that under `overruled` in its `hitl.json`, and is the one way such a trial
reaches the export.

## The plan leaves part of the instruction out

`tandem plan`, the session log and the UI list each clause the model could not express, with its reason. It
is almost always an object the instruction names that perception did not detect. Put it on the table, or
reword the task, before collecting: the dataset is labeled with the whole instruction either way.

## "I did it" is refused at a human step

While recording, a human phase has to be carried out through the executor (`t`, to take the arm) so the
episode has that stretch of demonstration. Set `hitl.allow_unrecorded_human_phase: true` to accept a step
done off the record, or collect with `--no-record`. `tandem executors list` says whether teleop is ready on
this machine.

## The planner shows as outdated

This version of tandem pins other commits than the ones the runtime was built from. Run
`tandem planners install tiptop`. It replaces only the source trees whose pin moved, then rebuilds against
the pixi environment already on disk.

## The runtime build failed

The full log is `~/.local/state/tandem/logs/runtime-build-<time>.log`. The usual causes are a missing `nvcc`,
a torch/CUDA mismatch, or running out of disk mid-compile. `tandem planners install tiptop` retries, and
skips a step already done.

## A leg fails with "No level patch of … is large enough"

`placement_support` found no observed patch of the goal surface that would hold the object with
`placement_support_margin` around it. From the camera's side a box's near wall and lid can hide most of its
floor: `placement_fill_occluded: true` counts the hidden floor, and a noisy floor needs a larger
`placement_flatness_tol`. `placement_support_required: false` places on the bounding box instead, which is
what the setting exists to avoid. The full reason, with the object's footprint and the margin, is in the
leg's `metadata.json` and in the session log. See
[surface-fitted placement](CONFIGURATION.md#surface-fitted-placement-placement_).

## A TAMP setting seems to do nothing

Run `tandem profile show <name> --planner`: that is exactly what the planner receives, so a key missing
from it never applied. Unknown keys are refused when the profile loads, and `tandem doctor` warns about a
key that is set but does nothing without another one.

## The videos won't scrub in the browser

They are served with HTTP Range support. If scrubbing fails, check that nothing is proxying `/api/media/`
without passing Range headers through.

## Where everything is after a session that went wrong

- `~/.local/state/tandem/logs/session-<id>.json`: the session's summary and log.
- `~/.local/state/tandem/sessions/<profile>/<id>/events.jsonl`: one line per event, in order. The event
  types are in [METHOD.md](METHOD.md#the-events-file).
- Beside it: each perception pass that recorded nothing, and `vlm/<trajectory id>/` for a trial that was
  never filed.
- `~/.local/state/tandem/logs/export.log`: what `tandem export lerobot` logged, appended per run.
