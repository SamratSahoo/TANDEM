// Collect — start a session, watch it, and label each rollout while looking at it.

import { api, streamSession } from "../api.js";
import { clear, fmtDuration, h, mount } from "../dom.js";
import { renderReview } from "../review.js";
import { reportError, toast } from "../app.js";

const PIPELINE = [
  ["warm", ["warming"]],
  ["perceive", ["rolling"]],
  ["execute", ["rolling"]],
  ["label", ["awaiting_label", "labeling"]],
  ["next", ["awaiting_task"]],
];

const HANDOVER_STATES = new Set(["handing_off", "teleop_handoff", "awaiting_human_phase"]);

const TERMINAL = new Set(["stopped", "failed"]);

export function renderCollect(host, state) {
  if (!state.profile) {
    mount(host, h("div.empty", h("div.big", "◈"), h("div", "No profile yet.")));
    return;
  }
  const shell = h("div.stack");
  mount(host, shell);

  // A live session is shown whatever the runtime probe says. The readiness gate exists to stop
  // you STARTING a session; hiding one that is already running would leave an operator with no
  // controls while the arm is still moving.
  api.sessions()
    .then((payload) => {
      const live = (payload.sessions || []).find(
        (session) => session.profile === state.profile && !TERMINAL.has(session.state)
      );
      if (live) attachSession(shell, state, live);
      else if (!state.runtimeReady) mount(shell, notReady());
      else mount(shell, startForm(shell, state));
    })
    .catch(() => mount(shell, state.runtimeReady ? startForm(shell, state) : notReady()));
}

function notReady() {
  return h("div.card",
    h("div.card-title", "The planner's runtime is not built"),
    h("div.card-hint",
      "Collection needs the runtime of the planner this profile uses (Settings shows which, and what it " +
      "is missing). Visualizing already-collected trajectories works without it."),
    h("div.alert.info", h("span.mono", "tandem planners install <planner>"),
      " on the workstation (`tandem init` does too) — a GPU planner's first build can take 5–20 minutes."));
}

// ---- start -----------------------------------------------------------------

function startForm(shell, state) {
  const profile = state.profiles.find((p) => p.name === state.profile) || {};
  const taskInput = h("input", { value: profile.goal || profile.prompt || "", placeholder: "what should the robot do?" });
  const episodesInput = h("input", { type: "number", min: "1", placeholder: String(profile.target || 20) });
  const executeInput = h("input", { type: "checkbox", checked: true, style: { width: "auto" } });
  const recordInput = h("input", { type: "checkbox", checked: true, style: { width: "auto" } });

  const startButton = h("button.primary.big", { onclick: start }, "Start session");

  async function start() {
    startButton.disabled = true;
    startButton.textContent = "Warming up…";
    try {
      const session = await api.createSession({
        profile: state.profile,
        task: taskInput.value.trim() || null,
        execute: executeInput.checked,
        record: recordInput.checked,
        episodes: episodesInput.value ? Number(episodesInput.value) : null,
      });
      attachSession(shell, state, session);
    } catch (error) {
      reportError(error, "Could not start the session");
      startButton.disabled = false;
      startButton.textContent = "Start session";
    }
  }

  return h("div.card",
    h("div.card-title", `Collect under ${state.profile}`),
    h("div.card-hint",
      "The driver warms up once — the planner, the cameras, the robot — and then loops rollouts " +
      "against that warm state. Warmup takes about a minute; after that each rollout starts immediately."),
    h("div.field", h("label", "Task"), taskInput,
      h("div.desc", "Also the language label stored with every episode.")),
    h("div.field-row",
      h("div.field", h("label", "Stop after"), episodesInput,
        h("div.desc", "Blank to keep going until you stop.")),
      h("div.field",
        h("label", "Options"),
        h("div.row", executeInput, h("span.small", "move the robot")),
        h("div.row", { style: { marginTop: "6px" } }, recordInput, h("span.small", "record camera video")),
        h("div.desc", "Unchecking “move the robot” perceives and plans without touching the arm."))),
    h("div.row", startButton,
      h("span.faint.small", "A preempt stops further plan steps, not the arm — use the E-stop for a hard halt."))
  );
}

// ---- live session ----------------------------------------------------------

function attachSession(shell, state, initial) {
  let summary = initial;
  let closeStream = null;
  let reviewedRollout = null;

  const pipelineHost = h("div.pipeline");
  const statHost = h("div.row.wrap", { style: { gap: "18px" } });
  const actionHost = h("div.row.wrap", { style: { gap: "8px" } });
  const noticeHost = h("div");
  const phaseHost = h("div");
  const reviewHost = h("div");
  const logHost = h("div.logs");

  let pinned = true;
  logHost.addEventListener("scroll", () => {
    pinned = logHost.scrollTop + logHost.clientHeight >= logHost.scrollHeight - 24;
  });

  mount(shell,
    h("div.card",
      h("div.card-head",
        h("div.card-title", h("span.dot.live"), " ", summary.task || state.profile),
        h("span.faint.small", { id: "sess-id" }, summary.id)),
      pipelineHost, h("div", { style: { height: "12px" } }), statHost,
      h("div", { style: { height: "14px" } }), noticeHost, actionHost),
    phaseHost,
    reviewHost,
    h("div.card", h("div.card-title", "Driver output"), logHost));

  function appendLog(line) {
    const kind = line.stream === "tandem" ? "l-tandem"
      : /ERROR|Traceback/.test(line.text) ? "l-err"
      : /WARNING/.test(line.text) ? "l-warn" : "";
    const stamp = new Date((line.at || Date.now() / 1000) * 1000).toLocaleTimeString();
    logHost.appendChild(h("div", h("span.ts", stamp), h("span", { class: kind }, line.text)));
    while (logHost.childElementCount > 3000) logHost.removeChild(logHost.firstChild);
    if (pinned) logHost.scrollTop = logHost.scrollHeight;
  }

  function paint() {
    renderPipeline(pipelineHost, summary.state);
    renderStats(statHost, summary);
    renderNotice(noticeHost, summary);
    renderPhase(phaseHost, summary);
    renderActions(actionHost, summary, state, { onEnded: () => renderCollect(shell.parentElement, state) });
    maybeShowReview();
  }

  // At the label prompt, show the rollout that just finished so the operator decides while
  // looking at it rather than from memory.
  function maybeShowReview() {
    const current = summary.current;
    if (summary.state === "awaiting_label" && current && current.id !== reviewedRollout) {
      reviewedRollout = current.id;
      api.trajectory(state.profile, current.id)
        .then((traj) => {
          const card = h("div.card",
            h("div.card-head",
              h("div.card-title", "Review this rollout"),
              h("span.faint.small", `${traj.n_frames} frames · ${fmtDuration(traj.duration_s)}`)));
          const inner = h("div");
          card.appendChild(inner);
          mount(reviewHost, card);
          renderReview(inner, { profile: state.profile, trajectory: traj });
        })
        .catch(() => clear(reviewHost));
    } else if (summary.state !== "awaiting_label" && summary.state !== "labeling") {
      if (reviewHost.childElementCount) clear(reviewHost);
      reviewedRollout = null;
    }
  }

  closeStream = streamSession(summary.id, (message) => {
    if (message.type === "log") appendLog(message);
    else if (message.type === "state") {
      summary = { ...summary, ...message };
      paint();
      if (TERMINAL.has(summary.state) && closeStream) {
        closeStream();
        closeStream = null;
      }
    } else if (message.type === "event") {
      // Events also arrive folded into the state frames; surfacing the interesting ones as
      // toasts means the operator does not have to be watching the log.
      if (message.event === "rollout_aborted") toast.info("Rollout aborted", "The session is still warm.");
      if (message.event === "teleop_handoff_start") toast.info("Handing the arm over…", "Releasing the cameras takes ~15s.");
      // An excluded trial never reaches the label prompt, so this is the one cue that the
      // demonstration just given is not in the dataset.
      if (message.event === "trial_excluded") toast.err("Trial excluded, not labeled", message.reason || "");
    }
  });

  // No separate snapshot fetch. The stream already opens with a state frame and the recent
  // log history, so asking for the same thing over HTTP as well raced it: whichever arrived
  // second appended a second copy of every line the two had in common.
  paint();
}

function renderPipeline(host, state) {
  clear(host);
  if (HANDOVER_STATES.has(state)) {
    const label = {
      handing_off: "handing the arm over…",
      teleop_handoff: "the human has the arm",
      awaiting_human_phase: "waiting on you",
    }[state];
    host.appendChild(h("div.step.now",
      h("span.pd", { style: { background: "var(--violet)" } }), label));
    return;
  }
  let reached = false;
  PIPELINE.forEach(([label, states], index) => {
    if (index) host.appendChild(h("span.sep", "·"));
    const isNow = states.includes(state) && !reached;
    if (isNow) reached = true;
    host.appendChild(h(`div.step${isNow ? ".now" : reached ? "" : ".done"}`, h("span.pd"), label));
  });
}

function renderStats(host, summary) {
  const elapsed = (Date.now() / 1000) - (summary.started_at || Date.now() / 1000);
  mount(host,
    stat(String(summary.success || 0), "success", "var(--green)"),
    stat(String((summary.labeled || 0) - (summary.success || 0)), "failure", "var(--red)"),
    stat(`${summary.labeled || 0}/${summary.target || "—"}`, "labeled"),
    // Not labeled and not in the dataset, so counted apart from both.
    summary.excluded ? stat(String(summary.excluded), "excluded", "var(--amber)") : null,
    summary.phase_progress
      ? stat(`${summary.phase_progress[0]}/${summary.phase_progress[1]}`, "phases", "var(--violet)")
      : null,
    stat(fmtDuration(elapsed), "elapsed"),
    h("div.spacer"),
    h("span.chip" + (TERMINAL.has(summary.state) ? "" : ".accent"), summary.state.replace(/_/g, " ")));
}

function stat(value, label, color) {
  return h("div",
    h("div", { style: { fontSize: "20px", fontWeight: 700, color: color || "var(--text)" } }, value),
    h("div.faint.small", label));
}

/**
 * The step the plan says only a person can do.
 *
 * The expectations are shown because they are the same list the model is about to be asked
 * about — being judged against a standard you were never told is the fastest way to make an
 * operator stop trusting the verification.
 */
function renderPhase(host, summary) {
  const phase = summary.human_phase;
  if (!phase) {
    clear(host);
    return;
  }

  const id = summary.id;
  const act = (fn, failure) => async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    try {
      await fn();
    } catch (error) {
      reportError(error, failure);
      button.disabled = false;
    }
  };

  const heading = phase.total
    ? `${phase.description} — step ${phase.index + 1} of ${phase.total}`
    : phase.description || "Your turn";

  // Only the answers the session will accept. While recording, a step done by hand has no
  // demonstration, so "I did it" is not offered (phase.by_hand); the step goes through whoever
  // carries out human steps (hitl.human_executor), the teleop rig unless the profile says otherwise.
  const executor = summary.human_executor || {};
  const byHand = phase.by_hand !== false;
  const isTeleop = (executor.name || "teleop") === "teleop";
  const stuck = !byHand && !executor.ready;

  mount(host,
    h("div.card", { style: { borderColor: "var(--violet)" } },
      h("div.card-head",
        h("div.card-title", h("span.chip.violet", "your turn"), " ", heading),
        phase.attempt > 1 ? h("span.chip.eval", `attempt ${phase.attempt}`) : null),

      phase.instructions ? h("div", { style: { fontSize: "15px", marginBottom: "14px" } }, phase.instructions) : null,

      phase.missing && phase.missing.length
        ? h("div.alert", { style: { marginBottom: "14px" } },
            h("div", { style: { fontWeight: 650, marginBottom: "6px" } },
              "That did not look done. Still expected:"),
            h("ul", { style: { margin: 0, paddingLeft: "18px" } },
              ...phase.missing.map((item) => h("li", item))))
        : null,

      phase.expected && phase.expected.length
        ? h("div", { style: { marginBottom: "16px" } },
            h("div.faint.small", { style: { marginBottom: "4px" } }, "When you are done, this should be true:"),
            h("ul.muted.small", { style: { margin: 0, paddingLeft: "18px" } },
              ...phase.expected.map((item) => h("li", item))))
        : null,

      stuck
        ? h("div.alert", { style: { marginBottom: "14px" } },
            h("strong", "This step is being recorded, and it cannot be done here. "),
            `${executor.display_name || "Its executor"} is not ready on this machine: `,
            (executor.unmet && executor.unmet.length ? executor.unmet.join("; ") : executor.error) || "it is not set up.")
        : null,

      h("div.row.wrap", { style: { gap: "8px" } },
        executor.ready
          ? h("button.violet.big", {
              title: isTeleop
                ? "Take the arm through the teleop rig, then hand it back."
                : `Hand this step to ${executor.display_name}, then take the arm back when it is done.`,
              onclick: act(() => api.teleopSwitch(id), "Could not hand the step over"),
            }, isTeleop ? "Take the arm" : `Run ${executor.display_name}`)
          : null,
        byHand
          ? h("button.primary.big", {
              title: "You did it by hand. The plan checks a photo before carrying on.",
              onclick: act(() => api.humanPhaseDone(id), "Could not complete the phase"),
            }, "✔ I did it")
          : null,
        h("div.spacer"),
        h("button.ghost", {
          title: "Give up on this step, and with it this attempt at the task.",
          onclick: act(() => api.humanPhaseAbort(id), "Could not abort the phase"),
        }, "Give up on this task"))));
}

function renderNotice(host, summary) {
  clear(host);
  if (summary.error) {
    host.appendChild(h("div.alert.err", summary.error));
  } else if (summary.handoff_error) {
    host.appendChild(h("div.alert", summary.handoff_error));
  } else if (summary.state === "awaiting_label") {
    const trial = summary.last_trial || {};
    if (trial.failure_stage) {
      host.appendChild(h("div.alert", { style: { marginBottom: "8px" } },
        h("strong", `Stopped at ${trial.failure_stage}: `), trial.reason || ""));
    }
    host.appendChild(h("div.alert.info", "Did that rollout do the task? Watch it below, then mark it."));
  } else if (summary.state === "awaiting_task" && summary.last_trial && summary.last_trial.excluded) {
    host.appendChild(h("div.alert",
      h("strong", "The last trial was excluded and not labeled. "),
      summary.last_trial.reason || "",
      h("div.small", { style: { marginTop: "4px" } },
        "Its legs are kept under failure/, marked excluded, with the failing checks in hitl.json.")));
  } else if (summary.state === "teleop_handoff") {
    host.appendChild(h("div.alert",
      "The arm is yours — the driver has released the robot and closed its cameras. " +
      "When you hand it back the same task replans from wherever you left the arm; it will not home first."));
  } else if (summary.state === "handing_off") {
    host.appendChild(h("div.alert",
      "Finishing the current plan step, then releasing the robot and cameras. Camera teardown takes about 15 seconds."));
  }

  // The plan covers less than the instruction asked for. Worth saying loudly: every later
  // phase is planned against this, and the episode is labeled with the whole instruction.
  for (const clause of summary.unrepresented || []) {
    host.appendChild(h("div.alert", { style: { marginTop: "8px" } },
      h("strong", "Not covered by the plan: "),
      clause.clause || JSON.stringify(clause),
      clause.reason ? h("div.small", { style: { marginTop: "4px" } }, clause.reason) : null));
  }

  if (host.childElementCount) host.appendChild(h("div", { style: { height: "12px" } }));
}

function renderActions(host, summary, state, { onEnded }) {
  clear(host);
  const id = summary.id;
  const ended = TERMINAL.has(summary.state);

  const act = (fn, failure) => async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    try {
      await fn();
    } catch (error) {
      reportError(error, failure);
      button.disabled = false;
    }
  };

  if (ended) {
    host.appendChild(h("div.row",
      h("span.chip", summary.state),
      h("span.faint.small", summary.end_reason ? `ended: ${summary.end_reason}` : ""),
      h("button.primary", { onclick: onEnded }, "Start another session")));
    return;
  }

  if (summary.state === "awaiting_label") {
    host.appendChild(h("button.ok.big", { onclick: act(() => api.label(id, true), "Could not label") }, "✔ Success"));
    host.appendChild(h("button.danger.big", { onclick: act(() => api.label(id, false), "Could not label") }, "✖ Failure"));
  }

  if (summary.state === "awaiting_task") {
    const taskInput = h("input", { placeholder: summary.task, style: { maxWidth: "320px" } });
    host.appendChild(taskInput);
    host.appendChild(h("button.primary", {
      onclick: act(() => api.continueSession(id, taskInput.value.trim() || null), "Could not continue"),
    }, "Collect another"));
  }

  host.appendChild(h("div.spacer"));

  if (summary.state === "teleop_handoff") {
    host.appendChild(h("button.violet", {
      onclick: act(() => api.teleopResume(id), "Could not return control"),
    }, "Return control to TAMP"));
  } else if (summary.teleop_available) {
    host.appendChild(h("button.ghost", {
      disabled: !summary.can_preempt || summary.teleop_pending,
      title: "Lends the arm to a human at the next plan-step boundary. Nothing is aborted.",
      onclick: act(() => api.teleopSwitch(id), "Could not request the hand-off"),
    }, summary.teleop_pending ? "Hand-off armed…" : "Switch to teleop"));
  }

  host.appendChild(h("button.warn", {
    disabled: !summary.can_preempt,
    title: summary.can_preempt
      ? "Abort this rollout. The session stays warm — no re-warm needed."
      : "Not available during a hand-off; return control first.",
    onclick: act(() => api.preempt(id), "Could not preempt"),
  }, "Preempt rollout"));

  host.appendChild(h("button.ghost", {
    onclick: act(() => api.stopSession(id), "Could not stop"),
  }, "Finish session"));
}
