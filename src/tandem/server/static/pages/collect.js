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

const TERMINAL = new Set(["stopped", "failed"]);

export function renderCollect(host, state) {
  if (!state.profile) {
    mount(host, h("div.empty", h("div.big", "◈"), h("div", "No profile yet.")));
    return;
  }
  if (!state.runtimeReady) {
    mount(host, notReady());
    return;
  }

  const shell = h("div.stack");
  mount(host, shell);

  api.sessions()
    .then((payload) => {
      const live = (payload.sessions || []).find(
        (session) => session.profile === state.profile && !TERMINAL.has(session.state)
      );
      if (live) attachSession(shell, state, live);
      else mount(shell, startForm(shell, state));
    })
    .catch(() => mount(shell, startForm(shell, state)));
}

function notReady() {
  return h("div.card",
    h("div.card-title", "The GPU runtime is not built"),
    h("div.card-hint",
      "Collection needs the planner stack: torch, cuRobo's compiled CUDA kernels, cuTAMP and tiptop. " +
      "Visualizing already-collected trajectories works without it."),
    h("div.alert.info", h("span.mono", "tandem init"), " on the workstation — the first build takes 5–20 minutes."));
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
      "The driver warms up once — cuRobo, SAM2, the cameras, the robot — and then loops rollouts " +
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
    }
  });

  api.session(summary.id).then((payload) => {
    summary = payload;
    clear(logHost);
    for (const line of payload.logs || []) appendLog(line);
    paint();
  }).catch(() => paint());

  paint();
}

function renderPipeline(host, state) {
  clear(host);
  if (state === "handing_off" || state === "teleop_handoff") {
    host.appendChild(h("div.step.now",
      h("span.pd", { style: { background: "var(--violet)" } }),
      state === "handing_off" ? "handing the arm over…" : "the human has the arm"));
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
    stat(fmtDuration(elapsed), "elapsed"),
    h("div.spacer"),
    h("span.chip" + (TERMINAL.has(summary.state) ? "" : ".accent"), summary.state.replace(/_/g, " ")));
}

function stat(value, label, color) {
  return h("div",
    h("div", { style: { fontSize: "20px", fontWeight: 700, color: color || "var(--text)" } }, value),
    h("div.faint.small", label));
}

function renderNotice(host, summary) {
  clear(host);
  if (summary.error) {
    host.appendChild(h("div.alert.err", summary.error));
  } else if (summary.handoff_error) {
    host.appendChild(h("div.alert", summary.handoff_error));
  } else if (summary.state === "awaiting_label") {
    host.appendChild(h("div.alert.info", "Did that rollout do the task? Watch it below, then mark it."));
  } else if (summary.state === "teleop_handoff") {
    host.appendChild(h("div.alert",
      "The arm is yours — the driver has released the robot and closed its cameras. " +
      "When you hand it back the same task replans from wherever you left the arm; it will not home first."));
  } else if (summary.state === "handing_off") {
    host.appendChild(h("div.alert",
      "Finishing the current plan step, then releasing the robot and cameras. Camera teardown takes about 15 seconds."));
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
