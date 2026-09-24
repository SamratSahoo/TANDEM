// Trajectories — the grid, and the review drawer.

import { api } from "../api.js";
import { clear, fmtBytes, fmtDuration, fmtTimestamp, h, mount } from "../dom.js";
import { renderReview } from "../review.js";
import { reportError, toast } from "../app.js";

const STATUS_ORDER = ["success", "failure", "eval"];

export function renderTrajectories(host, state) {
  if (!state.profile) {
    mount(host, h("div.empty",
      h("div.big", "◈"),
      h("div", "No profile yet."),
      h("div.faint.small", "Run tandem init to create one.")));
    return;
  }

  let filter = null;
  const listHost = h("div");
  const summaryHost = h("div.row.wrap", { style: { marginBottom: "16px" } });

  const filterTabs = h("div.pill-tabs",
    ...[["", "All"], ...STATUS_ORDER.map((s) => [s, s[0].toUpperCase() + s.slice(1)])].map(([value, label]) =>
      h("button", {
        class: value === (filter || "") ? "active" : "",
        onclick: (event) => {
          filter = value || null;
          for (const button of filterTabs.children) button.classList.remove("active");
          event.currentTarget.classList.add("active");
          load();
        },
      }, label))
  );

  mount(host,
    h("div.row", { style: { marginBottom: "14px" } }, filterTabs, h("div.spacer")),
    summaryHost,
    listHost);

  async function load() {
    mount(listHost, h("div.row", h("span.spin"), h("span.faint.small", "loading…")));
    try {
      const payload = await api.trajectories(state.profile, filter);
      renderSummary(summaryHost, payload);
      renderList(listHost, state.profile, payload, load);
      openDeepLink(state.profile, payload.trajectories || [], load);
    } catch (error) {
      mount(listHost, h("div.alert.err", error.message));
    }
  }

  load();
}

/** #/trajectories/<id> opens that rollout's drawer, so a link to one is shareable. */
function openDeepLink(profile, items, reload) {
  const wanted = (location.hash || "").split("/")[2];
  if (!wanted) return;
  const traj = items.find((t) => t.id === wanted);
  if (traj) openDrawer(profile, traj, reload);
}

function renderSummary(host, payload) {
  const counts = payload.counts || {};
  const done = counts.success || 0;
  const target = payload.target || 0;
  const pct = target ? Math.min(100, Math.round((done / target) * 100)) : 0;

  mount(host,
    h("div.card", { style: { flex: "1", minWidth: "260px" } },
      h("div.row",
        h("div",
          h("div.faint.small", "collected"),
          h("div", { style: { fontSize: "22px", fontWeight: 700 } }, `${done}`,
            h("span.faint", { style: { fontSize: "14px", fontWeight: 400 } }, ` / ${target}`))),
        h("div.spacer"),
        h("div.bar" + (done >= target && target ? ".done" : ""), { style: { width: "140px" } },
          h("span", { style: { width: `${pct}%` } })))),
    ...STATUS_ORDER.map((status) =>
      h("div.card", { style: { minWidth: "120px" } },
        h("div.faint.small", status),
        h("div", { style: { fontSize: "22px", fontWeight: 700 } }, String(counts[status] || 0))))
  );
}

function renderList(host, profile, payload, reload) {
  const items = payload.trajectories || [];
  if (!items.length) {
    mount(host, h("div.empty",
      h("div.big", "▦"),
      h("div", "Nothing collected yet."),
      h("div.faint.small", "Run a session from the Collect tab, or `tandem collect` in a terminal.")));
    return;
  }

  const table = h("table",
    h("thead", h("tr",
      h("th", "when"), h("th", ""), h("th", "task"),
      h("th", "frames"), h("th", "length"), h("th", "cams"), h("th", ""))),
    h("tbody", ...items.map((traj) => row(profile, traj, reload)))
  );
  mount(host, h("div.card", { style: { padding: "4px 6px" } }, table));
}

function row(profile, traj, reload) {
  const flags = [];
  if (traj.merged) {
    // Any leg that is not the planner's is a human phase's, a policy executor's included (review.js).
    const human = (traj.segments || []).filter((s) => (s.source || "tamp") !== "tamp").length;
    flags.push(h("span.chip.violet", `hand-off ×${human}`));
  }
  if (!traj.complete) flags.push(h("span.chip", "incomplete"));

  return h("tr.clickable", { onclick: () => openDrawer(profile, traj, reload) },
    h("td", h("span.mono", fmtTimestamp(traj.id))),
    h("td", h(`span.chip.${traj.status}`, traj.status)),
    h("td", { style: { maxWidth: "320px" } }, h("div.truncate", { title: traj.instruction }, traj.instruction || "—")),
    h("td.faint", String(traj.n_frames || "—")),
    h("td.faint", fmtDuration(traj.duration_s)),
    h("td.faint", String((traj.cameras || []).length)),
    h("td", h("div.row", ...flags))
  );
}

// ---- drawer ----------------------------------------------------------------

function openDrawer(profile, traj, reload) {
  if (document.querySelector(".drawer")) return; // a deep link and a click can race
  const body = h("div.drawer-body");
  const scrim = h("div.drawer-scrim", { onclick: close });
  const drawer = h("div.drawer",
    h("div.drawer-head",
      h("div",
        h("div", { style: { fontWeight: 650 } }, traj.id),
        h("div.faint.small", traj.instruction || "no instruction recorded")),
      h("div.spacer"),
      h(`span.chip.${traj.status}`, traj.status),
      h("button.icon.ghost", { onclick: close, title: "Close" }, "✕")),
    body);

  function close() {
    scrim.remove();
    drawer.remove();
    document.removeEventListener("keydown", onKey);
    if ((location.hash || "").split("/")[2]) history.replaceState(null, "", "#/trajectories");
  }
  function onKey(event) {
    if (event.key === "Escape") close();
  }
  document.addEventListener("keydown", onKey);

  history.replaceState(null, "", `#/trajectories/${traj.id}`);
  document.body.appendChild(scrim);
  document.body.appendChild(drawer);

  const copyLink = h("button.small.ghost", {
    title: "Copy a link to this trajectory",
    onclick: () => {
      navigator.clipboard?.writeText(location.href);
      toast.ok("Link copied");
    },
  }, "🔗 Link");
  drawer.querySelector(".drawer-head").insertBefore(copyLink, drawer.querySelector(".drawer-head").lastChild);

  body.appendChild(metaBlock(traj));
  const reviewHost = h("div", { style: { marginTop: "18px" } });
  body.appendChild(reviewHost);
  renderReview(reviewHost, { profile, trajectory: traj });

  body.appendChild(h("div.row.wrap", { style: { marginTop: "22px", gap: "8px" } },
    h("span.faint.small", "relabel"),
    ...STATUS_ORDER.map((status) =>
      h("button.small", {
        disabled: status === traj.status,
        onclick: async () => {
          try {
            await api.relabel(profile, traj.id, status);
            toast.ok(`Moved to ${status}`);
            close();
            reload();
          } catch (error) {
            reportError(error, "Could not relabel");
          }
        },
      }, status)),
    h("div.spacer"),
    h("button.small.danger", {
      onclick: async () => {
        if (!confirm(`Delete ${traj.id}? This cannot be undone.`)) return;
        try {
          await api.deleteTrajectory(profile, traj.id);
          toast.ok("Deleted");
          close();
          reload();
        } catch (error) {
          reportError(error, "Could not delete");
        }
      },
    }, "Delete")
  ));
}

function metaBlock(traj) {
  const meta = traj.meta || {};
  const pairs = [
    ["frames", `${traj.n_frames} at ${traj.fps} Hz`],
    ["length", fmtDuration(traj.duration_s)],
    ["cameras", (traj.camera_labels || []).join(", ") || "none"],
    ["plan", traj.has_plan ? "recorded" : "none"],
  ];
  if (traj.trajectory_id) pairs.push(["trajectory id", traj.trajectory_id]);
  if (meta.config_id) pairs.push(["config", meta.config_id]);
  if (traj.size_bytes) pairs.push(["size", fmtBytes(traj.size_bytes)]);
  pairs.push(["path", traj.path]);

  const summary = traj.summary || {};
  if (summary.gripper_events != null) pairs.push(["gripper events", String(summary.gripper_events)]);
  if (summary.joint_travel_rad != null) {
    pairs.push(["joint travel", `${summary.joint_travel_rad.toFixed(2)} rad`]);
  }

  return h("dl.kv", ...pairs.flatMap(([key, value]) => [
    h("dt", key),
    h("dd", key === "path" ? h("span.mono.faint", value) : String(value)),
  ]));
}
