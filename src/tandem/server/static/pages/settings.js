// Settings — paths, credentials (write-only), the planner and human-executor catalogs, runtime
// status, and the doctor report.

import { api } from "../api.js";
import { clear, h, mount } from "../dom.js";
import { reloadShell, reportError, toast } from "../app.js";

const GLYPH = { ok: ["✔", "var(--green)"], warn: ["!", "var(--amber)"], fail: ["✖", "var(--red)"], skip: ["·", "var(--faint)"] };

// Chip colour per status, as `tandem planners list` / `tandem executors list` colour them.
const PLANNER_CHIP = {
  "installed": ".success",
  "no runtime needed": ".success",
  "outdated": ".eval",
  "not installed": "",
  "broken": ".failure",
};
const EXECUTOR_CHIP = { "ready": ".success", "needs setup": ".eval", "broken": ".failure" };

export function renderSettings(host, state) {
  const paneHost = h("div.stack");
  mount(host, paneHost);

  function load() {
    api.settings()
      .then((payload) => mount(paneHost,
        credentialsCard(payload, load),
        pathsCard(payload),
        plannersCard(state),
        executorsCard(state),
        runtimeCard(),
        doctorCard(state)))
      .catch((error) => mount(paneHost, h("div.alert.err", error.message)));
  }

  load();
}

function profileQuery(state) {
  return state.profile ? `?profile=${encodeURIComponent(state.profile)}` : "";
}

// Choosing a planner changes which runtime the topbar's chip and the runtime card describe, so the
// whole shell reloads rather than just this card. `done` returns a title, or [title, detail].
async function choose(path, body, done, failed) {
  try {
    const result = await api.post(path, body);
    const said = done(result);
    const [title, detail] = Array.isArray(said) ? said : [said, undefined];
    toast.ok(title, detail);
    await reloadShell();
  } catch (error) {
    reportError(error, failed);
  }
}

// What a planner switch did to the profile's planner.options, said as `tandem planners use` says it:
// the old planner's settings leave profile.yml, kept beside it and restored by a switch back.
function switchDetail(result) {
  const parts = [];
  const dropped = Object.keys(result.dropped_options || {}).sort();
  if (dropped.length) {
    parts.push(`Removed planner.options ${dropped.join(", ")} (${result.previous}'s own settings); ` +
      `kept beside the profile, and restored by switching back to ${result.previous}.`);
  }
  const restored = Object.keys(result.restored_options || {}).sort();
  if (restored.length) parts.push(`Restored planner.options ${restored.join(", ")}.`);
  if (result.restore_problem) parts.push(result.restore_problem);
  return parts.join(" ") || undefined;
}

// Asked before a switch that takes settings out of the profile. The page knows them: the catalog carries
// the profile's planner.options keys.
function confirmSwitch(payload, row) {
  const keys = payload.profile_options || [];
  if (!keys.length) return true;
  const from = payload.profile_planner || "its current planner";
  return confirm(
    `Switch ${payload.profile} from ${from} to ${row.name}?\n\n` +
    `The planner.options ${keys.join(", ")} are ${from}'s own and leave the profile: they are kept ` +
    `beside it, and come back when you switch back to ${from}.`);
}

function plannersCard(state) {
  const body = h("div", h("div.row", h("span.spin"), h("span.faint.small", "loading…")));
  api.get(`/planners${profileQuery(state)}`).then((payload) => {
    clear(body);
    if (payload.profile_problem) {
      body.appendChild(h("div.alert", { style: { marginBottom: "12px" } },
        `Profile ${payload.profile}: ${payload.profile_problem}.`));
    }
    body.appendChild(h("table",
      h("thead", h("tr", h("th", ""), h("th", "planner"), h("th", "status"), h("th", ""), h("th", ""))),
      h("tbody", ...payload.planners.map((row) => plannerRow(row, payload)))));
  }).catch((error) => mount(body, h("div.alert.err", error.message)));

  return h("div.card",
    h("div.card-title", "Planners"),
    h("div.card-hint",
      "The task and motion planners tandem can drive: the ones it ships, and any an installed package " +
      "registers. ● marks the one this profile plans with. Installing one fetches its pinned sources and " +
      "builds its runtime — up to twenty minutes, in a terminal — so the command to run is shown here."),
    body);
}

function plannerRow(row, payload) {
  const profile = payload.profile;
  const name = h("div",
    h("div", h("strong", row.name), row.default ? h("span.chip.violet", { style: { marginLeft: "8px" } }, "default") : null),
    h("div.faint.small", row.display_name));
  const about = [];
  if (row.ok) about.push(h("div.small", row.summary || "—"));
  if (row.status !== "installed") about.push(h("div.faint.small", row.detail || ""));
  if (row.install_command) {
    about.push(h("div.mono", { style: { marginTop: "4px" } }, "$ " + row.install_command));
  }
  const actions = h("div.row", { style: { justifyContent: "flex-end" } });
  // Offered for a profile that does not load too: switching its planner works from the file as written,
  // and is how a profile naming a planner this machine no longer has is repaired.
  const repairable = !payload.profile_problem || payload.profile_exists !== false;
  if (row.ok && repairable && !row.active) {
    actions.appendChild(h("button.small", {
      onclick: () => {
        if (!confirmSwitch(payload, row)) return;
        choose(`/planners/${encodeURIComponent(row.name)}/use`, { profile },
          (result) => [`${result.profile} now plans with ${result.display_name}`, switchDetail(result)],
          "Could not switch planner");
      },
    }, `Use for ${profile}`));
  }
  if (row.ok && !row.default) {
    actions.appendChild(h("button.small.ghost", {
      onclick: () => choose(`/planners/${encodeURIComponent(row.name)}/default`, null,
        (result) => `New profiles plan with ${result.default_planner}`, "Could not change the default"),
    }, "Make default"));
  }
  return h("tr",
    h("td", { style: { width: "20px", color: "var(--accent)" } }, row.active ? "●" : ""),
    h("td", { style: { width: "160px" } }, name),
    h("td", { style: { width: "130px" } }, h("span.chip" + (PLANNER_CHIP[row.status] || ""), row.status)),
    h("td", ...about),
    h("td", actions));
}

function executorsCard(state) {
  const body = h("div", h("div.row", h("span.spin"), h("span.faint.small", "loading…")));
  api.get(`/executors${profileQuery(state)}`).then((payload) => {
    clear(body);
    if (payload.profile_problem) {
      body.appendChild(h("div.alert", { style: { marginBottom: "12px" } },
        `Profile ${payload.profile}: ${payload.profile_problem}.`));
    } else if (payload.phase_planning === false) {
      body.appendChild(h("div.alert.info", { style: { marginBottom: "12px" } },
        `Phase planning is off in ${payload.profile} (hitl.enabled), so it has no human phases until it is on.`));
    }
    body.appendChild(h("table",
      h("thead", h("tr", h("th", ""), h("th", "executor"), h("th", "status"), h("th", ""), h("th", ""))),
      h("tbody", ...payload.executors.map((row) => executorRow(row, payload)))));
  }).catch((error) => mount(body, h("div.alert.err", error.message)));

  return h("div.card",
    h("div.card-title", "Human executors"),
    h("div.card-hint",
      "Who carries out a human phase (hitl.human_executor). ● marks the one this profile hands its " +
      "human phases to. What each still needs on this machine is listed under it."),
    body);
}

function executorRow(row, payload) {
  const profile = payload.profile;
  const about = [];
  if (row.ok) about.push(h("div.small", row.summary || "—"));
  else about.push(h("div.small", { style: { color: "var(--red)" } }, row.error || "it will not load"));
  for (const unmet of row.unmet || []) about.push(h("div.faint.small", "· " + unmet));
  const actions = h("div.row", { style: { justifyContent: "flex-end" } });
  // As for planners: also offered for a profile that does not load, which this repairs.
  const repairable = !payload.profile_problem || payload.profile_exists !== false;
  if (row.ok && repairable && !row.active) {
    actions.appendChild(h("button.small", {
      onclick: () => choose(`/executors/${encodeURIComponent(row.name)}/use`, { profile },
        (result) => `${result.profile} hands human phases to ${result.display_name}`, "Could not switch executor"),
    }, `Use for ${profile}`));
  }
  return h("tr",
    h("td", { style: { width: "20px", color: "var(--accent)" } }, row.active ? "●" : ""),
    h("td", { style: { width: "160px" } },
      h("div", h("strong", row.name)), h("div.faint.small", row.display_name || "")),
    h("td", { style: { width: "130px" } }, h("span.chip" + (EXECUTOR_CHIP[row.status] || ""), row.status)),
    h("td", ...about),
    h("td", actions));
}

function credentialsCard(payload, reload) {
  const creds = payload.credentials || {};
  const geminiInput = h("input", { type: "password", placeholder: "paste a new key to replace" });
  const hfInput = h("input", { type: "password", placeholder: "paste a token to replace" });

  async function save() {
    const body = {};
    if (geminiInput.value.trim()) body.gemini_api_key = geminiInput.value.trim();
    if (hfInput.value.trim()) body.hf_token = hfInput.value.trim();
    if (!Object.keys(body).length) {
      toast.info("Nothing to save");
      return;
    }
    try {
      await api.saveSecrets(body);
      geminiInput.value = "";
      hfInput.value = "";
      toast.ok("Credentials updated", "Stored at mode 0600 in credentials.toml");
      reload();
    } catch (error) {
      reportError(error, "Could not save credentials");
    }
  }

  return h("div.card",
    h("div.card-title", "Credentials"),
    h("div.card-hint",
      "Written to credentials.toml at mode 0600, and never sent back out of this page — " +
      "only which source is in play and a masked preview."),
    h("div.field",
      h("label", "Gemini API key"),
      h("div.row", { style: { marginBottom: "6px" } },
        h("span.chip" + (creds.gemini && creds.gemini.source !== "none" ? ".success" : ".eval"),
          creds.gemini ? creds.gemini.source : "none"),
        h("span.mono.faint", creds.gemini ? creds.gemini.masked : "—")),
      geminiInput,
      h("div.desc",
        "Perception calls Gemini once per rollout to turn the task string into objects and goal " +
        "predicates. Collection cannot run without it. An exported GEMINI_API_KEY wins over the stored one.")),
    h("div.field",
      h("label", "HuggingFace token"),
      h("div.row", { style: { marginBottom: "6px" } },
        h("span.chip" + (creds.hf && creds.hf.source !== "none" ? ".success" : ""),
          creds.hf ? creds.hf.source : "none"),
        h("span.mono.faint", creds.hf ? creds.hf.masked : "—")),
      hfInput,
      h("div.desc", "Only needed to push an exported dataset.")),
    h("button.primary", { onclick: save }, "Save credentials"));
}

function pathsCard(payload) {
  const paths = payload.paths || {};
  return h("div.card",
    h("div.card-title", "Paths"),
    h("dl.kv",
      ...Object.entries(paths).flatMap(([key, value]) => [
        h("dt", key.replace(/_/g, " ")),
        h("dd", h("span.mono.faint", value)),
      ])));
}

function runtimeCard() {
  const body = h("div", h("div.row", h("span.spin"), h("span.faint.small", "checking…")));
  api.runtime().then((runtime) => {
    // The rows are the planner's own (sources, its environment, each build step), so this card
    // shows whichever planner the active profile uses without knowing any of them.
    const rows = [
      ["planner", runtime.title || runtime.planner || "—"],
      ...((runtime.rows && runtime.rows.length) ? runtime.rows : [["detail", runtime.detail || "—"]]),
      ["root", runtime.root || "—"],
    ];
    const sourceRows = (runtime.sources || []).map((source) => {
      const installed = source.installed
        ? String(source.installed).slice(0, 12) + (source.installed === source.commit ? "" : " (not the pin)")
        : "—";
      return h("tr", h("td", source.name),
        h("td.mono.faint", String(source.commit || "").slice(0, 12)),
        // The branch the pin follows (a fork's TANDEM, say), so a commit says where it came from.
        h("td.mono.faint", source.ref || "—"),
        h("td.mono.faint", installed),
        h("td.faint.small", source.url || ""));
    });

    clear(body);
    if (!runtime.ready) {
      const problem = (runtime.problems || []).join("; ") || "the runtime is not built";
      body.appendChild(h("div.alert", { style: { marginBottom: "12px" } },
        `Collection is unavailable — ${problem}. Run `,
        h("span.mono", `tandem planners install ${runtime.planner || ""}`.trim()),
        " on the workstation. Visualizing already-collected trajectories works without it."));
    }
    body.appendChild(h("dl.kv", ...rows.flatMap(([key, value]) => [
      h("dt", key), h("dd", key === "root" ? h("span.mono.faint", value) : String(value)),
    ])));
    if (sourceRows.length) {
      body.appendChild(h("div.section-title", { style: { marginTop: "16px" } }, "sources"));
      body.appendChild(h("table",
        h("thead", h("tr", h("th", "source"), h("th", "pinned"), h("th", "branch"), h("th", "installed"),
          h("th", "upstream"))),
        h("tbody", ...sourceRows)));
    }
  }).catch((error) => mount(body, h("div.alert.err", error.message)));

  return h("div.card", h("div.card-title", "Planner runtime"), body);
}

function doctorCard(state) {
  const body = h("div");
  const hardwareToggle = h("input", { type: "checkbox", style: { width: "auto" } });

  async function run() {
    mount(body, h("div.row", h("span.spin"), h("span.faint.small", "running checks…")));
    try {
      const payload = await api.doctor(state.profile, hardwareToggle.checked);
      const groups = new Map();
      for (const check of payload.checks) {
        if (!groups.has(check.group)) groups.set(check.group, []);
        groups.get(check.group).push(check);
      }
      clear(body);
      for (const [group, checks] of groups) {
        body.appendChild(h("div.section-title", group));
        body.appendChild(h("table", h("tbody", ...checks.map((check) => {
          const [glyph, color] = GLYPH[check.state] || GLYPH.skip;
          return h("tr",
            h("td", { style: { width: "28px", color, fontWeight: 700 } }, glyph),
            h("td", { style: { width: "190px" } }, check.name),
            h("td.faint", check.detail || ""));
        }))));
      }
      const problems = payload.checks.filter((c) => c.hint && c.state !== "ok");
      if (problems.length) {
        body.appendChild(h("div.section-title", { style: { marginTop: "16px" } }, "what to do"));
        for (const check of problems) {
          body.appendChild(h("div.alert" + (check.state === "fail" ? ".err" : ""),
            { style: { marginBottom: "8px" } },
            h("strong", check.name), " — ", check.hint));
        }
      }
    } catch (error) {
      mount(body, h("div.alert.err", error.message));
    }
  }

  const card = h("div.card",
    h("div.card-head",
      h("div.card-title", "Diagnostics"),
      h("div.row",
        h("label", { style: { margin: 0 } }, "probe hardware"),
        hardwareToggle,
        h("button.small", { onclick: run }, "Re-run"))),
    body);

  run();
  return card;
}
