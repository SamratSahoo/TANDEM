// Settings — paths, credentials (write-only), runtime status, and the doctor report.

import { api } from "../api.js";
import { clear, h, mount } from "../dom.js";
import { reportError, toast } from "../app.js";

const GLYPH = { ok: ["✔", "var(--green)"], warn: ["!", "var(--amber)"], fail: ["✖", "var(--red)"], skip: ["·", "var(--faint)"] };

export function renderSettings(host, state) {
  const paneHost = h("div.stack");
  mount(host, paneHost);

  function load() {
    api.settings()
      .then((payload) => mount(paneHost,
        credentialsCard(payload, load),
        pathsCard(payload),
        runtimeCard(),
        doctorCard(state)))
      .catch((error) => mount(paneHost, h("div.alert.err", error.message)));
  }

  load();
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
    const rows = [
      ["sources", runtime.sources_present ? "present" : "missing"],
      ["pixi env", runtime.env_built ? "built" : "not built"],
      ["curobo kernels", runtime.kernels_built ? "compiled" : "not compiled"],
      ["built", runtime.built_at || "—"],
      ["root", runtime.root],
    ];
    const vendor = runtime.vendor || {};
    const vendorRows = Object.entries(vendor)
      .filter(([, meta]) => meta && typeof meta === "object")
      .map(([name, meta]) =>
        h("tr", h("td", name), h("td.faint", meta.version || ""),
          h("td.mono.faint", String(meta.commit || "").slice(0, 12)),
          h("td.faint.small", meta.url || "")));

    clear(body);
    if (!runtime.ready) {
      const problem = (runtime.problems || []).join("; ") || "the runtime is not built";
      body.appendChild(h("div.alert", { style: { marginBottom: "12px" } },
        `Collection is unavailable — ${problem}. Run `,
        h("span.mono", "tandem init"),
        " on the workstation. Visualizing already-collected trajectories works without it."));
    }
    body.appendChild(h("dl.kv", ...rows.flatMap(([key, value]) => [
      h("dt", key), h("dd", key === "root" ? h("span.mono.faint", value) : String(value)),
    ])));
    if (vendorRows.length) {
      body.appendChild(h("div.section-title", { style: { marginTop: "16px" } }, "vendored sources"));
      body.appendChild(h("table",
        h("thead", h("tr", h("th", "component"), h("th", "version"), h("th", "commit"), h("th", "upstream"))),
        h("tbody", ...vendorRows)));
    }
  }).catch((error) => mount(body, h("div.alert.err", error.message)));

  return h("div.card", h("div.card-title", "GPU runtime"), body);
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
