// Profiles — the cards, and an editor that shows exactly what the planner will receive.

import { api } from "../api.js";
import { clear, h, mount } from "../dom.js";
import { reloadShell, reportError, toast } from "../app.js";

export function renderProfiles(host, state) {
  const listHost = h("div.grid");
  const editorHost = h("div");

  mount(host,
    h("div.row", { style: { marginBottom: "14px" } },
      h("div.spacer"),
      h("button.primary", { onclick: () => createDialog(state, refresh) }, "+ New profile")),
    listHost,
    editorHost);

  async function refresh() {
    await reloadShell();
    draw();
  }

  function draw() {
    clear(listHost);
    for (const profile of state.profiles) {
      listHost.appendChild(card(profile, state, refresh, editorHost));
    }
    if (!state.profiles.length) {
      mount(listHost, h("div.empty", h("div.big", "◈"), h("div", "No profiles yet.")));
    }
  }

  draw();
}

function card(profile, state, refresh, editorHost) {
  if (!profile.valid) {
    return h("div.card",
      h("div.card-title", profile.name),
      h("div.alert.err", { style: { marginTop: "8px" } }, profile.error));
  }

  const counts = profile.counts || {};
  const done = counts.success || 0;
  const pct = profile.target ? Math.min(100, Math.round((done / profile.target) * 100)) : 0;

  return h("div.card.clickable", { onclick: () => openEditor(editorHost, profile.name, refresh) },
    h("div.card-head",
      h("div.card-title", profile.name),
      profile.active ? h("span.chip.accent", "active") : null),
    h("div.faint.small.truncate", { title: profile.prompt }, profile.prompt || "no task set"),
    h("div.row", { style: { marginTop: "12px" } },
      h("div.bar" + (pct >= 100 ? ".done" : ""), h("span", { style: { width: `${pct}%` } })),
      h("span.faint.small", `${done}/${profile.target}`)),
    h("div.row", { style: { marginTop: "10px" } },
      h("span.chip", profile.robot),
      h("span.chip", `${(profile.cameras || []).length} cams`),
      profile.tamp_count ? h("span.chip.accent", `${profile.tamp_count} tamp`) : null,
      h("div.spacer"),
      !profile.active
        ? h("button.small.ghost", {
            onclick: async (event) => {
              event.stopPropagation();
              try {
                await api.setActive(profile.name);
                toast.ok(`Switched to ${profile.name}`);
                refresh();
              } catch (error) {
                reportError(error);
              }
            },
          }, "Use")
        : null)
  );
}

// ---- editor ----------------------------------------------------------------

async function openEditor(host, name, refresh) {
  mount(host, h("div.card", { style: { marginTop: "18px" } },
    h("div.row", h("span.spin"), h("span.faint.small", "loading…"))));

  let payload;
  try {
    payload = await api.profile(name);
  } catch (error) {
    mount(host, h("div.alert.err", error.message));
    return;
  }

  const profile = payload.profile;
  const yamlHost = h("textarea", {
    value: JSON.stringify(profile.tamp || {}, null, 2),
    style: { minHeight: "220px" },
  });
  const overridesHost = h("pre.mono", {
    style: { background: "var(--bg-alt)", padding: "12px", borderRadius: "7px", overflow: "auto", maxHeight: "260px" },
  }, JSON.stringify(payload.tamp_overrides || {}, null, 2));

  const promptInput = h("input", { value: profile.task.prompt || "" });
  const goalInput = h("input", { value: profile.task.goal || "", placeholder: "same as the task above" });
  const targetInput = h("input", { type: "number", min: "1", value: String(profile.task.target_episodes) });
  const descInput = h("input", { value: profile.description || "" });
  const hostInput = h("input", { value: profile.robot.host });
  const tdfInput = h("input", { type: "number", step: "0.05", min: "0.05", max: "1", value: String(profile.robot.time_dilation_factor) });

  const warningsHost = h("div");
  function drawWarnings(list) {
    clear(warningsHost);
    for (const warning of list || []) warningsHost.appendChild(h("div.alert", { style: { marginBottom: "8px" } }, warning));
  }
  drawWarnings(payload.warnings);

  async function save() {
    let tamp;
    try {
      tamp = JSON.parse(yamlHost.value || "{}");
    } catch (error) {
      toast.err("TAMP settings are not valid JSON", error.message);
      return;
    }
    const body = {
      ...profile,
      description: descInput.value,
      task: {
        ...profile.task,
        prompt: promptInput.value,
        goal: goalInput.value.trim() || null,
        target_episodes: Number(targetInput.value) || profile.task.target_episodes,
      },
      robot: {
        ...profile.robot,
        host: hostInput.value,
        time_dilation_factor: Number(tdfInput.value) || profile.robot.time_dilation_factor,
      },
      tamp,
    };
    try {
      const updated = await api.saveProfile(name, body);
      toast.ok(`Saved ${name}`);
      drawWarnings(updated.warnings);
      overridesHost.textContent = JSON.stringify(
        (await api.profile(name)).tamp_overrides || {}, null, 2);
      refresh();
    } catch (error) {
      reportError(error, "Could not save the profile");
    }
  }

  mount(host, h("div.card", { style: { marginTop: "18px" } },
    h("div.card-head",
      h("div.card-title", `Edit ${name}`),
      h("button.icon.ghost", { onclick: () => clear(host), title: "Close" }, "✕")),
    warningsHost,
    h("div.field", h("label", "Description"), descInput),
    h("div.field", h("label", "Task"), promptInput,
      h("div.desc", "The language label stored with every episode.")),
    h("div.field-row",
      h("div.field", h("label", "Planner goal"), goalInput,
        h("div.desc", "Only when the goal must differ from the label.")),
      h("div.field", h("label", "Target episodes"), targetInput),
      h("div.field", h("label", "Robot host"), hostInput),
      h("div.field", h("label", "Speed"), tdfInput,
        h("div.desc", "time_dilation_factor. 0.2 is 20% — raise it only once you trust the setup."))),
    h("div.field",
      h("label", "TAMP settings"),
      yamlHost,
      h("div.desc",
        "tiptop's own key names, so anything documented upstream works verbatim. " +
        "Unknown keys are rejected on save — a silently ignored override is the failure mode that looks like success.")),
    h("div.field",
      h("label", "What the planner receives"),
      overridesHost,
      h("div.desc", "Passed as --curobo-overrides. Paths are resolved to absolute here.")),
    h("div.row",
      h("button.primary", { onclick: save }, "Save"),
      h("div.spacer"),
      h("span.faint.small.mono", payload.dir))
  ));
}

// ---- create ----------------------------------------------------------------

function createDialog(state, refresh) {
  const nameInput = h("input", { placeholder: "fold-cloth" });
  const promptInput = h("input", { placeholder: "place the toy on the cloth and fold it" });
  const fromSelect = h("select",
    h("option", { value: "" }, "built-in template"),
    ...state.profiles.filter((p) => p.valid).map((p) => h("option", { value: p.name }, `copy of ${p.name}`)));

  const scrim = h("div.drawer-scrim", { onclick: close });
  const panel = h("div.card", {
    style: {
      position: "fixed", top: "18%", left: "50%", transform: "translateX(-50%)",
      width: "min(480px, 92vw)", zIndex: 42, boxShadow: "var(--shadow)",
    },
  },
    h("div.card-title", "New profile"),
    h("div.card-hint", "A profile is one collection setup and the trajectories it produces."),
    h("div.field", h("label", "Name"), nameInput,
      h("div.desc", "Lowercase letters, digits, - and _. This is also the directory name.")),
    h("div.field", h("label", "Start from"), fromSelect),
    h("div.field", h("label", "Task"), promptInput),
    h("div.row",
      h("button.primary", { onclick: create }, "Create"),
      h("button.ghost", { onclick: close }, "Cancel")));

  function close() {
    scrim.remove();
    panel.remove();
  }

  async function create() {
    try {
      await api.createProfile({
        name: nameInput.value.trim(),
        from: fromSelect.value || null,
        prompt: promptInput.value.trim() || null,
      });
      toast.ok(`Created ${nameInput.value.trim()}`);
      close();
      refresh();
    } catch (error) {
      reportError(error, "Could not create the profile");
    }
  }

  document.body.appendChild(scrim);
  document.body.appendChild(panel);
  nameInput.focus();
}
