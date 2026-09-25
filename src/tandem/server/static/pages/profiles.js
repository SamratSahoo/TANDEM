// Profiles — one YAML file each, the paper's five tasks among them: the cards, a new one from a task or a
// copy, and an editor that shows exactly what the planner will receive.
//
// The planner's own settings (planner.options) are edited as one JSON block and described by the
// planner itself (planner_view: a summary, what it receives, what is wrong), so this page knows no
// planner's schema and shows a planner registered tomorrow the same way it shows TiPToP. The robot and
// cameras are no profile's: they are this machine's rig (Settings).

import { api } from "../api.js";
import { clear, h, mount, prettyJson } from "../dom.js";
import { reloadShell, reportError, toast } from "../app.js";

export function renderProfiles(host, state) {
  const listHost = h("div.grid");
  const editorHost = h("div");

  // Old-layout profiles are not listed until they are moved; say so rather than show fewer than there are.
  const oldLayout = state.oldLayout || [];
  mount(host,
    oldLayout.length
      ? h("div.alert", { style: { marginBottom: "14px" } },
          `${oldLayout.length} profile(s) in the old layout (${oldLayout.join(", ")}) are not shown: run `,
          h("span.mono", "tandem init"), " (or ", h("span.mono", "tandem profile migrate"),
          ") to move them, their robot and cameras going to this machine's rig.")
      : null,
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
      mount(listHost, h("div.empty",
        h("div.big", "◈"),
        h("div", "No profiles yet. ", h("span.mono", "tandem init"), " adds the paper's five tasks; ",
          "+ New profile makes your own."),
        h("button", { style: { marginTop: "12px" }, onclick: addPaperTasks }, "Add the paper's five tasks")));
    }
  }

  async function addPaperTasks() {
    try {
      const result = await api.addPaperProfiles();
      toast.ok(`Added ${result.added.length} of the paper's tasks`, `Active profile: ${result.active}`);
      if ((result.held_back || []).length) {
        toast.info(`Not added: ${result.held_back.join(", ")}`,
          "A profile of that name is still in the old layout: tandem profile migrate moves it.");
      }
      refresh();
    } catch (error) {
      reportError(error, "Could not add the paper's tasks");
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
      h("span.chip", { title: profile.planner_summary || "" }, profile.planner),
      // One of the paper's five tasks, as the paper collected it unless it has been edited since.
      profile.builtin ? h("span.chip.violet", { title: profile.description || "" }, "paper") : null,
      // A profile collected with a plugin this machine lacks: browsable here, collects where it is installed.
      ...(profile.missing || []).map((what) =>
        h("span.chip.eval", { title: "not installed on this machine" }, `${what} missing`)),
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
  const view = payload.planner_view || {};
  const optionsHost = h("textarea", {
    value: prettyJson((profile.planner || {}).options || {}),
    style: { minHeight: "260px" },
  });
  const receivesHost = h("pre.mono", {
    style: { background: "var(--bg-alt)", padding: "12px", borderRadius: "7px", overflow: "auto", maxHeight: "260px" },
  }, JSON.stringify(view.receives || {}, null, 2));
  const receivesNote = h("div.desc", view.receives_note || "");

  const promptInput = h("input", { value: profile.task.prompt || "" });
  const goalInput = h("input", { value: profile.task.goal || "", placeholder: "same as the task above" });
  const targetInput = h("input", { type: "number", min: "1", value: String(profile.task.target_episodes) });
  const descInput = h("input", { value: profile.description || "" });

  const warningsHost = h("div");
  function drawWarnings(list) {
    clear(warningsHost);
    for (const warning of list || []) warningsHost.appendChild(h("div.alert", { style: { marginBottom: "8px" } }, warning));
  }
  drawWarnings(payload.warnings);

  async function save() {
    let options;
    try {
      options = JSON.parse(optionsHost.value || "{}");
    } catch (error) {
      toast.err("The planner's settings are not valid JSON", error.message);
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
      planner: { ...profile.planner, options },
    };
    try {
      const updated = await api.saveProfile(name, body);
      toast.ok(`Saved ${name}`);
      drawWarnings(updated.warnings);
      const saved = updated.planner_view || {};
      receivesHost.textContent = JSON.stringify(saved.receives || {}, null, 2);
      receivesNote.textContent = saved.receives_note || "";
      optionsHost.value = prettyJson((updated.profile.planner || {}).options || {});
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
      h("div.field", h("label", "Target episodes"), targetInput)),
    h("div.field",
      h("label", `Planner settings — ${(profile.planner || {}).backend || "?"}`),
      view.summary ? h("div.faint.small", { style: { marginBottom: "6px" } }, view.summary) : null,
      optionsHost,
      h("div.desc",
        "planner.options, checked by the planner itself when you save. A key it does not read is " +
        "rejected — a silently ignored setting is the failure mode that looks like success. " +
        "`tandem planners info <name>` lists what a planner reads.")),
    h("div.field",
      h("label", "What the planner receives"),
      receivesHost,
      receivesNote),
    h("div.row",
      h("button.primary", { onclick: save }, "Save"),
      h("div.spacer"),
      h("span.faint.small.mono", payload.file)),
    h("div.desc", { style: { marginTop: "8px" } },
      "The file holds every setting, phase planning (hitl) included: ",
      h("span.mono", `tandem profile edit ${name}`), " opens it. The robot and cameras are the rig's (Settings).")
  ));
}

// ---- create ----------------------------------------------------------------

function createDialog(state, refresh) {
  const nameInput = h("input", { placeholder: "my-task" });
  const promptInput = h("input", { placeholder: "put the cup on the plate" });
  // A new task on the paper's settings, or a copy of a profile: any here, and the paper's five whether or
  // not this machine has them yet (the server copies the packaged one).
  const here = state.profiles.filter((p) => p.valid).map((p) => p.name);
  const sources = [...here, ...(state.builtin || []).filter((name) => !here.includes(name))];
  const fromSelect = h("select",
    h("option", { value: "" }, "the paper's settings (a new task)"),
    ...sources.map((name) => h("option", { value: name }, `copy of ${name}`)));
  const promptDesc = h("div.desc");
  function describeTask() {
    promptDesc.textContent = fromSelect.value
      ? `Optional: the copy keeps ${fromSelect.value}'s task unless you give one.`
      : "What the robot and you are to do: the label stored with every episode, which phase planning splits into steps.";
  }
  fromSelect.addEventListener("change", describeTask);
  describeTask();

  const scrim = h("div.drawer-scrim", { onclick: close });
  const panel = h("div.card", {
    style: {
      position: "fixed", top: "18%", left: "50%", transform: "translateX(-50%)",
      width: "min(480px, 92vw)", zIndex: 42, boxShadow: "var(--shadow)",
    },
  },
    h("div.card-title", "New profile"),
    h("div.card-hint", "A profile is one task: its settings, in one YAML file, and the trajectories collected with it."),
    h("div.field", h("label", "Name"), nameInput,
      h("div.desc", "Lowercase letters, digits, - and _. It is also the file's name.")),
    h("div.field", h("label", "Start from"), fromSelect),
    h("div.field", h("label", "Task"), promptInput, promptDesc),
    h("div.row",
      h("button.primary", { onclick: create }, "Create"),
      h("button.ghost", { onclick: close }, "Cancel")));

  function close() {
    scrim.remove();
    panel.remove();
  }

  async function create() {
    const name = nameInput.value.trim();
    const prompt = promptInput.value.trim();
    // Said here as the server would say it, before a round trip: a new task needs its task.
    if (!fromSelect.value && !prompt) {
      toast.err("A new profile needs its task", "Say what the robot and you are to do, or start from a copy.");
      promptInput.focus();
      return;
    }
    try {
      const created = await api.createProfile({ name, from: fromSelect.value || null, prompt: prompt || null });
      toast.ok(`Created ${created.name}`, created.file);
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
