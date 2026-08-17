// Shell: sidebar, topbar, profile switcher, hash router, toasts.

import { api } from "./api.js";
import { clear, h, mount } from "./dom.js";
import { renderCollect } from "./pages/collect.js";
import { renderProfiles } from "./pages/profiles.js";
import { renderSettings } from "./pages/settings.js";
import { renderTrajectories } from "./pages/trajectories.js";

const ROUTES = [
  { id: "trajectories", label: "Trajectories", icon: "▦", render: renderTrajectories },
  { id: "collect", label: "Collect", icon: "●", render: renderCollect },
  { id: "profiles", label: "Profiles", icon: "◈", render: renderProfiles },
  { id: "settings", label: "Settings", icon: "⚙", render: renderSettings },
];

export const state = {
  profile: null,
  profiles: [],
  version: "",
  runtimeReady: false,
};

// ---- toasts ----------------------------------------------------------------

const toastHost = h("div.toasts");

export const toast = {
  ok: (title, message) => push("ok", title, message, 4000),
  err: (title, message) => push("err", title, message, 9000),
  info: (title, message) => push("", title, message, 5000),
};

function push(kind, title, message, ms) {
  const node = h(`div.toast${kind ? "." + kind : ""}`,
    h("div.t-title", title),
    message ? h("div.t-msg", message) : null);
  toastHost.appendChild(node);
  setTimeout(() => node.remove(), ms);
}

/** Report an API failure with its server-supplied hint, which is usually the useful half. */
export function reportError(error, fallback = "Something went wrong") {
  const message = (error && error.message) || fallback;
  const hint = error && error.hint;
  toast.err(message, hint || undefined);
}

// ---- routing ---------------------------------------------------------------

function currentRoute() {
  const id = (location.hash || "#/trajectories").replace(/^#\/?/, "").split("/")[0];
  return ROUTES.find((route) => route.id === id) || ROUTES[0];
}

export function navigate(routeId) {
  location.hash = `#/${routeId}`;
}

// ---- shell -----------------------------------------------------------------

const contentHost = h("div.content");
const titleHost = h("h1", "Trajectories");
const profileSelect = h("select", { style: { width: "auto", minWidth: "150px" } });
const runtimeChip = h("span.chip", "…");

function sidebar() {
  return h("aside.sidebar",
    h("div.brand",
      h("div.brand-logo", "t"),
      h("div", h("div.brand-name", "tandem"), h("div.brand-sub", "human-in-the-loop TAMP"))),
    ...ROUTES.map((route) =>
      h("button.nav-item", {
        id: `nav-${route.id}`,
        onclick: () => navigate(route.id),
      }, h("span.ic", route.icon), h("span", route.label))),
    h("div.nav-spacer"),
    h("div.nav-foot", { id: "version-foot" }, "")
  );
}

function topbar() {
  profileSelect.addEventListener("change", async () => {
    const name = profileSelect.value;
    try {
      await api.setActive(name);
      state.profile = name;
      toast.ok(`Switched to ${name}`);
      render();
    } catch (error) {
      reportError(error, "Could not switch profile");
    }
  });

  return h("header.topbar",
    titleHost,
    h("div.row",
      runtimeChip,
      h("span.faint.small", "profile"),
      profileSelect));
}

function setActiveNav(routeId) {
  for (const route of ROUTES) {
    const node = document.getElementById(`nav-${route.id}`);
    if (node) node.classList.toggle("active", route.id === routeId);
  }
}

function render() {
  const route = currentRoute();
  titleHost.textContent = route.label;
  setActiveNav(route.id);
  clear(contentHost);
  route.render(contentHost, state);
}

async function refreshShell() {
  try {
    const payload = await api.profiles();
    state.profiles = payload.profiles || [];
    state.profile = payload.active;
    clear(profileSelect);
    for (const profile of state.profiles) {
      profileSelect.appendChild(
        h("option", { value: profile.name, selected: profile.name === payload.active }, profile.name)
      );
    }
    if (!state.profiles.length) {
      profileSelect.appendChild(h("option", { value: "" }, "no profiles"));
    }
  } catch (error) {
    reportError(error, "Could not load profiles");
  }

  try {
    const runtime = await api.runtime();
    state.runtimeReady = !!runtime.ready;
    clear(runtimeChip);
    runtimeChip.className = `chip ${runtime.ready ? "success" : ""}`;
    runtimeChip.title = runtime.ready
      ? `Runtime ready at ${runtime.root}`
      : (runtime.problems || []).join("; ") || "The GPU runtime is not built";
    runtimeChip.appendChild(h("span.dot" + (runtime.ready ? ".live" : ".warn")));
    runtimeChip.appendChild(document.createTextNode(runtime.ready ? "runtime ready" : "visualize only"));
  } catch {
    state.runtimeReady = false;
  }

  try {
    const settings = await api.settings();
    state.version = settings.version;
    const foot = document.getElementById("version-foot");
    if (foot) foot.textContent = `v${settings.version}`;
  } catch {
    /* the footer version is cosmetic */
  }
}

export async function reloadShell() {
  await refreshShell();
  render();
}

async function boot() {
  const root = document.getElementById("root");
  mount(root, h("div.app", sidebar(), h("div.main", topbar(), contentHost)), toastHost);
  await refreshShell();
  render();
  window.addEventListener("hashchange", render);
}

boot();
