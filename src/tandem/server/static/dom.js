// A ~60-line DOM helper instead of a framework.
//
// The whole point of this UI is that it ships as package data with no build step and no
// npm — `pip install tandem-tamp` must be the only install anyone runs. A framework would
// mean either a bundler in the release process or a CDN fetch at runtime, and this app is
// four pages; it does not need one.

/** h("div.card", {onclick}, ...children) — tag with optional #id and .class shorthand. */
export function h(spec, props, ...children) {
  const [tag, ...rest] = String(spec).split(/(?=[.#])/);
  const el = document.createElement(tag || "div");
  for (const token of rest) {
    if (token[0] === ".") el.classList.add(token.slice(1));
    else if (token[0] === "#") el.id = token.slice(1);
  }
  if (props && (props.nodeType || Array.isArray(props) || typeof props !== "object")) {
    children.unshift(props);
    props = null;
  }
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key === "class") el.className += (el.className ? " " : "") + value;
    else if (key === "style" && typeof value === "object") Object.assign(el.style, value);
    else if (key === "html") el.innerHTML = value;
    else if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key in el && key !== "list") el[key] = value;
    else el.setAttribute(key, value);
  }
  append(el, children);
  return el;
}

export function append(parent, children) {
  for (const child of children.flat(4)) {
    if (child == null || child === false || child === true) continue;
    parent.appendChild(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return parent;
}

/** Same as h() but for SVG, which needs createElementNS or nothing renders. */
export function svg(spec, props, ...children) {
  const [tag, ...rest] = String(spec).split(/(?=[.#])/);
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const token of rest) if (token[0] === ".") el.classList.add(token.slice(1));
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else el.setAttribute(key, value);
  }
  for (const child of children.flat(4)) {
    if (child == null || child === false) continue;
    el.appendChild(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function mount(el, ...children) {
  clear(el);
  append(el, children);
  return el;
}

// ---- formatting ------------------------------------------------------------

export function fmtDuration(seconds) {
  if (!seconds && seconds !== 0) return "—";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

export function fmtBytes(bytes) {
  if (!bytes) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i++;
  }
  return `${value.toFixed(value < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

/** Trajectory ids are timestamps like 2026-08-16_21-14-02; show them like a human would. */
export function fmtTimestamp(id) {
  const match = /^(\d{4})-(\d{2})-(\d{2})[_T](\d{2})-(\d{2})-(\d{2})/.exec(id || "");
  if (!match) return id || "";
  const [, , month, day, hour, minute] = match;
  return `${month}/${day} ${hour}:${minute}`;
}

export function fmtNumber(value, digits = 2) {
  if (value == null || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits).replace(/\.?0+$/, "");
}

export function debounce(fn, ms = 200) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}
