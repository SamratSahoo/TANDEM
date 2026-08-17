// Hand-rolled SVG charts for a trajectory's per-frame series.
//
// Three stacked panels sharing one x scale and one crosshair, so a spike in commanded
// velocity lines up with the joint that moved and with the gripper event that caused it.
// Clicking anywhere seeks every camera video to that frame.

import { svg, h, clear, fmtNumber } from "./dom.js";

const W = 760;
const PAD_L = 54;
const PAD_R = 12;
const PLOT_W = W - PAD_L - PAD_R;
const N_JOINTS = 7;

// A perceptually spread ramp so seven overlapping lines stay tellable apart.
export const JOINT_COLORS = [
  "#4f9dff", "#3fb950", "#d29922", "#a371f7", "#f85149", "#2dd4bf", "#f472b6",
];

// Why a frame was dropped by the DROID non-idle filter, and how to shade it.
const FILTER_FILL = {
  idle: "rgba(125,136,150,0.16)",
  short: "rgba(210,153,34,0.14)",
  trim: "rgba(163,113,247,0.13)",
};
const FILTER_LABEL = {
  idle: "idle run",
  short: "run too short",
  trim: "trimmed tail",
};

function niceTicks(lo, hi, count = 4) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) || magnitude * 10;
  const ticks = [];
  for (let value = Math.ceil(lo / step) * step; value <= hi + 1e-9; value += step) ticks.push(value);
  return ticks;
}

function extent(seriesList) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const values of seriesList) {
    for (const value of values) {
      if (!Number.isFinite(value)) continue;
      if (value < lo) lo = value;
      if (value > hi) hi = value;
    }
  }
  if (!Number.isFinite(lo)) return [0, 1];
  if (lo === hi) return [lo - 0.5, hi + 0.5];
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

/** Pull column j out of a [F][7] series into a flat array. */
function column(rows, j) {
  if (!rows) return null;
  const out = new Array(rows.length);
  for (let i = 0; i < rows.length; i++) out[i] = rows[i][j];
  return out;
}

class Panel {
  constructor({ title, unit, lines, t, domain, height = 168, showX, dropSpans, onHover, onSeek }) {
    this.opts = { title, unit, lines, t, domain, height, showX, dropSpans, onHover, onSeek };
  }

  render() {
    const { title, unit, lines, t, domain, height, showX, dropSpans, onHover, onSeek } = this.opts;
    const plotTop = 8;
    const plotBottom = height - (showX ? 24 : 8);
    const plotH = plotBottom - plotTop;
    const tmax = t.length ? t[t.length - 1] : 1;
    const [lo, hi] = domain;
    const span = hi - lo || 1;

    const x = (i) => PAD_L + (tmax > 0 ? (t[i] / tmax) * PLOT_W : 0);
    const xt = (seconds) => PAD_L + (tmax > 0 ? (seconds / tmax) * PLOT_W : 0);
    const y = (v) => plotTop + (1 - (v - lo) / span) * plotH;

    const kids = [];

    // Shade the frames training would throw away, underneath everything else.
    for (const span_ of dropSpans || []) {
      const x0 = xt(span_.t0);
      const x1 = xt(span_.t1);
      if (x1 - x0 < 0.4) continue;
      kids.push(
        svg("rect", {
          x: x0,
          y: plotTop,
          width: Math.max(0.6, x1 - x0),
          height: plotH,
          fill: FILTER_FILL[span_.reason] || FILTER_FILL.idle,
        })
      );
    }

    // Gridlines + y labels.
    for (const tick of niceTicks(lo, hi)) {
      const yy = y(tick);
      kids.push(svg("line", { x1: PAD_L, x2: W - PAD_R, y1: yy, y2: yy, stroke: "#1e2631", "stroke-width": 1 }));
      kids.push(
        svg("text", { x: PAD_L - 7, y: yy + 3.5, "text-anchor": "end", fill: "#5c6875", "font-size": 10 },
          fmtNumber(tick, Math.abs(tick) < 1 ? 2 : 1))
      );
    }

    for (const line of lines) {
      if (!line.values) continue;
      const path = buildPath(line.values, x, y, line.kind);
      if (!path) continue;
      kids.push(
        svg("path", {
          d: path,
          fill: "none",
          stroke: line.color,
          "stroke-width": line.width || 1.35,
          "stroke-dasharray": line.dashed ? "3 3" : null,
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
          opacity: line.opacity ?? 1,
        })
      );
    }

    if (showX) {
      for (const tick of niceTicks(0, tmax, 6)) {
        const xx = xt(tick);
        kids.push(
          svg("text", { x: xx, y: height - 7, "text-anchor": "middle", fill: "#5c6875", "font-size": 10 },
            `${fmtNumber(tick, 1)}s`)
        );
      }
    }

    const cursor = svg("line", {
      x1: 0, x2: 0, y1: plotTop, y2: plotBottom,
      stroke: "#4f9dff", "stroke-width": 1, opacity: 0, "pointer-events": "none",
    });
    kids.push(cursor);

    const node = svg("svg.chart", {
      viewBox: `0 0 ${W} ${height}`,
      preserveAspectRatio: "none",
      style: `height:${height}px`,
      onmousemove: (event) => {
        const index = indexAt(event, node, t, tmax);
        if (index == null) return;
        cursor.setAttribute("x1", x(index));
        cursor.setAttribute("x2", x(index));
        cursor.setAttribute("opacity", 0.75);
        onHover && onHover(index);
      },
      onmouseleave: () => {
        cursor.setAttribute("opacity", 0);
        onHover && onHover(null);
      },
      onclick: (event) => {
        const index = indexAt(event, node, t, tmax);
        if (index != null && onSeek) onSeek(index);
      },
    }, ...kids);

    const header = h("div.row", { style: { marginBottom: "2px" } },
      h("span.small", { style: { fontWeight: 600 } }, title),
      unit ? h("span.faint.small", unit) : null,
      h("div.spacer"),
      h("div.chart-legend", ...(lines.filter((l) => l.label).map((l) =>
        h("span", h("span.sw", { style: { background: l.color, opacity: l.dashed ? 0.6 : 1 } }), l.label)
      )))
    );

    return h("div.chart-panel", header, node);
  }
}

function indexAt(event, node, t, tmax) {
  const rect = node.getBoundingClientRect();
  const fraction = (event.clientX - rect.left) / rect.width;
  const px = fraction * W;
  const seconds = ((px - PAD_L) / PLOT_W) * tmax;
  if (!t.length) return null;
  // Nearest sample, not the interpolated position: every plotted point is a real frame.
  let best = 0;
  let bestDelta = Infinity;
  for (let i = 0; i < t.length; i++) {
    const delta = Math.abs(t[i] - seconds);
    if (delta < bestDelta) {
      bestDelta = delta;
      best = i;
    }
  }
  return best;
}

function buildPath(values, x, y, kind) {
  let d = "";
  let open = false;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) {
      open = false;
      continue;
    }
    const px = x(i);
    const py = y(v);
    if (!open) {
      d += `M${px.toFixed(2)},${py.toFixed(2)}`;
      open = true;
    } else if (kind === "step") {
      d += `H${px.toFixed(2)}V${py.toFixed(2)}`;
    } else {
      d += `L${px.toFixed(2)},${py.toFixed(2)}`;
    }
  }
  return d || null;
}

/**
 * Render every panel for one series payload into `container`.
 * `onSeek(index)` receives the plotted-frame index the user clicked.
 */
export function renderSeries(container, series, { onSeek } = {}) {
  clear(container);
  const t = series.t || [];
  if (!t.length) {
    container.appendChild(h("div.empty", "This trajectory recorded no per-frame state."));
    return;
  }

  const dropSpans = (series.filter && series.filter.drop_spans) || [];
  const readout = h("div.small.faint", { style: { minHeight: "18px", fontFamily: "var(--mono)" } }, " ");

  const measured = series.joint_position;
  const commanded = series.cmd_joint_position;
  const velocity = series.cmd_joint_velocity;
  const gripper = series.gripper_position;
  const cmdGripper = series.cmd_gripper;

  const onHover = (index) => {
    if (index == null) {
      readout.textContent = " ";
      return;
    }
    const parts = [`t ${fmtNumber(t[index], 2)}s`, `frame ${index}`];
    if (series.filter && series.filter.reason) {
      const reason = series.filter.reason[index];
      parts.push(reason === "keep" ? "kept" : `dropped · ${FILTER_LABEL[reason] || reason}`);
    }
    if (gripper) parts.push(`gripper ${fmtNumber(gripper[index], 2)}`);
    readout.textContent = parts.join("   ");
  };

  const panels = [];

  if (measured) {
    const lines = [];
    for (let j = 0; j < N_JOINTS; j++) {
      lines.push({ values: column(measured, j), color: JOINT_COLORS[j], label: j === 0 ? "measured" : null });
    }
    if (commanded) {
      for (let j = 0; j < N_JOINTS; j++) {
        lines.push({
          values: column(commanded, j), color: JOINT_COLORS[j], dashed: true, opacity: 0.55,
          label: j === 0 ? "commanded" : null,
        });
      }
    }
    panels.push(new Panel({
      title: "joint position", unit: "rad", lines, t,
      domain: extent(lines.map((l) => l.values).filter(Boolean)),
      dropSpans, onHover, onSeek,
    }));
  }

  if (velocity) {
    const lines = [];
    for (let j = 0; j < N_JOINTS; j++) {
      lines.push({ values: column(velocity, j), color: JOINT_COLORS[j], label: j === 0 ? "commanded" : null });
    }
    panels.push(new Panel({
      title: "commanded joint velocity",
      unit: "DROID normalized · clipped to ±1 at training time",
      lines, t, domain: extent(lines.map((l) => l.values)), dropSpans, onHover, onSeek,
    }));
  }

  if (gripper || cmdGripper) {
    const lines = [];
    if (gripper) lines.push({ values: gripper, color: "#4f9dff", label: "measured", width: 1.6 });
    if (cmdGripper) {
      lines.push({ values: cmdGripper, color: "#d29922", kind: "step", label: "commanded (binary)", width: 1.6 });
    }
    panels.push(new Panel({
      title: "gripper", unit: "0 open · 1 closed",
      lines, t, domain: [-0.08, 1.08], height: 128, showX: true, dropSpans, onHover, onSeek,
    }));
  }

  if (!panels.length) {
    container.appendChild(h("div.empty", "No plottable channels in this trajectory."));
    return;
  }
  panels[panels.length - 1].opts.showX = true;

  for (const panel of panels) container.appendChild(panel.render());
  container.appendChild(readout);

  if (series.filter) container.appendChild(filterSummary(series.filter, series.n_frames));
  if (series.downsampled) {
    container.appendChild(
      h("div.faint.small", { style: { marginTop: "6px" } },
        `Plotted ${series.n_plotted} of ${series.n_frames} frames. The filter above ran at full resolution.`)
    );
  }
}

function filterSummary(filter, nFrames) {
  const kept = filter.n_kept || 0;
  const total = filter.n_frames || nFrames || 1;
  const pct = Math.round((kept / total) * 100);
  const dropped = filter.n_dropped || {};

  const legend = h("div.chart-legend", { style: { marginTop: "4px" } },
    ...Object.entries(FILTER_LABEL)
      .filter(([key]) => dropped[key])
      .map(([key, label]) =>
        h("span", h("span.sw", { style: { background: FILTER_FILL[key], height: "9px" } }),
          `${label} · ${dropped[key]}`))
  );

  const clipNote = filter.frac_clipped > 0.01
    ? h("div.alert", { style: { marginTop: "8px" } },
        `${(filter.frac_clipped * 100).toFixed(1)}% of commanded velocity elements exceed ±1 and are ` +
        `clipped in the dataset. The filter above reflects the clipped signal, so the shading and the ` +
        `plotted line diverge where that happens.`)
    : null;

  return h("div", { style: { marginTop: "10px" } },
    h("div.row",
      h("span.small", { style: { fontWeight: 600 } }, "non-idle filter"),
      h("span.faint.small", `${kept} of ${total} frames (${pct}%) survive to training`),
      h("div.spacer"),
      h("div.bar", { style: { width: "120px" } }, h("span", { style: { width: `${pct}%` } }))
    ),
    legend, clipNote);
}
