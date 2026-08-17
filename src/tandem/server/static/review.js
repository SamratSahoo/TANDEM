// The trajectory review: camera videos, the hand-off segment ribbon, and the charts.
//
// Used in two places that want exactly the same thing — the Trajectories drawer, and the
// live Collect page at the moment the operator is asked to label a rollout. Labeling while
// looking at what the robot just did is the whole point.

import { api } from "./api.js";
import { renderSeries } from "./charts.js";
import { clear, h, mount } from "./dom.js";

const CAMERA_LABELS = {
  "external_cam.mp4": "Exterior 1",
  "external_cam_2.mp4": "Exterior 2",
  "hand_cam.mp4": "Wrist",
};

export function renderReview(container, { profile, trajectory }) {
  clear(container);
  const videos = new Map();

  const cameras = trajectory.cameras || [];
  if (cameras.length) {
    const grid = h("div.video-grid",
      ...cameras.map((file) => {
        const video = h("video", {
          src: api.mediaUrl(profile, trajectory.id, file),
          controls: true,
          muted: true,
          playsInline: true,
          preload: "metadata",
        });
        videos.set(file, video);
        return h("div.video-cell", video, h("div.video-cap", CAMERA_LABELS[file] || file));
      })
    );
    container.appendChild(grid);
    container.appendChild(h("div.row", { style: { marginTop: "8px" } },
      h("button.small.ghost", { onclick: () => videos.forEach((v) => v.play()) }, "▶ Play all"),
      h("button.small.ghost", { onclick: () => videos.forEach((v) => v.pause()) }, "⏸ Pause all"),
      h("button.small.ghost", {
        onclick: () => videos.forEach((v) => { v.currentTime = 0; }),
      }, "↺ Restart")
    ));
  } else {
    container.appendChild(h("div.alert", "No camera video was recorded for this trajectory."));
  }

  const chartHost = h("div", h("div.row", h("span.spin"), h("span.faint.small", "loading series…")));
  const ribbonHost = h("div");
  container.appendChild(ribbonHost);
  container.appendChild(chartHost);

  let series = null;

  /**
   * Seek every video to a plotted frame.
   *
   * Three tiers, in order of trustworthiness:
   *   1. video_time  — each frame's position in the CONCATENATED clip. The only correct map
   *      for a merged hand-off trajectory, whose wall clock contains the gaps between legs
   *      (camera teardown, the human working) that the video does not.
   *   2. record_start/stop — the camera recording window brackets the state window, so a
   *      frame's absolute timestamp minus record_start is its position in the clip.
   *   3. proportional — legacy episodes with neither. Best effort.
   */
  function seekToIndex(index) {
    if (!series) return;
    for (const video of videos.values()) {
      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : null;
      let target;
      if (series.video_time && series.video_time[index] != null) {
        target = series.video_time[index];
      } else if (series.record_start != null && series.t0 != null) {
        target = series.t0 + series.t[index] - series.record_start;
      } else if (duration != null && series.t.length) {
        const span = series.t[series.t.length - 1] || 1;
        target = (series.t[index] / span) * duration;
      } else {
        target = series.t[index];
      }
      target = Math.max(0, duration != null ? Math.min(target, duration - 1e-3) : target);
      try {
        video.currentTime = target;
      } catch {
        /* seeking before metadata loads throws; the next click will work */
      }
    }
  }

  function seekToSeconds(seconds) {
    for (const video of videos.values()) {
      try {
        video.currentTime = Math.max(0, seconds);
      } catch {
        /* ignore */
      }
    }
  }

  api.series(profile, trajectory.id)
    .then((payload) => {
      series = payload;
      renderSeries(chartHost, payload, { onSeek: seekToIndex });
      if (payload.segments && payload.segments.length > 1) {
        mount(ribbonHost, segmentRibbon(payload.segments, seekToSeconds));
      }
    })
    .catch((error) => {
      mount(chartHost, h("div.alert.err", error.message || "Could not load the series for this trajectory."));
    });

  return { seekToIndex, seekToSeconds };
}

/**
 * Which stretch of the video came from the planner and which from a human.
 *
 * Only meaningful for a merged hand-off trajectory; an ordinary rollout has one segment and
 * no ribbon. Widths are proportional to each leg's share of the CLIP, which is what the
 * viewer is actually scrubbing.
 */
function segmentRibbon(segments, onSeek) {
  const total = segments[segments.length - 1].video_stop || 0;
  if (!(total > 0)) return h("div");

  const bar = h("div.seg-ribbon", { title: "Hand-off segments — click one to jump the videos to it" },
    ...segments.map((segment) => {
      const width = ((segment.video_stop - segment.video_start) / total) * 100;
      const human = segment.source === "teleop";
      const seconds = Math.round(segment.video_stop - segment.video_start);
      return h(`button.seg.${human ? "seg-human" : "seg-tamp"}`, {
        style: { width: `${width}%` },
        title: `${human ? "Human" : "TAMP"} · ${segment.timestamp} · ${seconds}s · ${segment.n_frames} frames`,
        onclick: () => onSeek(segment.video_start),
      }, width > 8 ? (human ? "human" : "TAMP") : "");
    })
  );

  const humanLegs = segments.filter((s) => s.source === "teleop").length;
  return h("div",
    bar,
    h("div.faint.small",
      `${segments.length} legs merged into one trajectory — ${humanLegs} hand-off${humanLegs === 1 ? "" : "s"}. ` +
      "Click a stretch to jump the videos to it.")
  );
}
