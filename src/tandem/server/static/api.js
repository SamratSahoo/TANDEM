// The JSON API, plus the SSE channel a live session streams over.

const BASE = "/api";

async function request(method, path, body) {
  const response = await fetch(BASE + path, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = { error: text };
  }
  if (!response.ok) {
    const error = new Error((payload && (payload.error || payload.detail)) || response.statusText);
    error.status = response.status;
    error.hint = payload && payload.hint;
    throw error;
  }
  return payload;
}

export const api = {
  get: (path) => request("GET", path),
  post: (path, body) => request("POST", path, body),
  put: (path, body) => request("PUT", path, body),
  del: (path) => request("DELETE", path),

  // ---- profiles
  profiles: () => request("GET", "/profiles"),
  profile: (name) => request("GET", `/profiles/${encodeURIComponent(name)}`),
  saveProfile: (name, body) => request("PUT", `/profiles/${encodeURIComponent(name)}`, body),
  createProfile: (body) => request("POST", "/profiles", body),
  deleteProfile: (name, purge) =>
    request("DELETE", `/profiles/${encodeURIComponent(name)}${purge ? "?purge=true" : ""}`),
  setActive: (name) => request("POST", "/profiles/active", { name }),

  // ---- trajectories
  trajectories: (profile, status) => {
    const params = new URLSearchParams({ profile });
    if (status) params.set("status", status);
    return request("GET", `/trajectories?${params}`);
  },
  trajectory: (profile, id) =>
    request("GET", `/trajectories/${encodeURIComponent(profile)}/${encodeURIComponent(id)}`),
  series: (profile, id) =>
    request("GET", `/trajectories/${encodeURIComponent(profile)}/${encodeURIComponent(id)}/series`),
  relabel: (profile, id, status, force = false) =>
    request("POST", `/trajectories/${encodeURIComponent(profile)}/${encodeURIComponent(id)}/relabel`, { status, force }),
  deleteTrajectory: (profile, id) =>
    request("DELETE", `/trajectories/${encodeURIComponent(profile)}/${encodeURIComponent(id)}`),
  mediaUrl: (profile, id, file) =>
    `${BASE}/media/${encodeURIComponent(profile)}/${encodeURIComponent(id)}/${encodeURIComponent(file)}`,

  // ---- sessions
  createSession: (body) => request("POST", "/sessions", body),
  sessions: () => request("GET", "/sessions"),
  session: (id) => request("GET", `/sessions/${id}`),
  label: (id, success) => request("POST", `/sessions/${id}/label`, { success }),
  continueSession: (id, task, more = true) => request("POST", `/sessions/${id}/continue`, { task, more }),
  preempt: (id) => request("POST", `/sessions/${id}/preempt`),
  stopSession: (id) => request("POST", `/sessions/${id}/stop`),
  forceStop: (id) => request("POST", `/sessions/${id}/force-stop`),
  teleopSwitch: (id) => request("POST", `/sessions/${id}/teleop-switch`),
  teleopResume: (id) => request("POST", `/sessions/${id}/teleop-resume`),
  humanPhaseDone: (id) => request("POST", `/sessions/${id}/human-phase/done`),
  humanPhaseAbort: (id) => request("POST", `/sessions/${id}/human-phase/abort`),

  // ---- settings
  settings: () => request("GET", "/settings"),
  saveSettings: (body) => request("PUT", "/settings", body),
  saveSecrets: (body) => request("PUT", "/settings/secrets", body),
  runtime: () => request("GET", "/runtime"),
  doctor: (profile, hardware) => {
    const params = new URLSearchParams();
    if (profile) params.set("profile", profile);
    if (hardware) params.set("hardware", "true");
    return request("GET", `/doctor?${params}`);
  },
};

/**
 * Subscribe to a session's event stream.
 *
 * EventSource reconnects on its own, which is what we want next to a robot: a dropped
 * connection must not look like a dead session.
 */
export function streamSession(sessionId, onMessage, onError) {
  const source = new EventSource(`${BASE}/sessions/${sessionId}/stream`);
  source.onmessage = (event) => {
    try {
      onMessage(JSON.parse(event.data));
    } catch {
      /* keepalive comments and partial frames are not worth reporting */
    }
  };
  source.onerror = () => onError && onError();
  return () => source.close();
}
