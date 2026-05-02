const defaultApiBase =
  typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:8000/api/v1`
    : "http://localhost:8000/api/v1";

const rawBase = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim();
const API_BASE = (rawBase && rawBase.length > 0 ? rawBase : defaultApiBase).replace(/\/+$/, "");

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  // #region agent log
  fetch("http://127.0.0.1:7835/ingest/448cfad2-0dbb-49ef-8467-dc5fbb19c120", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "ffe8cc" },
    body: JSON.stringify({
      sessionId: "ffe8cc",
      runId: "iphone-host-debug",
      hypothesisId: "H3_H4",
      location: "src/api/client.ts:4",
      message: "apiFetch request",
      data: {
        path,
        apiBase: API_BASE,
        method: init?.method || "GET",
        pageHost: typeof window !== "undefined" ? window.location.host : "no-window",
      },
      timestamp: Date.now(),
    }),
  }).catch(() => {});
  // #endregion

  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!response.ok) {
    const text = await response.text();
    // #region agent log
    fetch("http://127.0.0.1:7835/ingest/448cfad2-0dbb-49ef-8467-dc5fbb19c120", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "ffe8cc" },
      body: JSON.stringify({
        sessionId: "ffe8cc",
        runId: "iphone-host-debug",
        hypothesisId: "H4_H5",
        location: "src/api/client.ts:30",
        message: "apiFetch non-ok response",
        data: {
          path,
          status: response.status,
          statusText: response.statusText,
        },
        timestamp: Date.now(),
      }),
    }).catch(() => {});
    // #endregion
    throw new Error(text || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function getApiOrigin(): string {
  return API_BASE.replace(/\/api\/v1\/?$/, "");
}

/** WebSocket URL for demo emergency vitals stream (same host/port as REST API). */
export function getVitalsWebSocketUrl(): string {
  const origin = getApiOrigin();
  const wsProto = origin.startsWith("https") ? "wss" : "ws";
  const hostPath = origin.replace(/^https?:\/\//, "");
  return `${wsProto}://${hostPath}/api/v1/emergency/vitals/ws`;
}

export { API_BASE };
