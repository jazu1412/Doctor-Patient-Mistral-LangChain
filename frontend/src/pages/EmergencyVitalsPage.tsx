import { useCallback, useEffect, useRef, useState } from "react";
import { getVitalsWebSocketUrl } from "../api/client";

type VitalsPayload = {
  ts: number;
  vitals: {
    heart_rate_bpm: number;
    spo2_pct: number;
    blood_pressure: { systolic: number; diastolic: number };
    respiratory_rate: number;
    temperature_f: number;
  };
  urgency: "normal" | "high" | "critical";
  alert: boolean;
  message: string | null;
};

type ConnState = "connecting" | "live" | "closed" | "error";

export function EmergencyVitalsPage() {
  const [conn, setConn] = useState<ConnState>("connecting");
  const [latest, setLatest] = useState<VitalsPayload | null>(null);
  const [lastAlerts, setLastAlerts] = useState<{ ts: number; urgency: string; message: string }[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const attachHandlers = useCallback((ws: WebSocket) => {
    ws.onopen = () => setConn("live");
    ws.onclose = () => setConn("closed");
    ws.onerror = () => setConn("error");
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data as string) as VitalsPayload;
        setLatest(data);
        if (data.alert && data.message) {
          setLastAlerts((prev) => {
            const next = [{ ts: data.ts, urgency: data.urgency, message: data.message! }, ...prev];
            return next.slice(0, 12);
          });
        }
      } catch {
        // ignore malformed frames
      }
    };
  }, []);

  useEffect(() => {
    const url = getVitalsWebSocketUrl();
    const ws = new WebSocket(url);
    wsRef.current = ws;
    attachHandlers(ws);
    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [attachHandlers]);

  function reconnect() {
    setConn("connecting");
    wsRef.current?.close();
    const url = getVitalsWebSocketUrl();
    const ws = new WebSocket(url);
    wsRef.current = ws;
    attachHandlers(ws);
  }

  const v = latest?.vitals;
  const urgency = latest?.urgency ?? "normal";

  return (
    <section className="list emergencyVitalsRoot">
      <article className="card card--emergencyMonitor">
        <h2 className="emergencyMonitorTitle">Live vitals</h2>
        <p className="subtitle emergencyMonitorSub">
          Pulse, SpO₂, blood pressure, respiratory rate, and temperature — updated in real time. Simulated
          feed for training scenarios; not for clinical diagnosis.
        </p>
        <div className="emergencyStatusRow">
          <span className={`emergencyConn emergencyConn-${conn}`}>
            <span className="emergencyConnDot" aria-hidden />
            {conn === "connecting" && "Connecting…"}
            {conn === "live" && "Live stream"}
            {conn === "closed" && "Disconnected"}
            {conn === "error" && "Connection error"}
          </span>
          {(conn === "closed" || conn === "error") && (
            <button type="button" className="btn primary" onClick={reconnect}>
              Reconnect
            </button>
          )}
        </div>
      </article>

      {latest?.alert && latest.message && (
        <div className={`emergencyAlert urgency-${urgency}`} role="alert">
          <strong>{urgency === "critical" ? "Critical alert" : "High urgency"}</strong>
          <p>{latest.message}</p>
        </div>
      )}

      <div className="vitalsGrid vitalsGrid--emergency">
        <div className="vitalCard vitalCard--emergency">
          <span className="vitalLabel">Heart rate</span>
          <strong className="vitalValue">{v ? `${v.heart_rate_bpm} bpm` : "—"}</strong>
        </div>
        <div className="vitalCard vitalCard--emergency">
          <span className="vitalLabel">SpO₂</span>
          <strong className="vitalValue">{v ? `${v.spo2_pct}%` : "—"}</strong>
        </div>
        <div className="vitalCard vitalCard--emergency">
          <span className="vitalLabel">Blood pressure</span>
          <strong className="vitalValue">
            {v ? `${v.blood_pressure.systolic}/${v.blood_pressure.diastolic}` : "—"}
          </strong>
        </div>
        <div className="vitalCard vitalCard--emergency">
          <span className="vitalLabel">Resp. rate</span>
          <strong className="vitalValue">{v ? `${v.respiratory_rate} /min` : "—"}</strong>
        </div>
        <div className="vitalCard vitalCardWide vitalCard--emergency">
          <span className="vitalLabel">Temperature</span>
          <strong className="vitalValue">{v ? `${v.temperature_f} °F` : "—"}</strong>
        </div>
      </div>

      {lastAlerts.length > 0 && (
        <article className="card card--emergencyAlerts">
          <h3 className="sectionTitle emergencyAlertsTitle">Recent alerts</h3>
          <ul className="emergencyAlertLog">
            {lastAlerts.map((a) => (
              <li key={`${a.ts}-${a.message}`}>
                <span className={`tag urgency-${a.urgency}`}>{a.urgency}</span>
                {a.message}
              </li>
            ))}
          </ul>
        </article>
      )}
    </section>
  );
}
