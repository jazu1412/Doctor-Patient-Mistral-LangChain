import { useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { apiFetch } from "../api/client";
import type { AuthUser } from "../types";

type Props = { onLogin: (user: AuthUser) => void };

export function AuthPage({ onLogin }: Props) {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [role, setRole] = useState("patient");
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setMessage("");
    try {
      if (mode === "login") {
        const res = await apiFetch<{ ok: boolean; message: string; user?: AuthUser }>("/auth/login", {
          method: "POST",
          body: JSON.stringify({ email, password }),
        });
        setMessage(res.message);
        if (res.ok) {
          if (res.user) {
            onLogin(res.user);
          }
          // Always leave auth page after successful login.
          navigate("/", { replace: true });
          // Fallback hard redirect for cases where client-side navigation is blocked/stale.
          window.location.replace("/");
        }
      } else {
        const res = await apiFetch<{ ok: boolean; message: string }>("/auth/signup", {
          method: "POST",
          body: JSON.stringify({ email, password, full_name: fullName, role }),
        });
        setMessage(res.message);
      }
    } catch (err) {
      setMessage((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card">
      <div className="row">
        <button className={mode === "login" ? "btn primary" : "btn"} onClick={() => setMode("login")}>Login</button>
        <button className={mode === "signup" ? "btn primary" : "btn"} onClick={() => setMode("signup")}>Sign Up</button>
      </div>
      <form onSubmit={onSubmit} className="form">
        <label>Email<input value={email} onChange={(e) => setEmail(e.target.value)} required type="email" /></label>
        <label>Password<input value={password} onChange={(e) => setPassword(e.target.value)} required type="password" /></label>
        {mode === "signup" && (
          <>
            <label>Full Name<input value={fullName} onChange={(e) => setFullName(e.target.value)} /></label>
            <label>Role
              <select value={role} onChange={(e) => setRole(e.target.value)}>
                <option value="patient">Patient</option>
                <option value="doctor">Doctor</option>
              </select>
            </label>
          </>
        )}
        <button className="btn primary" disabled={busy} type="submit">{busy ? "Please wait..." : "Continue"}</button>
      </form>
      {message && <p className="status">{message}</p>}
    </section>
  );
}
