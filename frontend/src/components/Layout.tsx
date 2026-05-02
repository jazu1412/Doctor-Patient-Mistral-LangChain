import { Link, Outlet, useLocation } from "react-router-dom";
import type { AuthUser } from "../types";

type Props = {
  user: AuthUser | null;
  onLogout: () => void;
};

export function Layout({ user, onLogout }: Props) {
  const location = useLocation();
  const isDoctor = (user?.role || "").toLowerCase() === "doctor";
  const isPatient = Boolean(user && !isDoctor);
  const isEmergencyVitals = location.pathname === "/emergency-vitals";

  const doctorTabs = [{ to: "/", label: "Clinical Decision Matching System" }];
  const patientTabs = [
    { to: "/", label: "Find Doctor" },
    { to: "/appointments", label: "My Appointments" },
  ];

  const title = isDoctor
    ? "Clinical Decision Matching System"
    : isEmergencyVitals
      ? "Emergency vitals monitoring"
      : "Doctor-Patient Matching System";

  const subtitle = isDoctor
    ? "Review similar clinical cases to support informed medical decisions."
    : isEmergencyVitals
      ? "Live pulse, oxygen, blood pressure, respiration, and temperature."
      : "Enter your symptoms below to find the best matching doctor for your needs.";

  return (
    <div className="shell">
      <header className={`topbar ${isPatient && isEmergencyVitals ? "topbar--emergency" : ""}`}>
        <div className="topbarTopRow">
          <div className="topbarTitles">
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
          {isPatient && !isEmergencyVitals && (
            <Link
              to="/emergency-vitals"
              className="stealthEmergencyLink"
              title="Emergency vitals (hidden entry)"
              aria-label="Open emergency vitals monitoring"
            >
              ·
            </Link>
          )}
        </div>
        <nav>
          {user && isDoctor &&
            doctorTabs.map((tab) => (
              <Link
                className={location.pathname === tab.to ? "tab active" : "tab"}
                key={tab.to}
                to={tab.to}
              >
                {tab.label}
              </Link>
            ))}
          {user && isPatient && isEmergencyVitals && (
            <Link className="tab tab-emergencyHome" to="/">
              Home
            </Link>
          )}
          {user && isPatient && !isEmergencyVitals &&
            patientTabs.map((tab) => (
              <Link
                className={location.pathname === tab.to ? "tab active" : "tab"}
                key={tab.to}
                to={tab.to}
              >
                {tab.label}
              </Link>
            ))}
          {!user ? (
            <Link className={location.pathname === "/auth" ? "tab active" : "tab"} to="/auth">
              Login
            </Link>
          ) : (
            <button className="tab" onClick={onLogout} type="button">
              Logout
            </button>
          )}
        </nav>
      </header>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
