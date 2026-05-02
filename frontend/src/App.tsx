import { useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { AppointmentsPage } from "./pages/AppointmentsPage";
import { AuthPage } from "./pages/AuthPage";
import { BookingPage } from "./pages/BookingPage";
import { ClinicalDecisionPage } from "./pages/ClinicalDecisionPage";
import { EmergencyVitalsPage } from "./pages/EmergencyVitalsPage";
import type { AuthUser } from "./types";

function App() {
  const [user, setUser] = useState<AuthUser | null>(() => {
    const raw = localStorage.getItem("auth_user");
    return raw ? (JSON.parse(raw) as AuthUser) : null;
  });

  useEffect(() => {
    if (user) {
      localStorage.setItem("auth_user", JSON.stringify(user));
    } else {
      localStorage.removeItem("auth_user");
    }
  }, [user]);

  function handleLogout() {
    setUser(null);
  }

  return (
    <Routes>
      <Route element={<Layout user={user} onLogout={handleLogout} />}>
        <Route
          path="/"
          element={
            user ? (
              user.role.toLowerCase() === "doctor" ? (
                <ClinicalDecisionPage />
              ) : (
                <BookingPage user={user} />
              )
            ) : (
              <Navigate to="/auth" replace />
            )
          }
        />
        <Route
          path="/appointments"
          element={
            user ? (
              user.role.toLowerCase() === "doctor" ? (
                <Navigate to="/" replace />
              ) : (
                <AppointmentsPage user={user} />
              )
            ) : (
              <Navigate to="/auth" replace />
            )
          }
        />
        <Route
          path="/emergency-vitals"
          element={
            user ? (
              user.role.toLowerCase() === "doctor" ? (
                <Navigate to="/" replace />
              ) : (
                <EmergencyVitalsPage />
              )
            ) : (
              <Navigate to="/auth" replace />
            )
          }
        />
        <Route path="/auth" element={user ? <Navigate to="/" replace /> : <AuthPage onLogin={setUser} />} />
        <Route path="*" element={<Navigate to={user ? "/" : "/auth"} replace />} />
      </Route>
    </Routes>
  );
}

export default App;
