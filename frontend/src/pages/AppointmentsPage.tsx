import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../api/client";
import type { Appointment, AuthUser } from "../types";

type Props = { user: AuthUser | null };

export function AppointmentsPage({ user }: Props) {
  const query = useQuery({
    queryKey: ["appointments", user?.email],
    queryFn: () => apiFetch<{ appointments: Appointment[] }>(`/appointments/me?email=${encodeURIComponent(user?.email || "")}`),
    enabled: Boolean(user?.email),
  });

  if (!user) return <section className="card"><p className="status">Login to view appointments.</p></section>;
  if (query.isLoading) return <section className="card"><p className="status">Loading appointments...</p></section>;
  if (query.isError) return <section className="card"><p className="status">{(query.error as Error).message}</p></section>;

  return (
    <section className="card">
      <h2>My Appointments</h2>
      <div className="list">
        {(query.data?.appointments || []).map((appt) => (
          <div key={appt.appointment_id} className="listItem">
            <strong>{appt.doctor_name}</strong>
            <span>{appt.speciality}</span>
            <span>{appt.appointment_date} at {appt.slot_start_time}</span>
          </div>
        ))}
        {!query.data?.appointments?.length && <p className="status">No appointments booked yet.</p>}
      </div>
    </section>
  );
}
