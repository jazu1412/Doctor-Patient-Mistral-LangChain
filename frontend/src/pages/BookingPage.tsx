import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useMutation } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import { apiFetch } from "../api/client";
import type { AuthUser, DoctorMatch } from "../types";

type Props = { user: AuthUser | null };

function sanitizeRecommendationText(raw: string): string {
  return (raw || "")
    .replace(/\|\|+/g, " ")
    .replace(/^[\s|]*(?:-{3,}|={3,}|_{3,})[\s|]*$/gm, "")
    .replace(/(\s*[|]\s*){2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function hashString(input: string): number {
  let h = 0;
  for (let i = 0; i < input.length; i += 1) {
    h = (h * 31 + input.charCodeAt(i)) >>> 0;
  }
  return h;
}

function buildDoctorMapLink(doctor: DoctorMatch, doctorIndex: number, fallbackZip: string): string {
  const zip = doctor.zip_codes?.[0] || fallbackZip || "95192";
  // Demo centroid for San Jose State area (95192). Fallback keeps map usable.
  const baseLat = 37.3352;
  const baseLon = -121.8811;
  const seed = hashString(`${doctor.id}-${doctor.doctor_name}-${zip}`);
  const angle = ((seed % 360) * Math.PI) / 180;
  // Spread listed doctors apart for demo while keeping them nearby.
  const radiusKm = 0.45 + (doctorIndex % 5) * 0.35;
  const latOffset = (radiusKm / 111) * Math.cos(angle);
  const lonOffset = (radiusKm / (111 * Math.cos((baseLat * Math.PI) / 180))) * Math.sin(angle);
  const lat = (baseLat + latOffset).toFixed(6);
  const lon = (baseLon + lonOffset).toFixed(6);
  return `https://www.google.com/maps/search/?api=1&query=${lat},${lon}`;
}

export function BookingPage({ user }: Props) {
  const [symptoms, setSymptoms] = useState("");
  const [patientAge, setPatientAge] = useState<number>(27);
  const [patientGender, setPatientGender] = useState("female");
  const [patientZip, setPatientZip] = useState("");
  const [zipStatus, setZipStatus] = useState("Detecting ZIP from your location...");
  const [selected, setSelected] = useState<DoctorMatch | null>(null);
  const [date, setDate] = useState(new Date().toISOString().slice(0, 10));
  const [slot, setSlot] = useState("");
  const [message, setMessage] = useState("");
  const [topRecommendation, setTopRecommendation] = useState("");

  const searchContext = `${symptoms}\nPatient age: ${patientAge}\nPatient gender: ${patientGender}`;

  const matchMutation = useMutation({
    mutationFn: () =>
      apiFetch<{ doctors: DoctorMatch[] }>("/match/symptoms", {
        method: "POST",
        body: JSON.stringify({
          symptoms: searchContext,
          top_k: 5,
          patient_age: patientAge,
          patient_zip: patientZip,
        }),
      }),
  });

  const slotsMutation = useMutation({
    mutationFn: (doctorName: string) =>
      apiFetch<{ available_slots: string[] }>(
        `/doctors/${encodeURIComponent(doctorName)}/slots?appointment_date=${date}`,
      ),
  });

  const bookMutation = useMutation({
    mutationFn: () =>
      apiFetch<{ ok: boolean; message: string }>("/appointments", {
        method: "POST",
        body: JSON.stringify({
          doctor_name: selected?.doctor_name,
          speciality: selected?.speciality,
          email: user?.email,
          appointment_date: date,
          slot_start_time: slot,
        }),
      }),
    onSuccess: (res) => setMessage(res.message),
    onError: (err) => setMessage((err as Error).message),
  });

  const recommendationMutation = useMutation({
    mutationFn: () =>
      apiFetch<{ recommendation: string }>("/match/recommendation", {
        method: "POST",
        body: JSON.stringify({ symptoms: searchContext, top_k: 1 }),
      }),
    onSuccess: (res) => setTopRecommendation(res.recommendation || ""),
    onError: () => setTopRecommendation(""),
  });

  const doctors = useMemo(() => matchMutation.data?.doctors || [], [matchMutation.data]);

  useEffect(() => {
    if (!("geolocation" in navigator)) {
      setZipStatus("Location not supported in this browser. Enter ZIP manually.");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        try {
          const lat = position.coords.latitude;
          const lon = position.coords.longitude;
          const res = await apiFetch<{ zip_code: string }>(
            `/location/reverse-zip?lat=${encodeURIComponent(String(lat))}&lon=${encodeURIComponent(String(lon))}`,
            { method: "POST" },
          );
          if (res?.zip_code) {
            setPatientZip((prev) => prev || res.zip_code);
            setZipStatus(`ZIP auto-filled from current location: ${res.zip_code}`);
          } else {
            setZipStatus("Could not detect ZIP automatically. Enter ZIP manually.");
          }
        } catch {
          setZipStatus("Could not detect ZIP automatically. Enter ZIP manually.");
        }
      },
      () => {
        setZipStatus("Location permission denied. Enter ZIP manually.");
      },
      { enableHighAccuracy: false, timeout: 10000 },
    );
  }, []);

  async function runMatch(event: FormEvent) {
    event.preventDefault();
    setMessage("");
    setSelected(null);
    setSlot("");
    setTopRecommendation("");
    const result = await matchMutation.mutateAsync();
    if (result?.doctors?.length) {
      recommendationMutation.mutate();
    }
  }

  return (
    <section className="list">
      <article className="card">
        <h2>Describe your symptoms:</h2>
        <form className="form" onSubmit={runMatch}>
          <label>Describe your symptoms:
            <textarea required value={symptoms} onChange={(e) => setSymptoms(e.target.value)} rows={4} />
          </label>
          <div className="row">
            <label>Age<input type="number" value={patientAge} onChange={(e) => setPatientAge(Number(e.target.value))} /></label>
            <label>
              Gender
              <select value={patientGender} onChange={(e) => setPatientGender(e.target.value)}>
                <option value="female">Female</option>
                <option value="male">Male</option>
                <option value="other">Other</option>
              </select>
            </label>
            <label>
              ZIP
              <input value={patientZip} onChange={(e) => setPatientZip(e.target.value)} />
              <span className="status info" style={{ marginTop: 6 }}>{zipStatus}</span>
            </label>
          </div>
          <button className="btn primary" disabled={matchMutation.isPending}>
            {matchMutation.isPending ? "Matching..." : "Find Doctors"}
          </button>
        </form>
      </article>

      {matchMutation.isPending && (
        <article className="card loadingCard">
          <div className="matchingLoader" aria-live="polite" aria-busy="true">
            <div className="spinnerClock" />
            <h3>Matching doctors...</h3>
            <p>Analyzing symptoms and finding the best available specialists.</p>
          </div>
        </article>
      )}

      <article className="card">
        <h2>Matched doctors</h2>
        <div className="list">
          {doctors.map((doctor, doctorIndex) => (
            <button key={doctor.id} className={selected?.id === doctor.id ? "listItem active" : "listItem"} onClick={() => { setSelected(doctor); slotsMutation.mutate(doctor.doctor_name); }}>
              <strong>{doctor.doctor_name}</strong>
              <span>{doctor.speciality}</span>
              <a
                href={buildDoctorMapLink(doctor, doctorIndex, patientZip)}
                target="_blank"
                rel="noreferrer"
                className="mapLink"
                onClick={(event) => event.stopPropagation()}
              >
                View location on Google Maps
              </a>
            </button>
          ))}
          {!doctors.length && <p className="status">No doctor results yet.</p>}
        </div>
        {(recommendationMutation.isPending || topRecommendation) && (
          <div className="listItem ai-panel">
            <strong className="sectionTitle">Top AI Recommendation</strong>
            {recommendationMutation.isPending && <span>Generating recommendation...</span>}
            {!recommendationMutation.isPending && topRecommendation && (
              <div className="analysisMarkdown">
                <ReactMarkdown>{sanitizeRecommendationText(topRecommendation)}</ReactMarkdown>
              </div>
            )}
          </div>
        )}
        {selected && (
          <div className="form">
            <h3 style={{ margin: "4px 0" }}>Book slot with {selected.doctor_name}</h3>
            <label>Date<input type="date" value={date} onChange={(e) => setDate(e.target.value)} /></label>
            <label>Slot
              <select value={slot} onChange={(e) => setSlot(e.target.value)}>
                <option value="">Select available slot</option>
                {(slotsMutation.data?.available_slots || []).map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </label>
            <button className="btn primary" disabled={!user || !slot || bookMutation.isPending} onClick={() => bookMutation.mutate()}>
              {bookMutation.isPending ? "Booking..." : user ? "Book Appointment" : "Login to Book"}
            </button>
          </div>
        )}
      </article>
      {message && <p className="status">{message}</p>}
    </section>
  );
}
