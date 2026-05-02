export type AuthUser = {
  id: number;
  email: string;
  full_name: string;
  role: string;
};

export type DoctorMatch = {
  id: string;
  doctor_name: string;
  speciality: string;
  zip_codes: string[];
  similarity_score: number;
  rank_score: number;
  clinical_bonus: number;
  distance: number;
};

export type Appointment = {
  appointment_id: number;
  doctor_name: string;
  speciality: string;
  appointment_date: string;
  slot_start_time: string;
  status: string;
};
