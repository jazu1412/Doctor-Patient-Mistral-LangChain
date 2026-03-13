-- Google Cloud SQL (MySQL 8.4) schema for Doctor Appointments
-- Run this in Cloud SQL Studio: Connect to your instance -> Open Cloud SQL Studio -> paste and run

-- Create database (if not exists; Cloud SQL often has a default database)
-- CREATE DATABASE IF NOT EXISTS doctor_appointments_db;
-- USE doctor_appointments_db;

-- =============================================================================
-- 1. Doctors (200 doctors)
-- =============================================================================

USE doctor_appointments_db;

CREATE TABLE IF NOT EXISTS doctors (
    doctor_id INT NOT NULL PRIMARY KEY,
    doctor_name VARCHAR(255) NOT NULL,
    speciality VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT chk_doctor_id_range CHECK (doctor_id >= 1 AND doctor_id <= 200)
);

-- =============================================================================
-- 2. Users (default user: jas@gmail.com)
-- =============================================================================
CREATE TABLE IF NOT EXISTS users (
    user_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Insert default user
INSERT IGNORE INTO users (user_id, email) VALUES (1, 'jas@gmail.com');

-- =============================================================================
-- 3. Appointments
-- Slots: 9:00 AM to 5:00 PM (30-minute slots: 9:00, 9:30, 10:00, ..., 16:30)
-- One row per (doctor_id, appointment_date, slot_start_time) when booked
-- =============================================================================
CREATE TABLE IF NOT EXISTS appointments (
    appointment_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    doctor_id INT NOT NULL,
    user_id INT NOT NULL,
    appointment_date DATE NOT NULL,
    slot_start_time TIME NOT NULL COMMENT 'Start of 30-min slot, e.g. 09:00, 09:30, ..., 16:30',
    status ENUM('booked', 'cancelled') NOT NULL DEFAULT 'booked',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_doctor_date_slot (doctor_id, appointment_date, slot_start_time),
    CONSTRAINT fk_appt_doctor FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id) ON DELETE CASCADE,
    CONSTRAINT fk_appt_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    CONSTRAINT chk_slot_9to5 CHECK (
        slot_start_time >= '09:00:00' AND slot_start_time <= '16:30:00'
    )
);

-- =============================================================================
-- 4. Seed 200 doctors (placeholder names; sync from your app if needed)
-- =============================================================================
-- Insert doctors 1-200 with placeholder names (you can update names from ChromaDB later)
INSERT IGNORE INTO doctors (doctor_id, doctor_name, speciality)
WITH RECURSIVE n AS (
    SELECT 1 AS id
    UNION ALL
    SELECT id + 1 FROM n WHERE id < 200
)
SELECT id, CONCAT('Doctor ', id), 'General' FROM n
ON DUPLICATE KEY UPDATE doctor_name = VALUES(doctor_name), speciality = VALUES(speciality);

-- Indexes for common queries
CREATE INDEX idx_appointments_doctor_date ON appointments(doctor_id, appointment_date);
CREATE INDEX idx_appointments_user ON appointments(user_id, appointment_date);
CREATE INDEX idx_appointments_status ON appointments(status);
