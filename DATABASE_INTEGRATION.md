# Database Integration Guide

## Overview

The Doctor-Patient Matching System now integrates with **TimescaleDB** (PostgreSQL) to track doctor availability and manage appointments.

## Features

1. **Doctor Availability Tracking**: Doctors are stored in TimescaleDB with availability status
2. **Automatic Filtering**: Only available doctors are shown in search results
3. **Appointment Booking**: Patients can book appointments, which marks doctors as unavailable
4. **Database Sync**: Doctors from ChromaDB are automatically synced to TimescaleDB

## Database Schema

```sql
CREATE TABLE doctors (
    id SERIAL PRIMARY KEY,
    doctor_name VARCHAR(255) NOT NULL UNIQUE,
    speciality VARCHAR(255) NOT NULL,
    chroma_id VARCHAR(255),
    is_available BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Connection String

The system uses the following connection string format:

```
postgres://tsdbadmin:g9heeciuyqfflr2z@lowkayf0dw.u827f1jcvc.tsdb.cloud.timescale.com:35981/tsdb?sslmode=require
```

Or alternative format:

```
dbname=tsdb user=tsdbadmin password=g9heeciuyqfflr2z host=lowkayf0dw.u827f1jcvc.tsdb.cloud.timescale.com port=35981 sslmode=require
```

## Configuration

Add to your `.env` file:

```env
DB_CONNECTION=postgres://tsdbadmin:g9heeciuyqfflr2z@lowkayf0dw.u827f1jcvc.tsdb.cloud.timescale.com:35981/tsdb?sslmode=require
```

## How It Works

### 1. Database Initialization

The database schema is automatically created on first app run. You can also manually initialize:

```bash
python init_database.py
```

### 2. Doctor Sync

When a patient searches for doctors:
1. ChromaDB returns matching doctors based on symptoms
2. Doctors are automatically synced to TimescaleDB
3. Only available doctors are returned to the patient

### 3. Availability Filtering

The `find_best_doctor()` function:
- Queries ChromaDB for matching doctors
- Syncs doctors to TimescaleDB
- Filters results to show only available doctors
- Returns top-k available doctors

### 4. Appointment Booking

When a patient clicks "Book Appointment":
1. System checks doctor availability
2. If available, marks doctor as `is_available = FALSE`
3. Doctor is removed from future search results until released

### 5. Releasing Doctors

To make a doctor available again, you can:
- Use the `sync_release_doctor()` function
- Manually update the database: `UPDATE doctors SET is_available = TRUE WHERE doctor_name = 'Doctor Name'`

## Database Functions

### Available Functions

- `sync_init_database()`: Initialize database schema
- `sync_sync_doctors(doctors)`: Sync doctors from ChromaDB to TimescaleDB
- `sync_get_available_doctors(doctor_names)`: Get list of available doctors
- `sync_check_doctor_availability(doctor_name)`: Check if a doctor is available
- `sync_book_doctor(doctor_name)`: Mark doctor as unavailable
- `sync_release_doctor(doctor_name)`: Mark doctor as available
- `sync_get_all_doctors()`: Get all doctors with their status
- `sync_get_doctor_by_name(doctor_name)`: Get doctor information

## UI Features

### Sidebar Database Management

1. **View All Doctors**: Display all doctors in the database with their availability status
2. **Sync Doctors from ChromaDB**: Manually sync all doctors from ChromaDB to TimescaleDB

### Doctor Cards

Each doctor card shows:
- ✅ Available - Doctor is available for booking
- ⏸️ Currently Unavailable - Doctor is booked
- 📅 Book Appointment button - Only shown for available doctors

## Example Usage

```python
from database import sync_book_doctor, sync_release_doctor, sync_get_all_doctors

# Book a doctor
sync_book_doctor("Dr. John Smith")

# Release a doctor
sync_release_doctor("Dr. John Smith")

# Get all doctors
all_doctors = sync_get_all_doctors()
```

## Error Handling

The system includes fail-safe mechanisms:
- If database connection fails, all doctors are shown (no filtering)
- Warnings are displayed instead of errors to prevent app crashes
- Database operations are wrapped in try-except blocks

## Future Enhancements

Potential improvements:
- Appointment scheduling with time slots
- Patient information storage
- Appointment history tracking
- Doctor availability by time slots
- Automatic release after appointment completion


