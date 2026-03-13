# Google Cloud SQL – Appointments (9 AM–5 PM, default user jas@gmail.com)

## 1. Create the tables in Cloud SQL

1. In **Google Cloud Console** go to **SQL** and open your instance **doctor-appointments** (MySQL 8.4).
2. Use **Cloud SQL Studio** (or any MySQL client connected to the instance).
3. Create a database (if you prefer a dedicated DB):
   ```sql
   CREATE DATABASE IF NOT EXISTS doctor_appointments_db;
   USE doctor_appointments_db;
   ```
4. Run the full contents of **`cloud_sql_schema.sql`** in this project:
   - Creates `doctors` (200 rows, doctor_id 1–200),
   - `users` (inserts default user `jas@gmail.com`),
   - `appointments` (doctor_id, user_id, date, slot 9:00–16:30, status),
   - Indexes and foreign keys.

Slots are **30-minute intervals from 9:00 AM to 5:00 PM** (last slot 16:30–17:00).

## 2. Your Cloud SQL instance

| Setting | Value |
|--------|--------|
| **Connection name** | `linkedin-job-assistant-443903:us-west1:doctor-appointments` |
| **Public IP** | `34.11.152.7` |
| **Port** | `3306` (default MySQL TCP) |
| **Region** | `us-west1` |

Ensure the instance’s **Authorized networks** (in Cloud Console → SQL → your instance → Connections) include your app’s IP if you connect via public IP, or use **Cloud SQL Auth Proxy** and connect to `127.0.0.1:3306`.

## 3. Connect the app

If the app can reach your Cloud SQL instance, it will:

- Use **default user** `jas@gmail.com` (created automatically if missing).
- Let you **book appointments** by choosing a date and a time slot for a doctor.
- Show **My Appointments** in the sidebar when Cloud SQL is enabled.

Add to your **`.env`** (use your MySQL username and password from the instance):

```env
# Google Cloud SQL (MySQL) – doctor appointments
CLOUD_SQL_HOST=34.11.152.7
CLOUD_SQL_PORT=3306
CLOUD_SQL_USER=your_mysql_username
CLOUD_SQL_PASSWORD=your_mysql_password
CLOUD_SQL_DATABASE=doctor_appointments_db
```

- **CLOUD_SQL_USER** / **CLOUD_SQL_PASSWORD**: the MySQL user you created for this instance (e.g. in Cloud Console → instance → Users, or when setting up the instance).
- For **Cloud SQL Auth Proxy**: set `CLOUD_SQL_HOST=127.0.0.1` and run the proxy so it forwards to `34.11.152.7:3306`.

**Important:** The MySQL user must have **INSERT** and **SELECT** (and preferably **UPDATE**) on `doctor_appointments_db` and its tables. If the user has only SELECT, the connection test may succeed but booking will fail with an error like `(1044, 'Access denied...')`. In Cloud SQL Studio (as a privileged user) run:

```sql
GRANT SELECT, INSERT, UPDATE ON doctor_appointments_db.* TO 'your_mysql_username'@'%';
FLUSH PRIVILEGES;
```

If these variables are not set or the app cannot connect, booking falls back to the existing behavior (e.g. TimescaleDB only) and slot booking is skipped.

### If you see "Connection failed" in the sidebar

The app now shows the **real error** from MySQL (e.g. "Access denied", "Unknown database", "timed out"). Use it to fix:

- **Access denied for user** → Wrong username or password. In **Cloud Console → SQL → doctor-appointments → Users**, create or edit the MySQL user and set the password; use that exact username and password in `.env` as `CLOUD_SQL_USER` and `CLOUD_SQL_PASSWORD`. Avoid special characters in the password or quote the value in `.env`.
- **Unknown database** → The database in `CLOUD_SQL_DATABASE` does not exist. In Cloud SQL Studio run `CREATE DATABASE doctor_appointments_db;` then run `cloud_sql_schema.sql` with `USE doctor_appointments_db;` first.
- **Connection timed out** / **Can't connect** → Your IP may not be authorized yet, or a firewall is blocking outbound port 3306. In **Cloud Console → SQL → doctor-appointments → Connections → Authorized networks**, add your current public IP (e.g. 73.223.52.168). Wait a minute and try again.

## 4. App behavior

- **Doctor ID**: Each doctor from your app is mapped to a stable `doctor_id` in 1–200 (by name hash). The same doctor name always gets the same ID.
- **User**: One default user `jas@gmail.com`; the app gets/creates this user in `users` when booking.
- **Slots**: Only 9:00–16:30 in 30-minute steps. The UI shows only slots that are still free for the chosen doctor and date.
- **Booking**: Choosing date + time and confirming creates one row in `appointments` and marks that (doctor, date, slot) as booked.

## 5. Dependencies

Install the MySQL driver:

```bash
pip install pymysql>=1.1.0
```

(Already listed in `requirements.txt`.)

## 6. Same database as the app

The app uses the database in `CLOUD_SQL_DATABASE`. In Cloud SQL Studio, run `SELECT DATABASE();` and use that same name in `.env`. If you created tables in a different database, the sidebar will show a schema error and slot booking will not work. Set `CLOUD_SQL_DATABASE` to the database where `doctors`, `users`, and `appointments` exist.

## 7. Logs (when booking doesn’t create rows)

The app writes all Cloud SQL activity to **`cloud_sql_appointments.log`** in the project root and to the Streamlit terminal. After clicking **Confirm booking**, check the terminal and the log file for lines like `book_appointment called: …`, `executing INSERT …`, `commit done, lastrowid=…`, or any exception. The full error (e.g. Access denied, Duplicate entry) will be logged.

## 8. Quick check

- Run the schema in Cloud SQL Studio and confirm `doctors`, `users`, and `appointments` exist and `users` has `jas@gmail.com`.
- Set `.env`, restart the app, search for a doctor, open “Book slot with …”, pick a date and time, and confirm booking. Check “My Appointments” in the sidebar.
