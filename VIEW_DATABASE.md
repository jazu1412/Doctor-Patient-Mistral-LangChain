# How to View TimescaleDB

## Connection Details

**Host:** `lowkayf0dw.u827f1jcvc.tsdb.cloud.timescale.com`  
**Port:** `35981`  
**Database:** `tsdb`  
**User:** `tsdbadmin`  
**Password:** `g9heeciuyqfflr2z`  
**SSL Mode:** `require`

## Method 1: TimescaleDB Cloud Console (Easiest)

1. Go to [TimescaleDB Cloud Console](https://console.cloud.timescale.com/)
2. Sign in with your account
3. Select your service
4. Use the **SQL Editor** or **Data Explorer** to view and query data

## Method 2: Command Line (psql)

Connect using psql:

```bash
psql "postgres://tsdbadmin:g9heeciuyqfflr2z@lowkayf0dw.u827f1jcvc.tsdb.cloud.timescale.com:35981/tsdb?sslmode=require"
```

Or with individual parameters:

```bash
psql -h lowkayf0dw.u827f1jcvc.tsdb.cloud.timescale.com \
     -p 35981 \
     -U tsdbadmin \
     -d tsdb \
     --set=sslmode=require
```

### Useful SQL Queries

Once connected, you can run:

```sql
-- View all doctors
SELECT * FROM doctors;

-- View only available doctors
SELECT * FROM doctors WHERE is_available = TRUE;

-- View only unavailable doctors
SELECT * FROM doctors WHERE is_available = FALSE;

-- Count doctors by availability
SELECT is_available, COUNT(*) FROM doctors GROUP BY is_available;

-- View doctors with their specialities
SELECT doctor_name, speciality, is_available, created_at, updated_at 
FROM doctors 
ORDER BY doctor_name;
```

## Method 3: GUI Database Tools

### Option A: pgAdmin
1. Download [pgAdmin](https://www.pgadmin.org/download/)
2. Add new server with these connection details
3. Navigate to `tsdb` database → `Schemas` → `public` → `Tables` → `doctors`

### Option B: DBeaver
1. Download [DBeaver](https://dbeaver.io/download/)
2. Create new PostgreSQL connection
3. Enter connection details:
   - Host: `lowkayf0dw.u827f1jcvc.tsdb.cloud.timescale.com`
   - Port: `35981`
   - Database: `tsdb`
   - Username: `tsdbadmin`
   - Password: `g9heeciuyqfflr2z`
   - SSL: Enable

### Option C: TablePlus
1. Download [TablePlus](https://tableplus.com/)
2. Create new PostgreSQL connection
3. Enter the connection details above

## Method 4: Through Streamlit App

1. Run the app: `streamlit run app.py`
2. Open sidebar
3. Click **"📊 View All Doctors"** button
4. View data in an interactive table

## Method 5: Python Script

You can also create a simple Python script to view the data:

```python
from database import sync_get_all_doctors
import pandas as pd

# Get all doctors
doctors = sync_get_all_doctors()

# Display as DataFrame
if doctors:
    df = pd.DataFrame(doctors)
    print(df.to_string())
else:
    print("No doctors in database")
```

## Database Schema

The `doctors` table structure:

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

## Quick Connection String

For quick reference, use this connection string in any PostgreSQL client:

```
postgres://tsdbadmin:g9heeciuyqfflr2z@lowkayf0dw.u827f1jcvc.tsdb.cloud.timescale.com:35981/tsdb?sslmode=require
```
