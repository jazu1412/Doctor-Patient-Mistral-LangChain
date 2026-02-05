"""
Script to initialize the TimescaleDB database schema
Run this once to set up the database tables
"""
import asyncio
from database import init_database, close_pool

async def main():
    """Initialize database schema"""
    print("Initializing database schema...")
    try:
        await init_database()
        print("✅ Database schema initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing database: {str(e)}")
    finally:
        await close_pool()

if __name__ == "__main__":
    asyncio.run(main())


