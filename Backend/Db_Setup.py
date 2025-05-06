
from config import DATABASE_CONFIG
import psycopg2

def connect_to_database():
    """
    Establishes a connection to the PostgreSQL database using centralized configuration.
    """
    try:
        connection = psycopg2.connect(**DATABASE_CONFIG)
        print("Database connection successful!")
        return connection
    except Exception as e:
        print("Error while connecting to database:", str(e))
        return None
