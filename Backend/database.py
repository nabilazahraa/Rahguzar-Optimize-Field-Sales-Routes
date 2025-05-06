import psycopg2
import logging

logger = logging.getLogger("Database")

def connect_to_database(host, database, user, password, port):
    """
    Create a connection to the PostgreSQL database.
    """
    try:
        connection = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None
