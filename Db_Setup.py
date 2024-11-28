import psycopg2
from psycopg2 import sql

def connect_to_database(host, database, user, password, port=5432):
    """
    Connects to the PostgreSQL database and returns the connection object.
    
    Args:
        host (str): The hostname or IP address of the database server.
        database (str): The name of the database.
        user (str): The username for authentication.
        password (str): The password for authentication.
        port (int): The port number (default is 5432).

    Returns:
        connection: A psycopg2 connection object.
    """
    try:
        connection = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
        print("Database connection successful!")
        return connection
    except Exception as e:
        print("Error while connecting to database:", str(e))
        return None
