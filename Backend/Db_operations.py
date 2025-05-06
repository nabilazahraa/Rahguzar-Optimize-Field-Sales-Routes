from Db_Setup import connect_to_database

def fetch_data(query, params=None):
    connection = connect_to_database()  # No need to pass host, user, etc.
    if not connection:
        return None
    try:
        cursor = connection.cursor()
        cursor.execute(query, params)
        data = cursor.fetchall()
        return data
    except Exception as e:
        print(f"Error executing query: {e}")
        return None
    finally:
        if connection:
            cursor.close()
            connection.close()
