from flask import Blueprint, request, jsonify
import psycopg2
import logging
from Db_Setup import connect_to_database

update_status_bp = Blueprint('update_status', __name__)
logger = logging.getLogger('UpdateStatus')

@update_status_bp.route('/update_status', methods=['PATCH'])
@update_status_bp.route('/update_status', methods=['PATCH'])
def update_status():
    """
    API endpoint to update the status of a store (1 = Active, 0 = Inactive).
    """
    try:
        data = request.get_json()
        print("Received Data:", data)  # Debugging: Print received data

        storecode = data.get('storecode')
        status = data.get('status')

        if storecode is None or status not in [0, 1]:  # Ensure status is either 0 or 1
            return jsonify({"error": "Invalid request. storeid and status (0 or 1) are required"}), 400

        connection = connect_to_database()        
        if not connection:
            return jsonify({"error": "Database connection failed"}), 500

        cursor = connection.cursor()

        # Update the status in the database
        query = "UPDATE store_hierarchy SET status = %s WHERE storecode = %s"
        cursor.execute(query, (status, storecode))
        connection.commit()

        return jsonify({
            "success": True,
            "message": f"Store {storecode} status updated to {'Active' if status == 1 else 'Inactive'}",
            "updated_status": status
        }), 200
    except Exception as e:
        logger.error(f"Error updating store status: {e}")
        return jsonify({"error": "Database update failed"}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()
