from flask import Blueprint, jsonify, request
from Db_Setup import connect_to_database
import traceback
import logging

store_bp = Blueprint('store', __name__)
logger = logging.getLogger('StoreRoutes')

@store_bp.route('/get_stores', methods=['GET'])
def get_stores():
    """
    Fetch store details filtered by distributor ID.
    """
    try:
        distributor_id = request.args.get('distributorid', type=int)
        if distributor_id is None:
            return jsonify({"status": "error", "message": "distributorid is required."}), 400
        logger.info(f"Received distributor_id: {distributor_id}")

        connection = connect_to_database()
        if not connection:
            return jsonify({"status": "error", "message": "Database connection failed."}), 500

        try:
            cursor = connection.cursor()

            # SQL query with distributorid filter
            query = """
                SELECT 
                    sl.storeid,
                    sh.storecode,
                    sl.latitude,
                    sl.longitude,
                    sc.channeltypeid,
                    sc.subchannelid,
                    sl.areatype,
                    sl.townid,
                    sl.localityid,
                    sl.sublocalityid,
                    scl.storeclassificationoneid,
                    scl.storeclassificationtwoid,
                    scl.storeclassificationthreeid,
                    sh.status,
                    sh.storeperfectid,
                    sh.storefilertype
                FROM 
                    distributor_stores ds
                JOIN 
                    store_location sl ON ds.storeid = sl.storeid
                JOIN 
                    store_channel sc ON sl.storeid = sc.storeid
                JOIN 
                    store_classification scl ON sl.storeid = scl.storeid
                JOIN 
                    store_hierarchy sh ON sl.storeid = sh.storeid
                WHERE 
                    ds.distributorid = %s
            """
            cursor.execute(query, (distributor_id,))
            result = cursor.fetchall()

            # Convert result to list of dictionaries
            stores = [
                {
                    "storeid": row[0],
                    "storecode": row[1],
                    "latitude": row[2],
                    "longitude": row[3],
                    "channeltypeid": row[4],
                    "subchannelid": row[5],
                    "areatype": row[6],
                    "townid": row[7],
                    "localityid": row[8],
                    "sublocalityid": row[9],
                    "classificationoneid": row[10],
                    "classificationtwoid": row[11],
                    "classificationthreeid": row[12],
                    "status": row[13],
                    "storeperfectid": row[14],
                    "storefilertype": row[15],
                }
                for row in result
            ]

            return jsonify({"status": "success", "stores": stores}), 200
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            traceback_str = traceback.format_exc()
            logger.error(traceback_str)
            return jsonify({"status": "error", "message": "Error fetching data from the database."}), 500
        finally:
            cursor.close()
            connection.close()

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback_str = traceback.format_exc()
        logger.error(traceback_str)
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500

@store_bp.route('/get_filtered_stores', methods=['GET'])
def get_filtered_stores():
    """
    Fetch filtered store details based on distributor ID.
    Uses distributor_stores, store_channel, and store_hierarchy tables.
    Only includes stores with active status (status = 1).
    """
    try:
        distributor_id = request.args.get('distributorid', type=int)
        if distributor_id is None:
            return jsonify({"status": "error", "message": "distributorid is required."}), 400
        logger.info(f"Received distributor_id for filtered query: {distributor_id}")

        connection = connect_to_database()
        if not connection:
            return jsonify({"status": "error", "message": "Database connection failed."}), 500

        try:
            cursor = connection.cursor()

            query = """
                SELECT
                    ds.storeid,
                    ds.latitude,
                    ds.longitude,
                    sc.channeltypeid,
                    sh.storecode
                FROM distributor_stores ds
                JOIN store_channel sc ON ds.storeid = sc.storeid
                JOIN store_hierarchy sh ON ds.storeid = sh.storeid
                WHERE ds.distributorid = %s
                AND sh.status = 1
                AND sc.channeltypeid IN (1, 2)
            """
            cursor.execute(query, (distributor_id,))
            result = cursor.fetchall()

            stores = [
                {
                    "storeid": row[0],
                    "latitude": row[1],
                    "longitude": row[2],
                    "channeltypeid": row[3],
                    "storecode": row[4]
                }
                for row in result
            ]

            return jsonify({"status": "success", "stores": stores}), 200

        except Exception as e:
            logger.error(f"Error executing filtered query: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"status": "error", "message": "Error fetching filtered data from the database."}), 500
        finally:
            cursor.close()
            connection.close()

    except Exception as e:
        logger.error(f"Unexpected error in get_filtered_stores: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500

@store_bp.route('/get_raw_stores', methods=['GET'])
def get_raw_stores():
    """
    Fetch raw store data for a given distributor without applying filters on channeltypeid or lat/long.
    """
    try:
        distributor_id = request.args.get('distributorid', type=int)
        if distributor_id is None:
            return jsonify({"status": "error", "message": "distributorid is required."}), 400
        logger.info(f"Fetching raw store data for distributor_id {distributor_id}...")

        connection = connect_to_database()
        if not connection:
            return jsonify({"status": "error", "message": "Database connection failed."}), 500

        try:
            cursor = connection.cursor()

            query = """
                SELECT
                    ds.storeid,
                    ds.latitude,
                    ds.longitude,
                    sc.channeltypeid,
                    sh.storecode,
                    sh.status
                FROM distributor_stores ds
                JOIN store_channel sc ON ds.storeid = sc.storeid
                JOIN store_hierarchy sh ON ds.storeid = sh.storeid
                WHERE ds.distributorid = %s
                AND sh.status = 1
            """

            cursor.execute(query, (distributor_id,))
            rows = cursor.fetchall()

            stores = [
                {
                    "storeid": row[0],
                    "latitude": row[1],
                    "longitude": row[2],
                    "channeltypeid": row[3],
                    "storecode": row[4],
                    "status": row[5]
                }
                for row in rows
            ]

            logger.info(f"Fetched {len(stores)} raw stores for distributor_id {distributor_id}")
            return jsonify({"status": "success", "stores": stores}), 200

        except Exception as e:
            logger.error(f"Query error in get_raw_stores: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"status": "error", "message": "Error querying store data."}), 500
        finally:
            cursor.close()
            connection.close()

    except Exception as e:
        logger.error(f"Unexpected error in get_raw_stores: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500

