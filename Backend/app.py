
import traceback
import psycopg2
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from FinalPJPgenerator import PJPOptimizer, validate_parameters
from Db_operations import fetch_data
from stores import store_bp  # Import the store blueprint
from update_status import update_status_bp  # Import the update_status blueprint
import pandas as pd
import matplotlib
matplotlib.use('Agg')
# bcrypt
import bcrypt
from Db_Setup import connect_to_database

import metrics_copy
import os
import json
import time

# Add necessary imports
from datetime import datetime, timedelta
from flask import jsonify

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PJPOptimizer')

PLAN_ID = None

# Global variables to track distributor state
current_distributor = None  # Stores the currently active distributor ID
changed_distributor = False  # Flag indicating if distributor has changed


def convert_to_standard_types(obj):
    """
    Recursively convert numpy types and other non-standard types in the object to standard Python types.
    """
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, np.integer):
                new_key = int(k)
            elif isinstance(k, np.floating):
                new_key = float(k)
            elif isinstance(k, np.bool_):
                new_key = bool(k)
            elif isinstance(k, bytes):
                new_key = k.decode('utf-8')
            else:
                new_key = k
            new_dict[new_key] = convert_to_standard_types(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_to_standard_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    else:
        return obj


@app.route('/optimize', methods=['POST'])
def optimize():

    global changed_distributor

    data = request.get_json()

    distributor_id = data.get('distributor_id')
    num_orderbookers = data.get('num_orderbookers')
    plan_duration = data.get('plan_duration')  # 'day' or 'custom'
    store_type = data.get('store_type', 'both')
    holidays = data.get('holidays', [])
    wholesale_time = data.get('wholesale_time',40)
    retail_time = data.get('retail_time',20)

    # New fields for custom approach
    custom_days = data.get('custom_days')  # user input number of days if plan_duration == 'custom'
    replicate = data.get('replicate', False)  # boolean
    working_hours_per_day = data.get('working_hours_per_day', 8)  # Default to 8 if not provided
    selected_stores = data.get('selected_stores', None)  # List of selected stores
    start_time = time.time()  # Timer starts here

    # Parameter Validation
    if not isinstance(distributor_id, int):
        return jsonify({"status": "error", "message": "Invalid distributor_id. It must be an integer."}), 400
    if distributor_id <= 0:
        return jsonify({"status": "error", "message": "distributor_id must be a positive integer."}), 400

    if not isinstance(num_orderbookers, int):
        return jsonify({"status": "error", "message": "Invalid num_orderbookers. It must be an integer."}), 400
    if num_orderbookers < 0:
        return jsonify({"status": "error", "message": "num_orderbookers cannot be negative."}), 400

    # Only valid plan_duration is 'day' or 'custom'
    if plan_duration not in ['day', 'custom']:
        return jsonify({"status": "error", "message": "plan_duration must be 'day' or 'custom'."}), 400

    if store_type not in ['both', 'wholesale', 'retail']:
        return jsonify({"status": "error", "message": "store_type must be 'both', 'wholesale', or 'retail'."}), 400

    if not isinstance(wholesale_time, int) or wholesale_time <= 0:
        return jsonify({"status": "error", "message": "wholesale_time must be a positive integer."}), 400

    if not isinstance(retail_time, int) or retail_time <= 0:
        return jsonify({"status": "error", "message": "retail_time must be a positive integer."}), 400

    if not isinstance(working_hours_per_day, (int, float)) or working_hours_per_day <= 0 or working_hours_per_day > 24:
        return jsonify({"status": "error", "message": "working_hours_per_day must be a number between 1 and 24."}), 400

    # If plan_duration == 'custom', we need a valid custom_days
    if plan_duration == 'custom':
        if not isinstance(custom_days, int) or custom_days <= 0:
            return jsonify({"status": "error", "message": "custom_days must be a positive integer when plan_duration='custom'"}), 400
    
    try:
        # Build the PJPOptimizer
        optimizer = PJPOptimizer(
            distributor_id=distributor_id,
            plan_duration=plan_duration,
            num_orderbookers=num_orderbookers,
            store_type=store_type,
            retail_time=retail_time,
            wholesale_time=wholesale_time,
            holidays=holidays,
            custom_days=custom_days if plan_duration == 'custom' else None,
            replicate=replicate,
            working_hours_per_day=working_hours_per_day,  
            selected_stores= selected_stores 
        )


        distance_matrix, store_to_index, time_matrix = optimizer.get_distance_data() # send to metrics.py
        # optimizer.save_distance_matrix(distance_matrix)

        store_count = len(optimizer.store_data)
        if store_count == 0:
            return jsonify({"status": "error", "message": "No stores available for the given distributor."}), 400

        validate_parameters(num_orderbookers, store_count)

        # Generate PJP based on plan_duration
        pjp, plan_id = optimizer.generate_pjp()
        clusters = optimizer.get_clusters_data()
        response_data = {
            "status": "success",
            "pjp": convert_to_standard_types(pjp),
            "clusters": convert_to_standard_types(clusters),
            "available_days": optimizer.available_days,
            "plan_id": plan_id,

        }

        PLAN_ID = plan_id
        # Ensure plan_id is an integer (assuming generate_pjp returns it as such)
        distributorid = int(distributor_id)

        # End measuring response time
        end_time = time.time()  # Timer stops here
        response_time = round((end_time - start_time)/60, 2)  # Response time in minutes


        # Generate metrics JSON files
        metrics_copy.call_functions(distance_matrix, time_matrix, store_to_index, plan_id, distributorid, response_time, plan_duration, num_orderbookers)
     

        changed_distributor = True
        

        return jsonify(response_data), 200

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Error: {e}\n{traceback_str}")
        return jsonify({"status": "error", "message": "An internal error occurred."}), 500


@app.route('/get_orderbookers', methods=['GET'])
def get_orderbookers():
    try:
        distributor_id = request.args.get('distributor_id', type=int)
        if distributor_id is None or distributor_id <= 0:
            return jsonify({"status": "error", "message": "Distributor ID must be a positive integer."}), 400

        connection = connect_to_database()
        if not connection:
            return jsonify({"status": "error", "message": "Database connection failed."}), 500

        try:
            cursor = connection.cursor()
            query = """
                SELECT COUNT(DISTINCT appuser_ref) AS number_of_app_users
                FROM public.universe_stores
                WHERE distributor_ref = %s;
            """
            cursor.execute(query, (distributor_id,))
            result = cursor.fetchone()

            if result:
                number_of_orderbookers = result[0]
                return jsonify({"status": "success", "number_of_orderbookers": number_of_orderbookers}), 200
            else:
                return jsonify({"status": "success", "number_of_orderbookers": 0}), 200

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            traceback_str = traceback.format_exc()
            logger.error(traceback_str)
            return jsonify({"status": "error", "message": "Error fetching data from database."}), 500
        finally:
            cursor.close()
            connection.close()

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback_str = traceback.format_exc()
        logger.error(traceback_str)
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500


@app.route('/reroute-clusters', methods=['POST'])
def reroute_clusters():
    """
    Accepts modified store-level data (with possible reassignments).
    - If is_removed is True for a store, it is considered removed.
    - We unify old_cluster_id + cluster_id to determine the 'affected' clusters.
    - Then we re-run the scheduling for the affected clusters.
    """
    data = request.get_json()
    modified_clusters = data.get("modified_clusters", [])

    if not modified_clusters:
        return jsonify({"status": "error", "message": "No modified clusters provided."}), 400

    distributor_id = data.get("distributor_id")
    plan_duration = data.get("plan_duration")  # 'day' or 'custom'
    num_orderbookers = data.get("num_orderbookers")
    store_type = data.get("store_type", "both")
    retail_time = data.get("retail_time", 20)
    wholesale_time = data.get("wholesale_time", 40)
    holidays = data.get("holidays", [])

    custom_days = data.get('custom_days')
    replicate = data.get('replicate', False)

    try:
        existing_clusters_df = pd.DataFrame(modified_clusters)

        # Ensure the data contains 'storeid' and 'cluster_id' columns
        if "storeid" not in existing_clusters_df.columns or "cluster_id" not in existing_clusters_df.columns:
            return jsonify({"status": "error", "message": "Modified data must include 'storeid' and 'cluster_id'"}), 400

        # Identify removed stores based on the is_removed flag (if present)
        if "is_removed" in existing_clusters_df.columns:
            removed_store_ids = set(existing_clusters_df[existing_clusters_df["is_removed"] == True]["storeid"])
            # Keep only non-removed stores for further processing
            existing_clusters_df = existing_clusters_df[existing_clusters_df["is_removed"] != True]
        else:
            removed_store_ids = set()

        # Figure out the set of new cluster IDs
        new_cluster_ids = set(existing_clusters_df["cluster_id"].dropna().unique().tolist())

        # If there's an old_cluster_id column, include those too
        affected_cluster_ids = set(new_cluster_ids)
        if "old_cluster_id" in existing_clusters_df.columns:
            old_cids = existing_clusters_df["old_cluster_id"].dropna().unique().tolist()
            for cid in old_cids:
                affected_cluster_ids.add(cid)
        affected_cluster_ids = list(affected_cluster_ids)

        # Pass these to our PJPOptimizer with existing_clusters
        optimizer = PJPOptimizer(
            distributor_id=distributor_id,
            plan_duration=plan_duration,
            num_orderbookers=num_orderbookers,
            store_type=store_type,
            retail_time=retail_time,
            wholesale_time=wholesale_time,
            holidays=holidays,
            existing_clusters=existing_clusters_df,
            custom_days=custom_days if plan_duration == 'custom' else None,
            replicate=replicate
        )

        # Perform the partial re-route
        response = optimizer.reroute(affected_cluster_ids, removed_store_ids)

        # Return updated schedule + cluster info, ensuring removed stores are not included
        filtered_clusters = []
        for c in optimizer.get_clusters_data():
            new_stores = [st for st in c["stores"] if st["storeid"] not in removed_store_ids]
            c2 = {**c, "stores": new_stores}
            filtered_clusters.append(c2)

        final_schedule = response["schedule"]

        return jsonify({
            "status": "success",
            "pjp": convert_to_standard_types(final_schedule),
            "clusters": convert_to_standard_types(filtered_clusters),
            "available_days": optimizer.available_days
        }), 200

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Reroute error: {e}\n{traceback_str}")
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/login', methods=['POST'])
def login():
    global current_distributor, changed_distributor
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"status": "error", "message": "Username and password required"}), 400

    connection = connect_to_database()
    if not connection:
        return jsonify({"status": "error", "message": "Database connection failed"}), 500

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT distributor_id, username, password_hash FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        if not user:
            return jsonify({"status": "error", "message": "Invalid credentials"}), 401

        distributor_id, db_username, db_password = user
        if bcrypt.checkpw(password.encode(), db_password.encode()):
            # Check if distributor has changed
            if current_distributor != distributor_id:
                changed_distributor = True
                current_distributor = distributor_id
            else:
                changed_distributor = False
            return jsonify({"status": "success", "distributor_id": distributor_id}), 200
        else:
            return jsonify({"status": "error", "message": "Invalid credentials"}), 401
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500
    finally:
        cursor.close()
        connection.close()



@app.route('/get_pjp_plans', methods=['GET'])
def get_pjp_plans():
    """
    Fetches all plan IDs for a given distributor ID from the database.
    Returns a list with the IDs.
    """
    try:
        distributor_id = request.args.get('distributor_id', type=int)
        logger.info(f"Received distributor_id: {distributor_id}")

        if distributor_id is None:
            logger.error("Distributor ID is missing.")
            return jsonify({"status": "error", "message": "Distributor ID is required."}), 400

        # Connect to the database
        connection = connect_to_database()
        cursor = connection.cursor()

        # Query to fetch plans for the given distributor_id
        query = """
        SELECT pm.plan_id
        FROM plan_master pm
        WHERE pm.distributor_id = %s;
        """
        
        cursor.execute(query, (distributor_id,))
        rows = cursor.fetchall()
        logger.info(f"Fetched rows: {rows}")

        if not rows:
            logger.info(f"No plans found for distributor ID {distributor_id}")
            return jsonify({"status": "success", "plans": []}), 200

        # Convert the result to the expected format
        plans = [{"id": row[0], "name": f"{row[0]}"} for row in rows]
        logger.info(f"Formatted plans: {plans}")

        return jsonify({"status": "success", "plans": plans}), 200

    except Exception as e:
        logger.error(f"Error fetching plans: {e}")
        traceback_str = traceback.format_exc()
        logger.error(traceback_str)
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500
    
    finally:
        if connection:
            connection.close()

# @app.route('/get_orderbookers_for_distributor', methods=['GET'])
# def get_orderbookers_for_distributor():
#     try:
#         distributor_id = request.args.get('distributor_id', type=int)
#         if distributor_id is None or distributor_id <= 0:
#             return jsonify({"status": "error", "message": "Distributor ID must be a positive integer."}), 400

#         connection = connect_to_database()
#         if not connection:
#             return jsonify({"status": "error", "message": "Database connection failed."}), 500

#         try:
#             cursor = connection.cursor()
#             query = """
#                 SELECT DISTINCT ob_id AS orderbooker_id
#                 FROM public.orderbooker_kpi_history
#                 WHERE distributorid = %s;
#             """
#             cursor.execute(query, (distributor_id,))
#             result = cursor.fetchall()

#             if result:
#                 orderbookers = [{"id": row[0], "name": f"Orderbooker {row[0]}"} for row in result]
#                 return jsonify({"status": "success", "orderbookers": orderbookers}), 200
#             else:
#                 return jsonify({"status": "success", "orderbookers": []}), 200

#         except Exception as e:
#             logger.error(f"Error executing query in get_orderbookers_for_distributor: {e}")
#             traceback_str = traceback.format_exc()
#             logger.error(traceback_str)
#             return jsonify({"status": "error", "message": "Error fetching order bookers."}), 500
#         finally:
#             cursor.close()
#             connection.close()

#     except Exception as e:
#         logger.error(f"Unexpected error in get_orderbookers_for_distributor: {e}")
#         traceback_str = traceback.format_exc()
#         logger.error(traceback_str)
#         return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500
@app.route('/get_orderbookers_for_plan', methods=['GET'])
def get_orderbookers_for_plan():
    try:
        distributor_id = request.args.get('distributor_id', type=int)
        pjp_id = request.args.get('pjp_id', type=str)

        if not distributor_id or not pjp_id:
            return jsonify({"status": "error", "message": "Distributor ID and Plan ID are required."}), 400

        connection = connect_to_database()
        if not connection:
            return jsonify({"status": "error", "message": "Database connection failed."}), 500

        cursor = connection.cursor()
        query = """
            SELECT DISTINCT ob_id as orderbooker_id
            FROM orderbooker_kpi_history
            WHERE distributorid = %s AND plan_id = %s
        """
        cursor.execute(query, (distributor_id, pjp_id))
        result = cursor.fetchall()

        orderbookers = [{"id": -1, "name": "All order bookers"}]
        orderbookers += [{"id": row[0], "name": f"Orderbooker {row[0]}"} for row in result]

        return jsonify({"status": "success", "orderbookers": orderbookers}), 200

    except Exception as e:
        logger.error(f"Error in get_orderbookers_for_plan: {e}")
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500


@app.route('/get_kpi_data', methods=['GET'])
def get_kpi_data():
    """Fetch KPI data directly from the database."""
    distributor_id = request.args.get('distributor_id', type=int)
    orderbooker_id = request.args.get('orderbooker_id', type=int, default=None)
    pjp_id = request.args.get('pjp_id', type=str, default=None)

    if not distributor_id:
        return jsonify({"status": "error", "message": "distributor_id is required"}), 400
    if not pjp_id:
        return jsonify({"status": "error", "message": "pjp_id is required"}), 400

    # Log the received values for debugging
    logger.info(f"Raw pjp_id value: {request.args.get('pjp_id')}")
    logger.info(f"Received get_kpi_data: distributor_id={distributor_id}, orderbooker_id={orderbooker_id}, pjp_id={pjp_id}")

    connection = connect_to_database()
    if not connection:
        return jsonify({"status": "error", "message": "Database connection failed"}), 500

    try:
        cursor = connection.cursor()

        # Check distributor existence
        cursor.execute("SELECT 1 FROM users WHERE distributor_id = %s", (distributor_id,))
        if not cursor.fetchone():
            return jsonify({"status": "error", "message": f"Distributor ID {distributor_id} not found"}), 404

        # Check if pjp_id exists in plan_master
        cursor.execute("SELECT plan_duration FROM plan_master WHERE plan_id = %s AND distributor_id = %s", (pjp_id, distributor_id))
        plan_result = cursor.fetchone()
        if not plan_result:
            kpi_data = {
                "daily": 0,
                "total_distance_travelled": 0, "avg_distance_travelled": 0,
                "total_shops_visited": 0, "avg_shops_visited": 0,
                "running_time": 0, "total_workload": 0, "avg_workload": 0, "response_time": 0
            }
            return jsonify({"status": "success", "data": kpi_data}), 200

        daily_value = 1 if plan_result[0] == 1 else 0

        # Fetch KPI data
        kpi_fields = "total_distance_travelled, avg_distance_travelled, total_shops_visited, avg_shops_visited, running_time, total_workload, avg_workload, response_time"
        if orderbooker_id is None or orderbooker_id == -1:
            kpi_query = f"""
                SELECT {kpi_fields}
                FROM distributor_kpi_history
                WHERE distributorid = %s AND plan_id = %s
            """
            kpi_params = (distributor_id, pjp_id)
        else:
            kpi_query = f"""
                SELECT {kpi_fields}
                FROM orderbooker_kpi_history
                WHERE distributorid = %s AND ob_id = %s AND plan_id = %s
            """
            kpi_params = (distributor_id, orderbooker_id, pjp_id)

        cursor.execute(kpi_query, kpi_params)
        kpi_result = cursor.fetchone()

        if not kpi_result:
            kpi_data = {
                "daily": daily_value,
                "total_distance_travelled": 0, "avg_distance_travelled": 0,
                "total_shops_visited": 0, "avg_shops_visited": 0,
                "running_time": 0, "total_workload": 0, "avg_workload": 0, "response_time":0
            }
        else:
            total_distance, avg_distance, total_shops, avg_shops, running_time, total_workload, avg_workload, response_time = kpi_result
            kpi_data = {
                "daily": daily_value,
                "total_distance_travelled": float(total_distance) if total_distance else 0,
                "avg_distance_travelled": float(avg_distance) if avg_distance else 0,
                "total_shops_visited": int(total_shops) if total_shops else 0,
                "avg_shops_visited": float(avg_shops) if avg_shops else 0,
                "running_time": float(running_time) if running_time else 0,
                "total_workload": float(total_workload) if total_workload else 0,
                "avg_workload": float(avg_workload) if avg_workload else 0,
                "response_time": float(response_time) if response_time else 0
            }

        return jsonify({"status": "success", "data": kpi_data}), 200

    except Exception as e:
        logger.error(f"Error fetching KPI data: {e}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500
    finally:
        cursor.close()
        connection.close()

@app.route('/get_graph_data', methods=['GET'])
def get_graph_data():
    """Fetch graph data (distance, visits, workload) for the last 30 days."""
    distributor_id = request.args.get('distributor_id', type=int)
    orderbooker_id = request.args.get('orderbooker_id', type=int, default=None)
    pjp_id = request.args.get('pjp_id', type=str, default=None)

    if not distributor_id:
        return jsonify({"status": "error", "message": "distributor_id is required"}), 400
    if not pjp_id:
        return jsonify({"status": "error", "message": "pjp_id is required"}), 400

    # Log the received values for debugging
    logger.info(f"Received get_graph_data: distributor_id={distributor_id}, orderbooker_id={orderbooker_id}, pjp_id={pjp_id}")

    connection = connect_to_database()
    if not connection:
        return jsonify({"status": "error", "message": "Database connection failed"}), 500

    try:
        cursor = connection.cursor()

        # Check if pjp_id exists
        cursor.execute("SELECT 1 FROM plan_master WHERE plan_id = %s AND distributor_id = %s", (pjp_id, distributor_id))
        if not cursor.fetchone():
            empty_data = [{"date": datetime.now().strftime("%m/%d/%Y"), "value": 0}]
            return jsonify({
                "status": "success",
                "data": {
                    "total_distance": empty_data,
                    "store_visited": empty_data,
                    "workload": empty_data
                }
            }), 200

        # Determine latest date and 30-day window
        if orderbooker_id is None or orderbooker_id == -1:
            date_query = """
                SELECT MAX(metric_date)
                FROM graph_total_ob_metrics
                WHERE plan_id = %s
            """
            date_params = (pjp_id,)
        else:
            date_query = """
                SELECT MAX(metric_date)
                FROM graph_unique_ob_metrics
                WHERE plan_id = %s AND orderbooker_id = %s
            """
            date_params = (pjp_id, orderbooker_id)

        cursor.execute(date_query, date_params)
        latest_date = cursor.fetchone()[0]
        if not latest_date:
            empty_data = [{"date": datetime.now().strftime("%m/%d/%Y"), "value": 0}]
            return jsonify({
                "status": "success",
                "data": {
                    "total_distance": empty_data,
                    "store_visited": empty_data,
                    "workload": empty_data
                }
            }), 200

        start_date = latest_date - timedelta(days=29)

        # Fetch graph data
        if orderbooker_id is None or orderbooker_id == -1:
            graph_query = """
                SELECT metric_date, total_distance, total_visits, total_workload
                FROM graph_total_ob_metrics
                WHERE plan_id = %s AND metric_date >= %s
                ORDER BY metric_date ASC
            """
            graph_params = (pjp_id, start_date)
        else:
            graph_query = """
                SELECT metric_date, distance, visits, workload
                FROM graph_unique_ob_metrics
                WHERE plan_id = %s AND orderbooker_id = %s AND metric_date >= %s
                ORDER BY metric_date ASC
            """
            graph_params = (pjp_id, orderbooker_id, start_date)

        cursor.execute(graph_query, graph_params)
        graph_results = cursor.fetchall()

        total_distance_data = []
        store_visited_data = []
        workload_data = []

        for row in graph_results:
            metric_date = row[0].strftime("%m/%d/%Y")
            distance = float(row[1]) if row[1] is not None else 0
            visits = int(row[2]) if row[2] is not None else 0
            workload = float(row[3]) if row[3] is not None else 0

            total_distance_data.append({"date": metric_date, "value": distance})
            store_visited_data.append({"date": metric_date, "value": visits})
            workload_data.append({"date": metric_date, "value": workload})

        return jsonify({
            "status": "success",
            "data": {
                "total_distance": total_distance_data,
                "store_visited": store_visited_data,
                "workload": workload_data
            }
        }), 200

    except Exception as e:
        logger.error(f"Error fetching graph data: {e}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500
    finally:
        cursor.close()
        connection.close()
        
@app.route('/check_distributor_change', methods=['GET'])
def check_distributor_change():
    global changed_distributor
    try:
        distributor_id = request.args.get('distributor_id', type=int)
        if distributor_id is None:
            return jsonify({"status": "error", "message": "distributor_id is required"}), 400

        logger.info(f"Checking distributor change: current={current_distributor}, requested={distributor_id}")
        
        # Prepare response and reset changed_distributor after reporting
        response = {
            "status": "success",
            "distributor_changed": changed_distributor,
            "distributor_id": distributor_id
        }
        # Reset the flag after it's checked, unless the distributor actually differs
        if distributor_id == current_distributor:
            changed_distributor = False
        else:
            changed_distributor = True  # Set to true if different, will be reset on next check
        
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error in check_distributor_change: {e}")
        traceback_str = traceback.format_exc()
        logger.error(traceback_str)
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500

# Reset changed_distributor after dashboard action
@app.route('/apply_dashboard_action', methods=['POST'])
def apply_dashboard_action():
    global changed_distributor
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        distributor_id = data.get('distributor_id')
        orderbooker_id = data.get('orderbooker_id')
        pjp_id = data.get('pjp_id')

        if not distributor_id:
            return jsonify({"status": "error", "message": "distributor_id is required"}), 400

        logger.info(f"Applying dashboard action: distributor_id={distributor_id}, orderbooker_id={orderbooker_id}, pjp_id={pjp_id}")
        
        # Reset the changed flag after applying
        changed_distributor = False
        
        return jsonify({"status": "success", "message": "Action applied successfully"}), 200
    except Exception as e:
        logger.error(f"Error in apply_dashboard_action: {e}")
        traceback_str = traceback.format_exc()
        logger.error(traceback_str)
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500

# Register Blueprints
app.register_blueprint(store_bp, url_prefix='/api')
app.register_blueprint(update_status_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
