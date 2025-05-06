import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from Db_Setup import connect_to_database

#----------------------------------------------------------------
# Helper: Get all plan IDs for a given distributor
#----------------------------------------------------------------
def get_distributor_plans(distributor_id):
    """
    Fetches all plan IDs for a given distributor ID.
    Returns a DataFrame with the plan IDs.
    """
    try:
        connection = connect_to_database()
        cursor = connection.cursor()
        query = """
            SELECT pm.plan_id
            FROM plan_master pm
            WHERE pm.distributor_id = %s;
        """
        cursor.execute(query, (distributor_id,))
        plans_df = pd.DataFrame(cursor.fetchall(), columns=['plan_id'])
        return plans_df
    except Exception as e:
        print(f"Error fetching plans: {e}")
    finally:
        if connection:
            connection.close()

#----------------------------------------------------------------
# Compute Distributor-Level KPIs and update in database
#----------------------------------------------------------------
def calculate_distributor_kpi(db_conn, plan_id, distance_matrix, time_matrix, store_to_index, plan_duration, num_orderbookers): 
    """
    Computes distributor-level KPIs:
      - Total and average number of shop visits per day or per orderbooker
      - Total and average distance travelled
      - Total and average travel time (including service time)
      - Running time (placeholder)
    Returns a dictionary with the metrics.
    """
    cursor = db_conn.cursor()
    
    # Query store visit data and plan days for the given plan_id
    query = """
        SELECT 
            pp.orderbooker_id,
            psa.store_id,
            pp.plan_date
        FROM 
            plan_master pm
        JOIN 
            pjp_plans pp ON pm.plan_id = pp.plan_id
        JOIN 
            pjp_store_assignments psa ON pp.pjp_id = psa.pjp_id
        WHERE 
            pm.plan_id = %s AND pm.status = TRUE;
    """
    cursor.execute(query, (plan_id,))
    data = cursor.fetchall()
    if not data:
        print(f"No data found for Plan ID: {plan_id}")
        cursor.close()
        return None

    df = pd.DataFrame(data, columns=['orderbooker_id', 'store_id', 'plan_date'])
    
    # num_days = df['plan_date'].nunique()
        # ✅ Fetch max num_days for this plan from plan_master_ob_days
    cursor.execute("""
        SELECT MAX(num_days) FROM plan_master_ob_days WHERE plan_id = %s;
    """, (plan_id,))
    result = cursor.fetchone()
    num_days = result[0] if result and result[0] else 0

    total_shops = len(df)

    if plan_duration == 'custom':
        avg_shops = total_shops / num_days if num_days > 0 else 0
    elif plan_duration == 'day':
        avg_shops = total_shops / num_orderbookers if num_orderbookers > 0 else 0

    # --- Total distance calculation ---
    cursor.execute("SELECT pjp_id FROM pjp_plans WHERE plan_id = %s;", (plan_id,))
    pjp_ids = [row[0] for row in cursor.fetchall()]
    total_distance = 0
    
    for pjp_id in pjp_ids:
        cursor.execute("""
            SELECT store_id 
            FROM pjp_store_assignments 
            WHERE pjp_id = %s 
            ORDER BY visit_sequence;
        """, (pjp_id,))
        stores = [r[0] for r in cursor.fetchall()]
        pjp_distance = 0
        for i in range(len(stores) - 1):
            store_a, store_b = stores[i], stores[i+1]
            if store_a in store_to_index and store_b in store_to_index:
                idx_a = store_to_index[store_a]
                idx_b = store_to_index[store_b]
                pjp_distance += distance_matrix[idx_a][idx_b]
        total_distance += pjp_distance
    
    if plan_duration == 'custom':
        avg_distance = total_distance / num_days if num_days > 0 else 0
    elif plan_duration == 'day':
        avg_distance = total_distance / num_orderbookers if num_orderbookers > 0 else 0

    # --- Total travel time calculation (including service time) ---
    total_travel_time = 0
    for pjp_id in pjp_ids:
        cursor.execute("""
            SELECT store_id 
            FROM pjp_store_assignments 
            WHERE pjp_id = %s 
            ORDER BY visit_sequence;
        """, (pjp_id,))
        stores = [r[0] for r in cursor.fetchall()]
        pjp_travel_time = 0
        for i in range(len(stores) - 1):
            store_a, store_b = stores[i], stores[i+1]
            if store_a in store_to_index and store_b in store_to_index:
                idx_a = store_to_index[store_a]
                idx_b = store_to_index[store_b]
                pjp_travel_time += time_matrix[idx_a][idx_b]
            # Add service time for the current store (store_a)
            cursor.execute("SELECT channeltypeid FROM store_channel WHERE storeid = %s;", (store_a,))
            result = cursor.fetchone()
            if result:
                if result[0] == 1:
                    pjp_travel_time += 20
                elif result[0] == 2:
                    pjp_travel_time += 40
        # Add service time for the last store
        if stores:
            cursor.execute("SELECT channeltypeid FROM store_channel WHERE storeid = %s;", (stores[-1],))
            result = cursor.fetchone()
            if result:
                if result[0] == 1:
                    pjp_travel_time += 20
                elif result[0] == 2:
                    pjp_travel_time += 40
        total_travel_time += pjp_travel_time
    
    if plan_duration == 'custom':
        avg_travel_time = total_travel_time / num_days if num_days > 0 else 0
    elif plan_duration == 'day':
        avg_travel_time = total_travel_time / num_orderbookers if num_orderbookers > 0 else 0

    running_time = 1  # Placeholder value

    metrics = {
        "total_shops_visited": total_shops,
        "avg_shops_visited": round(avg_shops, 2),
        "total_distance_travelled": round(total_distance, 2),
        "avg_distance_travelled": round(avg_distance, 2),
        "total_travel_time": round(total_travel_time, 2),
        "avg_travel_time": round(avg_travel_time, 2),
        "running_time": running_time
    }

    cursor.close()
    return metrics


def update_distributor_kpi_in_db(db_conn, distributorid, plan_id, metrics, response_time):
    """
    Inserts or updates the distributor-level KPI metrics directly into the database.
    """
    cursor = db_conn.cursor()
    query = """
    INSERT INTO distributor_kpi_history (
        distributorid, plan_id,
        total_distance_travelled, avg_distance_travelled,
        total_shops_visited, avg_shops_visited,
        running_time, total_workload, avg_workload, response_time
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (distributorid, plan_id)
    DO UPDATE SET
        total_distance_travelled = EXCLUDED.total_distance_travelled,
        avg_distance_travelled = EXCLUDED.avg_distance_travelled,
        total_shops_visited = EXCLUDED.total_shops_visited,
        avg_shops_visited = EXCLUDED.avg_shops_visited,
        running_time = EXCLUDED.running_time,
        total_workload = EXCLUDED.total_workload,
        avg_workload = EXCLUDED.avg_workload,
        response_time = EXCLUDED.response_time;
    """
    # Convert any potential NumPy types to native Python types
    import numpy as np
    params = (
        distributorid, 
        plan_id,
        float(metrics["total_distance_travelled"]),
        float(metrics["avg_distance_travelled"]),
        int(metrics["total_shops_visited"]),
        float(metrics["avg_shops_visited"]),
        int(metrics["running_time"]),
        float(metrics["total_travel_time"]),
        float(metrics["avg_travel_time"]),
        float(response_time)
    )
    cursor.execute(query, params)
    db_conn.commit()
    cursor.close()
    print(f"Distributor KPI updated for distributor {distributorid}, plan {plan_id}.")
#----------------------------------------------------------------
# Compute Orderbooker-Level KPIs and update in database
#----------------------------------------------------------------
# def calculate_orderbooker_kpi(db_conn, plan_id, distance_matrix, time_matrix, store_to_index):
#     """
#     Computes orderbooker-level KPIs:
#       - Total & average distance travelled and shop visits per day
#       - Total & average travel time (including service time)
#     Returns a dictionary mapping each orderbooker_id to its metrics.
#     """
#     cursor = db_conn.cursor()
#     query = """
#         SELECT 
#             pp.orderbooker_id, pp.pjp_id, psa.store_id, pp.plan_date
#         FROM 
#             pjp_plans pp
#         JOIN 
#             pjp_store_assignments psa ON pp.pjp_id = psa.pjp_id
#         WHERE 
#             pp.plan_id = %s;
#     """
#     cursor.execute(query, (plan_id,))
#     data = cursor.fetchall()
#     if not data:
#         print(f"No data found for Plan ID: {plan_id}")
#         cursor.close()
#         return {}
#     df = pd.DataFrame(data, columns=['orderbooker_id', 'pjp_id', 'store_id', 'plan_date'])
#     orderbooker_metrics = {}
#     grouped = df.groupby("orderbooker_id")
#     for orderbooker_id, group in grouped:
#         unique_pjps = group["pjp_id"].unique()
#         total_distance = 0
#         total_travel_time = 0
#         for pjp_id in unique_pjps:
#             stores = group[group["pjp_id"] == pjp_id]["store_id"].tolist()
#             pjp_distance = 0
#             pjp_time = 0
#             for i in range(len(stores) - 1):
#                 store_a, store_b = stores[i], stores[i + 1]
#                 if store_a in store_to_index and store_b in store_to_index:
#                     idx_a, idx_b = store_to_index[store_a], store_to_index[store_b]
#                     pjp_distance += distance_matrix[idx_a][idx_b]
#                     pjp_time += time_matrix[idx_a][idx_b]
#                 cursor.execute("SELECT channeltypeid FROM store_channel WHERE storeid = %s;", (store_a,))
#                 res = cursor.fetchone()
#                 if res:
#                     if res[0] == 1:
#                         pjp_time += 20
#                     elif res[0] == 2:
#                         pjp_time += 40
#             if stores:
#                 cursor.execute("SELECT channeltypeid FROM store_channel WHERE storeid = %s;", (stores[-1],))
#                 res = cursor.fetchone()
#                 if res:
#                     if res[0] == 1:
#                         pjp_time += 20
#                     elif res[0] == 2:
#                         pjp_time += 40
#             total_distance += pjp_distance
#             total_travel_time += pjp_time
#         total_visits = len(group)
#         num_days = group["plan_date"].nunique()
#         avg_visits = total_visits / num_days if num_days > 0 else 0
#         avg_distance = total_distance / num_days if num_days > 0 else 0
#         avg_travel_time = total_travel_time / num_days if num_days > 0 else 0
#         orderbooker_metrics[int(orderbooker_id)] = {
#             "total_distance_travelled": round(total_distance, 2),
#             "avg_distance_travelled": round(avg_distance, 2),
#             "total_shops_visited": total_visits,
#             "avg_shops_visited": round(avg_visits, 2),
#             "total_travel_time": round(total_travel_time, 2),
#             "avg_travel_time": round(avg_travel_time, 2),
#             "running_time": 1
#         }
#     cursor.close()
#     return orderbooker_metrics

def calculate_orderbooker_kpi(db_conn, plan_id, distance_matrix, time_matrix, store_to_index):
    """
    Computes orderbooker-level KPIs:
      - Total & average distance travelled and shop visits per day
      - Total & average travel time (including service time)
    Returns a dictionary mapping each orderbooker_id to its metrics.
    """
    cursor = db_conn.cursor()
    query = """
        SELECT
            pp.orderbooker_id, pp.pjp_id, psa.store_id, pp.plan_date
        FROM
            pjp_plans pp
        JOIN
            pjp_store_assignments psa ON pp.pjp_id = psa.pjp_id
        WHERE
            pp.plan_id = %s;
    """
    cursor.execute(query, (plan_id,))
    data = cursor.fetchall()
    if not data:
        print(f"No data found for Plan ID: {plan_id}")
        cursor.close()
        return {}
    df = pd.DataFrame(data, columns=['orderbooker_id', 'pjp_id', 'store_id', 'plan_date'])
    orderbooker_metrics = {}
    grouped = df.groupby("orderbooker_id")
    for orderbooker_id, group in grouped:
        unique_pjps = group["pjp_id"].unique()
        total_distance = 0
        total_travel_time = 0
        for pjp_id in unique_pjps:
            stores = group[group["pjp_id"] == pjp_id]["store_id"].tolist()
            pjp_distance = 0
            pjp_time = 0
            for i in range(len(stores) - 1):
                store_a, store_b = stores[i], stores[i + 1]
                if store_a in store_to_index and store_b in store_to_index:
                    idx_a, idx_b = store_to_index[store_a], store_to_index[store_b]
                    pjp_distance += distance_matrix[idx_a][idx_b]
                    pjp_time += time_matrix[idx_a][idx_b]
                cursor.execute("SELECT channeltypeid FROM store_channel WHERE storeid = %s;", (store_a,))
                res = cursor.fetchone()
                if res:
                    if res[0] == 1:
                        pjp_time += 20
                    elif res[0] == 2:
                        pjp_time += 40
            if stores:
                cursor.execute("SELECT channeltypeid FROM store_channel WHERE storeid = %s;", (stores[-1],))
                res = cursor.fetchone()
                if res:
                    if res[0] == 1:
                        pjp_time += 20
                    elif res[0] == 2:
                        pjp_time += 40
            total_distance += pjp_distance
            total_travel_time += pjp_time
        total_visits = len(group)

        # ✅ Fetch assigned num_days for this orderbooker
        cursor.execute("""
            SELECT num_days FROM plan_master_ob_days WHERE plan_id = %s AND orderbooker_id = %s;
        """, (plan_id, orderbooker_id))
        result = cursor.fetchone()
        num_days = result[0] if result and result[0] else 0

        avg_visits = total_visits / num_days if num_days > 0 else 0
        avg_distance = total_distance / num_days if num_days > 0 else 0
        avg_travel_time = total_travel_time / num_days if num_days > 0 else 0
        orderbooker_metrics[int(orderbooker_id)] = {
            "total_distance_travelled": round(total_distance, 2),
            "avg_distance_travelled": round(avg_distance, 2),
            "total_shops_visited": total_visits,
            "avg_shops_visited": round(avg_visits, 2),
            "total_travel_time": round(total_travel_time, 2),
            "avg_travel_time": round(avg_travel_time, 2),
            "running_time": 1
        }
    cursor.close()
    return orderbooker_metrics



def update_orderbooker_kpi_in_db(db_conn, distributorid, plan_id, orderbooker_metrics, response_time):
    """
    Inserts or updates orderbooker-level KPI metrics into the database.
    """
    cursor = db_conn.cursor()
    query = """
    INSERT INTO orderbooker_kpi_history (
        distributorid, plan_id, ob_id,
        total_distance_travelled, avg_distance_travelled,
        total_shops_visited, avg_shops_visited,
        running_time, total_workload, avg_workload, response_time
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (distributorid, plan_id, ob_id)
    DO UPDATE SET
        total_distance_travelled = EXCLUDED.total_distance_travelled,
        avg_distance_travelled = EXCLUDED.avg_distance_travelled,
        total_shops_visited = EXCLUDED.total_shops_visited,
        avg_shops_visited = EXCLUDED.avg_shops_visited,
        running_time = EXCLUDED.running_time,
        total_workload = EXCLUDED.total_workload,
        avg_workload = EXCLUDED.avg_workload,
        response_time = EXCLUDED.response_time;
    """
    import numpy as np
    for ob_id, metrics in orderbooker_metrics.items():
        params = (
            distributorid, plan_id, ob_id,
            float(metrics["total_distance_travelled"]),
            float(metrics["avg_distance_travelled"]),
            int(metrics["total_shops_visited"]),
            float(metrics["avg_shops_visited"]),
            int(metrics["running_time"]),
            float(metrics["total_travel_time"]),
            float(metrics["avg_travel_time"]),
            float(response_time)
        )
        cursor.execute(query, params)
    db_conn.commit()
    cursor.close()
    print(f"Orderbooker KPI updated for distributor {distributorid}, plan {plan_id}.")

#----------------------------------------------------------------
# Compute Graph Metrics (Distributor Level) and update in database
#----------------------------------------------------------------
def calculate_graph_total_metrics(db_conn, plan_id, distance_matrix, time_matrix, store_to_index, plan_duration, num_orderbookers):
    """
    Computes daily graph metrics for distributor-level data:
      - For each date, computes total distance, unique shop visits, and total workload.
    Returns a list of dictionaries (one per day or per orderbooker based on plan_duration).
    """
    cursor = db_conn.cursor()
    query = """
        SELECT pp.plan_date, pp.orderbooker_id, psa.store_id, pp.pjp_id 
        FROM pjp_plans pp
        JOIN pjp_store_assignments psa ON pp.pjp_id = psa.pjp_id
        WHERE pp.plan_id = %s
        ORDER BY pp.plan_date, psa.visit_sequence;
    """
    cursor.execute(query, (str(plan_id),))
    data = cursor.fetchall()
    if not data:
        print(f"No data found for Plan ID: {plan_id}")
        cursor.close()
        return []
    
    df = pd.DataFrame(data, columns=['plan_date', 'orderbooker_id', 'store_id', 'pjp_id'])
    df["plan_date"] = df["plan_date"].astype(str)

    daily_metrics = {}

    for date, group in df.groupby("plan_date"):
        if plan_duration == 'custom':
            orderbooker_ids = [-1]  # Custom plan type: Aggregate metrics per day
        elif plan_duration == 'day':
            orderbooker_ids = group["orderbooker_id"].unique()
        else:
            raise ValueError("Invalid plan_duration. Must be 'custom' or 'day'.")

        for orderbooker_id in orderbooker_ids:
            total_distance = 0
            total_visits = 0
            total_workload = 0
            
            if plan_duration == 'custom':
                # ✅ Combine metrics across all orderbookers for the date
                pjp_ids = group["pjp_id"].unique()
            else:
                # ✅ Fetch PJP IDs for the specific orderbooker and date
                cursor.execute(
                    "SELECT DISTINCT pjp_id FROM pjp_plans WHERE plan_date = %s AND plan_id = %s AND orderbooker_id = %s;", 
                    (date, str(plan_id), int(orderbooker_id))
                )
                pjp_ids = [r[0] for r in cursor.fetchall()]
            
            for pjp_id in pjp_ids:
                # ✅ Fetch store visits for the PJP
                cursor.execute(
                    "SELECT store_id FROM pjp_store_assignments WHERE pjp_id = %s ORDER BY visit_sequence;", 
                    (pjp_id,)
                )
                stores = [r[0] for r in cursor.fetchall()]
                
                # Calculate distance and workload
                pjp_distance = 0
                pjp_workload = 0
                
                for i in range(len(stores) - 1):
                    store_a, store_b = stores[i], stores[i + 1]
                    if store_a in store_to_index and store_b in store_to_index:
                        idx_a, idx_b = store_to_index[store_a], store_to_index[store_b]
                        pjp_distance += distance_matrix[idx_a][idx_b]
                        pjp_workload += time_matrix[idx_a][idx_b]
                    
                    # ✅ Add service time based on store type
                    cursor.execute(
                        "SELECT channeltypeid FROM store_channel WHERE storeid = %s;", 
                        (store_a,)
                    )
                    res = cursor.fetchone()
                    if res:
                        if res[0] == 1:
                            pjp_workload += 20
                        elif res[0] == 2:
                            pjp_workload += 40
                
                # ✅ Include last store service time
                if stores:
                    cursor.execute(
                        "SELECT channeltypeid FROM store_channel WHERE storeid = %s;", 
                        (stores[-1],)
                    )
                    res = cursor.fetchone()
                    if res:
                        if res[0] == 1:
                            pjp_workload += 20
                        elif res[0] == 2:
                            pjp_workload += 40
                
                total_distance += pjp_distance
                total_workload += pjp_workload
            
            if plan_duration == 'custom':
                # ✅ Sum visits across all orderbookers for the day
                total_visits = group["store_id"].nunique()
                orderbooker_id = -1  # Set orderbooker_id to -1 for custom plans
            else:
                # ✅ Count unique store visits for the specific orderbooker
                total_visits = group[group["orderbooker_id"] == orderbooker_id]["store_id"].nunique()

            # ✅ Store results
            daily_metrics[(date, orderbooker_id)] = {
                "total_distance": round(total_distance, 2),
                "total_visits": total_visits,
                "total_workload": round(total_workload, 2)
            }

    cursor.close()

    # ✅ Convert to list format for DB update
    daily_metrics_list = [
        {
            "date": date,
            "orderbooker_id": orderbooker_id,
            "total_distance": metrics["total_distance"],
            "total_visits": metrics["total_visits"],
            "total_workload": metrics["total_workload"]
        }
        for (date, orderbooker_id), metrics in daily_metrics.items()
    ]

    return daily_metrics_list
    
    # ✅ Convert to list format
    daily_metrics_list = []
    for (date, orderbooker_id), values in daily_metrics.items():
        daily_metrics_list.append({
            "date": date,
            "orderbooker_id": orderbooker_id,
            "total_distance": values["total_distance"],
            "total_visits": values["total_visits"],
            "total_workload": values["total_workload"]
        })

    return daily_metrics_list

def update_graph_total_metrics_in_db(db_conn, plan_id, daily_metrics_list):
    """
    Inserts or updates the daily distributor-level graph metrics into the database.
    """
    cursor = db_conn.cursor()
    query = """
    INSERT INTO graph_total_ob_metrics (metric_date, plan_id, orderbooker_id, total_distance, total_visits, total_workload)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (metric_date, plan_id, orderbooker_id)
    DO UPDATE SET 
        total_distance = EXCLUDED.total_distance,
        total_visits = EXCLUDED.total_visits,
        total_workload = EXCLUDED.total_workload;
    """
    for record in daily_metrics_list:
        params = (
            record["date"],
            str(plan_id),  # ✅ Convert plan_id to string
            int(record["orderbooker_id"]),  # ✅ Convert to int
            float(record["total_distance"]),  # ✅ Convert to float
            int(record["total_visits"]),  # ✅ Convert to int
            float(record["total_workload"])  # ✅ Convert to float
        )
        cursor.execute(query, params)
    db_conn.commit()
    cursor.close()
    print(f"Graph total OB metrics updated for plan {plan_id}.")



#----------------------------------------------------------------
# Compute Graph Metrics (Orderbooker-Level) and update in database
#----------------------------------------------------------------
def calculate_graph_orderbooker_metrics(db_conn, plan_id, distance_matrix, time_matrix, store_to_index):
    """
    Computes daily graph metrics per orderbooker:
      - For each (orderbooker, date) computes distance, unique visits and workload.
    Returns a list of dictionaries.
    """
    cursor = db_conn.cursor()
    query = """
        SELECT pp.orderbooker_id, pp.plan_date, psa.store_id, pp.pjp_id 
        FROM pjp_plans pp
        JOIN pjp_store_assignments psa ON pp.pjp_id = psa.pjp_id
        WHERE pp.plan_id = %s
        ORDER BY pp.orderbooker_id, pp.plan_date, psa.visit_sequence;
    """
    cursor.execute(query, (plan_id,))
    data = cursor.fetchall()
    if not data:
        print(f"No data found for Plan ID: {plan_id}")
        cursor.close()
        return []
    df = pd.DataFrame(data, columns=['orderbooker_id', 'plan_date', 'store_id', 'pjp_id'])
    df["plan_date"] = df["plan_date"].astype(str)
    results = []
    for (orderbooker_id, date), group in df.groupby(["orderbooker_id", "plan_date"]):
        total_distance = 0
        total_workload = 0
        for pjp_id in group["pjp_id"].unique():
            stores = group[group["pjp_id"] == pjp_id]["store_id"].tolist()
            pjp_distance = 0
            pjp_workload = 0
            for i in range(len(stores) - 1):
                store_a, store_b = stores[i], stores[i + 1]
                if store_a in store_to_index and store_b in store_to_index:
                    idx_a, idx_b = store_to_index[store_a], store_to_index[store_b]
                    pjp_distance += distance_matrix[idx_a][idx_b]
                    pjp_workload += time_matrix[idx_a][idx_b]
                cursor.execute("SELECT channeltypeid FROM store_channel WHERE storeid = %s;", (store_a,))
                res = cursor.fetchone()
                if res:
                    if res[0] == 1:
                        pjp_workload += 20
                    elif res[0] == 2:
                        pjp_workload += 40
            if stores:
                cursor.execute("SELECT channeltypeid FROM store_channel WHERE storeid = %s;", (stores[-1],))
                res = cursor.fetchone()
                if res:
                    if res[0] == 1:
                        pjp_workload += 20
                    elif res[0] == 2:
                        pjp_workload += 40
            total_distance += pjp_distance
            total_workload += pjp_workload
        total_visits = group["store_id"].nunique()
        results.append({
            "orderbooker_id": int(orderbooker_id),
            "date": date,
            "distance": round(total_distance, 2),
            "visits": int(total_visits),
            "workload": round(total_workload, 2)
        })
    cursor.close()
    return results

def update_graph_orderbooker_metrics_in_db(db_conn, plan_id, orderbooker_metrics_list):
    """
    Inserts or updates the daily orderbooker-level graph metrics into the database.
    """
    cursor = db_conn.cursor()
    query = """
    INSERT INTO graph_unique_ob_metrics (plan_id, orderbooker_id, metric_date, distance, visits, workload)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (plan_id, orderbooker_id, metric_date)
    DO UPDATE SET 
        distance = EXCLUDED.distance,
        visits = EXCLUDED.visits,
        workload = EXCLUDED.workload;
    """
    import numpy as np
    for record in orderbooker_metrics_list:
        params = (
            plan_id,
            int(record["orderbooker_id"]),
            record["date"],
            float(record["distance"]),
            int(record["visits"]),
            float(record["workload"])
        )
        cursor.execute(query, params)
    db_conn.commit()
    cursor.close()
    print(f"Graph unique OB metrics updated for plan {plan_id}.")

#----------------------------------------------------------------
# Main function: Compute and update all metrics directly in DB
#----------------------------------------------------------------
def call_functions(distance_matrix, time_matrix, store_to_index, plan_id, distributor_id, response_time, plan_duration, num_orderbookers):
    """
    Calls all the metric computation functions and updates the corresponding database tables.
    """
    db_conn = connect_to_database()
    if not db_conn:
        print("Database connection failed.")
        return

    plan_id_list = get_distributor_plans(distributor_id)
    print("PLAN_ID_LIST", plan_id_list)

    # Distributor-level KPI computation and update
    distributor_metrics = calculate_distributor_kpi(db_conn, plan_id, distance_matrix, time_matrix, store_to_index, plan_duration, num_orderbookers)
    if distributor_metrics:
        update_distributor_kpi_in_db(db_conn, distributor_id, plan_id, distributor_metrics, response_time)

    # Orderbooker-level KPI computation and update
    orderbooker_kpi = calculate_orderbooker_kpi(db_conn, plan_id, distance_matrix, time_matrix, store_to_index)
    if orderbooker_kpi:
        update_orderbooker_kpi_in_db(db_conn, distributor_id, plan_id, orderbooker_kpi, response_time)

    # Graph metrics (distributor level) computation and update
    graph_total = calculate_graph_total_metrics(db_conn, plan_id, distance_matrix, time_matrix, store_to_index, plan_duration, num_orderbookers)
    if graph_total:
        update_graph_total_metrics_in_db(db_conn, plan_id, graph_total)

    # Graph metrics (orderbooker level) computation and update
    graph_orderbooker = calculate_graph_orderbooker_metrics(db_conn, plan_id, distance_matrix, time_matrix, store_to_index)
    if graph_orderbooker:
        update_graph_orderbooker_metrics_in_db(db_conn, plan_id, graph_orderbooker)

    db_conn.close()

