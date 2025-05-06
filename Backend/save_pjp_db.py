import psycopg2
import json
from datetime import datetime, timedelta
import uuid
import re
from Db_Setup import connect_to_database

class PJPDataSaver:
    def __init__(self, optimized_schedule):
        self.optimized_schedule = optimized_schedule

    def connect(self):
        return connect_to_database()

    def save_pjp(self, distributor_id, num_days):
        try:
            conn = self.connect()
            cursor = conn.cursor()

            plan_id = f"PLAN_{uuid.uuid4().hex[:8]}"
            created_date = datetime.now()

            insert_master = """
                INSERT INTO plan_master (plan_id, distributor_id, plan_duration, created_date, status)
                VALUES (%s, %s, %s, %s, TRUE)
            """
            cursor.execute(insert_master, (plan_id, distributor_id, num_days, created_date))

            start_date = datetime.now().date()

            # Fixing OB ID 0 Issues
            orderbooker_ids = list(map(int, self.optimized_schedule.keys()))
            starts_from_zero = 0 in orderbooker_ids

            for orderbooker_id, schedule in self.optimized_schedule.items():
                adjusted_ob_id = int(orderbooker_id) + 1 if starts_from_zero else int(orderbooker_id)

                num_days_for_ob = len(schedule)  # Get the number of days assigned to this orderbooker

                # ✅ Insert into the new `plan_master_ob_days` table
                insert_ob_days = """
                    INSERT INTO plan_master_ob_days (plan_id, orderbooker_id, num_days)
                    VALUES (%s, %s, %s)
                """
                cursor.execute(insert_ob_days, (plan_id, adjusted_ob_id, num_days_for_ob))

                for day_offset, stores in schedule.items():
                    try:
                        # Handle cases like "Day1"
                        if isinstance(day_offset, str):
                            day_match = re.match(r"Day(\d+)", day_offset)
                            if day_match:
                                day_offset = int(day_match.group(1)) - 1  # Convert to zero-based index
                            else:
                                try:
                                    day_offset = datetime.strptime(day_offset, "%Y-%m-%d").day - 1
                                except ValueError:
                                    raise ValueError(f"Invalid date format for monthly schedule: {day_offset}")

                        # Handle datetime object
                        if isinstance(day_offset, datetime):
                            day_offset = day_offset.day - 1

                        # ✅ Validate day offset within month range
                        max_days = (start_date.replace(month=start_date.month % 12 + 1, day=1) - timedelta(days=1)).day
                        if not (0 <= day_offset < max_days):
                            raise ValueError(f"Invalid day offset: {day_offset + 1} (Allowed range: 1 to {max_days})")

                        plan_date = start_date + timedelta(days=day_offset)
                        pjp_id = f"PJP_{uuid.uuid4().hex[:8]}"

                        insert_plan = """
                            INSERT INTO pjp_plans (pjp_id, plan_id, orderbooker_id, plan_date)
                            VALUES (%s, %s, %s, %s)
                        """
                        cursor.execute(insert_plan, (pjp_id, plan_id, adjusted_ob_id, plan_date))

                        insert_assignment = """
                            INSERT INTO pjp_store_assignments (pjp_id, store_id, visit_sequence)
                            VALUES (%s, %s, %s)
                        """
                        for sequence, store in enumerate(stores, 1):
                            cursor.execute(insert_assignment, (pjp_id, store['storeid'], sequence))

                    except Exception as inner_err:
                        print(f"Skipping invalid day_offset '{day_offset}': {inner_err}")
                        continue

            conn.commit()
            return plan_id
        

        except Exception as e:
            conn.rollback()
            print(f"Error: {e}")
            raise e
        finally:
            cursor.close()
            conn.close()
