import numpy as np
import random
import logging
import time
from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import colorsys
import warnings
from Db_operations import fetch_data
from datetime import datetime, timedelta
import copy
import requests
import concurrent.futures
from save_pjp_db import PJPDataSaver
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import math

warnings.filterwarnings("ignore")

INITIAL_PROXIMITY_THRESHOLD_KM = 1.0
MIN_PROXIMITY_THRESHOLD_KM = 0.1
PROXIMITY_DECREASE_FACTOR = 0.2


# ------------------------------------------------------------------
# ROUTE HELPERS  – single metric (travel minutes) everywhere
# ------------------------------------------------------------------
def _edge_cost(travel_min, idx_a, idx_b):
    """Return travel minutes between idx_a and idx_b."""
    return travel_min[idx_a, idx_b]

def _route_cost(travel_min, store_to_index, route):
    """Total travel minutes along `route`."""
    if len(route) <= 1:
        return 0.0
    cost = 0.0
    for i in range(len(route) - 1):
        cost += _edge_cost(travel_min,
                           store_to_index[route[i]],
                           store_to_index[route[i + 1]])
    return cost


class EAScheduleOptimizer:
    def __init__(
        self,
        store_data,
        available_days,
        travel_min,
        distance_matrix,
        store_to_index,
        working_minutes=480,
        logger=None
    ):
        """
        Evolutionary Algorithm (EA) for scheduling:
        - Minimizes total daily travel + service time
        - Enforces an working hour daily limit as a hard constraint
        - Ensures each store's required visit frequency
        - Encourages balanced distribution across days
        - Adds a small penalty for separating geographically close single-frequency stores
        """
        self.store_data = store_data
        self.available_days = available_days
        self.travel_min = travel_min
        self.distance_matrix = distance_matrix
        self.store_to_index = store_to_index
        self.logger = logger or logging.getLogger("EAScheduleOptimizer")

        self.all_store_ids = set(self.store_data['storeid'].unique())
        
        # Add route caching
        self._route_cache = {}
        
        # Store-level service time
        self.service_time_by_id = {}
        for _, row in store_data.iterrows():
            sid = row['storeid']
            w = row.get('workload_weight', 20)  # fallback if missing
            self.service_time_by_id[sid] = w

        # Required visits (channeltypeid => typically 1 or 2 visits)
        self.required_visits = {}
        for _, row in store_data.iterrows():
            sid = row['storeid']
            freq = int(row['channeltypeid'])
            self.required_visits[sid] = freq

        self.DAILY_LIMIT = working_minutes
        self.lower_ideal_minutes = working_minutes * 0.75
        self.upper_ideal_minutes = working_minutes * 0.875

        # Use the improved neighbor map method
        self._build_improved_neighbor_map()

     
    def _build_improved_neighbor_map(self):
        """
        Create a better neighborhood model with adaptive proximity and distance weighting.
        """
        self.neighbors_map = defaultdict(list)
        self.proximity_weights = defaultdict(dict)  # Store pair-wise proximity weights
        
        if self.distance_matrix is None:
            return
            
        # Calculate statistics on distance distribution
        all_distances = []
        for i in range(len(self.distance_matrix)):
            for j in range(i+1, len(self.distance_matrix)):
                all_distances.append(self.distance_matrix[i,j])
        
        # Calculate percentiles for adaptive thresholds
        distances_array = np.array(all_distances)
        p25 = np.percentile(distances_array, 25)
        p50 = np.percentile(distances_array, 50)
        p75 = np.percentile(distances_array, 75)
        
        # Use weighted proximity model with multiple thresholds
        tight_radius = p25
        medium_radius = p50
        loose_radius = p75
        
        self.logger.info(f"Using adaptive proximity radii: tight={tight_radius:.2f}km, medium={medium_radius:.2f}km, loose={loose_radius:.2f}km")
        
        for s in self.all_store_ids:
            s_idx = self.store_to_index[s]
            s_freq = self.required_visits[s]
            
            for t in self.all_store_ids:
                if t != s:
                    t_idx = self.store_to_index[t]
                    t_freq = self.required_visits[t]
                    dist_val = self.distance_matrix[s_idx, t_idx]
                    
                    # Calculate proximity weight based on distance thresholds
                    if dist_val <= tight_radius:
                        # Very close stores - strong relationship
                        self.neighbors_map[s].append(t)
                        self.proximity_weights[s][t] = 3.0
                    elif dist_val <= medium_radius:
                        # Medium distance - moderate relationship
                        self.neighbors_map[s].append(t)
                        self.proximity_weights[s][t] = 2.0
                    elif dist_val <= loose_radius:
                        # Further but still considered neighbors
                        self.neighbors_map[s].append(t)
                        self.proximity_weights[s][t] = 1.0
        
        # Ensure minimum number of neighbors for every store
        min_neighbors = 4
        for s in self.all_store_ids:
            if len(self.neighbors_map[s]) < min_neighbors:
                s_idx = self.store_to_index[s]
                
                # Find closest stores
                distances = [(t, self.distance_matrix[s_idx, self.store_to_index[t]]) 
                            for t in self.all_store_ids if t != s]
                distances.sort(key=lambda x: x[1])
                
                # Add closest ones with appropriate weights
                for t, dist in distances[:min_neighbors]:
                    if t not in self.neighbors_map[s]:
                        self.neighbors_map[s].append(t)
                        # Weight based on relative position in sorted list
                        self.proximity_weights[s][t] = 1.0                      
    def _initialize_population_density_based(self, n_individuals):
        """
        Create smarter initial population based on store density regions.
        """
        population = []
        
        # Create a rough density estimation using geographic grid
        lat_min = min(self.store_data['latitude'])
        lat_max = max(self.store_data['latitude'])
        lng_min = min(self.store_data['longitude'])
        lng_max = max(self.store_data['longitude'])
        
        grid_size = 10  # Adjust based on your specific geography
        lat_step = (lat_max - lat_min) / grid_size
        lng_step = (lng_max - lng_min) / grid_size
        
        # Map stores to grid cells
        grid_map = defaultdict(list)
        for _, store in self.store_data.iterrows():
            lat_idx = min(grid_size-1, int((store['latitude'] - lat_min) / lat_step))
            lng_idx = min(grid_size-1, int((store['longitude'] - lng_min) / lng_step))
            grid_map[(lat_idx, lng_idx)].append(store['storeid'])
        
        # Generate individuals by assigning grid cells to days
        for _ in range(n_individuals):
            individual = {day: [] for day in self.available_days}
            
            # First assign complete grid cells
            available_days = self.available_days.copy()
            grid_cells = list(grid_map.keys())
            random.shuffle(grid_cells)
            
            for cell in grid_cells:
                if not available_days:
                    available_days = self.available_days.copy()
                day = random.choice(available_days)
                
                # Add stores from this grid cell considering frequency requirements
                for store_id in grid_map[cell]:
                    freq = self.required_visits[store_id]
                    if freq == 1:
                        individual[day].append(store_id)
                    else:
                        # For higher frequency stores, distribute across days
                        days_for_store = [day]
                        remaining_days = [d for d in self.available_days if d != day]
                        if remaining_days and freq > 1:
                            extra_days = random.sample(remaining_days, min(freq-1, len(remaining_days)))
                            days_for_store.extend(extra_days)
                        for d in days_for_store:
                            individual[d].append(store_id)
            
            population.append(individual)
        
        return population

    def _initialize_population_capacity_based(self, n_individuals):
        """
        Greedy first‑fit decreasing by (service_time + avg_travel).
        """
        population = []
        # precompute avg travel per store
        avg_travel = {}
        for sid in self.all_store_ids:
            idx = self.store_to_index[sid]
            row = self.distance_matrix[idx]
            avg_travel[sid] = np.mean(row)

        # compute a “weight” per store
        weights = {sid: self.service_time_by_id[sid] + avg_travel[sid]
                   for sid in self.all_store_ids}

        for _ in range(n_individuals):
            bins = {d: [] for d in self.available_days}
            usage = {d: 0.0 for d in self.available_days}
            # sort stores descending by weight
            for sid in sorted(self.all_store_ids, key=lambda s: weights[s], reverse=True):
                # place in the day with most remaining capacity
                best_day = min(usage, key=lambda d: usage[d])
                bins[best_day].append(sid)
                usage[best_day] += weights[sid]
            population.append(bins)
        return population

    # ---------------------------------------------------------------------
    # 1) Population Initialization
    # ---------------------------------------------------------------------
    def _initialize_population_proximity_based(self, n_individuals):
        """
        Create initial population with strong emphasis on geographic proximity.
        """
        population = []
        
        for _ in range(n_individuals):
            # Start with empty days
            individual = {day: [] for day in self.available_days}
            unassigned_stores = list(self.all_store_ids)
            
            while unassigned_stores:
                # Pick a random store as seed
                seed_store = random.choice(unassigned_stores)
                unassigned_stores.remove(seed_store)
                
                # Choose a day with fewest assignments or randomly if all equal
                day_loads = {d: len(stores) for d, stores in individual.items()}
                candidate_days = [d for d in self.available_days 
                                if individual[d].count(seed_store) < self.required_visits[seed_store]]
                
                if not candidate_days:
                    continue
                    
                target_day = min(candidate_days, key=lambda d: day_loads[d])
                individual[target_day].append(seed_store)
                
                # Find its close neighbors that aren't yet fully assigned
                close_stores = []
                for neighbor in self.neighbors_map.get(seed_store, []):
                    if neighbor in unassigned_stores:
                        # Higher weight = closer store, should be processed earlier
                        weight = self.proximity_weights[seed_store].get(neighbor, 1.0)
                        close_stores.append((neighbor, weight))
                
                # Sort by proximity weight (descending)
                close_stores.sort(key=lambda x: x[1], reverse=True)
                
                # Try to assign close stores to the same day
                for store, _ in close_stores:
                    if (store in unassigned_stores and 
                        individual[target_day].count(store) < self.required_visits[store]):
                        individual[target_day].append(store)
                        unassigned_stores.remove(store)
                        
                        # For stores requiring multiple visits, assign other visits too
                        freq = self.required_visits[store]
                        if freq > 1:
                            remaining_visits = freq - 1
                            other_days = [d for d in self.available_days if d != target_day]
                            if other_days and remaining_visits > 0:
                                for _ in range(min(remaining_visits, len(other_days))):
                                    other_day = random.choice(other_days)
                                    individual[other_day].append(store)
                                    other_days.remove(other_day)
            
            population.append(individual)
        
        return population

    def initialize_population(self, population_size):
        """
        Create a mixed population with four strategies including the new proximity-based one.
        """
        pop = []
        # 1/5 from random
        pop.extend(self._initialize_population_random(population_size//5))
        # 1/5 from geographic
        pop.extend(self._initialize_population_geographic(population_size//5))
        # 1/5 from capacity‑based
        pop.extend(self._initialize_population_capacity_based(population_size//5))
        # 1/5 from density
        pop.extend(self._initialize_population_density_based(population_size//5))
        # New: 1/5 from proximity-based
        pop.extend(self._initialize_population_proximity_based(population_size - 4*(population_size//5)))
        
        return pop
        
    def _initialize_population_random(self, n_individuals):
        population = []
        for _ in range(n_individuals):
            individual = {day: [] for day in self.available_days}
            for store_id in self.all_store_ids:
                freq = self.required_visits[store_id]
                if len(self.available_days) >= freq:
                    chosen_days = random.sample(self.available_days, freq)
                else:
                    chosen_days = random.choices(self.available_days, k=freq)
                for d in chosen_days:
                    individual[d].append(store_id)
            population.append(individual)
        return population

    def _initialize_population_geographic(self, n_individuals):
        """
        Create smarter initial population based on geographic proximity.
        """
        population = []
        for _ in range(n_individuals):
            individual = {day: [] for day in self.available_days}
            remaining_stores = list(self.all_store_ids)
            day_index = 0
            
            while remaining_stores:
                current_day = self.available_days[day_index % len(self.available_days)]
                
                # Pick a random starting store if day is empty
                if not individual[current_day]:
                    store_id = random.choice(remaining_stores)
                    individual[current_day].append(store_id)
                    remaining_stores.remove(store_id)
                    
                    # Assign required visits for high frequency stores
                    freq = self.required_visits[store_id]
                    if freq > 1 and len(self.available_days) > 1:
                        next_day = self.available_days[(day_index + 1) % len(self.available_days)]
                        individual[next_day].append(store_id)
                
                # Find closest store to those already in this day
                elif remaining_stores:
                    day_stores = individual[current_day]
                    store_indices = [self.store_to_index[s] for s in day_stores]
                    
                    # Find closest remaining store
                    best_store = None
                    min_avg_dist = float('inf')
                    
                    for store in remaining_stores:
                        store_idx = self.store_to_index[store]
                        avg_distance = sum(self.distance_matrix[store_idx, idx] for idx in store_indices) / len(store_indices)
                        if avg_distance < min_avg_dist:
                            min_avg_dist = avg_distance
                            best_store = store
                    
                    if best_store:
                        individual[current_day].append(best_store)
                        remaining_stores.remove(best_store)
                        
                        # Handle second visit if required
                        freq = self.required_visits[best_store]
                        if freq > 1 and len(self.available_days) > 1:
                            next_day = self.available_days[(day_index + 1) % len(self.available_days)]
                            individual[next_day].append(best_store)
                
                day_index += 1
                
            population.append(individual)
        
        return population
    
    def _nearest_neighbor_route(self, store_list, restarts: int = 5):
        """
        Greedy NN with `restarts` random starts, keep the best.
        Caching is OFF during optimisation; post‑processing can
        wrap this call and cache on its side.
        """
        n = len(store_list)
        if n <= 1:
            return store_list[:]

        best_route = None
        best_cost  = float("inf")

        indices = [self.store_to_index[s] for s in store_list]

        for _ in range(restarts):
            unvisited = set(range(n))
            current_idx = random.choice(tuple(unvisited))
            route_idx   = [current_idx]
            unvisited.remove(current_idx)

            while unvisited:
                current_global = indices[current_idx]
                next_idx = min(unvisited,
                            key=lambda j: _edge_cost(self.travel_min,
                                                        current_global,
                                                        indices[j]))
                route_idx.append(next_idx)
                unvisited.remove(next_idx)
                current_idx = next_idx

            candidate = [store_list[i] for i in route_idx]
            c_cost    = _route_cost(self.travel_min, self.store_to_index, candidate)

            if c_cost < best_cost:
                best_cost  = c_cost
                best_route = candidate

        return best_route

    # ---------------------------------------------------------------------
    # 2) Evaluate Fitness
        # ---------------------------------------------------------------------
    def evaluate_fitness(self, individual):
        """
        Enhanced fitness function with better geographic cohesion penalties.
        """
        from collections import defaultdict
        visit_counts = defaultdict(int)
        for day, stores in individual.items():
            for sid in stores:
                visit_counts[sid] += 1

        # mismatch penalty - calculate this first for early return
        mismatch_penalty = 0.0
        for sid in self.all_store_ids:
            req = self.required_visits[sid]
            diff = abs(visit_counts[sid] - req)
            if diff > 0:
                mismatch_penalty += 5000 * diff
                    
        # If there are any frequency mismatches, return early with high penalty
        if mismatch_penalty > 0:
            return 1e9 + mismatch_penalty

        # Calculate adjusted time utilization with new penalties
        total_time = 0.0
        day_usage = []
        for day, stores in individual.items():
            if not stores:
                day_usage.append(0.0)
                continue

            day_time = self._compute_day_time(stores)
            if day_time > self.DAILY_LIMIT:
                overflow = day_time - self.DAILY_LIMIT
                # massive penalty
                return 1e9 + 10000 * overflow + mismatch_penalty

            total_time += day_time
            day_usage.append(day_time)
        
        # Enhanced day balancing with progressive penalties
        TARGET_UTIL = 0.95 * self.DAILY_LIMIT
        imbalance_penalty = 0.0
        
        # Calculate variance between days to minimize overall spread
        if len(day_usage) > 1:
            time_variance = np.var(day_usage) 
            imbalance_penalty += time_variance * 0.5
        
        # Progressive penalties for under-utilization
        for usage in day_usage:
            # Penalty increases more sharply as we get further from target
            util_ratio = usage / TARGET_UTIL
            if util_ratio < 0.7:  # Severely underutilized
                imbalance_penalty += (TARGET_UTIL - usage) * 2.0
            elif util_ratio < 0.85:  # Moderately underutilized
                imbalance_penalty += (TARGET_UTIL - usage) * 1.2
            elif util_ratio < 0.95:  # Slightly underutilized
                imbalance_penalty += (TARGET_UTIL - usage) * 0.8
        
        # Enhanced geographical penalty that considers proximity weights
        geographical_penalty = 0.0
        store_day_map = {}
        for d, s_list in individual.items():
            for sid in s_list:
                store_day_map[sid] = d

        # Calculate proximity-based cohesion penalty
        for s, neighbors in self.neighbors_map.items():
            day_s = store_day_map.get(s, None)
            if day_s is None:
                continue
                
            for nb in neighbors:
                day_nb = store_day_map.get(nb, None)
                if day_nb is not None and day_s != day_nb:
                    # Use the proximity weight to scale the penalty
                    proximity_weight = self.proximity_weights[s].get(nb, 1.0)
                    
                    # Stronger penalty for very close stores (higher weight)
                    s_idx = self.store_to_index[s]
                    nb_idx = self.store_to_index[nb]
                    
                    # Frequency considerations
                    freq_s = self.required_visits[s]
                    freq_nb = self.required_visits[nb]
                    freq_factor = 1.0
                    
                    # Special case: if both stores require only one visit, strongly prefer them on same day
                    if freq_s == 1 and freq_nb == 1:
                        freq_factor = 2.0
                    
                    geographical_penalty += 3000.0 * proximity_weight * freq_factor

        # Combine all components into final fitness
        fitness = total_time + mismatch_penalty + imbalance_penalty + geographical_penalty
        return fitness

    # ---------------------------------------------------------------------
    # 3) Local Search
    # ---------------------------------------------------------------------
    def local_search(self, individual, max_iterations=10):
        """
        Enhanced local search that prioritizes keeping nearby stores together.
        """
        best_schedule = {d: list(stores) for d, stores in individual.items()}
        best_fitness = self.evaluate_fitness(best_schedule)

        for _ in range(max_iterations):
            # Try to move each store to a day where its closest neighbors are
            improved = False
            
            # 1. First focus on very close neighbors (high proximity weight)
            for store in self.all_store_ids:
                # Skip if store has no neighbors
                if store not in self.neighbors_map:
                    continue
                    
                # Find which days this store is currently on
                current_days = []
                for d, stores in best_schedule.items():
                    if store in stores:
                        current_days.append(d)
                
                if not current_days:
                    continue  # Store not assigned to any day
                    
                # Get weighted neighbor day distribution
                neighbor_day_weight = defaultdict(float)
                for nb in self.neighbors_map[store]:
                    weight = self.proximity_weights[store].get(nb, 1.0)
                    for d, stores in best_schedule.items():
                        if nb in stores:
                            neighbor_day_weight[d] += weight
                
                if not neighbor_day_weight:
                    continue  # No neighbors assigned to any day
                
                # Find best day with most weighted neighbors
                best_neighbor_day = max(neighbor_day_weight.keys(), 
                                    key=lambda d: neighbor_day_weight[d])
                
                # If store isn't on this day already and this day has significant neighbors
                if (best_neighbor_day not in current_days and 
                    neighbor_day_weight[best_neighbor_day] >= 2.0):  # Threshold for "significant"
                    
                    # Try moving one instance to the neighbor-rich day
                    from_day = current_days[0]  # Move from first day it appears
                    
                    # Make the move
                    best_schedule[from_day].remove(store)
                    best_schedule[best_neighbor_day].append(store)
                    
                    new_fitness = self.evaluate_fitness(best_schedule)
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        improved = True
                    else:
                        # Revert
                        best_schedule[best_neighbor_day].remove(store)
                        best_schedule[from_day].append(store)
                        
            # 2. Try swapping stores between days to improve fitness
            if not improved:
                days = list(best_schedule.keys())
                if len(days) >= 2:
                    for _ in range(min(20, len(self.all_store_ids))):  # Limit iterations
                        d1, d2 = random.sample(days, 2)
                        if not best_schedule[d1] or not best_schedule[d2]:
                            continue
                            
                        store1 = random.choice(best_schedule[d1])
                        store2 = random.choice(best_schedule[d2])
                        
                        # Skip if frequencies would be violated
                        if (best_schedule[d1].count(store1) <= 1 or 
                            best_schedule[d2].count(store2) <= 1):
                            continue
                        
                        # Make the swap
                        best_schedule[d1].remove(store1)
                        best_schedule[d1].append(store2)
                        best_schedule[d2].remove(store2)
                        best_schedule[d2].append(store1)
                        
                        new_fitness = self.evaluate_fitness(best_schedule)
                        if new_fitness < best_fitness:
                            best_fitness = new_fitness
                            improved = True
                        else:
                            # Revert the swap
                            best_schedule[d1].remove(store2)
                            best_schedule[d1].append(store1)
                            best_schedule[d2].remove(store1)
                            best_schedule[d2].append(store2)
            
            if not improved:
                break

        return best_schedule

    # ---------------------------------------------------------------------
    # 4) Crossover
    # ---------------------------------------------------------------------
    def standard_crossover(self, parent1, parent2):
        """
        Day-based crossover using day utilization distance from global avg.
        """
        child = {}
        p1_util = self.calculate_day_utilization(parent1)
        p2_util = self.calculate_day_utilization(parent2)

        avg1 = sum(p1_util.values()) / len(p1_util) if p1_util else 0
        avg2 = sum(p2_util.values()) / len(p2_util) if p2_util else 0
        global_avg = (avg1 + avg2) / 2.0

        for d in self.available_days:
            d1_time = p1_util[d]
            d2_time = p2_util[d]
            dist1 = abs(d1_time - global_avg)
            dist2 = abs(d2_time - global_avg)
            child[d] = parent1[d][:] if dist1 < dist2 else parent2[d][:]
        return child

    def block_crossover(self, parent1, parent2):
        """
        Block-based: pick entire assignment for each day from either parent.
        """
        child = {}
        for d in self.available_days:
            if random.random() < 0.5:
                child[d] = parent1[d][:]
            else:
                child[d] = parent2[d][:]
        return child

    def crossover(self, parent1, parent2):
        """
        Randomly choose which crossover strategy to apply.
        """
        if random.random() < 0.5:
            return self.standard_crossover(parent1, parent2)
        else:
            return self.block_crossover(parent1, parent2)

    # ---------------------------------------------------------------------
    # 5) Mutation
    # ---------------------------------------------------------------------
    def mutate(self, individual, mutation_rate):
        """
        Improved mutation with geographic awareness.
        """
        new_ind = {d: list(stores) for d, stores in individual.items()}
        days_list = list(new_ind.keys())

        # Geographic-aware mutations
        for day in days_list:
            if random.random() < mutation_rate and len(new_ind[day]) > 0:
                store_to_move = random.choice(new_ind[day])
                
                # Find day with most geographic neighbors
                best_day = None
                max_neighbors = -1
                
                for target_day in days_list:
                    if target_day != day:
                        neighbor_count = sum(1 for nb in self.neighbors_map.get(store_to_move, []) 
                                            if nb in new_ind[target_day])
                        if neighbor_count > max_neighbors:
                            max_neighbors = neighbor_count
                            best_day = target_day
                
                # Move to best day or random day if no neighbors found
                target_day = best_day if best_day and random.random() < 0.7 else random.choice([d for d in days_list if d != day])
                new_ind[day].remove(store_to_move)
                new_ind[target_day].append(store_to_move)

        # Enhance balance between days
        new_ind = self.redistribute_workload(new_ind)
        return new_ind

    # ---------------------------------------------------------------------
    # Utility & TSP-like methods
    # ---------------------------------------------------------------------
    def redistribute_workload(self, individual):
        """
        Improved day balancing with better time utilization.
        """
        new_ind = {d: list(stores) for d, stores in individual.items()}
        day_util = self.calculate_day_utilization(new_ind)
        
        # Calculate global metrics
        total_time = sum(day_util.values())
        avg_time = total_time / len(day_util) if day_util else 0
        days_sorted = sorted(day_util.items(), key=lambda x: x[1])  # Sorted by time
        
        # Target zone: 90-98% of working limit for optimal utilization
        target_min = 0.90 * self.DAILY_LIMIT
        target_max = 0.98 * self.DAILY_LIMIT
        
        # First pass: handle overloaded days
        over_days = [d for d, ut in day_util.items() if ut > self.DAILY_LIMIT]
        under_days = [d for d, ut in day_util.items() if ut < target_min]
        
        # Similar to your existing implementation but with improved targeting
        while over_days and under_days:
            od = max(over_days, key=lambda d: day_util[d])
            
            while day_util[od] > self.DAILY_LIMIT and under_days:
                if not new_ind[od]:
                    break
                    
                # Use geographic proximity for store selection
                best_store = None
                best_target = None
                best_score = float('-inf')
                
                for store in new_ind[od]:
                    store_time = self.service_time_by_id[store]
                    
                    for ud in under_days:
                        # Calculate how well this store fits the underutilized day
                        fit_score = (target_min - day_util[ud]) - abs(store_time - (target_min - day_util[ud]))
                        
                        # Add geographic bonus if neighbors are on this day
                        geo_bonus = sum(1 for nb in self.neighbors_map.get(store, []) 
                                    if nb in new_ind[ud]) * 10
                        
                        total_score = fit_score + geo_bonus
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_store = store
                            best_target = ud
                
                if best_store and best_target:
                    new_ind[od].remove(best_store)
                    new_ind[best_target].append(best_store)
                    
                    # Update utilizations
                    st_time = self.service_time_by_id[best_store]
                    day_util[od] -= st_time
                    day_util[best_target] += st_time
                    
                    # Check if days need to be removed from lists
                    if day_util[od] <= self.DAILY_LIMIT:
                        over_days.remove(od)
                    if day_util[best_target] >= target_min:
                        under_days.remove(best_target)
                else:
                    break
        
        # Second pass: optimize time utilization toward target zone
        return new_ind

    def calculate_day_utilization(self, individual):
        day_util = {}
        for d, stores in individual.items():
            day_util[d] = self._compute_day_time(stores)
        return day_util

    def _two_opt(self, route):
        """Deterministic 2‑opt until no improving swap exists."""
        n = len(route)
        if n <= 3:
            return route

        improved = True
        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 2, n):
                    if j - i == 1:       # adjacent
                        continue
                    # cost before swap
                    a, b = route[i - 1], route[i]
                    c, d = route[j - 1], route[j % n]
                    old = (_edge_cost(self.travel_min, self.store_to_index[a], self.store_to_index[b]) +
                        _edge_cost(self.travel_min, self.store_to_index[c], self.store_to_index[d]))
                    new = (_edge_cost(self.travel_min, self.store_to_index[a], self.store_to_index[c]) +
                        _edge_cost(self.travel_min, self.store_to_index[b], self.store_to_index[d]))
                    if new < old:
                        route[i:j] = reversed(route[i:j])
                        improved = True
            # loop continues until no swap improves distance
            return route
    
    def _nearest_insertion_route(self, store_list, restarts: int = 3):
        """
        Fast NI with adaptive position‑sampling & full 2‑opt at the end.
        """
        if len(store_list) <= 2:
            return store_list[:]

        best_route = None
        best_cost  = float("inf")
        idx_map = [self.store_to_index[s] for s in store_list]
        m       = len(store_list)

        for _ in range(restarts):
            unvisited = set(range(m))
            start = random.choice(tuple(unvisited))
            route_idx = [start]
            unvisited.remove(start)

            # add the closest second point
            if unvisited:
                second = min(unvisited,
                            key=lambda j: _edge_cost(self.travel_min,
                                                    idx_map[start],
                                                    idx_map[j]))
                route_idx.append(second)
                unvisited.remove(second)

            # main NI loop — adaptive sample size
            while unvisited:
                best_gain = float("inf")
                best_j    = None
                best_pos  = None

                for j in unvisited:
                    # test at a subset of positions
                    sample_size = max(len(route_idx) // 3, 20)
                    positions   = (range(len(route_idx))
                                if len(route_idx) <= sample_size
                                else random.sample(range(len(route_idx)),
                                                    sample_size))

                    for i in positions:
                        nxt = (i + 1) % len(route_idx)
                        gain = (_edge_cost(self.travel_min, idx_map[route_idx[i]], idx_map[j]) +
                                _edge_cost(self.travel_min, idx_map[j],               idx_map[route_idx[nxt]]) -
                                _edge_cost(self.travel_min, idx_map[route_idx[i]],   idx_map[route_idx[nxt]]))
                        if gain < best_gain:
                            best_gain = gain
                            best_j    = j
                            best_pos  = nxt

                route_idx.insert(best_pos, best_j)
                unvisited.remove(best_j)

            candidate = [store_list[i] for i in route_idx]
            candidate = self._two_opt(candidate)        # full refinement
            c_cost    = _route_cost(self.travel_min, self.store_to_index, candidate)

            if c_cost < best_cost:
                best_cost  = c_cost
                best_route = candidate

        return best_route


    def _compute_day_time(self, store_list):
        """
        Travel minutes + service minutes using unified metric.
        """
        if not store_list:
            return 0.0
        route = self._nearest_neighbor_route(store_list)   # fast estimate
        travel = _route_cost(self.travel_min, self.store_to_index, route)
        service = sum(self.service_time_by_id[s] for s in route)
        return travel + service


    

    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        selected = random.sample(list(zip(population, fitness_scores)), tournament_size)
        return min(selected, key=lambda x: x[1])[0]

    # ---------------------------------------------------------------------
    # 6) Main EA loop
    # ---------------------------------------------------------------------
    def optimize_schedule(
        self,
        population_size=50,  # Increased from 40 
        generations=25,      # Increased from 15
        mutation_rate=0.3,   # Keep as-is, but the mutation function is enhanced
        patience=8,          # Slightly increased
        max_time=180         # Allow more time for better solutions
    ):
        start_time = time.time()

        population = self.initialize_population(population_size)
        best_individual = None
        best_fitness = float('inf')
        no_improvement_count = 0
        
        # For early stopping
        prev_best_fitness = float('inf')
        convergence_count = 0
        convergence_threshold = 5  # Stop if no improvement for 3 generations

        for gen in range(generations):
            elapsed = time.time() - start_time
            if elapsed > max_time:
                self.logger.info(f"Max time {max_time}s exceeded. Stopping at gen={gen+1}.")
                break

            # Evaluate fitness for all individuals
            fitness_scores = [self.evaluate_fitness(ind) for ind in population]

            # Find best in this generation
            improved = False
            for i, fit in enumerate(fitness_scores):
                if fit < best_fitness:
                    best_fitness = fit
                    best_individual = population[i]
                    no_improvement_count = 0
                    improved = True

            # Early stopping check
            if best_fitness >= prev_best_fitness * 0.995:  # 0.5% tolerance
                convergence_count += 1
                if convergence_count >= convergence_threshold:
                    self.logger.info(f"Early stopping at gen={gen+1}, no significant improvement for {convergence_threshold} generations")
                    break
            else:
                convergence_count = 0
            prev_best_fitness = best_fitness

            if not improved:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                # adapt mutation if no improvement
                mutation_rate = min(1.0, mutation_rate * 1.5)
                self.logger.info(
                    f"No improvement for {no_improvement_count} gens. "
                    f"Increasing mutation_rate to {mutation_rate:.2f}."
                )

            new_population = []
            for _ in range(population_size // 2):
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)

                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)

                # child1
                child1 = self.mutate(child1, mutation_rate)
                child1 = self.local_search(child1)

                # child2
                child2 = self.mutate(child2, mutation_rate)
                child2 = self.local_search(child2)

                new_population.append(child1)
                new_population.append(child2)

            # if pop size is odd, keep best leftover
            if population_size % 2 == 1:
                import numpy as np
                best_idx = np.argmin(fitness_scores)
                leftover = population[best_idx]
                leftover = self.local_search(leftover)
                new_population.append(leftover)

            population = new_population

            self.logger.info(f"Gen={gen+1}, BestFitness={best_fitness:.2f}, no_improve={no_improvement_count}")

        # final fix & schedule creation
        if best_individual is not None:
            best_individual = self.redistribute_workload(best_individual)
            best_individual = self._create_final_schedule(best_individual)
        else:
            best_individual = {d: [] for d in self.available_days}

        return best_individual

    def _create_final_schedule(self, best_individual):
        """
        Create the final schedule with optimized routes
        (only called once at the end of the optimization process)
        """
        final_schedule = {d: [] for d in self.available_days}
        store_info = {}
        for row in self.store_data.itertuples():
            store_info[row.storeid] = {
                'storeid': row.storeid,
                'storecode': getattr(row, 'storecode', ''),
                'latitude': row.latitude,
                'longitude': row.longitude,
                'channeltypeid': row.channeltypeid
            }

        from collections import defaultdict
        visit_id_counter = defaultdict(int)
        
        # Now optimize each day's route with nearest insertion
        for d in self.available_days:
            store_ids = best_individual[d]
            if len(store_ids) > 1:
                # Use nearest insertion for final routes instead of ACO
                optimized_route = self._nearest_insertion_route(store_ids)
                final_schedule[d] = [store_info[sid].copy() for sid in optimized_route]
                for i, sid in enumerate(optimized_route):
                    final_schedule[d][i]['visit_id'] = visit_id_counter[sid]
                    visit_id_counter[sid] += 1
            else:
                # Direct assignment for 0-1 stores
                for sid in store_ids:
                    si = store_info[sid].copy()
                    si['visit_id'] = visit_id_counter[sid]
                    visit_id_counter[sid] += 1
                    final_schedule[d].append(si)

        valid, mismatches = self.validate_schedule(final_schedule)
        if not valid:
            self.logger.warning("Final schedule validation failed. Mismatches:")
            self.logger.warning(mismatches)
        else:
            self.logger.info("Final schedule validation passed.")
            
        return final_schedule

    def validate_schedule(self, schedule):
        from collections import defaultdict
        actual_visits = defaultdict(int)
        for d, visits in schedule.items():
            for st in visits:
                sid = st['storeid']
                actual_visits[sid] += 1

        mismatches = {}
        for sid in self.all_store_ids:
            req = self.required_visits[sid]
            if actual_visits[sid] != req:
                mismatches[sid] = {
                    'required': req,
                    'actual': actual_visits[sid]
                }
        return (len(mismatches) == 0), mismatches


class PJPOptimizer:
    def __init__(
        self,
        distributor_id,
        plan_duration,
        num_orderbookers,
        store_type='both',
        retail_time=20,
        wholesale_time=40,
        holidays=None,
        logger=None,
        outlier_strategy='factor',
        existing_clusters=None,
        max_days=20,
        working_hours_per_day=8,
        custom_days=None,        # <--- New: Number of days if plan_duration != 'day'
        replicate=False,          # <--- New: Whether to replicate the final schedule 4x
        selected_stores=None
    ):
        """
        PJPOptimizer with robust scheduling. 
         - If plan_duration == 'day': only 1 day (Day1).
         - Else we use 'custom_days' as the base # of days: Day1..DayN.
         - We can replicate 4x if replicate=True (post-schedule).
        """
        self.distributor_id = distributor_id
        self.plan_duration = plan_duration.lower()
        self.num_orderbookers = num_orderbookers
        self.store_type = store_type.lower()
        self.logger = logger or self.setup_logger()
        self.holidays = holidays or []
        self.outlier_strategy = outlier_strategy
        self.max_days = max_days

        self.custom_days = custom_days if isinstance(custom_days, int) and custom_days > 0 else None
        self.replicate = replicate

        self.existing_clusters = existing_clusters if isinstance(existing_clusters, pd.DataFrame) else None
        self.working_minutes = working_hours_per_day * 60
        # Prepare initial day set
        if self.plan_duration == 'day':
            # single-day plan => just Day1
            self.available_days = ["Day1"]
        else:
            # custom multi-day plan => Day1..DayN
            if self.custom_days and self.custom_days > 0:
                self.available_days = [f"Day{i}" for i in range(1, self.custom_days + 1)]
            else:
                self.logger.error("Invalid or missing custom_days for multi-day schedule.")
                raise ValueError("Custom days must be a positive integer.")
        self.selected_stores = selected_stores if isinstance(selected_stores, list) else None

        # Fetch store data
        self.store_data = self.fetch_store_data()
        if self.store_data.empty:
            self.logger.warning("No valid store data found.")

        # Assign workload weights
        self.store_data['workload_weight'] = self.store_data['channeltypeid'].apply(
            lambda x: wholesale_time if x == 2 else retail_time
        )

        # Add service_time_by_id mapping
        self.service_time_by_id = {}
        for _, row in self.store_data.iterrows():
            self.service_time_by_id[row['storeid']] = row['workload_weight']

        if self.existing_clusters is not None and not self.existing_clusters.empty:
            # Use provided clusters
            self.logger.info("Using existing clusters provided by user. Skipping normal clustering steps.")
            self.store_data = pd.merge(
                self.store_data,
                self.existing_clusters[['storeid', 'cluster_id']],
                on='storeid',
                how='left',
                suffixes=('', '_user')
            )
            self.store_data['cluster'] = self.store_data['cluster_id']
            self.store_data['cluster'].fillna(-1, inplace=True)
            # --- Remove stores that were removed (i.e. cluster_id == -1)
            self.store_data = self.store_data[self.store_data['cluster'] != -1]
            self.store_data['cluster_before'] = self.store_data['cluster']
            self.store_data['cluster_after'] = self.store_data['cluster']

            self.initial_clusters = self.store_data['cluster'].unique()

            if not self.store_data.empty:
                self.store_to_index = {
                    store_id: idx for idx, store_id in
                    enumerate(self.store_data['storeid'].unique())
                }
                self.distance_matrix, self.travel_min = self.build_osrm_matrices(self.store_data)
            else:
                self.store_to_index = {}
                self.distance_matrix = None
                self.travel_min = None

            self.min_days_per_ob = self.compute_min_days_per_ob()
            if not self.store_data.empty:
                # We'll create a schedule next
                self.schedule = self.create_schedule()
                self.post_balance_schedule()
            else:
                self.schedule = {}
        else:
            # Normal flow
            self.initial_clusters = self.cluster_stores_geographically()
            self.store_data['cluster_before'] = self.store_data['cluster']

            if not self.store_data.empty:
                self.store_to_index = {
                    store_id: idx for idx, store_id in
                    enumerate(self.store_data['storeid'].unique())
                }
                self.distance_matrix, self.travel_min = self.build_osrm_matrices(self.store_data)
            else:
                self.store_to_index = {}
                self.distance_matrix = None
                self.travel_min = None

            self.balance_clusters()
            self.reassign_outlier_stores(strategy=self.outlier_strategy)

            # self.visualize_clusters(self.initial_clusters, "Initial Clusters", "initial_clusters.png")
            self.store_data['cluster_after'] = self.store_data['cluster']
            # self.visualize_clusters(self.initial_clusters, "Final Clusters", "final_clusters.png")

            if not self.store_data.empty:
                self.min_days_per_ob = self.compute_min_days_per_ob()
                self.schedule = self.create_schedule()
                self.post_balance_schedule()
                self.print_daily_work_times()
            else:
                self.schedule = {}


    def compute_min_days_per_ob(self):
        """
        Compute the minimum days required per OB based on 
        service_time + a realistic intra-cluster travel bound.
        """
        # 1) get each cluster's workload
        cluster_workloads = self.calculate_cluster_workloads('cluster')
        min_days = {}
        # 2) map sorted clusters → OB IDs
        sorted_clusters = sorted(cluster_workloads.keys())
        for idx, cid in enumerate(sorted_clusters):
            ob_id = idx + 1
            wl = cluster_workloads[cid]
            # ceil(total_minutes / working_minutes) 
            min_days[ob_id] = math.ceil(wl / self.working_minutes)
        return min_days   
    
    def post_balance_schedule(self,
                              util_floor_ratio: float = 0.5) -> None:
        """
        1) Remove days with no visits.
        2) Merge any two days whose combined load <= working_minutes,
           provided each day is under util_floor_ratio of working_minutes.
        This directly mutates self.schedule.
        """
        for ob_id, day_map in list(self.schedule.items()):
            # --- 1) remove empty days ---
            for day in list(day_map):
                if not day_map[day]:
                    del day_map[day]

            # helper to recompute utilization
            def day_util(d):
                store_ids = [v['storeid'] for v in day_map[d]]
                return self._compute_day_time(store_ids)

            merged = True
            while merged:
                merged = False
                # recompute & sort under‐utilized days
                utils = {d: day_util(d) for d in day_map}
                under = [d for d,u in utils.items()
                         if u < util_floor_ratio * self.working_minutes]
                # try every pair
                for i in range(len(under)):
                    for j in range(i+1, len(under)):
                        d1, d2 = under[i], under[j]
                        if utils[d1] + utils[d2] <= self.working_minutes:
                            # merge day d2 into d1
                            day_map[d1].extend(day_map[d2])
                            # re‐route & re‐optimize that one day
                            day_map[d1] = self._optimize_day_route(day_map[d1])
                            # delete the now‐redundant day
                            del day_map[d2]
                            merged = True
                            break
                    if merged:
                        break

            # finally, re‐index days to be consecutive if you like:
            # Day1, Day2, … in sorted order
            new_map = {}
            for idx, d in enumerate(sorted(day_map), start=1):
                new_map[f"Day{idx}"] = day_map[d]
            self.schedule[ob_id] = new_map

    def get_distance_data(self):
        """Returns both the distance matrix and store index map"""
        try:
            distance_matrix = self.distance_matrix
            store_index_map = self.store_to_index
            travel_min = self.travel_min
            return distance_matrix, store_index_map, travel_min
        
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None, None
        
        # ------------------------------------------------------------------
        #  OR‑Tools exact/near‑exact TSP for a single day
        # ------------------------------------------------------------------
    def _solve_tsp_ortools(self, store_ids: list[int], time_limit_sec: int = 3, fallback=None):
        """
        Solve an **open‑path** TSP (start = first store, end = last store).
        Falls back to `fallback` if the solver times out.
        """
        n = len(store_ids)
        if n <= 2:
            return store_ids[:]          # nothing to optimise

        # ── 1. build Routing index manager (open path) ──────────────────────
        starts = [0]                     # first store is the start
        ends   = [n - 1]                 # last store is the end
        manager = pywrapcp.RoutingIndexManager(n, 1, starts, ends)

        routing = pywrapcp.RoutingModel(manager)

        # distance callback (must return **int** cost)
        def dist_cb(from_idx, to_idx):
            a = store_ids[manager.IndexToNode(from_idx)]
            b = store_ids[manager.IndexToNode(to_idx)]
            return int(self.travel_min[self.store_to_index[a],
                                    self.store_to_index[b]] * 1000)  # milli‑minutes

        transit = routing.RegisterTransitCallback(dist_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit)

        # ── 2. search parameters ────────────────────────────────────────────
        params = pywrapcp.DefaultRoutingSearchParameters()
        
        # Use PATH_CHEAPEST_ARC with a better first solution strategy
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        
        # Add a more intensive local search metaheuristic
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        
        # Allow more time for higher quality solutions
        params.time_limit.FromSeconds(time_limit_sec)
        
        # Increase neighborhood size for better local search
        params.guided_local_search_lambda_coefficient = 0.1
        params.local_search_operators.use_relocate = pywrapcp.BOOL_TRUE
        params.local_search_operators.use_exchange = pywrapcp.BOOL_TRUE
        params.local_search_operators.use_two_opt = pywrapcp.BOOL_TRUE

        # ── 3. solve ────────────────────────────────────────────────────────
        solution = routing.SolveWithParameters(params)
        if solution:
            order = []
            idx = routing.Start(0)
            while not routing.IsEnd(idx):
                order.append(store_ids[manager.IndexToNode(idx)])
                idx = solution.Value(routing.NextVar(idx))
            order.append(store_ids[manager.IndexToNode(idx)])          # append end node
            return order

        # ── 4. fallback ─────────────────────────────────────────────────────
        if fallback is not None:
            return fallback(store_ids)
        return store_ids


        
    def _compute_day_time(self, store_list):
        """
        Travel minutes + service minutes using unified metric.
        """
        if not store_list:
            return 0.0
        route = self._nearest_neighbor_route(store_list)   # fast estimate
        travel = _route_cost(self.travel_min, self.store_to_index, route)
        service = sum(self.service_time_by_id[s] for s in route)
        return travel + service


    def setup_logger(self):
        logger = logging.getLogger('PJPOptimizer')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            file_handler = logging.FileHandler('pjp_optimizer.log')
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        return logger
    
    # Add this method to the PJPOptimizer class
    def _nearest_neighbor_route(self, store_list, restarts: int = 5):
        """
        Greedy NN with `restarts` random starts, keep the best.
        Caching is OFF during optimisation; post‑processing can
        wrap this call and cache on its side.
        """
        n = len(store_list)
        if n <= 1:
            return store_list[:]

        best_route = None
        best_cost  = float("inf")

        indices = [self.store_to_index[s] for s in store_list]

        for _ in range(restarts):
            unvisited = set(range(n))
            current_idx = random.choice(tuple(unvisited))
            route_idx   = [current_idx]
            unvisited.remove(current_idx)

            while unvisited:
                current_global = indices[current_idx]
                next_idx = min(unvisited,
                            key=lambda j: _edge_cost(self.travel_min,
                                                        current_global,
                                                        indices[j]))
                route_idx.append(next_idx)
                unvisited.remove(next_idx)
                current_idx = next_idx

            candidate = [store_list[i] for i in route_idx]
            c_cost    = _route_cost(self.travel_min, self.store_to_index, candidate)

            if c_cost < best_cost:
                best_cost  = c_cost
                best_route = candidate

        return best_route


    def fetch_store_data(self):
        """
        Grab store data from DB, filtering for valid lat/long and channeltypeid ∈ {1,2}.
        """
        self.logger.info(f"Fetching store data for distributor_id {self.distributor_id} with store_type '{self.store_type}'...")
        try:
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
            """

            if self.store_type == 'wholesale':
                query += " AND sc.channeltypeid = 2"
            elif self.store_type == 'retail':
                query += " AND sc.channeltypeid != 2"

            data = fetch_data(query, (self.distributor_id,))
            if not data or len(data) == 0:
                self.logger.warning(f"No store data found for distributor_id {self.distributor_id}, store_type '{self.store_type}'.")
                return pd.DataFrame()

            df = pd.DataFrame(data, columns=[
                "storeid", "latitude", "longitude", "channeltypeid", "storecode"
            ])
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            initial_count = len(df)
            df.dropna(subset=['latitude', 'longitude'], inplace=True)
            # Only keep selected stores if specified
            if self.selected_stores:
                df = df[df['storeid'].isin(self.selected_stores)]
                self.logger.info(f"Filtered store data to {len(df)} selected stores.")

            final_count = len(df)
            if final_count < initial_count:
                self.logger.warning(f"Dropped {initial_count - final_count} stores due to invalid coordinates.")

            valid_channels = [1, 2]
            df = df[df['channeltypeid'].isin(valid_channels)]
            self.logger.info(f"Fetched and validated {len(df)} records for distributor_id {self.distributor_id} with store_type '{self.store_type}'.")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching store data: {e}")
            return pd.DataFrame()

    # -------------------------- Clustering ------------------------------
    def haversine(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r
    

    def optimize_geographic_cohesion(self, schedule):
        """
        Post-process the schedule to further enhance geographic cohesion.
        """
        self.logger.info("Optimizing geographic cohesion of the schedule...")
        
        # For each orderbooker's schedule
        for ob_id, day_map in schedule.items():
            # Process each day
            for day, visits in day_map.items():
                store_ids = [v["storeid"] for v in visits]
                
                if len(store_ids) <= 1:
                    continue
                    
                # Use your existing OR-Tools TSP solver for efficient routing
                optimized_route = self._solve_tsp_ortools(
                    store_ids, 
                    time_limit_sec=5,
                    fallback=self._nearest_insertion_route
                )
                
                # Update the visits in the optimal route order
                store_to_visit = {v["storeid"]: v for v in visits}
                schedule[ob_id][day] = [store_to_visit[sid] for sid in optimized_route]
        
        self.logger.info("Geographic cohesion optimization completed.")
        return schedule
    
    # ------------------------------------------------------------------
    #  Post–scheduling route optimisation (uses OR‑Tools whenever possible)
    # ------------------------------------------------------------------
    def optimize_routes_post_scheduling(self, tsp_limit: int = 300) -> dict:
        """
        Iterate over every OB‑day pair and reorder the visits so that total
        travel time is (near‑)minimal.

        Parameters
        ----------
        tsp_limit : int, default 100
            If a day contains ≤ `tsp_limit` stops, the exact OR‑Tools solver
            is invoked (with a 5‑second time limit).  
            For larger tours we fall back to the fast nearest‑insertion + 2‑opt
            heuristic already present in the class.

        Returns
        -------
        dict
            The updated schedule (same structure: {ob_id: {day: [visits…]}}).
        """
        self.logger.info(
            "Running postscheduling route optimisation ORTools threshold "
            f"set to {tsp_limit} stops."
        )

        # Simple in‑memory cache so identical stop‑sets aren’t re‑solved
        if not hasattr(self, "_post_process_cache"):
            self._post_process_cache: dict[tuple[int, ...], list[int]] = {}

        for ob_id, days in self.schedule.items():
            for day, visits in days.items():

                # Nothing to optimise for 0‑ or 1‑stop days
                if len(visits) <= 1:
                    continue

                store_ids = [v["storeid"] for v in visits]
                route_key = tuple(sorted(store_ids))
                store_map = {v["storeid"]: v for v in visits}

                # ── 1. reuse cached tour ──────────────────────────────────
                if route_key in self._post_process_cache:
                    ordered = self._post_process_cache[route_key]

                # ── 2. solve with OR‑Tools if tour is small enough ────────
                elif len(store_ids) <= tsp_limit:
                    print("Solving TSP with ORTools...")
                    logging.info(f"Solving TSP with ORTools for {len(store_ids)} stores.")
                    ordered = self._solve_tsp_ortools(
                        store_ids,
                        time_limit_sec=5,                 # tighten / loosen as you wish
                        fallback=self._nearest_insertion_route,
                    )
                    self._post_process_cache[route_key] = ordered

                # ── 3. fall back to the existing NI + 2‑opt heuristic ─────
                else:
                    ordered = self._nearest_insertion_route(store_ids)
                    self._post_process_cache[route_key] = ordered

                # Write back in the new order
                self.schedule[ob_id][day] = [store_map[sid] for sid in ordered]
        self.schedule = self.optimize_geographic_cohesion(self.schedule)
        self.logger.info("Postscheduling route optimisation completed.")
        return self.schedule


    def cluster_stores_geographically(self):
        if self.store_data.empty:
            return []

        self.logger.info("Clustering stores based on geographic proximity.")
        proximity_threshold_km = INITIAL_PROXIMITY_THRESHOLD_KM

        while True:
            self.logger.info(f"Using proximity threshold: {proximity_threshold_km:.2f} km")

            G = nx.Graph()
            stores = self.store_data[['storeid', 'latitude', 'longitude']].values
            store_ids = self.store_data['storeid'].tolist()
            G.add_nodes_from(store_ids)

            for i in range(len(stores)):
                for j in range(i + 1, len(stores)):
                    s1 = stores[i]
                    s2 = stores[j]
                    dist = self.haversine(s1[2], s1[1], s2[2], s2[1])
                    if dist <= proximity_threshold_km:
                        G.add_edge(s1[0], s2[0])

            connected_components = list(nx.connected_components(G))
            num_cc = len(connected_components)

            if num_cc >= self.num_orderbookers:
                break
            else:
                proximity_threshold_km -= PROXIMITY_DECREASE_FACTOR
                if proximity_threshold_km < MIN_PROXIMITY_THRESHOLD_KM:
                    msg = (
                        f"Cannot form {self.num_orderbookers} clusters even at "
                        f"min threshold {MIN_PROXIMITY_THRESHOLD_KM} km."
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)
                self.logger.info(f"Reducing proximity threshold and retrying: {proximity_threshold_km:.2f}")

        group_mapping = {}
        for group_id, component in enumerate(connected_components):
            for store_id in component:
                group_mapping[store_id] = group_id
        self.store_data['group'] = self.store_data['storeid'].map(group_mapping)

        # KMeans
        group_centroids = self.store_data.groupby('group')[['latitude', 'longitude']].mean().reset_index()
        try:
            kmeans = KMeans(n_clusters=self.num_orderbookers, random_state=42)
            group_centroids['cluster'] = kmeans.fit_predict(group_centroids[['latitude', 'longitude']])
            group_centroids['cluster'] +=1
        except Exception as e:
            self.logger.error(f"Error during KMeans: {e}")
            raise

        group_to_cluster = dict(zip(group_centroids['group'], group_centroids['cluster']))
        self.store_data['cluster'] = self.store_data['group'].map(group_to_cluster)
        self.logger.info("Geographic clustering completed.")
        return self.store_data['cluster'].unique()

    def estimate_intra_cluster_travel_time(self, cluster_id):
        """
        Lower‐bound = max( MST*2 , quick NN‐route cost )
        """
        if self.distance_matrix is None:
            return 0.0

        # gather stores in this cluster
        c_stores = self.store_data[self.store_data['cluster'] == cluster_id]
        store_ids = c_stores['storeid'].tolist()
        if len(store_ids) <= 1:
            return 0.0

        # --- MST estimate ---
        indices = [ self.store_to_index[sid] for sid in store_ids ]
        sub = self.distance_matrix[np.ix_(indices, indices)]
        G = nx.Graph()
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                G.add_edge(i, j, weight=sub[i, j])
        mst = nx.minimum_spanning_tree(G)
        mst_sum = sum(data['weight'] for _,_,data in mst.edges(data=True))
        mst_est = mst_sum * 2

        # --- Quick NN route estimate ---
        nn_route = self._nearest_neighbor_route(store_ids)
        nn_cost = 0.0
        for i in range(len(nn_route) - 1):
            a = self.store_to_index[nn_route[i]]
            b = self.store_to_index[nn_route[i+1]]
            nn_cost += self.distance_matrix[a, b]

        # return the tighter lower bound
        return max(mst_est, nn_cost)


    # -------------------------- Balancing ------------------------------
    def define_workload_bounds(self, cluster_workloads, tolerance=0.01):
        total_workload = sum(cluster_workloads.values())
        n_clusters = len(cluster_workloads)
        if n_clusters == 0:
            return 0, 0
        avg_w = total_workload / n_clusters
        min_w = avg_w * (1 - tolerance)
        max_w = avg_w * (1 + tolerance)
        return min_w, max_w

    def identify_imbalanced_clusters(self, cluster_workloads, min_w, max_w):
        over = {c: w for c, w in cluster_workloads.items() if w > max_w}
        under = {c: w for c, w in cluster_workloads.items() if w < min_w}
        return over, under

    def log_cluster_workloads(self, cluster_workloads, message):
        self.logger.info(message)
        for c, w in cluster_workloads.items():
            self.logger.info(f"Cluster {c}: Combined Workload = {w:.2f}")

    def calculate_cluster_workloads(self, cluster_column):
        cluster_workloads = {}
        unique_clusters = self.store_data[cluster_column].unique()
        for c_id in unique_clusters:
            
            service_wl = self.store_data[self.store_data[cluster_column] == c_id]['workload_weight'].sum()
            travel_time = self.estimate_intra_cluster_travel_time(c_id)
            combined = float(service_wl + travel_time)
            cluster_workloads[int(c_id)] = combined
        return cluster_workloads        

    def balance_clusters(self, tolerance=0.01):
        """
        Attempt to reassign some stores among clusters to even out workloads.
        """
        self.logger.info("Balancing workloads among clusters...")

        cluster_workloads = self.calculate_cluster_workloads('cluster')
        total_workload = sum(cluster_workloads.values())
        n_clusters = len(cluster_workloads)
        avg_w = total_workload / n_clusters if n_clusters > 0 else 0
        min_w, max_w = self.define_workload_bounds(cluster_workloads, tolerance)
        over, under = self.identify_imbalanced_clusters(cluster_workloads, min_w, max_w)

        self.log_cluster_workloads(cluster_workloads, "Before balancing:")
        iteration = 0
        max_iterations = 100

        while over and under and iteration < max_iterations:
            self.logger.info(f"Balancing Iteration {iteration+1}")
            moved_any = False
            cluster_workloads = self.calculate_cluster_workloads('cluster')
            total_workload = sum(cluster_workloads.values())
            n_clusters = len(cluster_workloads)
            avg_w = total_workload / n_clusters if n_clusters > 0 else 0
            over, under = self.identify_imbalanced_clusters(cluster_workloads, min_w, max_w)

            for over_c in list(over.keys()):
                over_stores = self.store_data[self.store_data['cluster'] == over_c]
                for under_c in list(under.keys()):
                    under_stores = self.store_data[self.store_data['cluster'] == under_c]
                    if under_stores.empty:
                        continue
                    under_centroid = under_stores[['latitude', 'longitude']].mean().values

                    over_stores = over_stores.copy()
                    over_stores['distance_to_under'] = over_stores.apply(
                        lambda row: self.haversine(row['longitude'], row['latitude'], under_centroid[1], under_centroid[0]),
                        axis=1
                    )
                    sorted_stores = over_stores.sort_values('distance_to_under')

                    for _, store in sorted_stores.iterrows():
                        s_id = store['storeid']
                        w = store['workload_weight']
                        current_over_w = cluster_workloads[over_c]
                        current_under_w = cluster_workloads[under_c]
                        new_over_w = current_over_w - w
                        new_under_w = current_under_w + w
                        current_error = (current_over_w - avg_w)**2 + (current_under_w - avg_w)**2
                        new_error = (new_over_w - avg_w)**2 + (new_under_w - avg_w)**2
                        if new_error < current_error:
                            self.store_data.loc[self.store_data['storeid'] == s_id, 'cluster'] = under_c
                            self.logger.info(
                                f"Moved store {s_id} from cluster {over_c} to {under_c} (error reduced {current_error:.2f} => {new_error:.2f})."
                            )
                            moved_any = True
                            cluster_workloads[over_c] = new_over_w
                            cluster_workloads[under_c] = new_under_w
                            break

                if moved_any:
                    break

            if not moved_any:
                self.logger.info("No beneficial moves found in this iteration.")
                break

            iteration += 1
            min_w, max_w = self.define_workload_bounds(cluster_workloads, tolerance)
            over, under = self.identify_imbalanced_clusters(cluster_workloads, min_w, max_w)

        self.log_cluster_workloads(cluster_workloads, "After balancing:")
        if iteration == max_iterations and over:
            self.logger.warning("Max balancing iterations reached. Some clusters may still be imbalanced.")
        self.logger.info("Workload balancing completed.")

    # -------------------------- Outlier Reassignment ------------------------
    def reassign_outlier_stores(self, outlier_factor=2.0, strategy='factor'):
        self.logger.info(f"Reassigning outlier stores with strategy='{strategy}'.")
        if self.store_data.empty:
            return
        centroids = self.store_data.groupby('cluster')[['latitude', 'longitude']].mean()

        for c_id in self.store_data['cluster'].unique():
            c_stores = self.store_data[self.store_data['cluster'] == c_id]
            if c_stores.empty:
                continue
            centroid_lat = centroids.loc[c_id, 'latitude']
            centroid_lon = centroids.loc[c_id, 'longitude']

            distances = c_stores.apply(
                lambda row: self.haversine(row['longitude'], row['latitude'], centroid_lon, centroid_lat),
                axis=1
            )
            avg_dist = distances.mean()
            std_dist = distances.std()

            if strategy == 'std_dev':
                threshold = avg_dist + outlier_factor * std_dist
            else:
                threshold = avg_dist * outlier_factor

            outliers = c_stores[distances > threshold]

            for _, outlier_store in outliers.iterrows():
                best_cluster = self.find_best_cluster_for_outlier(outlier_store)
                if best_cluster is not None and best_cluster != c_id:
                    self.store_data.loc[self.store_data['storeid'] == outlier_store['storeid'], 'cluster'] = best_cluster
                    self.logger.info(
                        f"Store {outlier_store['storeid']} from cluster {c_id} => cluster {best_cluster} (outlier reassignment)."
                    )

        self.logger.info("Outlier reassignment completed.")

    def find_best_cluster_for_outlier(self, store_row):
        current_cluster = store_row['cluster']
        store_id = store_row['storeid']
        lat, lon = store_row['latitude'], store_row['longitude']

        self.store_data.loc[self.store_data['storeid'] == store_id, 'cluster'] = None

        best_cluster = current_cluster
        best_dist = float('inf')
        for c_id in self.store_data['cluster'].dropna().unique():
            c_stores = self.store_data[self.store_data['cluster'] == c_id]
            if c_stores.empty:
                continue
            clat = c_stores['latitude'].mean()
            clon = c_stores['longitude'].mean()
            d = self.haversine(lon, lat, clon, clat)
            if d < best_dist:
                best_dist = d
                best_cluster = c_id

        self.store_data.loc[self.store_data['storeid'] == store_id, 'cluster'] = current_cluster
        if best_cluster == current_cluster:
            return None
        else:
            return best_cluster

    # -------------------------- OSRM Matrix -------------------------------
    def build_osrm_matrices(self, df_stores):
        self.logger.info("Building OSRM distance & time matrices in small chunks...")

        store_ids = df_stores['storeid'].unique()
        coords = []
        for sid in store_ids:
            row = df_stores[df_stores['storeid'] == sid].iloc[0]
            coords.append((row['latitude'], row['longitude']))

        n = len(coords)
        distance_matrix = np.zeros((n, n), dtype=float)
        travel_min = np.zeros((n, n), dtype=float)

        chunk_size = 100
        self.logger.info(f"Using chunk_size={chunk_size}")

        for i_start in range(0, n, chunk_size):
            for j_start in range(0, n, chunk_size):
                i_end = min(i_start + chunk_size, n)
                j_end = min(j_start + chunk_size, n)

                block_i = list(range(i_start, i_end))
                block_j = list(range(j_start, j_end))

                union_indices = sorted(set(block_i) | set(block_j))
                source_indices = [union_indices.index(i) for i in block_i]
                dest_indices = [union_indices.index(j) for j in block_j]

                sub_coords_str = ";".join([
                    f"{coords[idx][1]},{coords[idx][0]}" for idx in union_indices
                ])

                if not source_indices or not dest_indices:
                    continue

                url = (
                    f"http://localhost:5002/table/v1/driving/{sub_coords_str}"
                    f"?sources={';'.join(map(str, source_indices))}"
                    f"&destinations={';'.join(map(str, dest_indices))}"
                    f"&annotations=distance,duration"
                )

                try:
                    r = requests.get(url)
                    r.raise_for_status()
                    data = r.json()

                    if 'distances' not in data or 'durations' not in data:
                        self.logger.error("OSRM response missing distances/durations.")
                        continue

                    dist_block = data['distances']
                    dur_block = data['durations']

                    for si, real_i in enumerate(block_i):
                        for sj, real_j in enumerate(block_j):
                            distance_matrix[real_i, real_j] = dist_block[source_indices.index(union_indices.index(real_i))][dest_indices.index(union_indices.index(real_j))] / 1000.0
                            travel_min[real_i, real_j] = dur_block[source_indices.index(union_indices.index(real_i))][dest_indices.index(union_indices.index(real_j))] / 60.0

                except Exception as e:
                    self.logger.error(f"Error fetching OSRM: {e}")

        return distance_matrix, travel_min

    # -------------------------- Visualization ------------------------------
    def generate_n_colors(self, n):
        hues = np.linspace(0, 1, n, endpoint=False)
        colors = [colorsys.hsv_to_rgb(h, 0.7, 0.9) for h in hues]
        hex_colors = [to_hex(c) for c in colors]
        return hex_colors

    def visualize_clusters(self, clusters, title, save_path, show_workload=False):
        if self.store_data.empty:
            self.logger.warning("No store data to visualize.")
            return

        if "Initial" in title:
            cluster_col = 'cluster_before'
        else:
            cluster_col = 'cluster_after'

        cluster_workloads = self.calculate_cluster_workloads(cluster_col)
        centroids = self.store_data.groupby(cluster_col)[['latitude', 'longitude']].mean().reset_index()

        num_clusters = len(clusters)
        colors = self.generate_n_colors(num_clusters)
        color_map = {cid: colors[i] for i, cid in enumerate(clusters)}

        plt.figure(figsize=(12, 8))
        for cid in clusters:
            cluster_data = self.store_data[self.store_data[cluster_col] == cid]
            plt.scatter(
                cluster_data['longitude'], cluster_data['latitude'],
                c=[color_map[cid]], alpha=0.6, edgecolor='k', s=50,
                label=f"Cluster {cid} (WL: {cluster_workloads.get(int(cid), 0):.2f})"
            )

        plt.title(title, fontsize=16)
        plt.xlabel("Longitude", fontsize=14)
        plt.ylabel("Latitude", fontsize=14)
        plt.grid(True)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, title="Clusters", fontsize=10, title_fontsize=12, loc='best')

        if show_workload:
            for _, row in centroids.iterrows():
                c_id = row[cluster_col]
                wl = cluster_workloads.get(int(c_id), 0)
                plt.text(row['longitude'], row['latitude'], f'W:{wl:.1f}',
                         fontsize=12, fontweight='bold', ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        plt.savefig(save_path)
        self.logger.info(f"Cluster visualization '{title}' saved to {save_path}")
        plt.close()

    # -------------------------- Scheduling ------------------------------

    def create_day_schedule(self):
        self.logger.info("Creating fast schedule for a single day plan (per cluster, using multiprocessing).")

        # Select the first available day
        day = self.available_days[0] if self.available_days else "Monday"

        # Initialize schedule
        schedule = {}

        # Group stores by cluster_id (since orderbooker_id is not available)
        cluster_stores = {
            int(cluster_id): self.store_data[self.store_data["cluster"] == cluster_id].to_dict("records")
            for cluster_id in self.store_data["cluster"].unique()
        }

        # Optimize routes in parallel using multiprocessing
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_cluster = {
                executor.submit(self._optimize_day_route, stores): cluster_id
                for cluster_id, stores in cluster_stores.items() if stores
            }

            for future in concurrent.futures.as_completed(future_to_cluster):
                cluster_id = future_to_cluster[future]
                try:
                    schedule[cluster_id] = {day: future.result()}
                    self.logger.info(f"Created Schedule for {cluster_id} ")
                except Exception as e:
                    self.logger.error(f"Error optimizing route for cluster {cluster_id}: {e}")
                    schedule[cluster_id] = {day: []}

        self.logger.info("Single day scheduling completed (optimized per cluster).")
        return schedule
    
    def create_schedule(self):
        """
        - Single‐day if plan_duration == 'day'
        - Otherwise: start from the strict lower‐bounds, then grow per‐OB only as needed
        """
        self.logger.info(f"Creating schedule. plan_duration={self.plan_duration}, replicate={self.replicate}")
        if self.plan_duration == 'day':
            sched = self.create_day_schedule()
            self.schedule = sched
            # self.optimize_routes_post_scheduling()
            return sched

        # --- Multi‐day branch ---
        # how many extra beyond lower‐bound we’re allowed
        max_extra = max(0, self.max_days - max(self.min_days_per_ob.values()))
        # initial per‐OB days = max(custom_days, lower_bound)
        initial_days_per_ob = {
            ob: [f"Day{i}" for i in range(1, max(self.custom_days or 0, self.min_days_per_ob[ob]) + 1)]
            for ob in range(1, self.num_orderbookers + 1)
        }
        self.logger.info(f"  Starting days_per_ob={initial_days_per_ob}, max_extra={max_extra}")

        sched, final_days = self._build_feasible_schedule(initial_days_per_ob, max_extra)
        self.updated_custom_days = max(len(d) for d in final_days.values())
        self.schedule = sched

        self.optimize_routes_post_scheduling()
        if self.replicate:
            # replicate using one OB’s final days (all are same length logic)
            self.schedule = self._replicate_schedule(self.schedule, list(final_days.values())[0])

        return self.schedule

        

    def _build_feasible_schedule(self, days_per_ob, max_extra_days=7):
        """
        Attempt schedules, adding days only if:
          • 2+ consecutive failures, OR
          • overload > 5% of DAILY_LIMIT
        """
        used_extra = {ob: 0 for ob in days_per_ob}
        infeas = {ob: {'fails': 0, 'over': 0.0} for ob in days_per_ob}
        overload_floor = 0.10 * self.working_minutes  # 5% of 480 = 24min

        while True:
            sched = self._attempt_schedule_flexible(days_per_ob)

            infeasible_obs = []
            for ob, days in days_per_ob.items():
                ov, freq_mismatch = self._get_ob_infeasibility_details(sched[ob], days)
                if ov > overload_floor or freq_mismatch:
                    infeasible_obs.append(ob)
                    infeas[ob]['fails'] += 1
                    infeas[ob]['over'] += ov
                else:
                    infeas[ob]['fails'] = 0
                    infeas[ob]['over'] = 0.0

            if not infeasible_obs:
                self.logger.info("All OBs feasible—done.")
                return sched, days_per_ob

            worst = max(infeasible_obs, key=lambda o: (infeas[o]['fails'], infeas[o]['over']))
            f = infeas[worst]['fails']
            ov = infeas[worst]['over']
            self.logger.info(f"OB {worst} infeasible: fails={f}, overload={ov:.1f}min (floor={overload_floor:.1f}min)")

            # now only add if 2+ fails OR true overload above floor
            if (f >= 2 or ov > overload_floor) and used_extra[worst] < max_extra_days:
                new_day = f"Day{len(days_per_ob[worst]) + 1}"
                days_per_ob[worst].append(new_day)
                used_extra[worst] += 1
                infeas[worst]['fails'] = 0
                self.logger.info(f" Added {new_day} for OB {worst}")
                continue

            self.logger.warning("No further days can be added (max reached or criteria not met).")
            return sched, days_per_ob




    def _get_ob_infeasibility_details(self, ob_schedule, ob_days):
        """
        Check schedule feasibility with detailed metrics.
        Returns: (total_time_overload, has_frequency_mismatch)
        """
        # Check daily time limit
        total_time_overload = 0
        for day in ob_days:
            visits = ob_schedule.get(day, [])
            store_ids = [v['storeid'] for v in visits]
            total_time = self._compute_day_time(store_ids)
            if total_time > self.working_minutes:
                total_time_overload += (total_time - self.working_minutes)
                
        # Check store frequencies
        visit_counts = defaultdict(int)
        for day, visits in ob_schedule.items():
            for visit in visits:
                visit_counts[visit['storeid']] += 1
                
        # Get required frequencies for this OB's stores
        freq_mismatch = False
        ob_stores = set(visit['storeid'] for visits in ob_schedule.values() for visit in visits)
        for store_id in ob_stores:
            req_freq = self.store_data.loc[
                self.store_data['storeid'] == store_id, 
                'channeltypeid'
            ].values[0]
            if visit_counts[store_id] != req_freq:
                freq_mismatch = True
                break
                
        return total_time_overload, freq_mismatch        


    
    # ------------------------------------------------------------------------


    def _is_ob_schedule_feasible(self, ob_schedule, ob_days):
        """Check if schedule is feasible for specific OB."""
        
        # Check daily time limit
        for day in ob_days:
            visits = ob_schedule.get(day, [])
            # total_time = self._compute_day_time(visits)
            total_time = self._compute_day_time([v['storeid'] for v in visits])
            if total_time > self.working_minutes:
                return False
                
        # Check store frequencies
        visit_counts = defaultdict(int)
        for day, visits in ob_schedule.items():
            for visit in visits:
                visit_counts[visit['storeid']] += 1
                
        # Get required frequencies for this OB's stores
        ob_stores = set(visit['storeid'] for visits in ob_schedule.values() for visit in visits)
        for store_id in ob_stores:
            req_freq = self.store_data.loc[
                self.store_data['storeid'] == store_id, 
                'channeltypeid'
            ].values[0]
            if visit_counts[store_id] != req_freq:
                return False
                
        return True

    def _attempt_schedule_flexible(self, days_per_ob):
        """Create schedule with different days per OB."""
        
        schedule = {}
        cluster_ids = list(self.store_data['cluster'].unique())
        
        results = {}
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {}
            for cid in cluster_ids:
                ob_id = cluster_ids.index(cid) + 1
                futures[cid] = executor.submit(
                    self.optimize_cluster,
                    cid=cid,
                    day_list=days_per_ob[ob_id]
                )
                
            for cid, future in futures.items():
                results[cid] = future.result()
        
        # Map clusters to OBs
        cluster_ids_sorted = sorted(cluster_ids)
        for i, cid in enumerate(cluster_ids_sorted):
            ob_id = i + 1
            cluster_sched = results[cid]
            schedule[ob_id] = {}
            for d in days_per_ob[ob_id]:
                schedule[ob_id][d] = list(cluster_sched.get(d, []))
                
        return schedule
    

    def _count_schedule_mismatches(self, schedule):
        from collections import defaultdict
        actual_visits = defaultdict(int)
        freq_map = {}
        for row in self.store_data.itertuples():
            freq_map[row.storeid] = row.channeltypeid

        for ob_id, day_map in schedule.items():
            for day, visits in day_map.items():
                for v in visits:
                    sid = v['storeid']
                    actual_visits[sid] += 1

        mismatch_count = 0
        for store_id, req_count in freq_map.items():
            act_count = actual_visits.get(store_id, 0)
            if act_count != req_count:
                mismatch_count += 1
        return mismatch_count

    def optimize_cluster(self, cid, day_list):
        """
        Run EA for a single cluster => day->stores
        """
        cl_stores = self.store_data[self.store_data['cluster'] == cid]
        if cl_stores.empty:
            return {d: [] for d in day_list}

        ea_scheduler = EAScheduleOptimizer(
            store_data=cl_stores,
            available_days=day_list,
            travel_min=self.travel_min,
            distance_matrix=self.distance_matrix,
            store_to_index=self.store_to_index,
            working_minutes=  self.working_minutes,
            logger=self.logger
        )

        best_sched = ea_scheduler.optimize_schedule(
            population_size=30,
            generations=15,
            mutation_rate=0.25,
            patience=10,
            max_time=120
        )
        return best_sched

    def _replicate_schedule(self, base_schedule, days_per_ob):
        """
        Replicate each OB's final feasible schedule 4 times (original + 3 copies).
        Uses OB-specific base days from `days_per_ob`.
        """
        self.logger.info("Replicating the schedule 4x from the base feasible schedule.")
        all_ob_ids = sorted(base_schedule.keys())
        final_schedule = {}

        for ob_id in all_ob_ids:
            ob_days = days_per_ob[ob_id]
            final_schedule[ob_id] = {}

            # Original days
            for d in ob_days:
                final_schedule[ob_id][d] = copy.deepcopy(base_schedule[ob_id].get(d, {}))

            base_count = len(ob_days)

            for replicate_idx in range(1, 4):  # 3 more times
                offset = replicate_idx * base_count
                for d in ob_days:
                    old_day_num = int(d.replace("Day", ""))
                    new_day_num = old_day_num + offset
                    new_day_key = f"Day{new_day_num}"

                    if d in base_schedule[ob_id]:
                        final_schedule[ob_id][new_day_key] = copy.deepcopy(base_schedule[ob_id][d])

        return final_schedule
    
    def _two_opt(self, route):
        """Deterministic 2‑opt until no improving swap exists."""
        n = len(route)
        if n <= 3:
            return route

        improved = True
        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 2, n):
                    if j - i == 1:       # adjacent
                        continue
                    # cost before swap
                    a, b = route[i - 1], route[i]
                    c, d = route[j - 1], route[j % n]
                    old = (_edge_cost(self.travel_min, self.store_to_index[a], self.store_to_index[b]) +
                        _edge_cost(self.travel_min, self.store_to_index[c], self.store_to_index[d]))
                    new = (_edge_cost(self.travel_min, self.store_to_index[a], self.store_to_index[c]) +
                        _edge_cost(self.travel_min, self.store_to_index[b], self.store_to_index[d]))
                    if new < old:
                        route[i:j] = reversed(route[i:j])
                        improved = True
            # loop continues until no swap improves distance
        return route

    
    def _nearest_insertion_route(self, store_list, restarts: int = 3):
        """
        Fast NI with adaptive position‑sampling & full 2‑opt at the end.
        """
        if len(store_list) <= 2:
            return store_list[:]

        best_route = None
        best_cost  = float("inf")
        idx_map = [self.store_to_index[s] for s in store_list]
        m       = len(store_list)

        for _ in range(restarts):
            unvisited = set(range(m))
            start = random.choice(tuple(unvisited))
            route_idx = [start]
            unvisited.remove(start)

            # add the closest second point
            if unvisited:
                second = min(unvisited,
                            key=lambda j: _edge_cost(self.travel_min,
                                                    idx_map[start],
                                                    idx_map[j]))
                route_idx.append(second)
                unvisited.remove(second)

            # main NI loop — adaptive sample size
            while unvisited:
                best_gain = float("inf")
                best_j    = None
                best_pos  = None

                for j in unvisited:
                    # test at a subset of positions
                    sample_size = max(len(route_idx) // 3, 20)
                    positions   = (range(len(route_idx))
                                if len(route_idx) <= sample_size
                                else random.sample(range(len(route_idx)),
                                                    sample_size))

                    for i in positions:
                        nxt = (i + 1) % len(route_idx)
                        gain = (_edge_cost(self.travel_min, idx_map[route_idx[i]], idx_map[j]) +
                                _edge_cost(self.travel_min, idx_map[j],               idx_map[route_idx[nxt]]) -
                                _edge_cost(self.travel_min, idx_map[route_idx[i]],   idx_map[route_idx[nxt]]))
                        if gain < best_gain:
                            best_gain = gain
                            best_j    = j
                            best_pos  = nxt

                route_idx.insert(best_pos, best_j)
                unvisited.remove(best_j)

            candidate = [store_list[i] for i in route_idx]
            candidate = self._two_opt(candidate)        # full refinement
            c_cost    = _route_cost(self.travel_min, self.store_to_index, candidate)

            if c_cost < best_cost:
                best_cost  = c_cost
                best_route = candidate

        return best_route


        # ------------------------------------------------------------------
    #  Wrapper that now calls the OR‑Tools solver
    # ------------------------------------------------------------------
    def _optimize_day_route(self, visits, ortools_limit=300):
        """
        Given a list of visit dicts for one day, reorder them so that
        travel time is (near‑)minimal.
        • Uses OR‑Tools if len(visits) ≤ `ortools_limit`;
          otherwise falls back to nearest‑insertion + 2‑opt.
        """
        if len(visits) <= 1:
            return visits

        store_ids = [v['storeid'] for v in visits]

        # Choose strategy
        if len(store_ids) <= ortools_limit:
            print("Using OR‑Tools for TSP...")
            ordered = self._solve_tsp_ortools(
                store_ids,
                fallback=self._nearest_insertion_route
            )
        else:                                   # huge tour – stay heuristic
            ordered = self._nearest_insertion_route(store_ids)

        # Re‑assemble the visit dicts in the new order
        lookup = {v['storeid']: v for v in visits}
        return [lookup[sid] for sid in ordered]


    def reroute_day(self, affected_cluster_ids, removed_store_ids):
        """
        Reroutes affected clusters for a single-day plan by:
        - Removing visits for removed_store_ids.
        - Reordering the remaining visits using `_optimize_day_route`.
        """
        self.logger.info(f"Starting rerouting for affected clusters: {affected_cluster_ids} (single-day plan)...")

        if "cluster" not in self.store_data.columns:
            raise ValueError("Error: 'cluster' column is missing in self.store_data!")
        if self.schedule is None:
            raise ValueError("Error: No existing schedule to reroute from. Please run an initial schedule first.")

        # Fetch the only available day
        if self.plan_duration == "day":
            day = self.available_days[0]  # Single-day (e.g., "Day1")
        else:
            raise ValueError("This reroute function is designed for single-day plans only.")

        updated_schedule = copy.deepcopy(self.schedule)

        # Remove affected stores from the current schedule for the single day
        for cluster_id in affected_cluster_ids:
            if cluster_id in updated_schedule and day in updated_schedule[cluster_id]:
                updated_schedule[cluster_id][day] = [
                    visit for visit in updated_schedule[cluster_id][day] if visit['storeid'] not in removed_store_ids
                ]

        self.logger.info(f"Reordering remaining visits for clusters {affected_cluster_ids} on {day}...")

        # Optimize visit order using `_optimize_day_route`
        for cluster_id, day_map in updated_schedule.items():
            if day in day_map and day_map[day]:
                updated_schedule[cluster_id][day] = self._optimize_day_route(day_map[day])

        self.schedule = updated_schedule
        self.logger.info(f"Single-day rerouting completed for clusters {affected_cluster_ids}.")
        return self.schedule

    
    def reroute(self, affected_cluster_ids, removed_store_ids):
        """
        Recalculate schedules for 'affected' clusters, removing 'removed_store_ids'.
        """
        self.logger.info(f"Starting rerouting for affected clusters: {affected_cluster_ids}...")
        if self.plan_duration == 'day':
            self.schedule = self.reroute_day(affected_cluster_ids, removed_store_ids)
        else:
            if "cluster" not in self.store_data.columns:
                raise ValueError("Error: 'cluster' column is missing in self.store_data!")
            if self.schedule is None:
                raise ValueError("Error: No existing schedule to reroute from. Please run an initial schedule first.")

            affected_clusters = self.store_data[
                (self.store_data['cluster'].isin(affected_cluster_ids)) & (~self.store_data['storeid'].isin(removed_store_ids))
            ]

            if affected_clusters.empty:
                self.logger.info("No stores found for the given affected clusters. Nothing to reroute.")
                return {"schedule": self.schedule}

            new_cluster_schedule = {}
            
            self.logger.info(f"Removing existing visits for clusters {affected_cluster_ids} from current schedule.")
            affected_storeids = set(affected_clusters['storeid'].tolist())
            updated_schedule = copy.deepcopy(self.schedule)

            # remove from existing schedule
            for ob_id, day_map in updated_schedule.items():
                for day, visits in day_map.items():
                    filtered_visits = [v for v in visits if v['storeid'] not in affected_storeids]
                    updated_schedule[ob_id][day] = filtered_visits

            self.logger.info(f"Re-scheduling stores for clusters {affected_cluster_ids}...")
            # figure out the day_list from the updated_schedule
            example_ob = next(iter(updated_schedule.keys()))
            day_list = list(updated_schedule[example_ob].keys())

            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = {}
                for cluster_id in affected_cluster_ids:
                    cluster_data = affected_clusters[affected_clusters['cluster'] == cluster_id]
                    if not cluster_data.empty:
                        futures[cluster_id] = executor.submit(
                            self._optimize_single_cluster_reroute,
                            cluster_data,
                            day_list
                        )
                    else:
                        new_cluster_schedule[cluster_id] = {d: [] for d in day_list}

                for cid, future in futures.items():
                    new_cluster_schedule[cid] = future.result()

            self.logger.info("Distributing new cluster schedules among orderbookers...")

            cluster_ids_sorted = sorted(new_cluster_schedule.keys())
            if len(cluster_ids_sorted) != self.num_orderbookers:
                self.logger.warning("Reroute: mismatch in cluster vs OB count. This might be an edge case.")

            final_schedule = copy.deepcopy(updated_schedule)  # start from the updated one

            # re-inject the newly scheduled visits for the affected clusters
            for i, cid in enumerate(cluster_ids_sorted):
                ob_id = i + 1
                cluster_sched = new_cluster_schedule[cid]
                for d in day_list:
                    final_schedule[ob_id][d].extend(cluster_sched.get(d, []))

            # route-optimize each OB/day
            for ob_id, days in final_schedule.items():
                for day, visits in days.items():
                    if visits:
                        final_schedule[ob_id][day] = self._optimize_day_route(visits)

            self.schedule = final_schedule
            self.logger.info(f"Dynamic rerouting completed for clusters {affected_cluster_ids}.")
        return {"schedule": self.schedule}

    def _optimize_single_cluster_reroute(self, cluster_data, day_list):
        ea_scheduler = EAScheduleOptimizer(
            store_data=cluster_data,
            available_days=day_list,
            travel_min=self.travel_min,
            distance_matrix=self.distance_matrix,
            store_to_index=self.store_to_index,
            logger=self.logger
        )
        best_sched = ea_scheduler.optimize_schedule(
            population_size=40,
            generations=60,
            mutation_rate=0.25,
            patience=10,
            max_time=120
        )
        for day, visits in best_sched.items():
            if visits:
                best_sched[day] = self._optimize_day_route(visits)
        return best_sched

    # -------------------------- Public API --------------------------------------
    def generate_pjp(self):
        """
        Always return 'schedule' in the same structure: pjp => {OB: {DayX: [stores...]}}

        """
        # Save PJP in DB
        pjp_saver = PJPDataSaver(self.schedule)
        if self.plan_duration == "day":
            plan_duration_int = 1
        else:
            plan_duration_int = self.updated_custom_days
        plan_id = pjp_saver.save_pjp(self.distributor_id, plan_duration_int)
        print(f"PJP data saved successfully under Plan ID: {plan_id}")  

        return self.schedule, plan_id

    def get_clusters_data(self):
        clusters_data = []
        if self.store_data.empty:
            return clusters_data

        cluster_col = 'cluster_after' if 'cluster_after' in self.store_data.columns else 'cluster'
        cw = self.calculate_cluster_workloads(cluster_col)

        for cid in sorted(self.store_data[cluster_col].unique()):
            c_stores = self.store_data[self.store_data[cluster_col] == cid]
            centroid_lat = c_stores['latitude'].mean()
            centroid_lon = c_stores['longitude'].mean()

            service_wl = c_stores['workload_weight'].sum()
            travel_t = self.estimate_intra_cluster_travel_time(cid)
            total_wl = service_wl + travel_t

            stores_list = c_stores[[
                'storeid', 'storecode', 'latitude', 'longitude', 'workload_weight'
            ]].to_dict('records')

            clusters_data.append({
                "cluster_id": int(cid),
                "centroid": {"latitude": float(centroid_lat), "longitude": float(centroid_lon)},
                "travel_time": float(travel_t),
                "service_workload": float(service_wl),
                "total_workload": float(total_wl),
                "stores": stores_list
            })

        return clusters_data
    
    def print_daily_work_times(self):
        if not self.schedule:
            return

        overall_travel_time = 0.0
        overall_service_time = 0.0
        overall_distance = 0.0

        print("\n=========== DAILY WORK SUMMARY ===========")
        for ob_id, day_schedule in self.schedule.items():
            print(f"\nOrder Booker {ob_id}:")

            ob_travel_time = 0.0
            ob_service_time = 0.0
            ob_distance = 0.0

            for day, visits in day_schedule.items():
                if visits:
                    # Don't recalculate route - use the existing order
                    store_ids = [visit['storeid'] for visit in visits]
                    
                    # If just one store, no travel needed
                    if len(store_ids) <= 1:
                        travel_time = 0.0
                        travel_distance = 0.0
                    else:
                        # Calculate travel time and distance using the existing order
                        travel_time = 0.0
                        travel_distance = 0.0
                        for i in range(len(store_ids) - 1):
                            idx1 = self.store_to_index[store_ids[i]]
                            idx2 = self.store_to_index[store_ids[i + 1]]
                            travel_time += self.travel_min[idx1, idx2]
                            travel_distance += self.distance_matrix[idx1, idx2]

                    # Calculate service time for each store
                    service_time = 0.0
                    for sid in store_ids:
                        service_time += self.service_time_by_id[sid]

                    total_time = travel_time + service_time

                    ob_travel_time += travel_time
                    ob_service_time += service_time
                    ob_distance += travel_distance

                    print(f"  {day}: Travel Time = {travel_time:.2f} min, "
                        f"Service Time = {service_time:.2f} min, "
                        f"Total Time = {total_time:.2f} min, "
                        f"Travel Distance = {travel_distance:.2f} km")
                else:
                    print(f"  {day}: No visits scheduled.")

            ob_total_time = ob_travel_time + ob_service_time
            print(f"\n>>> OB {ob_id} Totals:")
            print(f"    Travel Time   = {ob_travel_time:.2f} min ({ob_travel_time/60:.2f} hr)")
            print(f"    Service Time  = {ob_service_time:.2f} min ({ob_service_time/60:.2f} hr)")
            print(f"    Total Time    = {ob_total_time:.2f} min ({ob_total_time/60:.2f} hr)")
            print(f"    Distance      = {ob_distance:.2f} km")

            overall_travel_time += ob_travel_time
            overall_service_time += ob_service_time
            overall_distance += ob_distance

        overall_total_time = overall_travel_time + overall_service_time
        print("\n=========== OVERALL SUMMARY ===========")
        print(f"Total Travel Time   = {overall_travel_time:.2f} min ({overall_travel_time/60:.2f} hr)")
        print(f"Total Service Time  = {overall_service_time:.2f} min ({overall_service_time/60:.2f} hr)")
        print(f"Total Combined Time = {overall_total_time:.2f} min ({overall_total_time/60:.2f} hr)")
        print(f"Total Distance      = {overall_distance:.2f} km")



def validate_parameters(num_orderbookers, store_count):
    if num_orderbookers > store_count:
        raise ValueError("Number of orderbookers cannot exceed the number of stores.")
