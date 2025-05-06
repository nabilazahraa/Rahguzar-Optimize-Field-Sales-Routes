import numpy as np
import random
import logging
import time
from collections import defaultdict

class EAScheduleOptimizer:
    def __init__(
        self,
        store_data,
        available_days,
        time_matrix,
        distance_matrix,  # needed for geographic penalty
        store_to_index,
        logger=None
    ):
        """
        Evolutionary Algorithm (EA) for scheduling:
          - Minimizes total daily travel + service time
          - Enforces an 8-hour daily limit (480 minutes) as a hard constraint
          - Ensures each store's required visit frequency
          - Encourages balanced distribution across days
          - Adds a small penalty for separating geographically close single-frequency stores
        """
        self.store_data = store_data
        self.available_days = available_days
        self.time_matrix = time_matrix
        self.distance_matrix = distance_matrix
        self.store_to_index = store_to_index
        self.logger = logger or logging.getLogger("EAScheduleOptimizer")

        self.all_store_ids = set(self.store_data['storeid'].unique())

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

        self.DAILY_LIMIT = 480.0  # minutes => 8 hours

        # Precompute small set of neighbors for single-frequency stores
        self.neighbors_map = defaultdict(list)
        if self.distance_matrix is not None:
            single_freq_stores = [s for s in self.all_store_ids if self.required_visits[s] == 1]
            for s in single_freq_stores:
                s_idx = self.store_to_index[s]
                distances = []
                for t in single_freq_stores:
                    if t != s:
                        t_idx = self.store_to_index[t]
                        dist_val = self.distance_matrix[s_idx, t_idx]
                        distances.append((t, dist_val))
                distances.sort(key=lambda x: x[1])
                top_5 = [x[0] for x in distances[:5]]
                self.neighbors_map[s] = top_5

    # ---------------------------------------------------------------------
    # 1) Population Initialization
    # ---------------------------------------------------------------------
    def initialize_population(self, population_size):
        """
        Create a mixed population: half random, half geographic-based heuristic.
        """
        half_size = population_size // 2
        pop = []
        # half from random
        pop.extend(self._initialize_population_random(half_size))
        # half from geographic-based
        pop.extend(self._initialize_population_geographic(population_size - half_size))
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
        Sort stores by lat/lon, assign in round-robin manner across days.
        """
        population = []
        sorted_stores = sorted(
            list(self.all_store_ids),
            key=lambda s: (
                self.store_data.loc[self.store_data['storeid'] == s, 'latitude'].values[0],
                self.store_data.loc[self.store_data['storeid'] == s, 'longitude'].values[0]
            )
        )

        for _ in range(n_individuals):
            individual = {day: [] for day in self.available_days}
            day_index = 0
            for store_id in sorted_stores:
                freq = self.required_visits[store_id]
                d1 = self.available_days[day_index % len(self.available_days)]
                individual[d1].append(store_id)
                if freq > 1:
                    d2 = self.available_days[(day_index + 1) % len(self.available_days)]
                    individual[d2].append(store_id)
                day_index += 1
            population.append(individual)

        return population

    # ---------------------------------------------------------------------
    # 2) Evaluate Fitness
    # ---------------------------------------------------------------------
    def evaluate_fitness(self, individual):
        """
        1) Hard constraint: day_time > 480 => big penalty
        2) Weighted sum of total route time + mismatch penalty + day-balancing + geo penalty
        """
        from collections import defaultdict
        visit_counts = defaultdict(int)
        for day, stores in individual.items():
            for sid in stores:
                visit_counts[sid] += 1

        # mismatch penalty
        mismatch_penalty = 0.0
        for sid in self.all_store_ids:
            req = self.required_visits[sid]
            diff = abs(visit_counts[sid] - req)
            if diff > 0:
                mismatch_penalty += 5000 * diff

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

        # day-balancing penalty
        imbalance_penalty = 0.0
        LOWER_IDEAL = 360
        UPPER_IDEAL = 420
        for usage in day_usage:
            if usage < LOWER_IDEAL:
                imbalance_penalty += (LOWER_IDEAL - usage)
            elif usage > UPPER_IDEAL:
                imbalance_penalty += (usage - UPPER_IDEAL)

        # geographic penalty for single-freq neighbors on different days
        geographical_penalty = 0.0
        store_day = {}
        for d, s_list in individual.items():
            for sid in s_list:
                store_day[sid] = d

        for s, nb_list in self.neighbors_map.items():
            day_s = store_day.get(s, None)
            for nb in nb_list:
                day_nb = store_day.get(nb, None)
                if day_s is not None and day_nb is not None and day_s != day_nb:
                    geographical_penalty += 50.0

        fitness = total_time + mismatch_penalty + imbalance_penalty + geographical_penalty
        return fitness

    # ---------------------------------------------------------------------
    # 3) Local Search
    # ---------------------------------------------------------------------
    def local_search(self, individual, max_iterations=2):
        """
        A small greedy local search that tries moving a random store
        from one day to another if it improves fitness.
        """
        best_schedule = {d: list(stores) for d, stores in individual.items()}
        best_fitness = self.evaluate_fitness(best_schedule)

        for _ in range(max_iterations):
            days_list = list(best_schedule.keys())
            d1, d2 = random.sample(days_list, 2)
            if not best_schedule[d1]:
                continue

            store_to_move = random.choice(best_schedule[d1])
            best_schedule[d1].remove(store_to_move)
            best_schedule[d2].append(store_to_move)

            new_fitness = self.evaluate_fitness(best_schedule)
            if new_fitness < best_fitness:
                best_fitness = new_fitness
            else:
                # revert
                best_schedule[d2].remove(store_to_move)
                best_schedule[d1].append(store_to_move)

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
        Random store moves, random day swap, then fix over/under capacity.
        """
        new_ind = {d: list(stores) for d, stores in individual.items()}
        days_list = list(new_ind.keys())

        # random store-level moves
        for day in days_list:
            if random.random() < mutation_rate and len(new_ind[day]) > 0:
                store_to_move = random.choice(new_ind[day])
                possible_days = [d for d in days_list if d != day]
                target_day = random.choice(possible_days)
                new_ind[day].remove(store_to_move)
                new_ind[target_day].append(store_to_move)

        # random day swap
        if random.random() < 0.5 * mutation_rate and len(days_list) > 1:
            d1, d2 = random.sample(days_list, 2)
            new_ind[d1], new_ind[d2] = new_ind[d2], new_ind[d1]

        # post-mutation fix
        new_ind = self.redistribute_workload(new_ind)
        return new_ind

    # ---------------------------------------------------------------------
    # Utility & TSP-like methods
    # ---------------------------------------------------------------------
    def redistribute_workload(self, individual):
        new_ind = {d: list(stores) for d, stores in individual.items()}
        day_util = self.calculate_day_utilization(new_ind)

        over_days = [d for d, ut in day_util.items() if ut > self.DAILY_LIMIT]
        under_days = [d for d, ut in day_util.items() if ut < 0.7 * self.DAILY_LIMIT]

        while over_days and under_days:
            od = over_days.pop()
            while day_util[od] > self.DAILY_LIMIT and under_days:
                if not new_ind[od]:
                    break
                store_to_move = random.choice(new_ind[od])
                best_target = min(under_days, key=lambda x: day_util[x])
                new_ind[od].remove(store_to_move)
                new_ind[best_target].append(store_to_move)

                day_util = self.calculate_day_utilization(new_ind)
                if day_util[best_target] > 0.9 * self.DAILY_LIMIT:
                    under_days.remove(best_target)
                if day_util[od] <= self.DAILY_LIMIT:
                    break

        return new_ind

    def calculate_day_utilization(self, individual):
        day_util = {}
        for d, stores in individual.items():
            day_util[d] = self._compute_day_time(stores)
        return day_util

    def _compute_day_time(self, store_list):
        if not store_list:
            return 0.0
        route = self._nearest_neighbor(store_list)
        travel_time = 0.0
        for i in range(len(route) - 1):
            idx1 = self.store_to_index[route[i]]
            idx2 = self.store_to_index[route[i+1]]
            travel_time += self.time_matrix[idx1, idx2]
        service_time = sum(self.service_time_by_id[sid] for sid in route)
        return travel_time + service_time

    def _nearest_neighbor(self, store_list):
        """
        Basic TSP nearest-neighbor approach to get a route for these stores.
        """
        if len(store_list) <= 1:
            return store_list

        unvisited = set(store_list)
        route = []
        current = random.choice(list(unvisited))
        route.append(current)
        unvisited.remove(current)

        while unvisited:
            idx_current = self.store_to_index[current]
            next_store = min(
                unvisited,
                key=lambda s: self.time_matrix[idx_current, self.store_to_index[s]]
            )
            route.append(next_store)
            unvisited.remove(next_store)
            current = next_store

        return route

    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        selected = random.sample(list(zip(population, fitness_scores)), tournament_size)
        return min(selected, key=lambda x: x[1])[0]

    # ---------------------------------------------------------------------
    # 6) Main EA loop
    # ---------------------------------------------------------------------
    def optimize_schedule(
        self,
        population_size=50,
        generations=100,
        mutation_rate=0.2,
        patience=12,
        max_time=60
    ):
        start_time = time.time()

        population = self.initialize_population(population_size)
        best_individual = None
        best_fitness = float('inf')
        no_improvement_count = 0

        for gen in range(generations):
            elapsed = time.time() - start_time
            if elapsed > max_time:
                self.logger.info(f"Max time {max_time}s exceeded. Stopping at gen={gen+1}.")
                break

            fitness_scores = [self.evaluate_fitness(ind) for ind in population]

            # find best in this generation
            improved = False
            for i, fit in enumerate(fitness_scores):
                if fit < best_fitness:
                    best_fitness = fit
                    best_individual = population[i]
                    no_improvement_count = 0
                    improved = True

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
        for d in self.available_days:
            for sid in best_individual[d]:
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
