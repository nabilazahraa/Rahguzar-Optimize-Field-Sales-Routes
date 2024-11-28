import pandas as pd
import numpy as np
from k_means_constrained import KMeansConstrained
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import folium
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from Db_operations import fetch_data
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from scipy.cluster.hierarchy import linkage, fcluster


class EnhancedPJPClustering:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger(__name__)

        self.colors = [
            'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'pink',
            'brown', 'gray', 'olive', 'black', 'beige', 'lightblue'
        ]
        self.service_times = {1: 20, 2: 40}  # Service times for retail (1) and wholesale (2)
        self.visits_per_week = {1: 2, 2: 1}  # Weekly visit frequency for retail (1) and wholesale (2)
        self.working_hours_per_day = 8 * 60  # Total working minutes per day
        self.osrm_base_url = "https://router.project-osrm.org"
        self.osrm_cache = {}  # Cache for OSRM results to improve performance
        self.lock = threading.Lock()  # Lock for thread-safe cache access

    def fetch_store_data(self, distributor_id):
        self.logger.info(f"Fetching store data for distributor_id {distributor_id}...")
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
        data = fetch_data(query, (distributor_id,))
        if not data or len(data) == 0:
            self.logger.warning(f"No data found for distributor_id {distributor_id}.")
            return pd.DataFrame()
        self.logger.info(f"Fetched {len(data)} records for distributor_id {distributor_id}.")
        df = pd.DataFrame(data, columns=[
            "storeid", "latitude", "longitude", "channeltypeid", "storecode"
        ])
        return df

    def calculate_workload(self, df):
        self.logger.info("Calculating workloads...")
        df['visit_time'] = df['channeltypeid'].map(self.service_times)
        df['weekly_visits'] = df['channeltypeid'].map(self.visits_per_week)
        df['total_time'] = df['visit_time'] * df['weekly_visits']
        self.logger.debug("Workload calculation completed.")
        return df

    def calculate_haversine_distance_matrix(self, df):
        """
        Calculate Haversine distance matrix between stores using vectorized numpy operations
        
        Parameters:
        df (pd.DataFrame): DataFrame containing store coordinates
        
        Returns:
        np.array: Distance matrix in kilometers
        """
        self.logger.info("Calculating Haversine distance matrix using vectorized approach...")
        
        def haversine_vectorized(lat1, lon1, lat2, lon2):
            R = 6371  # Earth's radius in kilometers
            
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            
            dlat = lat2[:, np.newaxis] - lat1[np.newaxis, :]
            dlon = lon2[:, np.newaxis] - lon1[np.newaxis, :]
            
            a = np.sin(dlat/2)**2 + np.cos(lat1[np.newaxis, :]) * np.cos(lat2[:, np.newaxis]) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
        
        latitudes = df['latitude'].values
        longitudes = df['longitude'].values
        
        distance_matrix = haversine_vectorized(latitudes, longitudes, latitudes, longitudes)
        
        self.logger.info("Vectorized Haversine distance matrix calculation completed.")
        return distance_matrix

    def prepare_clustering_features(self, df, iteration=0):
        """
        Prepare features for clustering with dynamic weights and Haversine distance
        
        Parameters:
        df (pd.DataFrame): DataFrame with store data
        iteration (int): Current iteration of clustering
        
        Returns:
        np.array: Combined scaled features for clustering
        """
        self.logger.info("Preparing clustering features with dynamic weights and Haversine distance...")

        # Scale the geographic coordinates
        scaler_geo = StandardScaler()
        coords_scaled = scaler_geo.fit_transform(df[['latitude', 'longitude']])

        # Scale the workload
        scaler_workload = StandardScaler()
        workload_scaled = scaler_workload.fit_transform(df[['total_time']])

        # Calculate Haversine distance matrix
        haversine_matrix = self.calculate_haversine_distance_matrix(df)
        
        # Scale the Haversine distance matrix
        scaler_distance = StandardScaler()
        distance_scaled = scaler_distance.fit_transform(haversine_matrix)

        # Calculate the variance of the scaled features
        var_geo = np.var(coords_scaled)
        var_workload = np.var(workload_scaled)
        var_distance = np.var(distance_scaled)

        # Adjust weights to equalize the variances and consider iteration
        if var_geo != 0 and var_workload != 0 and var_distance != 0:
            geography_weight = (1 / var_geo) * (1.1 ** iteration)
            workload_weight = (1 / var_workload) * (0.9 ** iteration)
            distance_weight = (1 / var_distance) * (1.0 ** iteration)

            # Normalize weights
            total_weight = geography_weight + workload_weight + distance_weight
            geography_weight /= total_weight
            workload_weight /= total_weight
            distance_weight /= total_weight

            self.logger.info(f"Dynamic weights calculated: geography_weight={geography_weight:.4f}, workload_weight={workload_weight:.4f}, distance_weight={distance_weight:.4f}")
        else:
            geography_weight = 0.4
            workload_weight = 0.3
            distance_weight = 0.3
            self.logger.warning("Variance is zero for one of the features, using default weights.")

        # Apply weights
        coords_scaled *= geography_weight
        workload_scaled *= workload_weight
        distance_scaled *= distance_weight

        # Combine features
        combined_features = np.hstack([coords_scaled, workload_scaled, distance_scaled])
        
        self.logger.debug("Clustering features prepared with dynamic weights and Haversine distance.")
        return combined_features


    def get_osrm_travel_time_matrix(self, origins, destinations):
            max_waypoints = 100  # OSRM limit for number of waypoints per request
            max_table_size = 10000  # OSRM limit for total elements in the table (sources * destinations)
            num_origins = len(origins)
            num_destinations = len(destinations)
            travel_times = np.zeros((num_origins, num_destinations))

            self.logger.info("Starting travel time matrix calculation using OSRM...")
            start_time = time.time()

            # Function to split indices into chunks that comply with OSRM limits
            def chunk_indices(length, max_chunk_size):
                return [list(range(i, min(i + max_chunk_size, length))) for i in range(0, length, max_chunk_size)]

            origin_chunks = chunk_indices(num_origins, max_waypoints)
            destination_chunks = chunk_indices(num_destinations, max_waypoints)

            # Prepare tasks for parallel execution
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for origin_indices in origin_chunks:
                    chunk_origins = [origins[i] for i in origin_indices]
                    for dest_indices in destination_chunks:
                        chunk_destinations = [destinations[j] for j in dest_indices]

                        # Check if the total elements exceed OSRM limit
                        if len(origin_indices) * len(dest_indices) > max_table_size:
                            self.logger.warning("Further splitting required to comply with OSRM table size limit.")
                            continue

                        # Build the coordinate list
                        all_coords = chunk_origins + chunk_destinations
                        coords_str = ';'.join([f"{lon},{lat}" for lat, lon in all_coords])

                        # Build sources and destinations indices relative to the coordinate list
                        sources = ';'.join(map(str, range(len(chunk_origins))))
                        dest_offset = len(chunk_origins)
                        destinations_param = ';'.join(map(str, range(dest_offset, dest_offset + len(chunk_destinations))))

                        # Prepare the URL
                        url = f"{self.osrm_base_url}/table/v1/driving/{coords_str}?sources={sources}&destinations={destinations_param}&annotations=duration"

                        # Submit the task to the executor
                        futures.append(executor.submit(self._osrm_request, url, origin_indices, dest_indices))

                # Collect the results as they complete
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        durations, origin_indices_res, dest_indices_res = result
                        # Update the travel_times matrix
                        for i, orig_idx in enumerate(origin_indices_res):
                            for j, dest_idx in enumerate(dest_indices_res):
                                travel_times[orig_idx, dest_idx] = durations[i][j]
                    else:
                        self.logger.error("OSRM request failed. Falling back to Haversine approximation.")
                        return None

            end_time = time.time()
            self.logger.info(f"Travel time matrix calculation completed in {end_time - start_time:.2f} seconds.")
            return travel_times

    def _osrm_request(self, url, origin_indices, dest_indices):
            try:
                with self.lock:
                    if url in self.osrm_cache:
                        self.logger.debug(f"Cache hit for URL: {url}")
                        data = self.osrm_cache[url]
                    else:
                        self.logger.debug(f"Making OSRM request to URL: {url}")
                        response = requests.get(url)
                        response.raise_for_status()
                        data = response.json()
                        self.osrm_cache[url] = data
                if "durations" in data:
                    durations = np.array(data["durations"]) / 60  # Convert seconds to minutes
                    return durations, origin_indices, dest_indices
                else:
                    self.logger.error("OSRM response missing durations.")
                    return None
            except Exception as e:
                self.logger.error(f"Error querying OSRM: {e}")
                return None

    def approximate_travel_time(self, df, avg_speed_kmh=40):
        self.logger.info("Calculating travel time matrix using Haversine approximation...")
        num_stores = len(df)
        travel_time_matrix = np.zeros((num_stores, num_stores))
        for i in range(num_stores):
            for j in range(num_stores):
                if i == j:
                    travel_time_matrix[i][j] = 0
                else:
                    dist_km = geodesic((df.iloc[i]['latitude'], df.iloc[i]['longitude']),
                                       (df.iloc[j]['latitude'], df.iloc[j]['longitude'])).km
                    travel_time_matrix[i][j] = (dist_km / avg_speed_kmh) * 60  # Minutes
        self.logger.info("Haversine travel time matrix calculation completed.")
        return travel_time_matrix

    def calculate_travel_time_matrix(self, df):
        num_stores = len(df)
        self.logger.info(f"Calculating travel time matrix for {num_stores} stores...")
        if num_stores > 1000:
            self.logger.warning("Large number of stores detected. Using Haversine approximation to improve performance.")
            travel_time_matrix = self.approximate_travel_time(df)
            return travel_time_matrix

        origins = [(df.iloc[i]["latitude"], df.iloc[i]["longitude"]) for i in range(num_stores)]
        travel_time_matrix = self.get_osrm_travel_time_matrix(origins, origins)

        if travel_time_matrix is None or len(travel_time_matrix) == 0:
            self.logger.warning("OSRM failed. Falling back to Haversine approximation.")
            travel_time_matrix = self.approximate_travel_time(df)

        return travel_time_matrix
    
    def split_clusters_by_centroid_proximity(self, df):
        self.logger.info("Splitting clusters based on centroid proximity...")
        for cluster_id in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster_id].reset_index(drop=True)
            centroid = cluster_data[['latitude', 'longitude']].mean().values
            distances = cluster_data[['latitude', 'longitude']].apply(
                lambda row: geodesic((row['latitude'], row['longitude']), tuple(centroid)).km, axis=1
            )
            max_distance = distances.max()
            
            if max_distance > 10:  # Threshold in kilometers
                self.logger.info(f"Cluster {cluster_id} exceeds max centroid distance. Splitting...")
                num_sub_clusters = 2
                sub_kmeans = KMeansConstrained(
                    n_clusters=num_sub_clusters,
                    size_min=1,
                    size_max=len(cluster_data),
                    init='k-means++',
                    n_init=10,
                    max_iter=300
                )
                cluster_features = self.prepare_clustering_features(cluster_data)
                sub_labels = sub_kmeans.fit_predict(cluster_features)
                new_labels = [f"{cluster_id}_{label}" for label in sub_labels]
                df.loc[cluster_data.index, 'cluster'] = new_labels
        return df

    def reduce_cluster_overlap(self, df):
        self.logger.info("Reducing cluster overlaps...")
        for store_id, row in df.iterrows():
            min_distance = float('inf')
            best_cluster = row['cluster']
            
            for cluster_id in df['cluster'].unique():
                cluster_data = df[df['cluster'] == cluster_id]
                centroid = cluster_data[['latitude', 'longitude']].mean().values
                distance = geodesic((row['latitude'], row['longitude']), tuple(centroid)).km
                if distance < min_distance:
                    min_distance = distance
                    best_cluster = cluster_id
            
            df.at[store_id, 'cluster'] = best_cluster
        self.logger.info("Cluster overlap reduction completed.")
        return df


    def refine_with_travel_time(self, df, max_cluster_travel_time=120):
        num_clusters = df['cluster'].nunique()
        self.logger.info(f"Refining {num_clusters} clusters using travel times...")

        iteration = 0
        while True:
            clusters_to_split = []
            for cluster_id in df['cluster'].unique():
                self.logger.info(f"Processing cluster {cluster_id}...")
                cluster_data = df[df['cluster'] == cluster_id].reset_index(drop=True)
                travel_time_matrix = self.calculate_travel_time_matrix(cluster_data)

                if travel_time_matrix is not None:
                    max_travel_time = np.max(travel_time_matrix)
                    avg_travel_time = np.mean(travel_time_matrix)
                    self.logger.info(f"Cluster {cluster_id} - Max Travel Time: {max_travel_time:.2f} mins, Avg Travel Time: {avg_travel_time:.2f} mins")

                    if max_travel_time > max_cluster_travel_time:
                        self.logger.info(f"Cluster {cluster_id} exceeds max travel time. Marking for splitting.")
                        clusters_to_split.append(cluster_id)
                else:
                    self.logger.warning(f"Travel time matrix not available for cluster {cluster_id}. Skipping.")

            if not clusters_to_split or iteration >= 5:
                self.logger.info("No clusters exceed max travel time or max iterations reached. Refinement completed.")
                break

            # Split clusters that exceed max travel time
            for cluster_id in clusters_to_split:
                cluster_data = df[df['cluster'] == cluster_id].reset_index(drop=True)
                cluster_data = self.split_clusters_by_centroid_proximity(cluster_data)
                df.loc[cluster_data.index, 'cluster'] = cluster_data['cluster']

            iteration += 1


    def evaluate_final_clusters(self, df, features):
        """
        Evaluate the final formed clusters
        
        Parameters:
        df (pd.DataFrame): Dataframe with cluster labels
        features (np.array): Scaled features used for clustering
        """
        self.logger.info("\n--- Final Cluster Evaluation ---")
        
        # Cluster Composition
        cluster_sizes = df['cluster'].value_counts()
        self.logger.info("\nCluster Sizes:")
        for cluster, size in cluster_sizes.items():
            self.logger.info(f"Cluster {cluster}: {size} stores")
        
        # Workload Distribution
        cluster_workloads = df.groupby('cluster')['total_time'].agg(['mean', 'sum'])
        self.logger.info("\nCluster Workload Analysis:")
        self.logger.info(f"\n{cluster_workloads}")
        
        # Channel Type Distribution
        channel_distribution = df.groupby(['cluster', 'channeltypeid']).size().unstack(fill_value=0)
        self.logger.info("\nChannel Type Distribution per Cluster:")
        self.logger.info(f"\n{channel_distribution}")
        
        # Cluster Evaluation Metrics
        try:
            # For silhouette score, labels need to be integers
            cluster_labels = pd.factorize(df['cluster'])[0]
            silhouette = silhouette_score(features, cluster_labels)
            calinski = calinski_harabasz_score(features, cluster_labels)
            davies = davies_bouldin_score(features, cluster_labels)
            
            self.logger.info("\nClustering Metrics:")
            self.logger.info(f"Silhouette Score: {silhouette:.4f}")
            self.logger.info(f"Calinski-Harabasz Score: {calinski:.4f}")
            self.logger.info(f"Davies-Bouldin Score: {davies:.4f}")
        except Exception as e:
            self.logger.error(f"Error calculating clustering metrics: {e}")

    from scipy.cluster.hierarchy import linkage, fcluster

    def hierarchical_pre_clustering(self, df, max_distance=0.05):
        self.logger.info("Performing hierarchical pre-clustering...")
        coords = df[['latitude', 'longitude']].values

        # Perform hierarchical clustering
        Z = linkage(coords, method='ward')  # Ward's method for minimizing variance
        pre_clusters = fcluster(Z, max_distance, criterion='distance')  # Group based on max distance

        df['pre_cluster'] = pre_clusters
        self.logger.info(f"Formed {df['pre_cluster'].nunique()} pre-clusters using hierarchical clustering.")
        return df


    def capacitated_clustering(self, df, num_order_bookers, max_iterations=5):
        self.logger.info("Preparing data for clustering...")
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['total_time'] = pd.to_numeric(df['total_time'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude', 'total_time'])
        self.logger.info(f"Prepared data for {len(df)} stores.")

        for iteration in range(max_iterations):
            self.logger.info(f"\nIteration {iteration+1} of clustering...")

            # Hierarchical Pre-Clustering
            df = self.hierarchical_pre_clustering(df)

            combined_features = self.prepare_clustering_features(df, iteration)
            kmeans = KMeansConstrained(
                n_clusters=num_order_bookers,
                size_min=None,
                size_max=None,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=iteration
            )
            df['cluster'] = kmeans.fit_predict(combined_features)

            # Penalize clusters exceeding travel time
            clusters_to_adjust = []
            for cluster_id in df['cluster'].unique():
                cluster_data = df[df['cluster'] == cluster_id].reset_index(drop=True)
                travel_time_matrix = self.calculate_travel_time_matrix(cluster_data)
                if travel_time_matrix is not None:
                    max_travel_time = np.max(travel_time_matrix)
                    if max_travel_time > self.working_hours_per_day:
                        clusters_to_adjust.append(cluster_id)

            if not clusters_to_adjust:
                break

            self.logger.warning(f"Adjusting clusters: {clusters_to_adjust}")
            num_order_bookers += len(clusters_to_adjust)  # Increase clusters for the next iteration

        return df
    


    def visualize_clusters_matplotlib(self, df):
        """
        Create a scatter plot of clusters using Matplotlib
        """
        self.logger.info("Visualizing clusters using Matplotlib...")
        plt.figure(figsize=(12, 8))
        
        # Get unique clusters
        unique_clusters = df['cluster'].unique()
        
        # Create a color map
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        
        # Plot each cluster
        for i, cluster in enumerate(unique_clusters):
            cluster_data = df[df['cluster'] == cluster]
            plt.scatter(
                cluster_data['longitude'], 
                cluster_data['latitude'], 
                c=[colors[i]], 
                label=f'Cluster {cluster}', 
                alpha=0.7
            )
        
        plt.title('Store Clusters Visualization')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        self.logger.info("Matplotlib visualization completed.")

    def visualize_clusters_folium(self, df):
        """
        Create an interactive map of clusters using Folium
        """
        self.logger.info("Visualizing clusters using Folium...")
        # Calculate the center of the map
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        # Create a map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Unique clusters
        unique_clusters = df['cluster'].unique()
        
        # Create a color palette
        color_palette = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        color_dict = {cluster: f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}' 
                      for cluster, color in zip(unique_clusters, color_palette)}
        
        # Add markers for each store
        for _, row in df.iterrows():
            # Create popup content
            popup_content = f"""
            Store ID: {row['storeid']}
            Store Code: {row['storecode']}
            Channel Type: {row['channeltypeid']}
            Cluster: {row['cluster']}
            Total Time: {row.get('total_time', 'N/A')} minutes
            """
            
            # Add marker with cluster color
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=popup_content,
                color=color_dict[row['cluster']],
                fill=True,
                fillColor=color_dict[row['cluster']],
                fillOpacity=0.7
            ).add_to(m)
        
        # Save the map
        m.save('store_clusters_map.html')
        self.logger.info("Interactive map saved as 'store_clusters_map.html'")

    def cluster_and_visualize(self, distributor_id, num_order_bookers):
        self.logger.info("Starting clustering and visualization process...")
        df = self.fetch_store_data(distributor_id)
        if df.empty:
            self.logger.error("No store data available. Exiting...")
            return

        df = self.calculate_workload(df)

        self.logger.info("Starting clustering...")
        clustered_data = self.capacitated_clustering(df, num_order_bookers)

        self.logger.info("Refining clusters with travel times...")
        self.refine_with_travel_time(clustered_data)

        self.logger.info("Visualizing clusters...")
        # Matplotlib visualization
        self.visualize_clusters_matplotlib(clustered_data)
        
        # Folium visualization
        self.visualize_clusters_folium(clustered_data)

        self.logger.info("Clustering and visualization process completed.")


def main():
    clustering = EnhancedPJPClustering()
    distributor_id = 7
    num_order_bookers = 3
    clustering.cluster_and_visualize(distributor_id, num_order_bookers)


if __name__ == "__main__":
    main() 