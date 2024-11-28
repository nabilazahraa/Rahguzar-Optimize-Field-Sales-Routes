import pandas as pd
import numpy as np
from k_means_constrained import KMeansConstrained
from sklearn.preprocessing import StandardScaler
import folium
import matplotlib.pyplot as plt
from folium.plugins import HeatMap
from Db_operations import fetch_data


class EnhancedPJPClustering:
    def __init__(self):
        self.colors = [
            'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'pink',
            'brown', 'gray', 'olive', 'black', 'beige', 'lightblue'
        ]
        self.service_times = {1: 20, 2: 40}  # Service times for retail (1) and wholesale (2)
        self.visits_per_week = {1: 2, 2: 1}  # Weekly visit frequency for retail (1) and wholesale (2)
        self.working_hours_per_day = 8 * 60  # Total working minutes per day

    def fetch_store_data(self, distributor_id):
        """
        Fetch store data for the given distributor.
        Returns:
            DataFrame with storeid, latitude, longitude, channeltypeid, storecode.
        """
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
        df = pd.DataFrame(data, columns=[
            "storeid", "latitude", "longitude", "channeltypeid", "storecode"
        ])
        return df

    def calculate_workload(self, df):
        """
        Calculate service time and workload for each store.
        Returns:
            DataFrame with workload columns added.
        """
        df['visit_time'] = df['channeltypeid'].map(self.service_times)
        df['weekly_visits'] = df['channeltypeid'].map(self.visits_per_week)
        df['total_time'] = df['visit_time'] * df['weekly_visits']
        return df

    def prepare_clustering_features(self, df, geography_weight=0.5, workload_weight=0.5):
        """
        Prepares combined features for clustering based on geography and workload.
        """
        scaler = StandardScaler()

        # Scale geographical coordinates
        coords_scaled = scaler.fit_transform(df[['latitude', 'longitude']])
        coords_scaled *= geography_weight

        # Scale workload
        workload_scaled = scaler.fit_transform(df[['total_time']])
        workload_scaled *= workload_weight

        # Combine features
        combined_features = np.hstack([coords_scaled, workload_scaled])
        return combined_features

    def capacitated_clustering(self, df, num_clusters):
        """
        Perform capacitated K-Means clustering.
        Returns:
            DataFrame with cluster assignments.
        """
        # Ensure numeric values
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['total_time'] = pd.to_numeric(df['total_time'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude', 'total_time'])

        # Prepare combined features
        combined_features = self.prepare_clustering_features(df)

        # Initialize KMeansConstrained
        kmeans = KMeansConstrained(
            n_clusters=num_clusters,
            size_min=int(len(df) / num_clusters * 0.9),
            size_max=int(len(df) / num_clusters * 1.1),
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=42
        )

        # Perform clustering
        df['cluster'] = kmeans.fit_predict(combined_features)

        # Calculate cluster capacities
        avg_capacity = self.working_hours_per_day * 5 * num_clusters
        cluster_capacity = avg_capacity / num_clusters

        # Rebalance clusters
        df = self.rebalance_clusters(df, cluster_capacity)
        return df

    def rebalance_clusters(self, df, cluster_capacity):
        """
        Rebalance clusters to ensure workloads are within tolerable limits.
        """
        max_iterations = 100
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Compute workloads
            cluster_workloads = df.groupby('cluster')['total_time'].sum()
            overloaded_cluster = cluster_workloads.idxmax()
            underloaded_cluster = cluster_workloads.idxmin()

            if abs(cluster_workloads[overloaded_cluster] - cluster_capacity) <= 0.1 * cluster_capacity and \
               abs(cluster_workloads[underloaded_cluster] - cluster_capacity) <= 0.1 * cluster_capacity:
                break

            # Move store from overloaded to underloaded cluster
            overloaded_stores = df[df['cluster'] == overloaded_cluster]
            underloaded_centroid = df[df['cluster'] == underloaded_cluster][['latitude', 'longitude']].mean()

            overloaded_stores['distance_to_underloaded'] = np.sqrt(
                (overloaded_stores['latitude'] - underloaded_centroid['latitude']) ** 2 +
                (overloaded_stores['longitude'] - underloaded_centroid['longitude']) ** 2
            )

            store_to_move = overloaded_stores.sort_values(
                by=['distance_to_underloaded', 'total_time'],
                ascending=[True, False]
            ).iloc[0]

            df.loc[df['storeid'] == store_to_move['storeid'], 'cluster'] = underloaded_cluster

        return df

    def visualize_clusters(self, df, method_name="Capacitated Clustering"):
        """
        Visualize clusters using Folium and save as HTML.
        """
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        # Add HeatMap
        heat_data = df[['latitude', 'longitude', 'total_time']].values.tolist()
        HeatMap(heat_data, radius=15, blur=10).add_to(m)

        # Add cluster markers
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=self.colors[row['cluster'] % len(self.colors)],
                fill=True,
                popup=f"Store ID: {row['storeid']}, Cluster: {row['cluster']}, Workload: {row['total_time']} mins"
            ).add_to(m)

        m.save(f"{method_name.replace(' ', '_')}_clusters.html")
        print(f"Cluster map saved to {method_name.replace(' ', '_')}_clusters.html")

    def plot_clusters(self, df, method_name="Capacitated Clustering"):
        """
        Visualize clusters using matplotlib.
        """
        plt.figure(figsize=(10, 8))
        for cluster in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster]
            plt.scatter(
                cluster_data['longitude'], cluster_data['latitude'],
                label=f"Cluster {cluster} ({cluster_data['total_time'].sum():.1f} mins)",
                alpha=0.6
            )
        plt.title(f"Cluster Visualization - {method_name}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid()
        plt.show()

    def cluster_and_visualize(self, distributor_id, num_order_bookers):
        """
        Main function to perform clustering and visualization.
        """
        df = self.fetch_store_data(distributor_id)
        df = self.calculate_workload(df)
        clustered_data = self.capacitated_clustering(df, num_order_bookers)
        self.visualize_clusters(clustered_data)
        self.plot_clusters(clustered_data)


def main():
    clustering = EnhancedPJPClustering()
    distributor_id = 7
    num_order_bookers = 3
    clustering.cluster_and_visualize(distributor_id, num_order_bookers)


if __name__ == "__main__":
    main()