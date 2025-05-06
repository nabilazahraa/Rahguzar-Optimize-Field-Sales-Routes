

import React, { useEffect, useMemo, useState } from 'react';
import {
  Typography,
  Box,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import { MapContainer, TileLayer, Marker, Popup, useMap, ZoomControl } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import axios from 'axios';
// import LoadingSpinner from '../RerouteSpinner';

const createColoredMarker = (color) => {
  return new L.DivIcon({
    html: `<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" 
             viewBox="0 0 24 24" fill="${color}" stroke="#ffffff" stroke-width="2" 
             stroke-linecap="round" stroke-linejoin="round" class="feather feather-map-pin">
              <path d="M21 10c0 6-9 13-9 13s-9-7-9-13a9 9 0 1 1 18 0z"></path>
              <circle cx="12" cy="10" r="3"></circle>
           </svg>`,
    className: "",
    iconSize: [24, 24],
    iconAnchor: [12, 24],
    popupAnchor: [0, -24],
  });
};

const HighlightMap = ({ selectedCluster }) => {
  const map = useMap();
  useEffect(() => {
    if (selectedCluster && selectedCluster.centroid) {
      const { latitude, longitude } = selectedCluster.centroid;
      if (typeof latitude === 'number' && typeof longitude === 'number') {
        map.flyTo([latitude, longitude], 14, { duration: 1.5 });
      }
    }
  }, [selectedCluster, map]);
  return null;
};

function ClusterView({
  clusters,
  setClusters,
  setSchedule,
  distributorId,
  planDuration,
  numOrderbookers,
  storeType,
  retailTime,
  wholesaleTime,
  holidays,
  custom_days,
  replicate,
  setRerouteNeeded,
  rerouteNeeded,
  setStoreData,
  storeData

}) {

  const [selectedCluster, setSelectedCluster] = useState(null);
  const [selectedStore, setSelectedStore] = useState(null);
  // const [rerouteNeeded, setRerouteNeeded] = useState(false);
  // const [isRerouting, setIsRerouting] = useState(false);
  // const [storeData, setStoreData] = useState([]);
  const BASE_URL = process.env.REACT_APP_API_BASE_URL;

  const clusterColors = useMemo(() => {
    const colors = {};
    const hueStep = clusters.length ? 360 / clusters.length : 0;
    clusters.forEach((cluster, index) => {
      const hue = index * hueStep;
      colors[cluster.cluster_id] = `hsl(${hue}, 70%, 50%)`;
    });
    return colors;
  }, [clusters]);

  useEffect(() => {
    if (clusters && clusters.length > 0) {
      const initialStoreData = clusters.flatMap(cluster =>
        cluster.stores.map(store => ({
          ...store,
          cluster_id: cluster.cluster_id,
          old_cluster_id: cluster.cluster_id,
          is_removed: store.is_removed ? true : false
        }))
      );
      setStoreData(initialStoreData);
    }
  }, [clusters]);

  const getMarkerColor = (store) => {
    if (store.is_removed) return "gray";
    return clusterColors[store.cluster_id] || "gray";
  };

  const handleClusterChange = (store, newClusterId) => {
    if (!store || store.storeid === undefined) return;
    const updatedStoreData = storeData.map(s => {
      if (s.storeid === store.storeid) {
        return { ...s, cluster_id: newClusterId, is_removed: false };
      }
      return s;
    });
    setStoreData(updatedStoreData);
    setRerouteNeeded(true);
  };

  const handleRemoveToggle = (store) => {
    if (!store || store.storeid === undefined) return;
    const updatedStoreData = storeData.map(s => {
      if (s.storeid === store.storeid) {
        return { ...s, is_removed: !s.is_removed };
      }
      return s;
    });
    setStoreData(updatedStoreData);
    setRerouteNeeded(true);
  };

  const handleStartRerouting = () => {
    if (!rerouteNeeded) return;
    // setIsRerouting(true);

    axios.post(`${BASE_URL}/reroute-clusters`, {
      modified_clusters: storeData,
      distributor_id: distributorId,
      plan_duration: planDuration,
      num_orderbookers: numOrderbookers,
      store_type: storeType,
      retail_time: retailTime,
      wholesale_time: wholesaleTime,
      holidays: holidays,
      custom_days: planDuration === 'custom' ? (custom_days || 7) : null,
      replicate: planDuration === 'custom' ? replicate : false
    })
    .then(response => {
      if (response.data.status === "success") {
        const updatedClusters = response.data.clusters;
        setClusters(updatedClusters);
        if (response.data.pjp) {
          setSchedule(response.data.pjp);
        }
        const newStoreData = updatedClusters.flatMap(cluster =>
          cluster.stores.map(st => ({
            ...st,
            cluster_id: cluster.cluster_id,
            old_cluster_id: cluster.cluster_id,
            is_removed: st.is_removed ? true : false
          }))
        );
        setStoreData(newStoreData);
        // setRerouteNeeded(false);
      }
    })
    .catch(error => console.error("Error rerouting clusters:", error))
    // .finally(() => setIsRerouting(false));
  };

  const mapCenter = useMemo(() => {
    if (!clusters || clusters.length === 0) return [0, 0];
    const allCentroids = clusters.map(c => c.centroid);
    const avgLat = allCentroids.reduce((sum, c) => sum + c.latitude, 0) / allCentroids.length;
    const avgLon = allCentroids.reduce((sum, c) => sum + c.longitude, 0) / allCentroids.length;
    return [avgLat, avgLon];
  }, [clusters]);

  const displayedClusters = clusters.filter(c => c.cluster_id !== -1);
  // {isRerouting && (
  //   <Box
  //     sx={{
  //       position: 'fixed',
  //       top: 0,
  //       left: 0,
  //       width: '100vw',
  //       height: '100vh',
  //       backgroundColor: 'rgba(255, 255, 255, 0.7)',
  //       zIndex: 3000,
  //       display: 'flex',
  //       alignItems: 'center',
  //       justifyContent: 'center',
  //     }}
  //   >
  //     <LoadingSpinner size={60} />
  //   </Box>
  // )}
  
  return (
    
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* MapWrapper */}
      <Box sx={{ height: '100%', width: '100%', position: 'relative' }}>
        {/* Floating Reroute Button on Map */}
        {/* {rerouteNeeded && (
          <Box sx={{ position: 'absolute', top: 72, right: 16, zIndex: 1000 }}>

            <Button
              variant="contained"
              color="primary"
              onClick={handleStartRerouting}
              disabled={isRerouting}
            >
              {isRerouting ? <LoadingSpinner size={24} color="inherit" /> : "Start Dynamic Re-Routing"}
            </Button>
          </Box>
        )} */}
        
        <MapContainer center={mapCenter} zoom={12} style={{ height: '100%', width: '100%' }}>
          <TileLayer
            url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
            attribution='&copy; OpenStreetMap contributors'
          />
          <ZoomControl position="bottomright" />
          <HighlightMap selectedCluster={selectedCluster} />
          {storeData.map((store, idx) => (
            <Marker
              key={idx}
              position={[store.latitude, store.longitude]}
              icon={createColoredMarker(getMarkerColor(store))}
              eventHandlers={{ click: () => setSelectedStore(store) }}
            >
              {selectedStore?.storeid === store.storeid && (
                <Popup onClose={() => setSelectedStore(null)}>
                  <Typography variant="subtitle2"><strong>Store {store.storecode}</strong></Typography>
                  <Typography variant="body2">Store ID: {store.storeid}</Typography>
                  <Typography variant="body2">
                    Cluster: {store.cluster_id}
                    {store.is_removed && " (Removed)"}
                  </Typography>
                  <FormControl fullWidth sx={{ mt: 1 }} disabled={store.is_removed}>
                    <InputLabel sx={{ backgroundColor: 'white' }}>Move Store</InputLabel>
                    <Select
                      value={store.cluster_id}
                      onChange={(event) => handleClusterChange(store, event.target.value)}
                      displayEmpty
                    >
                      <MenuItem value="" disabled>Select a cluster</MenuItem>
                      {displayedClusters.map((c) => (
                        <MenuItem key={c.cluster_id} value={c.cluster_id}>
                          Cluster {c.cluster_id}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  <Box sx={{ mt: 1 }}>
                    {!store.is_removed ? (
                      <Button variant="outlined" color="secondary" onClick={() => handleRemoveToggle(store)}>
                        Remove Store
                      </Button>
                    ) : (
                      <Button variant="outlined" color="primary" onClick={() => handleRemoveToggle(store)}>
                        Unremove Store
                      </Button>
                    )}
                  </Box>
                </Popup>
              )}
            </Marker>
          ))}
        </MapContainer>
      </Box>

      {/* Cluster Legend */}
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2, mt: 2, p: 2 }}>
        {displayedClusters.map(cluster => (
          <Box
            key={cluster.cluster_id}
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1,
              padding: "6px 12px",
              borderRadius: "8px",
              backgroundColor: "#f5f5f5",
              cursor: "pointer",
            }}
            onClick={() => setSelectedCluster(cluster)}
          >
            <Box
              sx={{
                width: 20,
                height: 20,
                backgroundColor: clusterColors[cluster.cluster_id],
                borderRadius: "50%",
                border: "1px solid #ccc",
              }}
            />
            <Typography variant="body2">
              Cluster {cluster.cluster_id}
            </Typography>
          </Box>
        ))}
      </Box>
    </Box>
    
  );
}

export default ClusterView;
