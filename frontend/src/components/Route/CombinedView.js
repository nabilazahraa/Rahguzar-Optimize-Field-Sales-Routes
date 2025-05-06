import React, { useState, useMemo, useRef, useEffect } from 'react';
// import { Tooltip } from 'recharts';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import Snackbar from '@mui/material/Snackbar';
import MuiAlert from '@mui/material/Alert';



import { exportScheduleToExcel } from './exportutils';
import SettingsIcon from '@mui/icons-material/Settings';
import {
  Box,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Paper,
  Chip,
  IconButton,
  Divider,
  Button,
  Modal,
  Grid,
  Card,
  useTheme,
  TextField,
  InputAdornment,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import Checkbox from '@mui/material/Checkbox';
import ListItemText from '@mui/material/ListItemText';
import MenuIcon from '@mui/icons-material/Menu';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import GetAppIcon from '@mui/icons-material/GetApp';
import BarChartIcon from '@mui/icons-material/BarChart';
import CloseIcon from '@mui/icons-material/Close';
import SearchIcon from '@mui/icons-material/Search';
import {
  MapContainer,
  TileLayer,
  Marker,
  Polyline,
  Popup,
  ZoomControl,
  useMap,
} from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import axios from 'axios';
import { format, parseISO, isValid } from 'date-fns';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Brush,
  Tooltip
} from 'recharts';

import ClusterView from './ClusterMap';
import LoadingSpinner from '../RerouteSpinner';
import Collapse from '@mui/material/Collapse';

const CollapsibleSidebar = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: 20,
  left: 16,
  width: DRAWER_WIDTH - 50,
  maxHeight: '85vh',
  overflowY: 'auto',
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.spacing(1),
  boxShadow: theme.shadows[3],
  padding: theme.spacing(2),
  zIndex: 1001,
  transition: 'all 0.3s ease-in-out',
}));

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

// Constants
const DRAWER_WIDTH = 350;

// CSS for animations and transitions
const StyledCSS = () => (
  <style jsx global>{`
    @keyframes pulse {
      0% {
        transform: scale(1);
        opacity: 1;
      }
      50% {
        transform: scale(1.1);
        opacity: 0.9;
      }
      100% {
        transform: scale(1);
        opacity: 1;
      }
    }
    
    .marker-icon {
      transition: all 0.3s ease-out;
      transform-origin: bottom center;
    }
    
    .pulse {
      animation: pulse 1.5s infinite;
    }
    
    .highlighted-marker {
      z-index: 1000 !important;
    }
    
    .leaflet-marker-icon {
      transition: transform 0.3s ease;
    }
    
    .store-card-transition {
      transition: all 0.3s ease;
    }
    
    .store-card-highlighted {
      transform: translateX(8px);
      position: relative;
    }
    
    .store-card-highlighted::before {
      content: '';
      position: absolute;
      left: -8px;
      top: 0;
      height: 100%;
      width: 4px;
      background-color: inherit;
      border-radius: 4px 0 0 4px;
    }
  `}</style>
);

// MapHandler component to access the map instance properly
const MapHandler = ({ onMapReady }) => {
  const map = useMap();
  
  useEffect(() => {
    if (map && onMapReady) {
      onMapReady(map);
    }
  }, [map, onMapReady]);
  
  return null;
};

// FlyToMarker component to animate map focus
const FlyToMarker = ({ store, map }) => {
  useEffect(() => {
    if (store && store.latitude && store.longitude && map) {
      map.flyTo([store.latitude, store.longitude], 16, {
        animate: true,
        duration: 1.5
      });
    }
  }, [store, map]);
  
  return null;
};

// Styled Components
const ControlsContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.shape.borderRadius,
}));

const StoreListContainer = styled(Box)(({ theme }) => ({
  flexGrow: 1,
  overflowY: 'auto',
  padding: theme.spacing(2),
}));
const getRouteFromOSRM = async (coordinates) => {
  if (!coordinates || coordinates.length < 2) return coordinates;

  const fullRoute = [];

  for (let i = 0; i < coordinates.length - 1; i++) {
    const from = coordinates[i];
    const to = coordinates[i + 1];

    // Format for OSRM: lon,lat
    const pairStr = `${from[1]},${from[0]};${to[1]},${to[0]}`;

    try {
      const res = await axios.get(
        `http://localhost:5002/route/v1/driving/${pairStr}?overview=full&geometries=geojson`
      );

      const segment = res.data?.routes?.[0]?.geometry?.coordinates;
      if (segment && segment.length > 0) {
        // Convert to lat,lon and append
        const latLngSegment = segment.map(([lon, lat]) => [lat, lon]);

        // Avoid repeating the last point of previous segment
        if (i > 0) latLngSegment.shift();

        fullRoute.push(...latLngSegment);
      } else {
        // If no valid segment, push fallback straight line
        fullRoute.push(from, to);
      }
    } catch (err) {
      console.error(`OSRM error between ${i} and ${i + 1}:`, err);
      fullRoute.push(from, to); // fallback to straight line
    }
  }

  return fullRoute;
};

const StoreCard = styled(Paper)(({ theme, clusterColor, isHighlighted }) => ({
  padding: theme.spacing(1.8),
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(2),
  cursor: 'pointer',
  border: `2px solid ${isHighlighted ? clusterColor : 'transparent'}`,
  background: isHighlighted ? `${clusterColor}22` : '#ffffff',
  boxShadow: isHighlighted ? `0 4px 12px ${clusterColor}66` : theme.shadows[1],
  borderRadius: theme.spacing(1.5),
  marginBottom: theme.spacing(1.5),
  transition: 'all 0.2s ease-in-out',
  position: 'relative',
  '&:hover': {
    transform: 'translateY(-3px)',
    boxShadow: `0 6px 14px ${clusterColor}33`,
    background: `${clusterColor}11`,
  },
}));

const NumberBadge = styled(Box)(({ theme, clusterColor }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  minWidth: '40px',
  height: '40px',
  fontSize: '18px',
  fontWeight: 'bold',
  borderRadius: '50%',
  backgroundColor: clusterColor || theme.palette.primary.main,
  color: theme.palette.getContrastText(clusterColor || theme.palette.primary.main),
}));

const MapWrapper = styled(Box)(() => ({
  height: '100vh',
  width: '100vw',
  position: 'absolute',
  top: 0,
  left: 0,
}));

const MenuButton = styled(IconButton)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(2),
  left: theme.spacing(2),
  zIndex: 1000,
  backgroundColor: theme.palette.background.paper,
  boxShadow: theme.shadows[3],
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  },
}));

const SummaryCard = styled(Card)(({ theme }) => ({
  minWidth: 200,
  textAlign: 'center',
  padding: theme.spacing(2),
  borderRadius: theme.spacing(2),
  boxShadow: theme.shadows[2],
  backgroundColor: theme.palette.background.default,
}));

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  borderRadius: theme.spacing(2),
  boxShadow: theme.shadows[1],
  backgroundColor: theme.palette.background.paper,
}));

const modalStyle = (theme) => ({
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: '85%',
  maxHeight: '85vh',
  bgcolor: theme.palette.background.paper,
  boxShadow: 24,
  p: 4,
  overflowY: 'auto',
  borderRadius: '16px',
});

const formatDateWithDay = (dateStr) => {
  const date = parseISO(dateStr);
  if (!isValid(date)) return dateStr;
  return format(date, 'EEEE, dd/MM');
};

const isDateString = (str) => {
  const date = parseISO(str);
  return isValid(date);
};

const generateColorPalette = (numColors) => {
  const palette = [];
  for (let i = 0; i < numColors; i++) {
    const hue = (i * 360) / numColors;
    palette.push(`hsl(${hue}, 70%, 50%)`);
  }
  return palette;
};

const createColoredMarker = (color, isHighlighted = false, size = 36) => {
  try {
    const scale = isHighlighted ? 1.4 : 1;
    const strokeWidth = isHighlighted ? 3 : 2;
    const stroke = isHighlighted ? '#fff' : '#ffffff';
    const shadow = isHighlighted ? 'drop-shadow(0 0 6px rgba(0,0,0,0.5))' : '';
    const zIndex = isHighlighted ? 1000 : 400;
    
    return new L.DivIcon({
      html: `
        <svg xmlns="http://www.w3.org/2000/svg" 
             width="${size * scale}" 
             height="${size * scale}" 
             viewBox="0 0 24 24" 
             fill="${color}" 
             stroke="${stroke}" 
             stroke-width="${strokeWidth}" 
             stroke-linecap="round" 
             stroke-linejoin="round" 
             class="marker-icon ${isHighlighted ? 'pulse' : ''}"
             style="filter: ${shadow}; z-index: ${zIndex};">
          <path d="M21 10c0 6-9 13-9 13s-9-7-9-13a9 9 0 1 1 18 0z"></path>
          <circle cx="12" cy="10" r="3"></circle>
        </svg>
      `,
      className: `custom-marker ${isHighlighted ? 'highlighted-marker' : ''}`,
      iconSize: [size * scale, size * scale],
      iconAnchor: [(size * scale) / 2, size * scale],
      popupAnchor: [0, -size * scale],
    });
  } catch (error) {
    console.error("Error creating marker:", error);
    return new L.Icon.Default();
  }
};

function CombinedView({
  pjp,
  monthly_pjp,
  clusters,
  distributorId,
  planDuration,
  numOrderbookers,
  storeType,
  retailTime,
  wholesaleTime,
  holidays,
  custom_days,
  replicate,
  setSchedule,
  setClusters,
  planId,
  resetPlan,
}) {
  const theme = useTheme();
  const mapRef = useRef(null);
  
  const [open, setOpen] = useState(true);
  const [statsModalOpen, setStatsModalOpen] = useState(false);
  const [showClusterMap, setShowClusterMap] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const [selectedOrderbookers, setSelectedOrderbookers] = useState(
    pjp ? [Object.keys(pjp)[0]] : [Object.keys(monthly_pjp)[0]]
  );
  const [selectedDay, setSelectedDay] = useState(
    pjp
      ? Object.keys(pjp[Object.keys(pjp)[0]] || {})[0]
      : Object.keys(monthly_pjp[Object.keys(monthly_pjp)[0]] || {})[0]
  );
  const [showRoute, setShowRoute] = useState(true);
  const [highlightedStore, setHighlightedStore] = useState(null);
  const [mapInstance, setMapInstance] = useState(null);

  const [clusterStoreData, setClusterStoreData] = useState([]);
  const [rerouteNeeded, setRerouteNeeded] = useState(false);
  const [isRerouting, setIsRerouting] = useState(false);

  const isMonthly = !!monthly_pjp;
  const currentPjp = isMonthly ? monthly_pjp : pjp;

  const [showResultsPopup, setShowResultsPopup] = useState(true);
  const [expandedOrderbookers, setExpandedOrderbookers] = useState({});
  const toggleOrderbookerCollapse = (ob) => {
    setExpandedOrderbookers((prev) => ({
      ...prev,
      [ob]: !prev[ob],
    }));
  };
  const [osrmRoutes, setOsrmRoutes] = useState({});

  useEffect(() => {
    if (clusters?.length > 0) {
      const initial = clusters.flatMap((cluster) =>
        cluster.stores.map((store) => ({
          ...store,
          cluster_id: cluster.cluster_id,
          old_cluster_id: cluster.cluster_id,
          is_removed: store.is_removed || false,
        }))
      );
      setClusterStoreData(initial);
    }
  }, [clusters]);
  const BASE_URL = process.env.REACT_APP_API_BASE_URL;

  // Reset highlighted store when changing day or orderbooker
  useEffect(() => {
    setHighlightedStore(null);
  }, [selectedDay, selectedOrderbookers]);

  const handleReroute = () => {
    if (!rerouteNeeded) return;
    setIsRerouting(true);

    axios
      .post(`${BASE_URL}/reroute-clusters`, {
        modified_clusters: clusterStoreData,
        distributor_id: distributorId,
        plan_duration: planDuration,
        num_orderbookers: numOrderbookers,
        store_type: storeType,
        retail_time: retailTime,
        wholesale_time: wholesaleTime,
        holidays: holidays,
        custom_days: planDuration === 'custom' ? custom_days || 7 : null,
        replicate: planDuration === 'custom' ? replicate : false,
      })
      .then((res) => {
        if (res.data.status === 'success') {
          // Update clusters and schedule with the new data from the backend
          setClusters(res.data.clusters);
          setSchedule(res.data.pjp);
          setRerouteNeeded(false);
        } else {
          console.error('Reroute error:', res.data.message);
        }
      })
      .catch((err) => {
        console.error('Rerouting failed:', err);
      })
      .finally(() => setIsRerouting(false));
  };

  const availableDays = useMemo(() => {
    const days = new Set();
    selectedOrderbookers.forEach((ob) => {
      const dayKeys = isMonthly ? Object.keys(monthly_pjp[ob] || {}) : Object.keys(pjp[ob] || {});
      dayKeys.forEach((d) => days.add(d));
    });
    return [...days].sort((a, b) =>
      isDateString(a) && isDateString(b) ? new Date(a) - new Date(b) : a.localeCompare(b)
    );
  }, [selectedOrderbookers, pjp, monthly_pjp, isMonthly]);

  useEffect(() => {
    const valid = selectedOrderbookers.some((ob) => {
      const days = isMonthly ? Object.keys(monthly_pjp[ob] || {}) : Object.keys(pjp[ob] || {});
      return days.includes(selectedDay);
    });
    if (!valid && availableDays.length > 0) {
      setSelectedDay(availableDays[0] || '');
    }
  }, [selectedOrderbookers, availableDays, selectedDay, pjp, monthly_pjp, isMonthly]);

  const handleOrderbookerChange = (e) => {
    const { value } = e.target;
    if (value.includes('all')) {
      const allSelected = selectedOrderbookers.length === Object.keys(currentPjp).length;
      setSelectedOrderbookers(allSelected ? [] : Object.keys(currentPjp));
    } else {
      setSelectedOrderbookers(
        typeof value === 'string' ? value.split(',') : value.filter((v) => v !== 'all')
      );
    }
  };

  const handleStoreClick = (store) => {
    setHighlightedStore(prevStore => 
      prevStore?.storeid === store.storeid ? null : store
    );
    
    if (mapInstance) {
      mapInstance.setView([store.latitude, store.longitude], 16, {
        animate: true,
        duration: 0.5
      });
    }
  };

  const isAllSelected = selectedOrderbookers.length === Object.keys(currentPjp).length;
  const clusterChartData = useMemo(() => {
    return clusters.map((cluster) => ({
      name: `Cluster ${cluster.cluster_id}`,
      cluster_id: cluster.cluster_id,
      value: cluster.stores.length,
    }));
  }, [clusters]);
  
  const filteredStoresByOrderbooker = useMemo(() => {
    const result = {};
    selectedOrderbookers.forEach((ob) => {
      const stores = currentPjp[ob]?.[selectedDay] || [];
      let filteredStores = stores.map((store, i) => ({ ...store, order: i + 1 }));
      
      // Apply search filter if there's a query
      if (searchQuery.trim()) {
        filteredStores = filteredStores.filter(store => 
          (store.storecode?.toLowerCase() || '').includes(searchQuery.toLowerCase()) ||
          (String(store.storeid) || '').toLowerCase().includes(searchQuery.toLowerCase())
        );
      }
      
      result[ob] = filteredStores;
    });
    return result;
  }, [currentPjp, selectedOrderbookers, selectedDay, searchQuery]);

  const filteredStoresWithCluster = useMemo(() => {
    const result = [];
    Object.entries(filteredStoresByOrderbooker).forEach(([ob, stores]) => {
      stores.forEach((store) => {
        const cluster = clusters.find((c) => c.stores.some((s) => s.storeid === store.storeid));
        result.push({ ...store, cluster_id: cluster?.cluster_id ?? 'Unknown', orderbooker: ob });
      });
    });
    return result;
  }, [filteredStoresByOrderbooker, clusters]);

  const obKeys = useMemo(() => {
    return Object.keys(currentPjp).sort((a, b) => parseInt(a) - parseInt(b));
  }, [currentPjp]);
  
  const orderbookerColors = useMemo(() => {
    const palette = generateColorPalette(obKeys.length);
    return obKeys.reduce((acc, ob, idx) => {
      acc[ob] = palette[idx];
      return acc;
    }, {});
  }, [obKeys]);
  
  const clusterColors = useMemo(() => {
    return clusters.reduce((acc, c, i) => {
      const hue = (i * 360) / clusters.length;
      acc[c.cluster_id] = `hsl(${hue}, 70%, 50%)`;
      return acc;
    }, {});
  }, [clusters]);

  const routePositionsPerOrderbooker = useMemo(() => {
    const routes = {};
    Object.keys(filteredStoresByOrderbooker).forEach((ob) => {
      routes[ob] = filteredStoresByOrderbooker[ob].map((s) => [s.latitude, s.longitude]);
    });
    return routes;
  }, [filteredStoresByOrderbooker]);

  useEffect(() => {
    const fetchRoutes = async () => {
      const newRoutes = {};
      for (const ob of selectedOrderbookers) {
        const coords = filteredStoresByOrderbooker[ob].map(s => [s.latitude, s.longitude]);
        if (coords.length > 1) {
          const route = await getRouteFromOSRM(coords);
          newRoutes[ob] = route;
        }
      }
      setOsrmRoutes(newRoutes);
    };
  
    if (showRoute) {
      fetchRoutes();
    }
  }, [selectedOrderbookers, filteredStoresByOrderbooker, showRoute]);
  

  const center = useMemo(() => {
    const all = filteredStoresWithCluster;
    if (!all.length) return [24.8607, 67.0011]; // Karachi fallback
    const lat = all.reduce((sum, s) => sum + s.latitude, 0) / all.length;
    const lon = all.reduce((sum, s) => sum + s.longitude, 0) / all.length;
    return [lat, lon];
  }, [filteredStoresWithCluster]);

  const barChartData = useMemo(() => {
    return obKeys.map((ob) => {
      const total = Object.values(currentPjp[ob] || {}).reduce((acc, day) => acc + day.length, 0);
      return { orderbooker: `OB ${ob}`, ob, totalStores: total };
    });
  }, [currentPjp, obKeys]);
  
  const clusterBarChartData = useMemo(() => {
    return clusters
      .filter((cluster) => cluster.cluster_id !== -1)
      .map((cluster) => ({
        cluster_id: cluster.cluster_id,                      
        label: `Orderbooker ${cluster.cluster_id}`,          
        total_workload: cluster.total_workload,
      }));
  }, [clusters]);
  
  const totalVisits = barChartData.reduce((acc, ob) => acc + ob.totalStores, 0);
  const avgVisits = totalVisits > 0 ? (totalVisits / Math.max(1, barChartData.length)).toFixed(2) : "0.00";
  const maxVisits = barChartData.length > 0 ? Math.max(...barChartData.map((ob) => ob.totalStores)) : 0;
  const [showSnackbar, setShowSnackbar] = useState(true);
  return (
    <Box sx={{ position: 'relative', width: '100%', height: '100vh', overflow: 'visible' }}>
      <StyledCSS />
      
      {showResultsPopup && (
        
          <Snackbar
          open={showSnackbar}
          autoHideDuration={5000}
          onClose={() => setShowSnackbar(false)}
          anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
        >
          <MuiAlert onClose={() => setShowSnackbar(false)} severity="success" sx={{ width: '100%' }}>
            Plan Generated Successfully! ID: {planId}
          </MuiAlert>
        </Snackbar>
     
      )}
      
      {/* Control Bar */}
      <Box
    
        sx={{
          position: 'absolute',
          top: 25,
          right: 10,
          zIndex: 2000,
          p: 1,
          display: 'flex',
          gap: 2,
          flexWrap: 'wrap',
          backgroundColor: theme.palette.background.paper,
          boxShadow: theme.shadows[3],
          borderRadius: 2,
          pointerEvents: 'auto',
          transform: 'translateZ(0)',
        }}
      >
        {/* <Typography variant="body2" sx={{ color: theme.palette.success.main, fontWeight: 'bold' }}>
  Plan ID: {planId} generated successfully!
</Typography> */}

        <Button
          variant="outlined"
          color="primary"
          startIcon={<SettingsIcon />}
          onClick={resetPlan}
        >
          Configure Plan
        </Button>
        <Button
          variant="contained"
          startIcon={<BarChartIcon />}
          onClick={() => setStatsModalOpen(true)}
          sx={{ backgroundColor: theme.palette.success.main }}
        >
          Statistics
        </Button>
        
        <Button
          variant="contained"
          startIcon={<GetAppIcon />}
          onClick={() => exportScheduleToExcel(pjp, monthly_pjp, isMonthly)}
        >
          Export Schedule
        </Button>
        <Button
          variant="contained"
          onClick={() => {
            if (showClusterMap) {
              setRerouteNeeded(false); // Hide map → reset reroute flag
            }
            setShowClusterMap(!showClusterMap);
          }}
          sx={{ backgroundColor: theme.palette.warning.main }}
        >
          {showClusterMap ? 'Hide Cluster Map' : 'Show Cluster Map'}
        </Button>
        <Button 
        variant="contained" 
        color="primary" 
        onClick={handleReroute} 
        disabled={!rerouteNeeded || isRerouting}
      >
        Dynamic Re-Route
      </Button>
      

      </Box>
     {isRerouting && (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        backgroundColor: 'rgba(255, 255, 255, 0.7)',
        zIndex: 3000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <LoadingSpinner size={60} />
    </Box>
  )}
  
      {/* Map */}
      {showClusterMap ? (
        <Box sx={{ height: '100vh', width: '100vw', position: 'absolute', top: 0, left: 0 }}>
          <ClusterView
            clusters={clusters}
            setClusters={setClusters}
            setSchedule={setSchedule}
            distributorId={distributorId}
            planDuration={planDuration}
            numOrderbookers={numOrderbookers}
            storeType={storeType}
            retailTime={retailTime}
            wholesaleTime={wholesaleTime}
            holidays={holidays}
            custom_days={custom_days}
            replicate={replicate}
            storeData={clusterStoreData}
  setStoreData={setClusterStoreData}
  rerouteNeeded={rerouteNeeded}
  setRerouteNeeded={setRerouteNeeded}
          />
        </Box>
      ) : (
        <MapWrapper>
          <MapContainer
            center={center}
            zoom={12}
            style={{ height: '100%', width: '100%' }}
            zoomControl={false}
            attributionControl={true}
          >
            <MapHandler onMapReady={map => {
              mapRef.current = map;
              setMapInstance(map);
            }} />
            
            <TileLayer 
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' 
            />
            
            <ZoomControl position="bottomright" />
            
            {filteredStoresWithCluster.map((store) => {
              if (!store || !store.latitude || !store.longitude) {
                return null;
              }
              
              const isHighlighted = highlightedStore?.storeid === store.storeid;
              return (
                <Marker
                  key={`${store.storeid}-${store.orderbooker}`}
                  position={[store.latitude, store.longitude]}
                  icon={createColoredMarker(
                    clusterColors[store.cluster_id] || 'gray',
                    isHighlighted,
                    36
                  )}
                  eventHandlers={{
                    click: () => {
                      handleStoreClick(store);
                    },
                    // mouseover: (e) => {
                    //   if (e.target && e.target._icon) {
                    //     e.target._icon.style.transform = 'scale(1.1) translateZ(0)';
                    //     e.target._icon.style.zIndex = '1000';
                    //   }
                    // },
                    // mouseout: (e) => {
                    //   if (e.target && e.target._icon && highlightedStore?.storeid !== store.storeid) {
                    //     e.target._icon.style.transform = 'scale(1) translateZ(0)';
                    //     e.target._icon.style.zIndex = '400';
                    //   }
                    // }
                  }}
                >
                  <Popup>
                    <Box sx={{ p: 1, minWidth: '180px' }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1 }}>
                        {store.storecode || `Store ${store.storeid}`}
                      </Typography>
                      <Grid container spacing={1}>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Order: {store.order}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Cluster: {store.cluster_id}
                          </Typography>
                        </Grid>
                        <Grid item xs={12}>
                          <Typography variant="body2" color="text.secondary">
                            Location: {store.latitude.toFixed(4)}, {store.longitude.toFixed(4)}
                          </Typography>
                        </Grid>
                        <Grid item xs={12} sx={{ mt: 1 }}>
                          <Button 
                            fullWidth
                            size="small" 
                            variant="contained"
                            onClick={() => handleStoreClick(store)}
                          >
                            {isHighlighted ? 'Deselect' : 'Select'}
                          </Button>
                        </Grid>
                      </Grid>
                    </Box>
                  </Popup>
                </Marker>
              );
            })}
            
            {showRoute &&
              selectedOrderbookers.map((ob) => {
                const positions = routePositionsPerOrderbooker[ob];
                const color = orderbookerColors[ob] || 'blue';
                return positions.length > 1 ? (
                  <Polyline
  key={`route-${ob}`}
  positions={osrmRoutes[ob] || []}
  pathOptions={{ color, weight: 4, opacity: 0.9 }}
/>

                ) : null;
              })}
              
            {/* Add FlyToMarker component */}
            {highlightedStore && mapInstance && <FlyToMarker store={highlightedStore} map={mapInstance} />}
          </MapContainer>
        </MapWrapper>
      )}

      {/* Collapse Sidebar */}
      <Collapse in={open} orientation="horizontal">
        <CollapsibleSidebar>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" fontWeight="bold">
              Store Routes
            </Typography>
            <IconButton onClick={() => setOpen(false)} size="small">
              <ChevronLeftIcon fontSize="small" />
            </IconButton>
          </Box>
          <Divider sx={{ mb: 2 }} />
          <ControlsContainer>
            <FormControl fullWidth variant="outlined" size="small" sx={{ mb: 2 }}>
              <InputLabel id="orderbooker-label">Orderbooker</InputLabel>
              <Select
                labelId="orderbooker-label"
                multiple
                value={selectedOrderbookers}
                onChange={handleOrderbookerChange}
                label="Orderbooker"
                renderValue={(selected) =>
                  isAllSelected ? (
                    'All Orderbookers'
                  ) : (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selected.map((value) => (
                        <Chip key={value} label={`OB ${value}`} sx={{ backgroundColor: '#0F0B5A', color: '#fff' }} />
                      ))}
                    </Box>
                  )
                }
              >
                <MenuItem value="all">
                  <Checkbox checked={isAllSelected} />
                  <ListItemText primary="Select All" />
                </MenuItem>
                {Object.keys(currentPjp).map((ob) => (
                  <MenuItem key={ob} value={ob}>
                    <Checkbox checked={selectedOrderbookers.includes(ob)} />
                    <ListItemText primary={`Orderbooker ${ob}`} />
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth variant="outlined" size="small" sx={{ mb: 2 }}>
              <InputLabel id="day-label">Day</InputLabel>
              <Select
                labelId="day-label"
                value={selectedDay}
                onChange={(e) => setSelectedDay(e.target.value)}
                label="Day"
              >
                {availableDays.map((day) => (
                  <MenuItem key={day} value={day}>
                    {isMonthly || isDateString(day) ? formatDateWithDay(day) : day}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControlLabel
              control={<Switch checked={showRoute} onChange={() => setShowRoute(!showRoute)} />}
              label="Show Route"
            />
{/*             
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
              <Button 
                size="small" 
                variant="outlined" 
                disabled={!highlightedStore}
                onClick={() => setHighlightedStore(null)}
                startIcon={<CloseIcon fontSize="small" />}
              >
                Clear Selection
              </Button>
            </Box> */}
          </ControlsContainer>

          <Divider sx={{ mb: 2 }} />
          
          <Box sx={{ mb: 2, px: 2 }}>
            <TextField
              fullWidth
              size="small"
              placeholder="Search stores..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon fontSize="small" />
                  </InputAdornment>
                ),
              }}
            />
          </Box>

          <StoreListContainer>
            {!selectedOrderbookers.length ? (
              <Typography variant="body1" sx={{ textAlign: 'center', py: 2 }}>
                Please select at least one orderbooker to view stores
              </Typography>
            ) : (
              selectedOrderbookers.map((ob) => (
                <Box key={ob} sx={{ mb: 3 }}>
                  <Box
  sx={{
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    mb: 1.5,
    pb: 1,
    borderBottom: '1px solid',
    borderColor: 'divider',
    cursor: 'pointer',
  }}
  onClick={() => toggleOrderbookerCollapse(ob)}
>
  <Typography
    variant="subtitle1"
    fontWeight="bold"
    sx={{
      display: 'flex',
      alignItems: 'center',
      '&::before': {
        content: '""',
        display: 'inline-block',
        width: 12,
        height: 12,
        borderRadius: '50%',
        backgroundColor: orderbookerColors[ob] || 'gray',
        marginRight: 1,
      },
    }}
  >
    Orderbooker {ob}
  </Typography>
  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
    <Typography variant="body2" color="text.secondary">
      {filteredStoresByOrderbooker[ob]?.length || 0}
    </Typography>
    {expandedOrderbookers[ob] ? (
      <ExpandLessIcon fontSize="small" />
    ) : (
      <ExpandMoreIcon fontSize="small" />
    )}
  </Box>
</Box>

                  

              <Collapse in={!!expandedOrderbookers[ob]}>
                {filteredStoresByOrderbooker[ob]?.length > 0 ? (
                  filteredStoresByOrderbooker[ob].map((store) => (
                    <StoreCard
                      key={store.storeid}
                      clusterColor={clusterColors[store.cluster_id]}
                      isHighlighted={highlightedStore?.storeid === store.storeid}
                      onClick={() => handleStoreClick(store)}
                      className={`store-card-transition ${highlightedStore?.storeid === store.storeid ? 'store-card-highlighted' : ''}`}
                      sx={{
                        boxShadow: highlightedStore?.storeid === store.storeid 
                          ? `0 4px 12px ${clusterColors[store.cluster_id]}66` 
                          : 1
                      }}
                    >
                      <NumberBadge clusterColor={clusterColors[store.cluster_id]}>
                        {store.order}
                      </NumberBadge>
                      <Box sx={{ flexGrow: 1 }}>
                        <Typography variant="subtitle2" fontWeight="bold">
                          {store.storecode || `Store ${store.storeid}`}
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 2, mt: 0.5 }}>
                        </Box>
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                          {store.latitude.toFixed(4)}, {store.longitude.toFixed(4)}
                        </Typography>
                      </Box>
                    </StoreCard>
                  ))
                ) : (
                  <Paper sx={{ p: 2, textAlign: 'center', backgroundColor: '#f5f5f5', borderRadius: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      No stores for this day
                    </Typography>
                  </Paper>
                )}
              </Collapse>

                </Box>
              ))
            )}
          </StoreListContainer>
        </CollapsibleSidebar>
      </Collapse>

      {/* Show when collapsed */}
      {!open && (
        <MenuButton onClick={() => setOpen(true)} sx={{ top: 20, left: 16 }}>
          <MenuIcon />
        </MenuButton>
      )}

      {/* Statistics Modal */}
      <Modal open={statsModalOpen} onClose={() => setStatsModalOpen(false)}>
        <Box sx={modalStyle(theme)}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
            <Typography variant="h5">Permanent Journey Plans - Statistics</Typography>
            <IconButton onClick={() => setStatsModalOpen(false)}>
              <CloseIcon />
            </IconButton>
          </Box>
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} sm={6} md={3}>
              <SummaryCard>
                <Typography variant="h6">Total OBs</Typography>
                <Typography variant="h4">{barChartData.length}</Typography>
              </SummaryCard>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <SummaryCard>
                <Typography variant="h6">Total Visits</Typography>
                <Typography variant="h4">{totalVisits}</Typography>
              </SummaryCard>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <SummaryCard>
                <Typography variant="h6">Avg Visits/OB</Typography>
                <Typography variant="h4">{avgVisits}</Typography>
              </SummaryCard>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <SummaryCard>
                <Typography variant="h6">Max Visits</Typography>
                <Typography variant="h4">{maxVisits}</Typography>
              </SummaryCard>
            </Grid>
          </Grid>
          <Grid container spacing={4}>
            <Grid item xs={12} md={6}>
              <StyledPaper>
                <Typography variant="h6" align="center">
                  Distribution of Visits
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie data={clusterChartData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label>
                      {clusterChartData.map((entry, i) => (
                        <Cell key={`cell-${i}`} fill={clusterColors[entry.cluster_id] || '#ccc'} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </StyledPaper>
            </Grid>
            <Grid item xs={12} md={6}>
              <StyledPaper>
                <Typography variant="h6" align="center">
                  Workload Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={clusterBarChartData}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="label" />
                    <YAxis />
                    <RechartsTooltip />
                    <Legend />
                    <Brush dataKey="label" height={30} stroke="#8884d8" />
                    <Bar dataKey="total_workload" name="Total Workload (Service + Travel time)">
                      {clusterBarChartData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={clusterColors[entry.cluster_id] || '#ccc'}
                          cursor="pointer"
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </StyledPaper>
            </Grid>
          </Grid>
        </Box>
      </Modal>
    </Box>
  );
}

export default CombinedView;