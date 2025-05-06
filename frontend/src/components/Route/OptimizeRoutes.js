import React, { useEffect, useState, useRef } from 'react';
import InputForm from './InputForm';
import CombinedView from './CombinedView';
import ErrorAlert from '../ErrorAlert';
import LoadingSpinner from '../LoadingSpinner';
import theme from '../../theme';
import {
  Paper,
  Box,
  Typography,
  IconButton,
  Button,
  Divider,
  Card,
  CardContent,
  Chip,
  Modal,
  Fade,
  useMediaQuery,
  Stack,
  Tooltip,
} from '@mui/material';
import { styled, alpha } from '@mui/material/styles';
import { useNavigate } from 'react-router-dom';
import { MapContainer, TileLayer, Marker, Popup, Rectangle, useMap, ZoomControl } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import 'leaflet-draw';
import 'leaflet-draw/dist/leaflet.draw.css';
import CloseIcon from '@mui/icons-material/Close';
import StorefrontIcon from '@mui/icons-material/Storefront';
import SettingsIcon from '@mui/icons-material/Settings';
import CheckBoxOutlineBlankIcon from '@mui/icons-material/CheckBoxOutlineBlank';
import CheckBoxIcon from '@mui/icons-material/CheckBox';
import SelectAllIcon from '@mui/icons-material/SelectAll';
import InfoIcon from '@mui/icons-material/Info';
import { motion } from "framer-motion";

// Styled Components
const MapWrapper = styled(Box)(({ theme }) => ({
  position: 'relative',
  height: 'calc(100vh - 100px)', // Reduced height to prevent scrolling
  width: '100%',
  overflow: 'hidden', // Prevent scrolling
  '& .leaflet-container': {
    height: '100%',
    width: '100%',
    zIndex: 0,
  },
  '& .leaflet-control-zoom': {
    borderRadius: theme.shape.borderRadius,
    boxShadow: theme.shadows[3],
  },
  '& .leaflet-popup-content-wrapper': {
    borderRadius: theme.shape.borderRadius,
    padding: theme.spacing(0.5),
    boxShadow: theme.shadows[2],
    minWidth: 220,
  },
  '& .leaflet-popup-tip': {
    boxShadow: theme.shadows[2],
  },
}));

const ControlPanel = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(2.5),
  left: theme.spacing(2.5),
  zIndex: 1000,
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1.5),
  maxWidth: '300px',
}));

const StyledLegend = styled(Card)(({ theme }) => ({
  position: 'absolute',
  bottom: theme.spacing(20), // Moved up significantly to ensure visibility
  right: theme.spacing(3),
  zIndex: 1000,
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[6],
  maxWidth: '250px',
  backgroundColor: alpha(theme.palette.background.paper, 0.95),
  backdropFilter: 'blur(5px)',
}));

const LegendHeader = styled(Box)(({ theme }) => ({
  backgroundColor: alpha(theme.palette.primary.main, 0.9),
  color: theme.palette.primary.contrastText,
  padding: theme.spacing(1, 2),
  borderTopLeftRadius: theme.shape.borderRadius,
  borderTopRightRadius: theme.shape.borderRadius,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
}));

const LegendItem = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(1),
  '&:not(:last-child)': {
    borderBottom: `1px solid ${alpha(theme.palette.divider, 0.5)}`,
  },
}));

const MarkerIcon = styled(Box)(({ color, theme }) => ({
  display: 'inline-block',
  width: 24,
  height: 24,
  marginRight: theme.spacing(1.5),
  backgroundColor: 'transparent',
  backgroundImage: `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="${encodeURIComponent(
    color
  )}" stroke="%23ffffff" stroke-width="2"><path d="M21 10c0 6-9 13-9 13s-9-7-9-13a9 9 0 1 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>')`,
  backgroundRepeat: 'no-repeat',
  backgroundSize: 'contain',
}));

const ActionButton = styled(Button)(({ theme, colorScheme = 'primary' }) => ({
  borderRadius: theme.shape.borderRadius * 2,
  fontWeight: 500,
  textTransform: 'none',
  boxShadow: theme.shadows[2],
  padding: theme.spacing(1, 2),
  transition: 'all 0.3s ease',
  '&:hover': {
    boxShadow: theme.shadows[4],
    transform: 'translateY(-2px)',
  },
}));

const StyledPopupContent = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1.5),
}));

const StyledChip = styled(Chip)(({ theme }) => ({
  fontWeight: 600,
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[1],
  '& .MuiChip-deleteIcon': {
    color: alpha(theme.palette.text.primary, 0.7),
    '&:hover': {
      color: theme.palette.error.main,
    },
  },
}));

const WelcomePanel = styled(Paper)(({ theme }) => ({
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  zIndex: 1400,
  padding: theme.spacing(4),
  borderRadius: theme.shape.borderRadius * 2,
  maxWidth: 450,
  boxShadow: theme.shadows[8],
  textAlign: 'center',
  background: 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)',
}));

const ConfigCard = styled(Card)(({ theme }) => ({
  width: '90%',
  maxWidth: 800,
  maxHeight: '90vh',
  overflow: 'auto',
  padding: theme.spacing(3, 4),
  borderRadius: theme.shape.borderRadius * 2,
  boxShadow: theme.shadows[10],
  backgroundImage: 'linear-gradient(to bottom, #ffffff, #fafafa)',
  '&::-webkit-scrollbar': {
    width: '10px',
  },
  '&::-webkit-scrollbar-track': {
    background: alpha(theme.palette.primary.main, 0.05),
    borderRadius: '10px',
  },
  '&::-webkit-scrollbar-thumb': {
    background: alpha(theme.palette.primary.main, 0.2),
    borderRadius: '10px',
    '&:hover': {
      background: alpha(theme.palette.primary.main, 0.3),
    }
  }
}));

// Fix for default marker icons in Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Create colored marker function
function createColoredMarker(color) {
  return new L.DivIcon({
    html: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" 
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
}

// Component to handle area selection
function AreaSelector({ stores, selectedStores, setSelectedStores, selectionActive }) {
  const map = useMap();
  const [startPoint, setStartPoint] = useState(null);
  const [endPoint, setEndPoint] = useState(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const selectionRectRef = useRef(null);
  const rectLayerRef = useRef(null);

  useEffect(() => {
    if (!selectionActive) return;

    const onMouseDown = (e) => {
      if (!selectionActive) return;
      
      // Check if this is the left mouse button
      if (e.button !== 0) return;
      
      // Get the map container element
      const mapContainer = map.getContainer();
      
      // Calculate click position relative to the map container
      const rect = mapContainer.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      // Convert to map coordinates
      const containerPoint = L.point(x, y);
      const latlng = map.containerPointToLatLng(containerPoint);
      
      setIsSelecting(true);
      setStartPoint(latlng);
      setEndPoint(latlng);
      
      // Prevent map drag during selection
      map.dragging.disable();
      
      // Prevent default behavior
      e.stopPropagation();
      e.preventDefault();
    };

    const onMouseMove = (e) => {
      if (!isSelecting) return;
      
      // Calculate mouse position relative to the map container
      const mapContainer = map.getContainer();
      const rect = mapContainer.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      // Convert to map coordinates
      const containerPoint = L.point(x, y);
      const latlng = map.containerPointToLatLng(containerPoint);
      
      setEndPoint(latlng);
      
      // Update the selection rectangle visually during drag
      if (rectLayerRef.current) {
        map.removeLayer(rectLayerRef.current);
      }
      
      if (startPoint) {
        const bounds = L.latLngBounds(startPoint, latlng);
        rectLayerRef.current = L.rectangle(bounds, {
          color: 'rgba(65, 105, 225, 0.6)',
          weight: 2,
          fillOpacity: 0.3,
          dashArray: '5, 5',
          lineCap: 'round'
        }).addTo(map);
      }
    };

    const onMouseUp = (e) => {
      if (!isSelecting) return;
      
      setIsSelecting(false);
      map.dragging.enable();
      
      // Clean up temporary rectangle
      if (rectLayerRef.current) {
        map.removeLayer(rectLayerRef.current);
        rectLayerRef.current = null;
      }
      
      if (!startPoint || !endPoint) return;
      
      // Get the bounds of the rectangle
      const bounds = L.latLngBounds(startPoint, endPoint);
      
      // Find stores within the selection area
      const storesInArea = stores.filter(store => {
        if (!store.latitude || !store.longitude) return false;
        const lat = parseFloat(store.latitude);
        const lng = parseFloat(store.longitude);
        if (isNaN(lat) || isNaN(lng)) return false;
        
        return bounds.contains(L.latLng(lat, lng));
      });
      
      // Add the selected stores to our list
      const storeIds = storesInArea.map(store => store.storeid);
      
      if (storeIds.length > 0) {
        // Toggle selection: if all selected stores are already in selectedStores, remove them; otherwise add them
        const allAlreadySelected = storeIds.every(id => selectedStores.includes(id));
        
        if (allAlreadySelected) {
          setSelectedStores(prev => prev.filter(id => !storeIds.includes(id)));
        } else {
          setSelectedStores(prev => {
            const newSelection = [...prev];
            storeIds.forEach(id => {
              if (!newSelection.includes(id)) {
                newSelection.push(id);
              }
            });
            return newSelection;
          });
        }
      }
      
      // Clear the selection points
      setStartPoint(null);
      setEndPoint(null);
    };

    // Add event listeners to the map container element
    const mapContainer = map.getContainer();
    mapContainer.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);

    return () => {
      mapContainer.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      
      // Clean up any remaining rectangle
      if (rectLayerRef.current) {
        map.removeLayer(rectLayerRef.current);
        rectLayerRef.current = null;
      }
    };
  }, [map, selectionActive, isSelecting, startPoint, endPoint, stores, selectedStores, setSelectedStores]);

  // Calculate the bounds for the Rectangle component
  const bounds = startPoint && endPoint ? [
    [Math.min(startPoint.lat, endPoint.lat), Math.min(startPoint.lng, endPoint.lng)],
    [Math.max(startPoint.lat, endPoint.lat), Math.max(startPoint.lng, endPoint.lng)]
  ] : null;

  return bounds ? (
    <Rectangle 
      bounds={bounds} 
      pathOptions={{ color: 'rgba(65, 105, 225, 0.6)', fillOpacity: 0.3, weight: 2, dashArray: '5, 5' }}
      ref={selectionRectRef}
    />
  ) : null;
}

// Component to set map bounds based on stores
function FitBoundsToStores({ stores }) {
  const map = useMap();
  
  useEffect(() => {
    if (stores.length > 0) {
      const validStores = stores.filter(
        store => store.latitude && store.longitude && 
               !isNaN(parseFloat(store.latitude)) && 
               !isNaN(parseFloat(store.longitude))
      );
      
      if (validStores.length > 0) {
        // Create bounds from all valid store coordinates
        const points = validStores.map(store => [
          parseFloat(store.latitude),
          parseFloat(store.longitude)
        ]);
        
        const bounds = L.latLngBounds(points);
        map.fitBounds(bounds, { padding: [100, 100] }); // Increased padding to ensure stores aren't at edges
      }
    }
  }, [stores, map]);
  
  return null;
}

// Add this new component to disable map interaction when needed
function MapInteractionController({ disableInteraction }) {
  const map = useMap();
  
  useEffect(() => {
    if (disableInteraction) {
      map.dragging.disable();
      map.touchZoom.disable();
      map.doubleClickZoom.disable();
      map.scrollWheelZoom.disable();
      map.boxZoom.disable();
      map.keyboard.disable();
      if (map.tap) map.tap.disable();
    } else {
      map.dragging.enable();
      map.touchZoom.enable();
      map.doubleClickZoom.enable();
      map.scrollWheelZoom.enable();
      map.boxZoom.enable();
      map.keyboard.enable();
      if (map.tap) map.tap.enable();
    }
  }, [map, disableInteraction]);
  
  return null;
}

function Optimize() {
  const [pjp, setPjp] = useState(null);
  const [clusters, setClusters] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);
  const [stores, setStores] = useState([]);
  const [mapCenter, setMapCenter] = useState([24.8607, 67.0011]);
  const [selectedStores, setSelectedStores] = useState([]);
  const [configOpen, setConfigOpen] = useState(false);
  const [showWelcomePanel, setShowWelcomePanel] = useState(true);
  const [storeStats, setStoreStats] = useState({ total: 0, retail: 0, wholesale: 0 });
  const [mapSelection, setMapSelection] = useState(false);
  const [areaSelection, setAreaSelection] = useState(false);
  const [showLegend, setShowLegend] = useState(true);
  const mapRef = useRef(null);
  const [planId, setPlanId] = useState(null);
  const BASE_URL = process.env.REACT_APP_API_BASE_URL;
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const navigate = useNavigate();

  useEffect(() => {
    const storedDistributorId = localStorage.getItem('distributor_id');
    const shouldResetPlan = localStorage.getItem("reset_plan");
  
    if (!storedDistributorId) {
      navigate('/');
    } else {
      fetchStoreData(parseInt(storedDistributorId, 10));
    }
  
    if (shouldResetPlan === "true") {
      resetPlan();
      localStorage.removeItem("reset_plan");
    }
  }, [navigate]);
  

  const fetchStoreData = async (distributorId) => {
    try {
      const response = await fetch(`${BASE_URL}/api/get_stores?distributorid=${distributorId}`);
      console.log('Response:', response); // Log the response object
      if (!response.ok) throw new Error(`Failed to fetch store data. Status: ${response.status}`);
      const data = await response.json();

      if (!data.stores || data.stores.length === 0) throw new Error('No stores found for this distributor.');
      setStores(data.stores);

      // Calculate store statistics
      const retail = data.stores.filter(store => store.channeltypeid === 1).length;
      const wholesale = data.stores.filter(store => store.channeltypeid !== 1).length;
      setStoreStats({
        total: data.stores.length,
        retail,
        wholesale
      });

      const validStores = data.stores.filter(
        (store) => store.latitude && store.longitude && !isNaN(parseFloat(store.latitude)) && !isNaN(parseFloat(store.longitude))
      );

      if (validStores.length > 0) {
        const avgLat = validStores.reduce((sum, s) => sum + parseFloat(s.latitude), 0) / validStores.length;
        const avgLng = validStores.reduce((sum, s) => sum + parseFloat(s.longitude), 0) / validStores.length;
        setMapCenter([avgLat, avgLng]);
      }
    } catch (err) {
      console.error('Error fetching store data:', err);
      setError(err.message);
    }
  };

  const handleFormSubmit = async (formData) => {
    // Add selected stores to form data if in map selection mode
    const finalFormData = {
      ...formData,
      selected_stores: (mapSelection || areaSelection) ? selectedStores : []
    };

    setError(null);
    setLoading(true);
    setPjp(null);
    setClusters([]);
    setData(finalFormData);
    setShowWelcomePanel(false);
    setConfigOpen(false);
    setPlanId(null);
    
    try {
      const response = await fetch(`${BASE_URL}/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(finalFormData),
      });
      const result = await response.json();
      setLoading(false);
      if (result.status === 'success') {
        setPjp(result.pjp || null);
        setClusters(result.clusters || []);
        setPlanId(result.plan_id || `PJP-${Math.floor(Math.random() * 10000)}`);  // fallback for safety
      } else {
        setError(result.message || 'Unknown error occurred.');
      }
    } catch (err) {
      setLoading(false);
      setError(err.message);
    }
  };
  
  const toggleStoreSelection = (storeId) => {
    setSelectedStores(prev => {
      if (prev.includes(storeId)) {
        return prev.filter(id => id !== storeId);
      } else {
        return [...prev, storeId];
      }
    });
  };

  const toggleMapSelection = () => {
    if (areaSelection && !mapSelection) {
      setAreaSelection(false);
    }
    setMapSelection(!mapSelection);
  };
  
  const toggleAreaSelection = () => {
    if (mapSelection && !areaSelection) {
      setMapSelection(false);
    }
    setAreaSelection(!areaSelection);
  };

  const clearAllSelections = () => {
    setSelectedStores([]);
  };

  const getMarkerIcon = (store) => {
    const color = selectedStores.includes(store.storeid) 
      ? '#ff3b30' // red for selected
      : (store.channeltypeid === 1 ? '#007aff' : '#34c759'); // blue for retail, green for wholesale
      
    return createColoredMarker(color);
  };

  const renderLegend = () => (
    <Fade in={showLegend}>
      <StyledLegend elevation={6}>
        <LegendHeader>
          <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>Map Legend</Typography>
          <IconButton 
            size="small" 
            onClick={() => setShowLegend(false)} 
            sx={{ color: 'inherit', p: 0.5 }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        </LegendHeader>
        <CardContent sx={{ p: 0 }}>
          <LegendItem>
            <MarkerIcon color="#007aff" />
            <Typography variant="body2">Retail Store</Typography>
          </LegendItem>
          <LegendItem>
            <MarkerIcon color="#34c759" />
            <Typography variant="body2">Wholesale Store</Typography>
          </LegendItem>
          {(mapSelection || areaSelection) && (
            <LegendItem>
              <MarkerIcon color="#ff3b30" />
              <Typography variant="body2">Selected Store</Typography>
            </LegendItem>
          )}
        </CardContent>
      </StyledLegend>
    </Fade>
  );

  // Add overlay component to block map interactions
  const renderOverlay = () => (
    <Box 
      sx={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 1100,
        backgroundColor: 'rgba(0, 0, 0, 0.3)',
        backdropFilter: 'blur(2px)',
      }}
    />
  );

  const resetPlan = () => {
    setPjp(null);
    setClusters([]);
    setShowWelcomePanel(false);
  };

  const renderWelcomePanel = () => (
    <Fade in={showWelcomePanel}>
      <WelcomePanel elevation={8}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Typography 
            variant="h5" 
            component="h2" 
            sx={{ 
              color: 'primary.main', 
              fontWeight: 'bold', 
              mb: 2.5,
              backgroundImage: 'linear-gradient(45deg, #161181, #3F3A9D)',
              backgroundClip: 'text',
              textFillColor: 'transparent',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
           Start Your Optimized Journey with Rahguzar
          </Typography>
          
          <motion.img 
            src="/3.png"
            alt="Welcome Graphic"
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.6, ease: "easeOut" }}
            style={{ 
              width: 250, 
              marginBottom: 24,
              borderRadius: '12px',
              boxShadow: '0 8px 24px rgba(22, 17, 129, 0.12)'
            }}
          />
          
          <Typography variant="body1" sx={{ mb: 4, color: 'text.secondary' }}>
            Configure your plan settings and select stores on the map to begin optimization.
          </Typography>
          
          <Button 
            variant="contained" 
            color="primary"
            onClick={() => setShowWelcomePanel(false)}
            sx={{ 
              borderRadius: 4,
              px: 3,
              py: 1.2,
              fontWeight: 600,
              boxShadow: 3,
              background: 'linear-gradient(45deg, #161181, #3F3A9D)',
              transition: 'all 0.3s ease',
              '&:hover': {
                boxShadow: 6,
                transform: 'translateY(-2px)'
              }
            }}
          >
            Get Started
          </Button>
        </motion.div>
      </WelcomePanel>
    </Fade>
  );
  
  const renderSelectionControls = () => (
    <ControlPanel>
      <ActionButton
        variant="contained"
        color="primary"
        startIcon={<SettingsIcon />}
        onClick={() => setConfigOpen(true)}
        fullWidth
      >
        Configure Plan
      </ActionButton>
      
      {!pjp && (
        <Stack spacing={1.5} width="100%">
          <ActionButton
            variant="contained"
            startIcon={mapSelection ? <CheckBoxIcon /> : <CheckBoxOutlineBlankIcon />}
            onClick={toggleMapSelection}
            colorScheme={mapSelection ? 'primary' : 'default'}
            sx={{ 
              backgroundColor: mapSelection ? '#4E89AE' : alpha('#E1ECF4', 0.9),
              color: mapSelection ? '#ffffff' : '#1e3a5f',
              '&:hover': {
                backgroundColor: mapSelection ? '#3A6E8C' : alpha('#d0e4f2', 0.9)
              }
            }}
            fullWidth
          >
            {mapSelection ? "Point Selection: ON" : "Enable Store Selection"}
          </ActionButton>
          
          <ActionButton
            variant="contained"
            startIcon={areaSelection ? <CheckBoxIcon /> : <SelectAllIcon />}
            onClick={toggleAreaSelection}
            colorScheme={areaSelection ? 'success' : 'default'}
            sx={{ 
              backgroundColor: areaSelection ? '#59A96A' : alpha('#E2F3E4', 0.9),
              color: areaSelection ? '#ffffff' : '#2f5233',
              '&:hover': {
                backgroundColor: areaSelection ? '#4B8859' : alpha('#cbe8ce', 0.9)
              }
            }}
            fullWidth
          >
            {areaSelection ? "Area Selection: ON" : "Enable Area Selection"}
          </ActionButton>
  
          {(mapSelection || areaSelection) && selectedStores.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <StyledChip 
                label={`${selectedStores.length} Stores Selected`}
                sx={{ 
                  width: '100%',
                  justifyContent: 'space-between',
                  backgroundColor: alpha('#FFD166', 0.9),
                  color: '#664500',
                  fontWeight: 500,
                  py: 0.5
                }}
                onDelete={clearAllSelections}
              />
            </motion.div>
          )}

          {!showLegend && (
            <Button
              variant="outlined"
              color="primary"
              size="small"
              onClick={() => setShowLegend(true)}
              startIcon={<InfoIcon />}
              sx={{ alignSelf: 'flex-start', mt: 1 }}
            >
              Show Legend
            </Button>
          )}
        </Stack>
      )}
    </ControlPanel>
  );

  return (
    <Box sx={{ 
      position: 'relative', 
      width: '100%', 
      height: '100vh', 
      overflow: 'hidden' // Prevent scrolling at container level
    }}>
      {!(pjp && clusters.length > 0) ? (
        <MapWrapper>
          <MapContainer 
            center={mapCenter} 
            zoom={12} 
            style={{ height: '100%', width: '100%' }}
            ref={mapRef}
            zoomControl={false}
            scrollWheelZoom={true} // Enable zoom for better interaction
          >
            <TileLayer 
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
            <ZoomControl position="bottomright" />
            <FitBoundsToStores stores={stores} />
            <MapInteractionController disableInteraction={showWelcomePanel} />
            <AreaSelector 
              stores={stores} 
              selectedStores={selectedStores} 
              setSelectedStores={setSelectedStores} 
              selectionActive={areaSelection && !showWelcomePanel}
            />
            {stores.map((store) => {
              if (store.latitude && store.longitude && !isNaN(parseFloat(store.latitude)) && !isNaN(parseFloat(store.longitude))) {
                const position = [parseFloat(store.latitude), parseFloat(store.longitude)];
                return (
                  <Marker
                    key={store.storeid}
                    position={position}
                    icon={getMarkerIcon(store)} 
                    eventHandlers={{
                      click: (e) => {
                        if (mapSelection && !showWelcomePanel) {
                          toggleStoreSelection(store.storeid);
                          L.DomEvent.stopPropagation(e);
                        }
                      }
                    }}
                  >
                    <Popup>
                      <StyledPopupContent>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                          Store: {store.storecode}
                        </Typography>
                        <Divider sx={{ mb: 1.5 }} />
                        <Stack spacing={1}>
                          <Typography variant="body2">
                            <strong>Channel:</strong> {store.channeltypeid === 1 ? 'Retail' : 'Wholesale'}
                          </Typography>
                          <Typography variant="body2">
                            <strong>Status:</strong> {store.status === 1 ? 'Active' : 'Inactive'}
                          </Typography>
                        </Stack>
                        {(mapSelection || areaSelection) && !showWelcomePanel && (
                          <Button 
                            variant={selectedStores.includes(store.storeid) ? "contained" : "outlined"}
                            size="small" 
                            color={selectedStores.includes(store.storeid) ? "error" : "primary"}
                            onClick={() => toggleStoreSelection(store.storeid)}
                            sx={{ mt: 2, width: '100%', borderRadius: 2 }}
                          >
                            {selectedStores.includes(store.storeid) ? "Remove Selection" : "Select Store"}
                          </Button>
                        )}
                      </StyledPopupContent>
                    </Popup>
                  </Marker>
                );
              }
              return null;
            })}
          </MapContainer>
  
          {showWelcomePanel && renderOverlay()}
          {showWelcomePanel && renderWelcomePanel()}
          {!showWelcomePanel && showLegend && renderLegend()}
          {!showWelcomePanel && renderSelectionControls()}
  
          {/* Config Modal */}
          <Modal
            open={configOpen}
            onClose={() => setConfigOpen(false)}
            closeAfterTransition
            BackdropProps={{ timeout: 500 }}
            sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          >
            <Fade in={configOpen}>
              <ConfigCard>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Typography 
                    variant="h5" 
                    sx={{ 
                      fontWeight: 600, 
                      color: 'primary.main',
                      backgroundImage: 'linear-gradient(45deg, #161181, #3F3A9D)',
                      backgroundClip: 'text',
                      textFillColor: 'transparent',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                    }}
                  >
                    Configure PJP Settings
                  </Typography>
                  <IconButton 
                    onClick={() => setConfigOpen(false)} 
                    sx={{ 
                      bgcolor: 'rgba(0, 0, 0, 0.04)',
                      '&:hover': {
                        bgcolor: 'rgba(0, 0, 0, 0.08)',
                      }
                    }}
                  >
                    <CloseIcon />
                  </IconButton>
                </Box>
  
                <Box sx={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  mb: 3,
                  flexDirection: isMobile ? 'column' : 'row',
                  gap: isMobile ? 1 : 0
                }}>
                  <Tooltip title="Total number of stores available">
                    <StyledChip 
                      icon={<StorefrontIcon />} 
                      label={`${storeStats.total} Total Stores`} 
                      color="primary" 
                      variant="outlined" 
                    />
                  </Tooltip>
                  <Tooltip title="Number of retail stores">
                    <StyledChip 
                      label={`${storeStats.retail} Retail`} 
                      color="info" 
                      variant="outlined" 
                    />
                  </Tooltip>
                  <Tooltip title="Number of wholesale stores">
                    <StyledChip 
                      label={`${storeStats.wholesale} Wholesale`} 
                      color="success" 
                      variant="outlined" 
                    />
                  </Tooltip>
                </Box>
  
                <Divider sx={{ mb: 3 }} />
  
                {(mapSelection || areaSelection) && selectedStores.length > 0 && (
                  <Box 
                    sx={{ 
                      mb: 3, 
                      p: 2.5, 
                      borderRadius: 2, 
                      bgcolor: alpha(theme.palette.info.main, 0.1),
                      border: `1px solid ${alpha(theme.palette.info.main, 0.3)}`,
                    }}
                  >
                    <Typography variant="subtitle1" sx={{ color: 'info.dark', fontWeight: 600, mb: 1 }}>
                      Store Selection Mode: {selectedStores.length} stores selected
                    </Typography>
                    <Typography variant="body2" sx={{ color: 'info.dark' }}>
                      Only selected stores will be included in the optimization.
                    </Typography>
                  </Box>
                )}
  
                <InputForm 
                  onSubmit={handleFormSubmit} 
                  selectedStores={selectedStores}
                  mapSelectionEnabled={mapSelection || areaSelection}
                />
              </ConfigCard>
            </Fade>
          </Modal>
  
          {/* Error and Loading indicators */}
          <Box sx={{ position: 'absolute', top: 20, right: 20, zIndex: 1300 }}>
            {loading && <LoadingSpinner />}
            {error && <ErrorAlert message={error} />}
          </Box>
        </MapWrapper>
      ) : (
        <Paper 
          sx={{ 
            position: 'relative',
            height: '100vh',
            width: '100%',
            zIndex: 1300,
            bgcolor: 'background.default',
            borderRadius: 0,
            overflow: 'auto'
          }}
        >
          <Box sx={{ p: 0 }}>
            <Box>
              <CombinedView 
                pjp={pjp} 
                clusters={clusters}  
                setClusters={setClusters}
                setSchedule={setPjp}
                distributorId={data?.distributor_id}
                planDuration={data?.plan_duration}
                numOrderbookers={data?.num_orderbookers}
                storeType={data?.store_type}
                retailTime={data?.retail_time}
                wholesaleTime={data?.wholesale_time}
                holidays={data?.holidays}
                custom_days={data?.custom_days}
                replicate={data?.replicate}
                selectedStores={data?.selected_stores}
                planId={planId}
                resetPlan={resetPlan}
              />
            </Box>
          </Box>
        </Paper>
      )}
    </Box>
  );
}

export default Optimize;