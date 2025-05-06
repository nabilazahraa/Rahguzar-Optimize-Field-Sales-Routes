import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Button,
  Container,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Chip,
  Grid,
  Divider,
  Drawer,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Collapse,
  Badge,
  TextField,
  InputAdornment
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import FilterListIcon from '@mui/icons-material/FilterList';
import SearchIcon from '@mui/icons-material/Search';
import ExpandLess from '@mui/icons-material/ExpandLess';
import ExpandMore from '@mui/icons-material/ExpandMore';
import { useNavigate } from 'react-router-dom';

function StoreManagement() {
  const [stores, setStores] = useState([]);
  const [filteredStores, setFilteredStores] = useState([]);
  const [selectedStore, setSelectedStore] = useState(null);
  const [distributorId, setDistributorId] = useState('');
  const [filterValues, setFilterValues] = useState({});
  const [selectedFilters, setSelectedFilters] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [expandedCategories, setExpandedCategories] = useState({});
  const [activeFilterCount, setActiveFilterCount] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const distributorId = localStorage.getItem("distributor_id");
    if (!distributorId) {
      navigate("/");
    }
  }, [navigate]);

  useEffect(() => {
    const storedDistributorId = localStorage.getItem("distributor_id");
    if (storedDistributorId) {
      setDistributorId(parseInt(storedDistributorId, 10));
      fetchStoreData(parseInt(storedDistributorId, 10));
    } else {
      console.error("No distributor ID found in localStorage. Please log in.");
    }
  }, []);

  // Calculate total active filters
  useEffect(() => {
    const totalFilters = Object.values(selectedFilters).reduce(
      (count, values) => count + (values?.length || 0), 
      0
    );
    setActiveFilterCount(totalFilters);
  }, [selectedFilters]);

  // Apply filters whenever stores, selectedFilters, or searchQuery changes
  useEffect(() => {
    if (stores.length > 0) {
      applyFiltersEffect();
    }
  }, [stores, selectedFilters, searchQuery]);

  const fetchStoreData = async (id) => {
    setLoading(true);
    setError(null);
    try {
      console.log("Fetching stores for Distributor ID:", id);
      const response = await fetch(
        `${BASE_URL}/api/get_stores?distributorid=${id}`
      );
      if (!response.ok) {
        throw new Error(`Failed to fetch store data. Status: ${response.status}`);
      }
      const data = await response.json();
      console.log("Fetched store data:", data);
  
      if (!data.stores || data.stores.length === 0) {
        throw new Error("No stores found for this distributor.");
      }
  
      setStores(data.stores);
      setFilteredStores(data.stores);
  
      // Extract unique values for filtering
      const uniqueValues = {};
      Object.keys(data.stores[0]).forEach((key) => {
        if (!["latitude", "longitude"].includes(key)) {
          uniqueValues[key] = [...new Set(data.stores.map((store) => store[key]))];
        }
      });
      setFilterValues(uniqueValues);
    } catch (err) {
      console.error("Error fetching store data:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };

  const toggleCategory = (category) => {
    setExpandedCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }));
  };

  const handleFilterChange = (key, value) => {
    const currentValues = selectedFilters[key] || [];
    const newValues = currentValues.includes(value)
      ? currentValues.filter(item => item !== value)
      : [...currentValues, value];
    
    setSelectedFilters(prev => ({
      ...prev,
      [key]: newValues
    }));
  };

  const handleSearchChange = (event) => {
    setSearchQuery(event.target.value);
  };

  // Apply filters as a side effect (used by useEffect)
  const applyFiltersEffect = () => {
    let results = stores;
    
    // Apply search query filter
    if (searchQuery.trim()) {
      results = results.filter(store => 
        store.storecode.toString().toLowerCase().includes(searchQuery.toLowerCase())
      );
    }
    
    // Apply selected filters
    Object.entries(selectedFilters).forEach(([key, values]) => {
      if (values && values.length > 0) {
        results = results.filter((store) => values.includes(store[key]));
      }
    });
    
    setFilteredStores(results);
  };

  // Apply filters from drawer button click
  const applyFilters = () => {
    applyFiltersEffect();
    setDrawerOpen(false); // Close drawer after applying filters
  };

  const clearFilters = () => {
    setSelectedFilters({});
    setSearchQuery('');
    setFilteredStores(stores);
  };

  const handleStoreClick = (store) => {
    setSelectedStore(store);
  };

  const handleDialogClose = () => {
    setSelectedStore(null);
  };
  const BASE_URL = process.env.REACT_APP_API_BASE_URL;

  const handleStatusEdit = async (event, store) => {
    event.stopPropagation();
    const updatedStatus = store.status === 1 ? 0 : 1;
    try {
      const response = await fetch(`${BASE_URL}/api/update_status`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          storecode: store.storecode,
          status: updatedStatus,
        }),
      });
  
      if (!response.ok) {
        throw new Error('Failed to update status');
      }
      
      // Update both the main stores array and the filtered stores array
      const updatedStores = stores.map((s) =>
        s.storecode === store.storecode ? { ...s, status: updatedStatus } : s
      );
      
      setStores(updatedStores);
      
      // Also update selected store if it's open in dialog
      if (selectedStore && selectedStore.storecode === store.storecode) {
        setSelectedStore({...selectedStore, status: updatedStatus});
      }
      
      // After updating, the useEffect will automatically reapply filters
    } catch (error) {
      console.error('Error updating status:', error);
      alert('Failed to update status');
    }
  };

  // Helper to format filter keys for display
  const formatFilterKey = (key) => {
    return key
      .replace(/([A-Z])/g, ' $1') // Add space before capital letters
      .replace(/^./, str => str.toUpperCase()) // Capitalize first letter
      .replace(/id$/i, 'ID'); // Format ID correctly
  };

  // Selected filter display
  const renderActiveFilters = () => {
    const hasActiveFilters = Object.values(selectedFilters).some(arr => arr && arr.length > 0) || searchQuery.trim();
    
    if (!hasActiveFilters) return null;
    
    return (
      <Box sx={{ mt: 2, mb: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>Active Filters:</Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {searchQuery.trim() && (
            <Chip
              label={`Store Code: ${searchQuery}`}
              onDelete={() => setSearchQuery('')}
              color="primary"
              variant="outlined"
              size="small"
            />
          )}
          {Object.entries(selectedFilters).map(([key, values]) => (
            values && values.length > 0 && values.map(value => (
              <Chip
                key={`${key}-${value}`}
                label={`${formatFilterKey(key)}: ${
                  key === 'status' 
                    ? (value === 1 ? 'Active' : 'Inactive')
                    : key === 'channelid'
                      ? (value === 1 ? 'Retail' : 'Wholesale')
                      : value
                }`}
                onDelete={() => handleFilterChange(key, value)}
                color="primary"
                variant="outlined"
                size="small"
              />
            ))
          ))}
          {hasActiveFilters && (
            <Chip
              label="Clear All"
              onClick={clearFilters}
              color="secondary"
              size="small"
            />
          )}
        </Box>
      </Box>
    );
  };

  const priorityFilters = ['status', 'channelid', 'subchannelid', 'areatype'];

  // Update filter values when stores change
  useEffect(() => {
    if (stores.length > 0) {
      const uniqueValues = {};
      Object.keys(stores[0]).forEach((key) => {
        if (!["latitude", "longitude"].includes(key)) {
          uniqueValues[key] = [...new Set(stores.map((store) => store[key]))];
        }
      });
      setFilterValues(uniqueValues);
    }
  }, [stores]);

  return (
    <Container
      maxWidth={false}
      sx={{
        py: 0,
        px: 6,
        backgroundColor: 'background.default',
        minHeight: '100vh',
      }}
    >
      <Box sx={{ padding: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h5">Store Management</Typography>
          <Badge badgeContent={activeFilterCount} color="primary">
            <Button
              variant="outlined"
              startIcon={<FilterListIcon />}
              onClick={toggleDrawer}
            >
              Filters
            </Button>
          </Badge>
        </Box>

        <Paper sx={{ p: 2, mb: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <TextField
          fullWidth
          placeholder="Search by Store Code"
          value={searchQuery}
          onChange={handleSearchChange}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
            endAdornment: searchQuery && (
              <InputAdornment position="end">
                <IconButton size="small" onClick={() => setSearchQuery('')}>
                  <CloseIcon fontSize="small" />
                </IconButton>
              </InputAdornment>
            )
          }}
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              applyFiltersEffect();
            }
          }}
          sx={{ flex: 1 }}
        />
       
      </Box>
    </Paper>


        {/* Active Filter Chips */}
        {renderActiveFilters()}

        <Paper sx={{ padding: 3, mt: 2 }}>
          {/* Loading and Error States */}
          {loading ? (
            <Box sx={{ textAlign: 'center', mt: 3 }}>
              <CircularProgress />
            </Box>
          ) : error ? (
            <Typography color="error">{error}</Typography>
          ) : filteredStores.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="body1" color="textSecondary">No stores found matching your filters</Typography>
            </Box>
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Store Code</strong></TableCell>
                    <TableCell><strong>Latitude</strong></TableCell>
                    <TableCell><strong>Longitude</strong></TableCell>
                    <TableCell><strong>Channel ID</strong></TableCell>
                    <TableCell><strong>Status</strong></TableCell>
                    <TableCell><strong>Actions</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredStores.map((store) => (
                    <TableRow
                      key={store.storeid}
                      onClick={() => handleStoreClick(store)}
                      sx={{ cursor: 'pointer', '&:hover': { backgroundColor: '#f5f5f5' } }}
                    >
                      <TableCell>{store.storecode}</TableCell>
                      <TableCell>{store.latitude}</TableCell>
                      <TableCell>{store.longitude}</TableCell>
                      <TableCell>{store.channeltypeid === 1 ? 'Retail' : 'Wholesale'}</TableCell>
                      <TableCell>
                        <Chip 
                          label={store.status === 1 ? 'Active' : 'Inactive'} 
                          color={store.status === 1 ? 'success' : 'default'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Button
                          variant="outlined"
                          color={store.status === 1 ? 'error' : 'success'}
                          size="small"
                          onClick={(event) => handleStatusEdit(event, store)}
                        >
                          {store.status === 1 ? 'Set Inactive' : 'Set Active'}
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Paper>
      </Box>

      {/* Filter Drawer */}
      <Drawer
        anchor="right"
        open={drawerOpen}
        onClose={toggleDrawer}
        sx={{
          '& .MuiDrawer-paper': {
            width: { xs: '100%', sm: 360 },
            padding: 2,
          },
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 2 }}>
          <Typography variant="h6">Filters</Typography>
          <IconButton onClick={toggleDrawer}>
            <CloseIcon />
          </IconButton>
        </Box>
        <Divider />
        
        <Box sx={{ p: 2, overflow: 'auto' }}>
          {/* Priority filters first */}
          {priorityFilters.map(key => (
            filterValues[key] && (
              <Box key={key} sx={{ mb: 2 }}>
                <Box 
                  sx={{ 
                    display: 'flex', 
                    justifyContent: 'space-between',
                    cursor: 'pointer',
                    mb: 1
                  }}
                  onClick={() => toggleCategory(key)}
                >
                  <Typography variant="subtitle1">{formatFilterKey(key)}</Typography>
                  {expandedCategories[key] ? <ExpandLess /> : <ExpandMore />}
                </Box>
                
                <Collapse in={expandedCategories[key] === true} timeout="auto">
                  <FormGroup>
                    {filterValues[key].map((value) => (
                      <FormControlLabel
                        key={value}
                        control={
                          <Checkbox
                            checked={(selectedFilters[key] || []).includes(value)}
                            onChange={() => handleFilterChange(key, value)}
                          />
                        }
                        label={
                          key === 'status' 
                            ? (value === 1 ? 'Active' : 'Inactive')
                            : key === 'channelid'
                              ? (value === 1 ? 'Retail' : 'Wholesale')
                              : value
                        }
                      />
                    ))}
                  </FormGroup>
                </Collapse>
                <Divider sx={{ mt: 1 }} />
              </Box>
            )
          ))}
          
          {/* Other filters */}
          {Object.entries(filterValues)
            .filter(([key]) => !priorityFilters.includes(key))
            .map(([key, values]) => (
              <Box key={key} sx={{ mb: 2 }}>
                <Box 
                  sx={{ 
                    display: 'flex', 
                    justifyContent: 'space-between',
                    cursor: 'pointer',
                    mb: 1
                  }}
                  onClick={() => toggleCategory(key)}
                >
                  <Typography variant="subtitle1">{formatFilterKey(key)}</Typography>
                  {expandedCategories[key] ? <ExpandLess /> : <ExpandMore />}
                </Box>
                
                <Collapse in={expandedCategories[key] === true} timeout="auto">
                  <FormGroup>
                    {values.map((value) => (
                      <FormControlLabel
                        key={value}
                        control={
                          <Checkbox
                            checked={(selectedFilters[key] || []).includes(value)}
                            onChange={() => handleFilterChange(key, value)}
                          />
                        }
                        label={value}
                      />
                    ))}
                  </FormGroup>
                </Collapse>
                <Divider sx={{ mt: 1 }} />
              </Box>
            ))}
        </Box>
        
        <Box sx={{ p: 2, borderTop: '1px solid rgba(0, 0, 0, 0.12)' }}>
          <Button
            variant="contained"
            color="primary"
            fullWidth
            onClick={applyFilters}
          >
            Apply Filters
          </Button>
          <Button
            variant="outlined"
            color="secondary"
            fullWidth
            onClick={clearFilters}
            sx={{ mt: 1 }}
          >
            Clear All
          </Button>
        </Box>
      </Drawer>

      {/* Store Details Dialog */}
      {selectedStore && (
        <Dialog open onClose={handleDialogClose} maxWidth="sm" fullWidth>
          <DialogTitle>
            Store Details
            <IconButton
              aria-label="close"
              onClick={handleDialogClose}
              sx={{
                position: 'absolute',
                right: 8,
                top: 8,
              }}
            >
              <CloseIcon />
            </IconButton>
          </DialogTitle>
          <DialogContent dividers>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body1"><strong>Store ID:</strong> {selectedStore.storeid}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body1"><strong>Store Code:</strong> {selectedStore.storecode}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body1"><strong>Latitude:</strong> {selectedStore.latitude}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body1"><strong>Longitude:</strong> {selectedStore.longitude}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body1">
                  <strong>Channel:</strong> {selectedStore.channeltypeid === 1 ? 'Retail' : 'Wholesale'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body1">
                  <strong>Status:</strong> {selectedStore.status === 1 ? 'Active' : 'Inactive'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body1"><strong>Subchannel ID:</strong> {selectedStore.subchannelid}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body1"><strong>Area Type:</strong> {selectedStore.areatype}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body1"><strong>Town ID:</strong> {selectedStore.townid}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body1"><strong>Locality ID:</strong> {selectedStore.localityid}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body1"><strong>Sub Locality ID:</strong> {selectedStore.sublocalityid}</Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="body1"><strong>Classifications:</strong></Typography>
                <Box sx={{ pl: 2 }}>
                  <Typography variant="body2">Classification One: {selectedStore.classificationoneid}</Typography>
                  <Typography variant="body2">Classification Two: {selectedStore.classificationtwoid}</Typography>
                  <Typography variant="body2">Classification Three: {selectedStore.classificationthreeid}</Typography>
                </Box>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body1"><strong>Store Perfect ID:</strong> {selectedStore.storeperfectid}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body1"><strong>Store Filer Type:</strong> {selectedStore.storefilertype}</Typography>
              </Grid>
            </Grid>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleDialogClose} color="primary">
              Close
            </Button>
          </DialogActions>
        </Dialog>
      )}
    </Container>
  );
}

export default StoreManagement;