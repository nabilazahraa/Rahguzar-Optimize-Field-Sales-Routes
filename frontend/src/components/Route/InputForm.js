import React, {useEffect, useState } from 'react';
import { useFormik } from 'formik';
import * as Yup from 'yup';
import { motion } from 'framer-motion';
import {
  TextField,
  Typography,
  Box,
  Grid,
  Paper,
  InputAdornment,
  IconButton,
  Tooltip,
  RadioGroup,
  FormControlLabel,
  Radio,
  Select,
  FormControl,
  MenuItem,
  Divider,
  Chip,
  Alert,
  ListItemText,
  Checkbox,
} from '@mui/material';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import RemoveCircleOutlineIcon from '@mui/icons-material/RemoveCircleOutline';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import EventIcon from '@mui/icons-material/Event';
import StorefrontIcon from '@mui/icons-material/Storefront';
import GroupIcon from '@mui/icons-material/Group';
import PersonIcon from '@mui/icons-material/Person';
import CalendarMonthIcon from '@mui/icons-material/CalendarMonth';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import LocalMallIcon from '@mui/icons-material/LocalMall';
import WeekendIcon from '@mui/icons-material/Weekend';
import { styled, useTheme } from '@mui/material/styles';

// Constants
const STORE_TYPES = [
  { value: 'both', label: 'Both Wholesale and Retail' },
  { value: 'wholesale', label: 'Wholesale Only' },
  { value: 'retail', label: 'Retail Only' },
];

const DAYS_OF_WEEK = [
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
  'Sunday',
];
const BASE_URL = process.env.REACT_APP_API_BASE_URL;

// Styled Components
const FormPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  borderRadius: theme.spacing(2),
  boxShadow: theme.shadows[3],
  backgroundColor: theme.palette.background.paper,
}));

const SectionTitle = styled(Typography)(({ theme }) => ({
  fontWeight: 'bold',
  color: theme.palette.primary.main,
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  marginBottom: theme.spacing(2),
}));

const StyledButton = styled(motion.button)(({ theme }) => ({
  backgroundColor: theme.palette.primary.main,
  color: '#fff',
  padding: theme.spacing(1.5, 4),
  fontSize: '1rem',
  fontWeight: 600,
  border: 'none',
  borderRadius: theme.spacing(1),
  cursor: 'pointer',
  outline: 'none',
  transition: 'background-color 0.3s ease',
  '&:hover': {
    backgroundColor: theme.palette.primary.dark,
  },
}));

const ErrorText = styled(Typography)(({ theme }) => ({
  color: theme.palette.error.main,
  fontSize: '0.875rem',
  marginTop: theme.spacing(0.5),
}));

const CounterButton = styled(IconButton)(({ theme }) => ({
  color: theme.palette.primary.main,
}));

const StyledTextField = styled(TextField)(({ theme }) => ({
  '& input[type=number]::-webkit-outer-spin-button, & input[type=number]::-webkit-inner-spin-button': {
    '-webkit-appearance': 'none',
    margin: 0,
  },
  '& input[type=number]': {
    '-moz-appearance': 'textfield',
  },
}));

const StyledRadioGroup = styled(RadioGroup)(({ theme }) => ({
  '& .MuiFormControlLabel-root': {
    border: `1px solid ${theme.palette.divider}`,
    borderRadius: theme.spacing(1),
    margin: theme.spacing(0.5),
    padding: theme.spacing(0.5, 2),
    transition: 'all 0.2s ease',
  },
  '& .MuiFormControlLabel-root:hover': {
    backgroundColor: theme.palette.action.hover,
  },
  '& .Mui-checked + .MuiTypography-root': {
    fontWeight: 600,
  },
}));

const StyledDivider = styled(Divider)(({ theme }) => ({
  margin: theme.spacing(4, 0),
}));

function InputForm({ onSubmit, selectedStores = [], mapSelectionEnabled = false }) {
  const theme = useTheme();
  const [fetchingOrderbookers, setFetchingOrderbookers] = useState(false);
  const [fetchOrderbookersError, setFetchOrderbookersError] = useState('');
  
  // Validation schema
  const validationSchema = Yup.object({
    num_orderbookers: Yup.number()
    .typeError('Number of Orderbookers must be a number')
    .integer('Must be an integer')
    .min(1, 'Must be at least 1')
    .max(20, 'Cannot exceed 20 orderbookers')  // <--- hardcoded limit
    .required('Number of Orderbookers is required'),


    plan_duration: Yup.string()
      .oneOf(['day', 'custom'], 'Invalid Plan Duration')
      .required('Plan Duration is required'),

    custom_days: Yup.number()
      .when('plan_duration', {
        is: 'custom',
        then: () => Yup.number()
          .typeError('Number of days must be a number')
          .integer('Must be an integer')
          .min(1, 'Minimum 1 day')
          .required('Number of days is required for a custom plan')
      }),

    store_type: Yup.string()
      .when('mapSelectionEnabled', {
        is: false,
        then: () => Yup.string()
          .oneOf(['both', 'wholesale', 'retail'], 'Invalid Store Type')
          .required('Store Type is required'),
        otherwise: () => Yup.string().optional(),
      }),

    wholesale_time: Yup.number()
      .when(['store_type', 'mapSelectionEnabled'], {
        is: (store_type, mapSelectionEnabled) => {
          return !mapSelectionEnabled && (store_type === 'both' || store_type === 'wholesale') || mapSelectionEnabled;
        },
        then: () => Yup.number()
          .typeError('Wholesale Time must be a number')
          .integer('Wholesale Time must be an integer')
          .min(1, 'Wholesale Time must be at least 1 minute')
          .required('Wholesale Time is required')
      }),

    retail_time: Yup.number()
      .when(['store_type', 'mapSelectionEnabled'], {
        is: (store_type, mapSelectionEnabled) => {
          return !mapSelectionEnabled && (store_type === 'both' || store_type === 'retail') || mapSelectionEnabled;
        },
        then: () => Yup.number()
          .typeError('Retail Time must be a number')
          .integer('Retail Time must be an integer')
          .min(1, 'Retail Time must be at least 1 minute')
          .required('Retail Time is required')
      }),

    holidays: Yup.array()
      .of(Yup.string().oneOf(DAYS_OF_WEEK, 'Invalid day selected'))
      .nullable(),

    replicate: Yup.bool(),
    
    working_hours: Yup.number()
      .typeError('Working hours must be a number')
      .integer('Working hours must be an integer')
      .min(1, 'Working hours must be at least 1')
      .max(24, 'Working hours cannot exceed 24')
      .required('Working hours are required'),
  });

  const formik = useFormik({
    initialValues: {
      distributor_id: localStorage.getItem('distributor_id') || '',
      num_orderbookers: '',
      plan_duration: 'day',
      custom_days: '',
      store_type: 'both',
      holidays: [],
      wholesale_time: 20,
      retail_time: 10,
      replicate: false,
      working_hours: 8,
      mapSelectionEnabled: mapSelectionEnabled,
    },
    validationSchema,
    onSubmit: (values, { setSubmitting, resetForm }) => {
      // Convert and pass to onSubmit
      if (values.num_orderbookers === 0) {
        setFetchOrderbookersError('No orderbookers available for the selected store type.');
        setSubmitting(false);
        return;
      }
      
      // Prepare data for submission
      const submissionData = {
        distributor_id: parseInt(values.distributor_id, 10),
        num_orderbookers: parseInt(values.num_orderbookers, 10),
        plan_duration: values.plan_duration,
        custom_days:
          values.plan_duration === 'custom'
            ? parseInt(values.custom_days, 10)
            : undefined,
        holidays: values.plan_duration === 'custom' ? values.holidays : [],
        working_hours: parseInt(values.working_hours, 10),
        replicate: values.replicate,
      };
      
      // Only include store type and related fields if map selection is not enabled
      if (!mapSelectionEnabled) {
        submissionData.store_type = values.store_type;
        
        if (values.store_type === 'both' || values.store_type === 'wholesale') {
          submissionData.wholesale_time = parseInt(values.wholesale_time, 10);
        }
        
        if (values.store_type === 'both' || values.store_type === 'retail') {
          submissionData.retail_time = parseInt(values.retail_time, 10);
        }
      } else {
        // When map selection is enabled, we still need to include these values for the API
        // but they'll be applied only to the selected stores
        submissionData.wholesale_time = parseInt(values.wholesale_time, 10);
        submissionData.retail_time = parseInt(values.retail_time, 10);
        submissionData.store_type = 'both'; // Default value
      }
      
      onSubmit(submissionData);
      setSubmitting(true);
    },
  });
  
  useEffect(() => {
    // Update mapSelectionEnabled in formik values when prop changes
    formik.setFieldValue('mapSelectionEnabled', mapSelectionEnabled);
  }, [mapSelectionEnabled]);
  
  useEffect(() => {
    const fetchOrderbookers = async () => {
      setFetchingOrderbookers(true);
      setFetchOrderbookersError('');
      try {
        const response = await fetch(`${BASE_URL}/get_orderbookers?distributor_id=${formik.values.distributor_id}`);
        const data = await response.json();
        if (data.status === 'success') {
          formik.setFieldValue('num_orderbookers', data.number_of_orderbookers);
        } else {
          setFetchOrderbookersError(data.message || 'Error fetching data');
        }
      } catch (error) {
        setFetchOrderbookersError('Error fetching data from server.');
      } finally {
        setFetchingOrderbookers(false);
      }
    };

    const timer = setTimeout(() => {
      if (formik.values.distributor_id && !formik.errors.distributor_id) {
        fetchOrderbookers();
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [formik.values.distributor_id, formik.errors.distributor_id]);


  // Animation variants
  const formVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6, ease: 'easeOut' },
    },
  };

  const buttonVariants = {
    rest: { scale: 1 },
    hover: {
      scale: 1.05,
      boxShadow: `0px 4px 15px ${theme.palette.primary.main}`,
    },
    tap: { scale: 0.95 },
  };

  // Increment and Decrement
  const handleIncrement = (field) => {
    const currentValue = Number(formik.values[field]) || 0;
    
    // Special handling for working_hours field that has a limit
    if (field === 'working_hours' && currentValue < 24) {
      formik.setFieldValue(field, currentValue + 1);
    } else if (field !== 'working_hours') {
      formik.setFieldValue(field, currentValue + 1);
    }
  };

  const handleDecrement = (field) => {
    const currentValue = Number(formik.values[field]) || 0;
    
    if (field === 'working_hours' && currentValue > 1) {
      formik.setFieldValue(field, currentValue - 1);
    } else if (field !== 'working_hours' && currentValue > 1) {
      formik.setFieldValue(field, currentValue - 1);
    }
  };

  const handleHolidaysChange = (event) => {
    const { value } = event.target;
    formik.setFieldValue('holidays', value);
  };

  return (
    <FormPaper elevation={3}>
      <motion.div variants={formVariants} initial="hidden" animate="visible">
        <Box component="form" onSubmit={formik.handleSubmit}>
          <Typography
            variant="h5"
            gutterBottom
            sx={{ 
              fontWeight: 700, 
              color: theme.palette.primary.main,
              textAlign: 'center',
              mb: 4 
            }}
          >
            Route Plan Configuration
          </Typography>

          {mapSelectionEnabled && selectedStores.length > 0 && (
            <Alert 
              severity="info" 
              sx={{ mb: 3 }}
              variant="outlined"
            >
              <Typography variant="body2">
                <strong>{selectedStores.length} stores</strong> have been selected from the map. 
                Store type selection is disabled as the route plan will be generated only for the selected stores.
              </Typography>
            </Alert>
          )}

          {/* Team Configuration Section */}
          <SectionTitle variant="h6">
            <GroupIcon /> Team Configuration
          </SectionTitle>
          <Grid container spacing={3}>
            {/* Number of Orderbookers */}
            <Grid item xs={12} sm={6}>
              <StyledTextField
                label="Number of Orderbookers"
                variant="outlined"
                fullWidth
                required
                type="number"
                id="num_orderbookers"
                name="num_orderbookers"
                value={formik.values.num_orderbookers}
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <PersonIcon />
                    </InputAdornment>
                  ),
                  endAdornment: (
                    <InputAdornment position="end">
                      <Tooltip title="Decrement">
                        <span>
                          <CounterButton
                            aria-label="decrement num_orderbookers"
                            onClick={() => handleDecrement('num_orderbookers')}
                            disabled={
                              Number(formik.values.num_orderbookers) <= 1
                            }
                          >
                            <RemoveCircleOutlineIcon />
                          </CounterButton>
                        </span>
                      </Tooltip>
                      <Tooltip title="Increment">
                        <CounterButton
                          aria-label="increment num_orderbookers"
                          onClick={() => handleIncrement('num_orderbookers')}
                        >
                          <AddCircleOutlineIcon />
                        </CounterButton>
                      </Tooltip>
                    </InputAdornment>
                  ),
                  inputProps: { min: 1 },
                }}
                error={
                  formik.touched.num_orderbookers &&
                  Boolean(formik.errors.num_orderbookers)
                }
                helperText={
                  formik.touched.num_orderbookers && formik.errors.num_orderbookers
                }
                sx={{
                  backgroundColor: '#fff',
                  borderRadius: 1,
                }}
              />
            </Grid>

            {/* Working Hours */}
            <Grid item xs={12} sm={6}>
              <StyledTextField
                label="Working Hours per Day"
                variant="outlined"
                fullWidth
                required
                type="number"
                id="working_hours"
                name="working_hours"
                value={formik.values.working_hours}
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <AccessTimeIcon />
                    </InputAdornment>
                  ),
                  endAdornment: (
                    <InputAdornment position="end">
                      <Tooltip title="Decrement">
                        <span>
                          <CounterButton
                            aria-label="decrement working hours"
                            onClick={() => handleDecrement('working_hours')}
                            disabled={formik.values.working_hours <= 1}
                          >
                            <RemoveCircleOutlineIcon />
                          </CounterButton>
                        </span>
                      </Tooltip>
                      <Tooltip title="Increment">
                        <span>
                          <CounterButton
                            aria-label="increment working hours"
                            onClick={() => handleIncrement('working_hours')}
                            disabled={formik.values.working_hours >= 24}
                          >
                            <AddCircleOutlineIcon />
                          </CounterButton>
                        </span>
                      </Tooltip>
                    </InputAdornment>
                  ),
                  inputProps: { min: 1, max: 24 },
                }}
                error={
                  formik.touched.working_hours &&
                  Boolean(formik.errors.working_hours)
                }
                helperText={
                  formik.touched.working_hours && formik.errors.working_hours
                }
                sx={{
                  backgroundColor: '#fff',
                  borderRadius: 1,
                }}
              />
            </Grid>
          </Grid>

          <StyledDivider />

          {/* Store Configuration Section */}
          {/* When map selection is not enabled, show store type selection */}
          {!mapSelectionEnabled ? (
            <>
              <SectionTitle variant="h6">
                <StorefrontIcon /> Store Configuration
              </SectionTitle>
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 500 }}>
                  Store Type
                </Typography>
                <StyledRadioGroup
                  row
                  aria-label="store_type"
                  name="store_type"
                  value={formik.values.store_type}
                  onChange={formik.handleChange}
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                  }}
                >
                  {STORE_TYPES.map((type) => (
                    <FormControlLabel
                      key={type.value}
                      value={type.value}
                      control={<Radio color="primary" />}
                      label={
                        <Typography variant="body1">{type.label}</Typography>
                      }
                      sx={{
                        backgroundColor: 
                          formik.values.store_type === type.value 
                            ? `${theme.palette.primary.light}20` 
                            : 'transparent',
                        flex: 1,
                        textAlign: 'center',
                      }}
                    />
                  ))}
                </StyledRadioGroup>
                {formik.touched.store_type && formik.errors.store_type && (
                  <ErrorText>{formik.errors.store_type}</ErrorText>
                )}
              </Box>

              {/* Visit Time Configuration - Show based on store type when map selection is disabled */}
              <Grid container spacing={3} sx={{ mt: 1 }}>
                {/* Wholesale Time Input - Only show for 'both' or 'wholesale' store types */}
                {(formik.values.store_type === 'both' || formik.values.store_type === 'wholesale') && (
                  <Grid item xs={12} sm={formik.values.store_type === 'both' ? 6 : 12}>
                    <StyledTextField
                      label="Wholesale Visit Time (minutes)"
                      variant="outlined"
                      fullWidth
                      required
                      type="number"
                      id="wholesale_time"
                      name="wholesale_time"
                      value={formik.values.wholesale_time}
                      onChange={formik.handleChange}
                      onBlur={formik.handleBlur}
                      InputProps={{
                        startAdornment: (
                          <InputAdornment position="start">
                            <LocalMallIcon />
                          </InputAdornment>
                        ),
                        endAdornment: (
                          <InputAdornment position="end">
                            <Tooltip title="Decrement">
                              <span>
                                <CounterButton
                                  aria-label="decrement wholesale_time"
                                  onClick={() => handleDecrement('wholesale_time')}
                                  disabled={
                                    Number(formik.values.wholesale_time) <= 1
                                  }
                                >
                                  <RemoveCircleOutlineIcon />
                                </CounterButton>
                              </span>
                            </Tooltip>
                            <Tooltip title="Increment">
                              <CounterButton
                                aria-label="increment wholesale_time"
                                onClick={() => handleIncrement('wholesale_time')}
                              >
                                <AddCircleOutlineIcon />
                              </CounterButton>
                            </Tooltip>
                          </InputAdornment>
                        ),
                        inputProps: { min: 1 },
                      }}
                      error={
                        formik.touched.wholesale_time &&
                        Boolean(formik.errors.wholesale_time)
                      }
                      helperText={
                        formik.touched.wholesale_time && formik.errors.wholesale_time
                      }
                      sx={{
                        backgroundColor: '#fff',
                        borderRadius: 1,
                      }}
                    />
                  </Grid>
                )}

                {/* Retail Time Input - Only show for 'both' or 'retail' store types */}
                {(formik.values.store_type === 'both' || formik.values.store_type === 'retail') && (
                  <Grid item xs={12} sm={formik.values.store_type === 'both' ? 6 : 12}>
                    <StyledTextField
                      label="Retail Visit Time (minutes)"
                      variant="outlined"
                      fullWidth
                      required
                      type="number"
                      id="retail_time"
                      name="retail_time"
                      value={formik.values.retail_time}
                      onChange={formik.handleChange}
                      onBlur={formik.handleBlur}
                      InputProps={{
                        startAdornment: (
                          <InputAdornment position="start">
                            <ShoppingCartIcon />
                          </InputAdornment>
                        ),
                        endAdornment: (
                          <InputAdornment position="end">
                            <Tooltip title="Decrement">
                              <span>
                                <CounterButton
                                  aria-label="decrement retail_time"
                                  onClick={() => handleDecrement('retail_time')}
                                  disabled={
                                    Number(formik.values.retail_time) <= 1
                                  }
                                >
                                  <RemoveCircleOutlineIcon />
                                </CounterButton>
                              </span>
                            </Tooltip>
                            <Tooltip title="Increment">
                              <CounterButton
                                aria-label="increment retail_time"
                                onClick={() => handleIncrement('retail_time')}
                              >
                                <AddCircleOutlineIcon />
                              </CounterButton>
                            </Tooltip>
                          </InputAdornment>
                        ),
                        inputProps: { min: 1 },
                      }}
                      error={
                        formik.touched.retail_time &&
                        Boolean(formik.errors.retail_time)
                      }
                      helperText={
                        formik.touched.retail_time && formik.errors.retail_time
                      }
                      sx={{
                        backgroundColor: '#fff',
                        borderRadius: 1,
                      }}
                    />
                  </Grid>
                )}
              </Grid>
            </>
          ) : (
            /* When map selection is enabled, show only visit time configuration */
            <>
              <SectionTitle variant="h6">
                <AccessTimeIcon /> Visit Time Configuration
              </SectionTitle>
              <Grid container spacing={3}>
                {/* Wholesale Time Input */}
                <Grid item xs={12} sm={6}>
                  <StyledTextField
                    label="Wholesale Visit Time (minutes)"
                    variant="outlined"
                    fullWidth
                    required
                    type="number"
                    id="wholesale_time"
                    name="wholesale_time"
                    value={formik.values.wholesale_time}
                    onChange={formik.handleChange}
                    onBlur={formik.handleBlur}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <LocalMallIcon />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <Tooltip title="Decrement">
                            <span>
                              <CounterButton
                                aria-label="decrement wholesale_time"
                                onClick={() => handleDecrement('wholesale_time')}
                                disabled={
                                  Number(formik.values.wholesale_time) <= 1
                                }
                              >
                                <RemoveCircleOutlineIcon />
                              </CounterButton>
                            </span>
                          </Tooltip>
                          <Tooltip title="Increment">
                            <CounterButton
                              aria-label="increment wholesale_time"
                              onClick={() => handleIncrement('wholesale_time')}
                            >
                              <AddCircleOutlineIcon />
                            </CounterButton>
                          </Tooltip>
                        </InputAdornment>
                      ),
                      inputProps: { min: 1 },
                    }}
                    error={
                      formik.touched.wholesale_time && 
                      Boolean(formik.errors.wholesale_time)
                    }
                    helperText={
                      formik.touched.wholesale_time && formik.errors.wholesale_time
                    }
                    sx={{
                      backgroundColor: '#fff',
                      borderRadius: 1,
                    }}
                  />
                </Grid>

                {/* Retail Time Input */}
                <Grid item xs={12} sm={6}>
                  <StyledTextField
                    label="Retail Visit Time (minutes)"
                    variant="outlined"
                    fullWidth
                    required
                    type="number"
                    id="retail_time"
                    name="retail_time"
                    value={formik.values.retail_time}
                    onChange={formik.handleChange}
                    onBlur={formik.handleBlur}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <ShoppingCartIcon />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <Tooltip title="Decrement">
                            <span>
                              <CounterButton
                                aria-label="decrement retail_time"
                                onClick={() => handleDecrement('retail_time')}
                                disabled={
                                  Number(formik.values.retail_time) <= 1
                                }
                              >
                                <RemoveCircleOutlineIcon />
                              </CounterButton>
                            </span>
                          </Tooltip>
                          <Tooltip title="Increment">
                            <CounterButton
                              aria-label="increment retail_time"
                              onClick={() => handleIncrement('retail_time')}
                            >
                              <AddCircleOutlineIcon />
                            </CounterButton>
                          </Tooltip>
                        </InputAdornment>
                      ),
                      inputProps: { min: 1 },
                    }}
                    error={
                      formik.touched.retail_time && 
                      Boolean(formik.errors.retail_time)
                    }
                    helperText={
                      formik.touched.retail_time && formik.errors.retail_time
                    }
                    sx={{
                      backgroundColor: '#fff',
                      borderRadius: 1,
                    }}
                  />
                </Grid>
              </Grid>
            </>
          )}
          
          <StyledDivider />

          {/* Schedule Configuration Section */}
          <SectionTitle variant="h6">
            <EventIcon /> Schedule Configuration
          </SectionTitle>
          
          {/* Plan Duration */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 500 }}>
              Plan Duration
            </Typography>
            <StyledRadioGroup
              row
              name="plan_duration"
              value={formik.values.plan_duration}
              onChange={formik.handleChange}
              sx={{
                display: 'flex',
                justifyContent: 'center',
                gap: 4
              }}
            >
              <FormControlLabel
                value="day"
                control={<Radio color="primary" />}
                label="Single Day"
                sx={{
                  backgroundColor: 
                    formik.values.plan_duration === 'day' 
                      ? `${theme.palette.primary.light}20` 
                      : 'transparent',
                }}
              />
              <FormControlLabel
                value="custom"
                control={<Radio color="primary" />}
                label="Custom Days"
                sx={{
                  backgroundColor: 
                    formik.values.plan_duration === 'custom' 
                      ? `${theme.palette.primary.light}20` 
                      : 'transparent',
                }}
              />
            </StyledRadioGroup>
            {formik.touched.plan_duration && formik.errors.plan_duration && (
              <ErrorText>{formik.errors.plan_duration}</ErrorText>
            )}
          </Box>


          {/* Custom Days - show only if plan_duration == 'custom' */}
          {formik.values.plan_duration === 'custom' && (
            <Grid container spacing={3}>
              <Grid item xs={12} sm={12}>
                <StyledTextField
                  label="Number of Days"
                  variant="outlined"
                  fullWidth
                  required
                  type="number"
                  id="custom_days"
                  name="custom_days"
                  value={formik.values.custom_days}
                  onChange={formik.handleChange}
                  onBlur={formik.handleBlur}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <CalendarMonthIcon />
                      </InputAdornment>
                    ),
                    endAdornment: (
                      <InputAdornment position="end">
                        <Tooltip title="Decrement">
                          <span>
                            <CounterButton
                              aria-label="decrement custom days"
                              onClick={() => handleDecrement('custom_days')}
                              disabled={formik.values.custom_days <= 1}
                            >
                              <RemoveCircleOutlineIcon />
                            </CounterButton>
                          </span>
                        </Tooltip>
                        <Tooltip title="Increment">
                          <CounterButton
                            aria-label="increment custom days"
                            onClick={() => handleIncrement('custom_days')}
                          >
                            <AddCircleOutlineIcon />
                          </CounterButton>
                        </Tooltip>
                      </InputAdornment>
                    ),
                    inputProps: { min: 1 },
                  }}
                  error={
                    formik.touched.custom_days &&
                    Boolean(formik.errors.custom_days)
                  }
                  helperText={
                    formik.touched.custom_days && formik.errors.custom_days
                  }
                  sx={{
                    backgroundColor: '#fff',
                    borderRadius: 1,
                  }}
                />
              </Grid>
            </Grid>
          )}

          {/* Holidays - Only show for custom plan duration */}
          {/* {formik.values.plan_duration === 'custom' && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 500, display: 'flex', alignItems: 'center', gap: 1 }}>
                <WeekendIcon fontSize="small" /> Select Holidays
              </Typography>
              <FormControl
                variant="outlined"
                fullWidth
                sx={{
                  backgroundColor: '#fff',
                  borderRadius: 1,
                }}
                error={
                  formik.touched.holidays && Boolean(formik.errors.holidays)
                }
              >
                <Select
                  labelId="holidays-label"
                  id="holidays"
                  name="holidays"
                  multiple
                  value={
                    Array.isArray(formik.values.holidays)
                      ? formik.values.holidays
                      : []
                  }
                  onChange={handleHolidaysChange}
                  onBlur={formik.handleBlur}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selected.map((value) => (
                        <Chip key={value} label={value} size="small" />
                      ))}
                    </Box>
                  )}
                >
                  {DAYS_OF_WEEK.map((day) => (
                    <MenuItem key={day} value={day}>
                      <Checkbox
                        checked={formik.values.holidays.indexOf(day) > -1}
                      />
                      <ListItemText primary={day} />
                    </MenuItem>
                  ))}
                </Select>
                {formik.touched.holidays && formik.errors.holidays && (
                  <ErrorText>{formik.errors.holidays}</ErrorText>
                )}
              </FormControl>
            </Box>
          )} */}


          {/* Replicate Checkbox - only show for custom duration */}
          {formik.values.plan_duration === 'custom' && (
            <Box sx={{ mt: 3 }}>
              <FormControlLabel
                control={
                  <Checkbox
                    id="replicate"
                    name="replicate"
                    color="primary"
                    checked={formik.values.replicate}
                    onChange={formik.handleChange}
                  />
                }
                label={
                  <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <EventIcon fontSize="small" /> Replicate 4X (e.g. if days=7, final schedule=28 days)
                  </Typography>
                }
              />
            </Box>
          )}

          {/* Submit Button */}
          <Box textAlign="center" sx={{ mt: 5 }}>
            <StyledButton
              type="submit"
              variants={buttonVariants}
              initial="rest"
              whileHover="hover"
              whileTap="tap"
              animate="rest"
              sx={{ fontSize: '1.1rem', px: 6, py: 1.5 }}
            >
              Generate Route Plan
            </StyledButton>
          </Box>
        </Box>
      </motion.div>
    </FormPaper>
  );
}

export default InputForm;