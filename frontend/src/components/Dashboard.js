

import { useEffect, useCallback, useState, useRef, memo } from "react";
import { useNavigate } from "react-router-dom";
import {
  Card, Row, Col, Statistic, Typography, Select, Button, message, Tooltip, 
  Radio, Tabs, Space, Divider, Modal, Drawer
} from 'antd';
import { 
  ArrowUpOutlined, ArrowDownOutlined, InfoCircleOutlined, 
  DashboardOutlined, LineChartOutlined, BarChartOutlined,
  SettingOutlined
} from '@ant-design/icons';
import BarChart from './BarChart';
import EnhancedLineChart from './interactiveLineChart.js';
import debounce from 'lodash/debounce';
import LoadingSpinner_dashboard from './LoadingSpinner_dashboard';
import './AnalyticsDashboard.css';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;
const MemoizedBarChart = memo(BarChart);

export const clearDashboardState = () => {
  localStorage.removeItem('dashboardDropdownOptions');
  localStorage.removeItem('dashboardPjpOptions');
  localStorage.removeItem('dashboardSelectedValue');
  localStorage.removeItem('dashboardSelectedPjp');
  localStorage.removeItem('dashboardKpiData');
  localStorage.removeItem('dashboardGraphData');
  localStorage.removeItem('dashboardIsDataApplied');
};

function Dashboard() {
  const navigate = useNavigate();
  const hasInitialized = useRef(false);

  const [pjpOptions, setPjpOptions] = useState(() =>
    JSON.parse(localStorage.getItem('dashboardPjpOptions')) || []
  );
  const [dropdownOptions, setDropdownOptions] = useState([]);

  const [selectedValue, setSelectedValue] = useState(null);
  const [selectedPjp, setSelectedPjp] = useState(() => {
    const saved = localStorage.getItem('dashboardSelectedPjp');
    return saved ? JSON.parse(saved).id : null;
  });

  const [tempSelectedValue, setTempSelectedValue] = useState(null);
  const [tempSelectedPjp, setTempSelectedPjp] = useState(null);

  const [kpiData, setKpiData] = useState(() =>
    JSON.parse(localStorage.getItem('dashboardKpiData')) || null
  );
  const [graphData, setGraphData] = useState(() =>
    JSON.parse(localStorage.getItem('dashboardGraphData')) || {
      total_distance: [],
      store_visited: [],
      workload: []
    }
  );
  const [loading, setLoading] = useState(false);
  const [loadingPjp, setLoadingPjp] = useState(false);
  const [error, setError] = useState(null);
  const [isDataApplied, setIsDataApplied] = useState(() =>
    JSON.parse(localStorage.getItem('dashboardIsDataApplied')) || false
  );
  
  // Control panel drawer state
  const [controlsVisible, setControlsVisible] = useState(false);

  const currentDistributorId = localStorage.getItem("distributor_id");

  // Chart metric states
  const [distanceDifference, setDistanceDifference] = useState(null);
  const [distanceDateRange, setDistanceDateRange] = useState('');
  const [distanceAverage, setDistanceAverage] = useState(null);
  const [visitsDifference, setVisitsDifference] = useState(null);
  const [visitsDateRange, setVisitsDateRange] = useState('');
  const [visitsAverage, setVisitsAverage] = useState(null);
  const [workloadDifference, setWorkloadDifference] = useState(null);
  const [workloadDateRange, setWorkloadDateRange] = useState('');
  const [workloadAverage, setWorkloadAverage] = useState(null);

  // Chart handlers
  const handleDistanceDifference = useCallback((difference) => setDistanceDifference(difference), []);
  const handleDistanceDateRange = useCallback((minDate, maxDate) => setDistanceDateRange(`${minDate} - ${maxDate}`), []);
  const handleDistanceAverage = useCallback((average) => setDistanceAverage(average), []);
  const handleVisitsDifference = useCallback((difference) => setVisitsDifference(difference), []);
  const handleVisitsDateRange = useCallback((minDate, maxDate) => setVisitsDateRange(`${minDate} - ${maxDate}`), []);
  const handleVisitsAverage = useCallback((average) => setVisitsAverage(average), []);
  const handleWorkloadDifference = useCallback((difference) => setWorkloadDifference(difference), []);
  const handleWorkloadDateRange = useCallback((minDate, maxDate) => setWorkloadDateRange(`${minDate} - ${maxDate}`), []);
  const handleWorkloadAverage = useCallback((average) => setWorkloadAverage(average), []);
  const BASE_URL = process.env.REACT_APP_API_BASE_URL;

  // Chart type states and active tab
  const [activeChartType, setActiveChartType] = useState('bar');
  const [activeTab, setActiveTab] = useState('2');

  const resetDashboard = useCallback(() => {
    const distributorId = localStorage.getItem("distributor_id");
    setDropdownOptions([]);
    setPjpOptions([]);
    setSelectedValue(null);
    setSelectedPjp(null);
    setTempSelectedValue(null);
    setTempSelectedPjp(null);
    setKpiData(null);
    setGraphData({ total_distance: [], store_visited: [], workload: [] });
    setError(null);
    setDistanceDifference(null);
    setDistanceDateRange('');
    setDistanceAverage(null);
    setVisitsDifference(null);
    setVisitsDateRange('');
    setVisitsAverage(null);
    setWorkloadDifference(null);
    setWorkloadDateRange('');
    setWorkloadAverage(null);
    setIsDataApplied(false);
    clearDashboardState();
    if (distributorId) localStorage.setItem("distributor_id", distributorId);
  }, []);

  const fetchOrderbookersForPlan = useCallback(async (planId) => {
    if (!planId || !currentDistributorId) return;
    try {
      const response = await fetch(
        `${BASE_URL}/get_orderbookers_for_plan?distributor_id=${currentDistributorId}&pjp_id=${planId}`
      );
      const result = await response.json();
      if (result.status === "success") {
        const orderbookers = result.orderbookers;
        setDropdownOptions(orderbookers);
        localStorage.setItem('dashboardDropdownOptions', JSON.stringify(orderbookers));
        
        // Auto-select the first orderbooker
        if (orderbookers.length > 0) {
          setTempSelectedValue(orderbookers[0].id);
        } else {
          setTempSelectedValue(null);
        }
      } else {
        setDropdownOptions([]);
        setTempSelectedValue(null);
      }
    } catch (error) {
      setError("Failed to fetch orderbookers for plan");
    }
  }, [currentDistributorId]);
  
  const fetchPjpOptions = useCallback(async () => {
    if (!currentDistributorId) return;
    try {
      const response = await fetch(`${BASE_URL}/get_pjp_plans?distributor_id=${currentDistributorId}`);
      const result = await response.json();
      if (result.status === "success") {
        const plans = result.plans;
        setPjpOptions(plans);
        localStorage.setItem('dashboardPjpOptions', JSON.stringify(plans));

        // Auto-select most recent plan
        if (plans.length > 0) {
          const mostRecent = plans[plans.length - 1];
          setTempSelectedPjp(mostRecent.id);
          fetchOrderbookersForPlan(mostRecent.id);
          if (mostRecent && plans.length === 1) {
            setSelectedPjp(mostRecent.id);
          }
        }
      }
    } catch (err) {
      setError("Failed to load PJP options");
    }
  }, [currentDistributorId, fetchOrderbookersForPlan]);

  const fetchDashboardData = useCallback(async (distributorId, orderbookerId, pjpId) => {
    if (!pjpId) {
      setError("Please select a PJP plan to load data.");
      return;
    }
    setError(null);
    try {
      const kpiUrl = `${BASE_URL}/get_kpi_data?distributor_id=${distributorId}` +
        (orderbookerId !== null ? `&orderbooker_id=${orderbookerId}` : '') +
        `&pjp_id=${pjpId}`;

      const graphUrl = `${BASE_URL}//get_graph_data?distributor_id=${distributorId}` +
        (orderbookerId !== null ? `&orderbooker_id=${orderbookerId}` : '') +
        `&pjp_id=${pjpId}`;

      const [kpiResponse, graphResponse] = await Promise.all([
        fetch(kpiUrl),
        fetch(graphUrl)
      ]);

      const [kpiResult, graphResult] = await Promise.all([
        kpiResponse.json(),
        graphResponse.json()
      ]);

      if (kpiResult.status === "success") {
        setKpiData(kpiResult.data);
        localStorage.setItem('dashboardKpiData', JSON.stringify(kpiResult.data));
      }
      if (graphResult.status === "success") {
        setGraphData(graphResult.data);
        localStorage.setItem('dashboardGraphData', JSON.stringify(graphResult.data));
      }
      setIsDataApplied(true);
      localStorage.setItem('dashboardIsDataApplied', JSON.stringify(true));
    } catch (error) {
      setError("Failed to load dashboard data");
    } finally {
    }
  }, []);

  useEffect(() => {
    if (!currentDistributorId) {
      navigate("/");
      return;
    }
  
    // Always reset dashboard on load
    clearDashboardState();
    resetDashboard();
    fetchAndAutoApplyMostRecentPlan();
    // Then fetch plans again
    fetchPjpOptions();
  }, [currentDistributorId, fetchPjpOptions, navigate, resetDashboard]);

  const fetchAndAutoApplyMostRecentPlan = useCallback(async () => {
    try {
      const response = await fetch(`${BASE_URL}/get_pjp_plans?distributor_id=${currentDistributorId}`);
      const result = await response.json();
  
      if (result.status === "success" && result.plans.length > 0) {
        const mostRecent = result.plans[result.plans.length - 1];
        setPjpOptions(result.plans);
        setTempSelectedPjp(mostRecent.id);
        setSelectedPjp(mostRecent.id);
        localStorage.setItem('dashboardPjpOptions', JSON.stringify(result.plans));
        localStorage.setItem('dashboardSelectedPjp', JSON.stringify({ id: mostRecent.id, name: mostRecent.name }));
  
        // Fetch orderbookers for the selected plan
        const obResponse = await fetch(`${BASE_URL}/get_orderbookers_for_plan?distributor_id=${currentDistributorId}&pjp_id=${mostRecent.id}`);
        const obResult = await obResponse.json();
  
        if (obResult.status === "success" && obResult.orderbookers.length > 0) {
          const firstOB = obResult.orderbookers[0];
          setDropdownOptions(obResult.orderbookers);
          setTempSelectedValue(firstOB.id);
          setSelectedValue(firstOB.id);
          localStorage.setItem('dashboardDropdownOptions', JSON.stringify(obResult.orderbookers));
          localStorage.setItem('dashboardSelectedValue', JSON.stringify({ id: firstOB.id, name: firstOB.name }));
  
          // Auto-fetch dashboard data
          await fetchDashboardData(currentDistributorId, firstOB.id, mostRecent.id);
          setIsDataApplied(true);
          localStorage.setItem('dashboardIsDataApplied', JSON.stringify(true));
        }
      }
    } catch (error) {
      setError("Auto-apply failed while loading the most recent plan and orderbooker");
    }
  }, [currentDistributorId, fetchDashboardData]);
  
  const handlePjpChange = (value) => {
    setTempSelectedPjp(value);
    fetchOrderbookersForPlan(value);
    setTempSelectedValue(null);
    setDropdownOptions([]);
  };

  const handleDashboardForChange = useCallback((value) => {
    setTempSelectedValue(value);
  }, []);

  const handleApplyClick = useCallback(debounce(async (event) => {
    event.preventDefault();
    if (!tempSelectedPjp) {
      message.warning("Please select a plan first");
      return;
    }
    if (!tempSelectedValue) {
      message.warning("Please select an order booker");
      return;
    }
    if (!currentDistributorId) {
      navigate("/");
      return;
    }

    setSelectedValue(tempSelectedValue);
    setSelectedPjp(tempSelectedPjp);

    localStorage.setItem('dashboardSelectedValue', JSON.stringify({
      id: tempSelectedValue,
      name: dropdownOptions.find(opt => opt.id === tempSelectedValue)?.name
    }));

    localStorage.setItem('dashboardSelectedPjp', JSON.stringify({
      id: tempSelectedPjp,
      name: pjpOptions.find(opt => opt.id === tempSelectedPjp)?.name
    }));

    await fetchDashboardData(
      currentDistributorId,
      tempSelectedValue === -1 ? null : tempSelectedValue,
      tempSelectedPjp
    );
    
    // Close the controls drawer after applying filters
    setControlsVisible(false);
  }, 200), [tempSelectedValue, tempSelectedPjp, fetchDashboardData, navigate, dropdownOptions, pjpOptions, currentDistributorId]);
  
  const isDailyPlan = kpiData?.daily === 1;

  const getSelectedOrderBookerName = () => {
    if (!selectedValue) return "";
    const selected = dropdownOptions.find(opt => opt.id === selectedValue);
    return selected ? selected.name : "";
  };

  const getSelectedPjpName = () => {
    if (!selectedPjp) return "";
    const selected = pjpOptions.find(opt => opt.id === selectedPjp);
    return selected ? selected.name : "";
  };

  // Function to render KPI card
  // const renderKpiCard = (title, value, tooltip) => (
  //   <Card className="kpi-card" bordered={false} style={{ marginBottom: '10px', boxShadow: '0 1px 4px rgba(0,0,0,0.1)' }}>
  //     <Statistic
  //       title={
  //         <Tooltip title={tooltip}>
  //           <div style={{ display: 'flex', alignItems: 'center', fontSize: '14px', color: '#555' }}>
  //             {title}
  //             <InfoCircleOutlined style={{ marginLeft: '5px', fontSize: '12px', color: '#161181' }} />
  //           </div>
  //         </Tooltip>
  //       }
  //       value={value}
  //       valueStyle={{ color: '#161181', fontWeight: 'bold', fontSize: '18px' }}
  //     />
  //   </Card>
  // );
  const renderKpiCard = (title, value, tooltip) => (
    <Card 
      className="kpi-card" 
      bordered={false} 
      style={{ 
        marginBottom: '10px', 
        boxShadow: '0 1px 4px rgba(0,0,0,0.1)', 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100%' 
      }}
    >
      <Statistic
        title={
          <Tooltip title={tooltip}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center', 
              fontSize: '14px', 
              color: '#555', 
              textAlign: 'center' 
            }}>
              {title}
              <InfoCircleOutlined style={{ marginLeft: '5px', fontSize: '12px', color: '#161181' }} />
            </div>
          </Tooltip>
        }
        value={value}
        valueStyle={{ 
          color: '#161181', 
          fontWeight: 'bold', 
          fontSize: '18px', 
          textAlign: 'center', 
          width: '100%' 
        }}
      />
    </Card>
  );
  

  // Function to render the average metrics in the top section
//   const renderAverageMetricsRow = () => (
//     <Row gutter={[16, 16]} style={{ marginBottom: '20px' }}>
//       <Col xs={24} sm={8}>
//         <Card bordered={false} style={{ boxShadow: '0 1px 4px rgba(0,0,0,0.1)', height: '100%' }}>
//         <Statistic
//   title={
//     <Tooltip title="Average distance travelled per day across the selected PJP plan">
//       <div style={{ textAlign: 'center', fontSize: '14px', color: '#555' }}>
//         Average Distance
//         <InfoCircleOutlined style={{ marginLeft: '5px', fontSize: '12px', color: '#161181' }} />
//       </div>
//     </Tooltip>
//   }
//   value={
//     typeof distanceAverage === 'number'
//       ? distanceAverage.toFixed(2)
//       : typeof kpiData?.avg_distance_travelled === 'number'
//       ? kpiData.avg_distance_travelled.toFixed(2)
//       : 'N/A'
//   }
//   suffix="km"
//   valueStyle={{ 
//     color: '#161181', 
//     fontWeight: 'bold', 
//     fontSize: '20px',
//     textAlign: 'center',
//     width: '100%'
//   }}
// />

//         </Card>
//       </Col>
//       <Col xs={24} sm={8}>
//         <Card bordered={false} style={{ boxShadow: '0 1px 4px rgba(0,0,0,0.1)', height: '100%' }}>
//           <Statistic
//             title={
//               <Tooltip title="Average number of stores visited per day">
//                 <div style={{ display: 'flex', alignItems: 'center', fontSize: '14px', color: '#555' }}>
//                   Average Stores Visited
//                   <InfoCircleOutlined style={{ marginLeft: '5px', fontSize: '12px', color: '#161181' }} />
//                 </div>
//               </Tooltip>
//             }
//             value={visitsAverage ? Math.round(visitsAverage) : Math.round(kpiData?.avg_shops_visited) || 'N/A'}
//             valueStyle={{ color: '#161181', fontWeight: 'bold', fontSize: '20px' }}
//           />
//         </Card>
//       </Col>
//       <Col xs={24} sm={8}>
//         <Card bordered={false} style={{ boxShadow: '0 1px 4px rgba(0,0,0,0.1)', height: '100%' }}>
//           <Statistic
//             title={
//               <Tooltip title="Average workload per day, calculated based on time and effort">
//                 <div style={{ display: 'flex', alignItems: 'center', fontSize: '14px', color: '#555' }}>
//                   Average Workload
//                   <InfoCircleOutlined style={{ marginLeft: '5px', fontSize: '12px', color: '#161181' }} />
//                 </div>
//               </Tooltip>
//             }
//             value={workloadAverage ? Math.round(workloadAverage) : Math.round(kpiData?.avg_workload) || 'N/A'}
//             suffix="min"
//             valueStyle={{ color: '#161181', fontWeight: 'bold', fontSize: '20px' }}
//           />
//         </Card>
//       </Col>
//     </Row>
//   );

  const renderAverageMetricsRow = () => (
    <Row gutter={[16, 16]} style={{ marginBottom: '20px' }}>
      <Col xs={24} sm={8}>
        <Card bordered={false} style={{ boxShadow: '0 1px 4px rgba(0,0,0,0.1)', height: '100%' }}>
          <Statistic
            title={
              <Tooltip title="Average distance travelled per day across the selected PJP plan">
                <div style={{ textAlign: 'center', fontSize: '14px', color: '#555' }}>
                  Average Distance per Day
                  <InfoCircleOutlined style={{ marginLeft: '5px', fontSize: '12px', color: '#161181' }} />
                </div>
              </Tooltip>
            }
            value={
              typeof distanceAverage === 'number'
                ? distanceAverage.toFixed(2)
                : typeof kpiData?.avg_distance_travelled === 'number'
                ? kpiData.avg_distance_travelled.toFixed(2)
                : 'N/A'
            }
            suffix="km"
            valueStyle={{ 
              color: '#161181', 
              fontWeight: 'bold', 
              fontSize: '20px', 
              textAlign: 'center', 
              width: '100%' 
            }}
          />
        </Card>
      </Col>
  
      <Col xs={24} sm={8}>
        <Card bordered={false} style={{ boxShadow: '0 1px 4px rgba(0,0,0,0.1)', height: '100%' }}>
          <Statistic
            title={
              <Tooltip title="Average number of stores visited per day">
                <div style={{ textAlign: 'center', fontSize: '14px', color: '#555' }}>
                  Average Stores Visited per Day
                  <InfoCircleOutlined style={{ marginLeft: '5px', fontSize: '12px', color: '#161181' }} />
                </div>
              </Tooltip>
            }
            value={
              visitsAverage 
                ? Math.round(visitsAverage) 
                : Math.round(kpiData?.avg_shops_visited) || 'N/A'
            }
            valueStyle={{ 
              color: '#161181', 
              fontWeight: 'bold', 
              fontSize: '20px', 
              textAlign: 'center', 
              width: '100%' 
            }}
          />
        </Card>
      </Col>
  
      <Col xs={24} sm={8}>
        <Card bordered={false} style={{ boxShadow: '0 1px 4px rgba(0,0,0,0.1)', height: '100%' }}>
          <Statistic
            title={
              <Tooltip title="Average workload per day, calculated based on time and effort">
                <div style={{ textAlign: 'center', fontSize: '14px', color: '#555' }}>
                  Average Workload per Day
                  <InfoCircleOutlined style={{ marginLeft: '5px', fontSize: '12px', color: '#161181' }} />
                </div>
              </Tooltip>
            }
            value={
              workloadAverage 
                ? Math.round(workloadAverage) 
                : Math.round(kpiData?.avg_workload) || 'N/A'
            }
            suffix="min"
            valueStyle={{ 
              color: '#161181', 
              fontWeight: 'bold', 
              fontSize: '20px', 
              textAlign: 'center', 
              width: '100%' 
            }}
          />
        </Card>
      </Col>
    </Row>
  );
  
  return (
    <div className="dashboard-container">
      {/* Header with selection details and control button */}
      <Row style={{ padding: '12px 20px' }} align="middle" justify="space-between">
        <Col>
          <Title level={4} style={{ margin: 0, color: '#161181' }}>
            <DashboardOutlined /> Performance Dashboard
          </Title>
          {isDataApplied && (
            <Text type="secondary" style={{ fontSize: '13px' }}>
              Plan: <Text strong>{getSelectedPjpName()}</Text> | 
              Order Booker: <Text strong>{getSelectedOrderBookerName()}</Text>
            </Text>
          )}
        </Col>
        <Col>
          <Button 
            type="primary" 
            icon={<SettingOutlined />} 
            onClick={() => setControlsVisible(true)}
            style={{ backgroundColor: '#161181', borderColor: '#161181' }}
          >
            Dashboard Controls
          </Button>
        </Col>
      </Row>

      {/* Controls Drawer */}
      <Modal
  title="Dashboard Controls"
  open={controlsVisible}
  onCancel={() => setControlsVisible(false)}
  footer={null}
  centered
  width={400}
  bodyStyle={{ paddingTop: 12 }}
>
  <Space direction="vertical" style={{ width: '100%' }} size="large">
    <div>
      <Text className="filter-title" strong>Select Plan:</Text>
      <Select
        style={{ width: '100%', marginTop: '8px' }}
        size="middle"
        placeholder="Select a PJP plan"
        value={tempSelectedPjp}
        onChange={handlePjpChange}
        disabled={loading || pjpOptions.length === 0}
      >
        {pjpOptions.map(option => (
          <Option key={option.id} value={option.id}>{option.name}</Option>
        ))}
      </Select>
    </div>

    <div>
      <Text className="filter-title" strong>Order Booker:</Text>
      <Select
        style={{ width: '100%', marginTop: '8px' }}
        size="middle"
        placeholder={tempSelectedPjp ? "Select an order booker" : "Select a plan first"}
        value={tempSelectedValue}
        onChange={handleDashboardForChange}
        disabled={loading || !tempSelectedPjp || dropdownOptions.length === 0}
      >
        {dropdownOptions.map(option => (
          <Option key={option.id} value={option.id}>{option.name}</Option>
        ))}
      </Select>
    </div>

    <Button
      block
      size="middle"
      type="primary"
      style={{ 
        backgroundColor: '#161181', 
        borderColor: '#161181',
        marginTop: '16px'
      }}
      onClick={handleApplyClick}
      loading={loading}
      disabled={!tempSelectedPjp || !tempSelectedValue}
    >
      Apply Filters
    </Button>
  </Space>
</Modal>


    

      {loading && <LoadingSpinner_dashboard />}
      
      {error && (
        <Row style={{ margin: '24px 0' }}>
          <Col span={24}>
            <Card>
              <Typography.Text type="danger">{error}</Typography.Text>
            </Card>
          </Col>
        </Row>
      )}

      {!loading && !error && kpiData && isDataApplied && (
        <div style={{ padding: '0 20px' }}>
          {/* Average Metrics Row - Placed where dashboard controls were */}
          {renderAverageMetricsRow()}

          <Row gutter={[16, 0]}>
            {/* Sidebar with KPIs */}
            <Col xs={24} lg={6} style={{ marginBottom: '16px' }}>
              <Card
                className="kpi-sidebar"
                title={
                  <div style={{ fontSize: '16px', fontWeight: 600, color: '#161181' }}>
                    Performance Overview
                  </div>
                }
                style={{ height: '100%' }}
                bordered
              >
                {/* <Title level={5} style={{ color: '#161181', marginBottom: '10px' }}>Key Metrics</Title> */}
                {renderKpiCard(
                  "Total Distance", 
                  kpiData.total_distance_travelled ? `${kpiData.total_distance_travelled} km` : 'N/A',
                  "Total distance travelled by the order booker across all trips"
                )}
                {renderKpiCard(
                  "Total Store Visits", 
                  kpiData.total_shops_visited ?? 'N/A',
                  "Total number of stores visited by the order booker"
                )}
                
                {renderKpiCard(
                  "Visit Coverage", 
                 '100%',
                  "Reflects how well the PJP covers planned store visits. Higher coverage means better plan compliance and market reach."
                )}
                {renderKpiCard(
                  "Running Time", 
                  kpiData.response_time ? `${kpiData.response_time.toFixed(2)} min` : 'N/A',
                  "Total time spent running the selected PJP plan (in minutes)"
                )}
                
              </Card>
            </Col>

            {/* Main Content Area with Charts */}
            <Col xs={24} lg={18}>
              <Card 
                className="performance-charts"
                title={
                  <div style={{ fontSize: '18px', fontWeight: 600, color: '#161181' }}>
                    Performance Trends Analysis
                  </div>
                }
                bordered
              >
                <Tabs activeKey={activeTab} onChange={setActiveTab} type="card">
                
                  <TabPane tab="Distance" key="2">
                    <Row>
                      <Col span={24} style={{ height: '400px' }}>
                          <MemoizedBarChart
                            data={graphData.total_distance || []}
                            chartType="totalDistance"
                            useOrderbookerLabels={isDailyPlan}
                            onDifferenceCalculated={handleDistanceDifference}
                            onDateRangeCalculated={handleDistanceDateRange}
                            onAverageCalculated={handleDistanceAverage}
                            // Add props for improved axis labels
                            yAxisLabel="Distance (km)"
                            improveAxisLabels={true}
                          />
                        
                      </Col>
                    </Row>
                    <div style={{ 
  marginTop: '8px', 
  textAlign: 'left', 
  fontStyle: 'italic', 
  fontSize: '13px', 
  color: '#666' 
}}>
  Period: {distanceDateRange || 'No data available'}
</div>

                  </TabPane>
                  
                  <TabPane tab="Store Visits" key="3">
                    <Row>
                      <Col span={24} style={{ height: '400px' }}>
                          <MemoizedBarChart
                            data={graphData.store_visited || []}
                            chartType="totalVisits"
                            useOrderbookerLabels={isDailyPlan}
                            onDifferenceCalculated={handleVisitsDifference}
                            onDateRangeCalculated={handleVisitsDateRange}
                            onAverageCalculated={handleVisitsAverage}
                            // Add props for improved axis labels
                            yAxisLabel="Number of Stores"
                            improveAxisLabels={true}
                          />
                      </Col>
                    </Row>
                    <div style={{ 
  marginTop: '8px', 
  textAlign: 'left', 
  fontStyle: 'italic', 
  fontSize: '13px', 
  color: '#666' 
}}>
  Period: {visitsDateRange || 'No data available'}
</div>

                  </TabPane>
                  
                  <TabPane tab="Workload" key="4">
                    <Row>
                      <Col span={24} style={{ height: '400px' }}>
                          <MemoizedBarChart
                            data={graphData.workload || []}
                            chartType="workLoad"
                            useOrderbookerLabels={isDailyPlan}
                            onDifferenceCalculated={handleWorkloadDifference}
                            onDateRangeCalculated={handleWorkloadDateRange}
                            onAverageCalculated={handleWorkloadAverage}
                            // Add props for improved axis labels
                            yAxisLabel="Time (minutes)"
                            improveAxisLabels={true}
                          />
                      </Col>
                    </Row>
                    <div style={{ 
  marginTop: '8px', 
  textAlign: 'left', 
  fontStyle: 'italic', 
  fontSize: '13px', 
  color: '#666' 
}}>
  Period: {workloadDateRange || 'No data available'}
</div>

                  </TabPane>
                </Tabs>
              </Card>
            </Col>
          </Row>
        </div>
      )}
    </div>
  );
}

export default Dashboard;