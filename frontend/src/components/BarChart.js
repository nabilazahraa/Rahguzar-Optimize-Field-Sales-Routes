import React, { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

const BarChart = ({
  data,
  onDifferenceCalculated,
  onDateRangeCalculated,
  onAverageCalculated,
  chartType,
  useOrderbookerLabels = false,
  yAxisLabel = ''
}) => {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  const getChartConfig = (type) => {
    switch (type) {
      case 'totalDistance':
        return { 
          label: 'Distance (km)', 
          colors: ['#161181', '#4E158D']  // Dark blue to purple radial gradient
        };
      case 'totalVisits':
        return { 
          label: 'Visits', 
          colors: ['#161181', '#4E158D']  // Dark blue to purple radial gradient
        };
      case 'workLoad':
        return { 
          label: 'Workload', 
          colors: ['#161181', '#4E158D']  // Dark blue to purple radial gradient
        };
      case 'salesComparison':
        return {
          labels: ['Last Week', '2nd Last Week'],
          colorSets: [
            ['#161181', '#4E158D'],  // First dataset gradient
            ['#0E3060', '#2A5294']   // Second dataset gradient - complementary blue
          ]
        };
      default:
        return { 
          label: 'Value', 
          colors: ['#161181', '#4E158D'] 
        };
    }
  };

  const renderChart = () => {
    if (!data || (Array.isArray(data) && data.length === 0)) {
      console.warn('No data provided to render chart');
      return;
    }

    let labels, datasets;

    try {
      if (chartType === 'salesComparison') {
        const lastWeek = data.lastWeek || [];
        const secondLastWeek = data.secondLastWeek || [];
        labels = lastWeek.map(item => `Day ${item.day}`);

        const lastWeekValues = lastWeek.map(item => item.value);
        const secondLastWeekValues = secondLastWeek.map(item => item.value);

        const lastWeekAvg = lastWeekValues.reduce((sum, val) => sum + val, 0) / lastWeekValues.length || 0;
        const secondLastWeekAvg = secondLastWeekValues.reduce((sum, val) => sum + val, 0) / secondLastWeekValues.length || 0;
        const difference = secondLastWeekAvg ? ((lastWeekAvg - secondLastWeekAvg) / secondLastWeekAvg) * 100 : 0;

        if (onDifferenceCalculated) onDifferenceCalculated(difference.toFixed(1));

        const allDates = [...lastWeek, ...secondLastWeek].map(item => new Date(item.date));
        if (!useOrderbookerLabels) {
          const minDate = new Date(Math.min(...labels.map(date => new Date(date))));
          const maxDate = new Date(Math.max(...labels.map(date => new Date(date))));
          const formatDate = (date) =>
            `${(date.getMonth() + 1).toString().padStart(2, '0')}/${date.getDate().toString().padStart(2, '0')}/${date.getFullYear()}`;
          if (onDateRangeCalculated) onDateRangeCalculated(formatDate(minDate), formatDate(maxDate));
        } else {
          if (onDateRangeCalculated) onDateRangeCalculated(null, null);
        }
        
        // const formatDate = (date) =>
        //   `${(date.getMonth() + 1).toString().padStart(2, '0')}/${date.getDate().toString().padStart(2, '0')}/${date.getFullYear()}`;
        // if (onDateRangeCalculated) onDateRangeCalculated(formatDate(minDate), formatDate(maxDate));
        // if (onAverageCalculated) onAverageCalculated(lastWeekAvg.toFixed(1));

        const config = getChartConfig(chartType);
        datasets = [
          {
            label: config.labels[0],
            data: lastWeekValues,
            backgroundColor: (context) => {
              const chart = context.chart;
              const { ctx, chartArea } = chart;
              if (!chartArea) return;
              const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
              gradient.addColorStop(0, config.colorSets[0][0]);
              gradient.addColorStop(1, config.colorSets[0][1]);
              return gradient;
            },
            borderColor: config.colorSets[0][0],
            borderWidth: 1,
            borderRadius: 5,
            hoverBackgroundColor: config.colorSets[0][1]
          },
          {
            label: config.labels[1],
            data: secondLastWeekValues,
            backgroundColor: (context) => {
              const chart = context.chart;
              const { ctx, chartArea } = chart;
              if (!chartArea) return;
              const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
              gradient.addColorStop(0, config.colorSets[1][0]);
              gradient.addColorStop(1, config.colorSets[1][1]);
              return gradient;
            },
            borderColor: config.colorSets[1][0],
            borderWidth: 1,
            borderRadius: 5,
            hoverBackgroundColor: config.colorSets[1][1]
          }
        ];
      } else {
        labels = data.map((item, index) => {
          if (useOrderbookerLabels) {
            return item.name || `Orderbooker ${index + 1}`;
          } else {
            return item.date;
          }
        });
        
        const values = data.map(item => item.value);

        const latestValue = values[values.length - 1] || 0;
        const secondLatestValue = values[values.length - 2] || 0;
        const difference = secondLatestValue ? ((latestValue - secondLatestValue) / secondLatestValue) * 100 : 0;

        if (onDifferenceCalculated) onDifferenceCalculated(difference.toFixed(1));

        const minDate = new Date(Math.min(...labels.map(date => new Date(date))));
        const maxDate = new Date(Math.max(...labels.map(date => new Date(date))));
        const formatDate = (date) =>
          `${(date.getMonth() + 1).toString().padStart(2, '0')}/${date.getDate().toString().padStart(2, '0')}/${date.getFullYear()}`;
        if (onDateRangeCalculated) onDateRangeCalculated(formatDate(minDate), formatDate(maxDate));

        const total = values.reduce((sum, value) => sum + value, 0);
        const average = total / values.length || 0;
        if (onAverageCalculated) onAverageCalculated(average.toFixed(1));

        const config = getChartConfig(chartType);
        datasets = [{
          label: config.label,
          data: values,
          backgroundColor: (context) => {
            const chart = context.chart;
            const { ctx, chartArea } = chart;
            if (!chartArea) return;
            const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
            gradient.addColorStop(0, config.colors[0]);
            gradient.addColorStop(1, config.colors[1]);
            return gradient;
          },
          borderColor: config.colors[0],
          borderWidth: 1,
          borderRadius: 5,
          hoverBackgroundColor: config.colors[1]
        }];
      }

      if (chartInstance.current) chartInstance.current.destroy();

      const ctx = chartRef.current.getContext('2d');
      chartInstance.current = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          layout: {
            padding: {
              top: 20,
              bottom: 20,
              left: 10,
              right: 10
            }
          },
          scales: {
            x: {
              grid: { 
                display: false,
                drawBorder: false
              },
              title: { 
                display: true, 
                text: chartType === 'salesComparison'
                  ? 'Days'
                  : useOrderbookerLabels
                    ? 'Order Booker'
                    : 'Date',
                font: { 
                  size: 14, 
                  weight: 'bold',
                  color: '#161181'
                }
              },
              ticks: { 
                font: { 
                  size: 12,
                  color: '#4E158D'
                },
                padding: 10
              }
            },
            y: {
              beginAtZero: true,
              grid: {
                color: 'rgba(22, 17, 129, 0.1)',
                drawBorder: false
              },
              title: { 
                display: !!yAxisLabel,
                text: yAxisLabel,
                font: { 
                  size: 14, 
                  weight: 'bold',
                  color: '#161181'
                }
              },
              ticks: { 
                font: { 
                  size: 12,
                  color: '#4E158D'
                },
                padding: 10
              }
            }
          }
,          
          plugins: {
            legend: { 
              display: chartType === 'salesComparison',
              labels: {
                font: {
                  size: 12,
                  weight: 'bold'
                },
                color: '#161181'
              }
            },
            tooltip: {
              backgroundColor: 'rgba(22, 17, 129, 0.9)',
              borderColor: '#4E158D',
              borderWidth: 1,
              cornerRadius: 5,
              callbacks: {
                label: (context) => `${context.dataset.label}: ${context.parsed.y || 0}`,
                title: (tooltipItems) => {
                  const index = tooltipItems[0].dataIndex;
                  return chartType === 'salesComparison'
                    ? (tooltipItems[0].datasetIndex === 0 ? data.lastWeek[index].date : data.secondLastWeek[index].date)
                    : labels[index];
                }
              },
              titleFont: { 
                size: 14, 
                weight: 'bold',
                color: 'white'
              },
              bodyFont: { 
                size: 12,
                color: 'white'
              }
            }
          },
          animation: {
            duration: 1000,
            easing: 'easeOutQuart'
          },
          hover: {
            mode: 'nearest',
            intersect: true
          }
        }
      });
    } catch (error) {
      console.error('Error rendering chart:', error);
    }
  };

  useEffect(() => {
    renderChart();
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [data, chartType, onDifferenceCalculated, onDateRangeCalculated, onAverageCalculated]);

  return (
    <div style={{ 
      width: '100%', 
      height: '450px', 
      borderRadius: '12px', 
      boxShadow: '0 10px 25px rgba(22, 17, 129, 0.1)',
      backgroundColor: 'rgba(22, 17, 129, 0.02)',
      marginBottom: '-10px'
    }}> 
      <canvas ref={chartRef}></canvas>
    </div>
  );
};

export default BarChart;