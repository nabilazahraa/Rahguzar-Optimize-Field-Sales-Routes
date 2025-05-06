import React, { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';
import 'chartjs-plugin-zoom';

const EnhancedLineChart = ({ 
  data, 
  onDifferenceCalculated, 
  onDateRangeCalculated, 
  onAverageCalculated,
  chartType
}) => {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  const renderChart = () => {
    if (!data || data.length === 0) return;

    const labels = data.map(item => item.date);
    const values = data.map(item => item.value);

    // Calculate metrics
    const latestValue = values[values.length - 1];
    const secondLatestValue = values[values.length - 2];
    const difference = secondLatestValue ? ((latestValue - secondLatestValue) / secondLatestValue) * 100 : 0;
    if (onDifferenceCalculated) onDifferenceCalculated(difference.toFixed(1));

    const minDate = new Date(Math.min(...labels.map(date => new Date(date))));
    const maxDate = new Date(Math.max(...labels.map(date => new Date(date))));
    const formatDate = (date) =>
      `${(date.getMonth() + 1).toString().padStart(2, '0')}/${date.getDate().toString().padStart(2, '0')}/${date.getFullYear()}`;
    if (onDateRangeCalculated) onDateRangeCalculated(formatDate(minDate), formatDate(maxDate));

    const total = values.reduce((sum, value) => sum + value, 0);
    const average = total / values.length;
    if (onAverageCalculated) onAverageCalculated(average.toFixed(1));

    // Calculate Y-axis range: 40% below lowest and 40% above highest
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const range = maxValue - minValue;
    const padding = range * 0.4;
    const yMinUnrounded = Math.max(0, minValue - padding);
    const yMaxUnrounded = maxValue + padding;

    const magnitude = Math.max(Math.abs(minValue), Math.abs(maxValue));
    let roundTo = magnitude >= 1000 ? 100 : magnitude >= 100 ? 10 : magnitude >= 10 ? 10 : 1;

    const yMin = Math.floor(yMinUnrounded / roundTo) * roundTo;
    const yMax = Math.ceil(yMaxUnrounded / roundTo) * roundTo;

    const ctx = chartRef.current.getContext('2d');
    
    // Gradient for line and point
    const gradientLine = ctx.createLinearGradient(0, 0, 0, 400);
    gradientLine.addColorStop(0, '#4e73df');
    gradientLine.addColorStop(1, '#224abe');

    // Gradient for fill
    const gradientFill = ctx.createLinearGradient(0, 0, 0, 400);
    gradientFill.addColorStop(0, 'rgba(78, 115, 223, 0.3)');
    gradientFill.addColorStop(1, 'rgba(34, 74, 190, 0)');

    if (chartInstance.current) chartInstance.current.destroy();

    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: chartType === 'totalDistance' ? 'Distance (km)' : 
                   chartType === 'totalVisits' ? 'Visits' : 'Workload',
            data: values,
            borderColor: gradientLine,
            backgroundColor: gradientFill,
            fill: true,
            tension: 0.4,
            pointRadius: 5,
            pointHoverRadius: 8,
            pointBackgroundColor: '#fff',
            pointBorderColor: gradientLine,
            pointBorderWidth: 2,
            pointHoverBackgroundColor: gradientLine,
            pointHoverBorderColor: '#fff',
            borderWidth: 3,
          },
          // Extended data points dataset (vertical lines to yMin)
          {
            type: 'scatter',
            label: 'Extended Points',
            data: values.map((value, index) => ({ x: labels[index], y: yMin })),
            backgroundColor: 'rgba(78, 115, 223, 0.5)',
            borderColor: 'rgba(78, 115, 223, 0.5)',
            pointRadius: 4,
            pointStyle: 'triangle',
            showLine: false,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 1500,
          easing: 'easeOutCubic',
        },
        scales: {
          x: {
            type: 'category',
            grid: { display: false },
            title: { 
              display: true, 
              text: 'Date', 
              font: { size: 16, weight: 'bold', family: 'Arial' },
              color: '#333'
            },
            ticks: { 
              font: { size: 14, family: 'Arial' },
              color: '#666',
              autoSkip: true,
              maxRotation: 45, // Rotate labels up to 45 degrees when needed
              minRotation: 0,  // Start at 0 degrees
              maxTicksLimit: 10, // Limit to 10 ticks maximum
            }
          },
          y: {
            min: yMin,
            max: yMax,
            grid: { 
              color: 'rgba(0, 0, 0, 0.05)',
              borderDash: [5, 5]
            },
            title: { 
              display: true, 
              text: chartType === 'totalDistance' ? 'Distance (km)' : 
                    chartType === 'totalVisits' ? 'Visits' : 'Workload (min)',
              font: { size: 16, weight: 'bold', family: 'Arial' },
              color: '#333'
            },
            ticks: {
              font: { size: 14, family: 'Arial' },
              color: '#666',
              callback: function(value) {
                return Math.round(value / roundTo) * roundTo;
              }
            }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(78, 115, 223, 0.9)',
            titleFont: { size: 16, weight: 'bold', family: 'Arial' },
            bodyFont: { size: 14, family: 'Arial' },
            padding: 12,
            cornerRadius: 6,
            boxPadding: 6,
            callbacks: {
              label: (context) => {
                if (context.datasetIndex === 0) {
                  return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}`;
                } else {
                  return `Extended Point`;
                }
              },
              title: (tooltipItems) => labels[tooltipItems[0].dataIndex]
            }
          },
          zoom: {
            pan: {
              enabled: true,
              mode: 'x'
            },
            zoom: {
              wheel: { enabled: true },
              pinch: { enabled: true },
              mode: 'x'
            }
          }
        },
        hover: {
          mode: 'nearest',
          intersect: true,
          animationDuration: 400
        },
        elements: {
          point: {
            hitRadius: 10
          }
        }
      },
      plugins: [{
        // Custom plugin to draw vertical lines from extended points to the main line
        afterDraw: (chart) => {
          const ctx = chart.ctx;
          const datasets = chart.data.datasets;
          const xAxis = chart.scales['x'];
          const yAxis = chart.scales['y'];

          const yMinPos = yAxis.getPixelForValue(yMin);

          ctx.save();
          ctx.beginPath();
          ctx.setLineDash([5, 5]);
          ctx.strokeStyle = 'rgba(78, 115, 223, 0.5)';

          datasets[1].data.forEach((point, index) => {
            const mainDataPoint = datasets[0].data[index];
            const xPos = xAxis.getPixelForValue(point.x);
            const mainYPos = yAxis.getPixelForValue(mainDataPoint);
            const extendedYPos = yMinPos;

            ctx.moveTo(xPos, extendedYPos);
            ctx.lineTo(xPos, mainYPos);
          });

          ctx.stroke();
          ctx.restore();
        }
      }]
    });

    // Removed the average line annotation to eliminate the horizontal line
    // chartInstance.current.update();
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
    <div style={{ width: '100%', height: '450px', marginBottom: '-10px' }}>
      {(!data || data.length === 0) && (
        <div style={{ textAlign: 'center', padding: '20px', color: '#666' }}>
          No data available to display the chart
        </div>
      )}
      <canvas ref={chartRef} style={{ display: data && data.length > 0 ? 'block' : 'none' }}></canvas>
    </div>
  );
};

export default EnhancedLineChart;