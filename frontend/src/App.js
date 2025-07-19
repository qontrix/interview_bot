import React, { useEffect, useState } from 'react';
import Plotly from 'plotly.js-dist-min';
import './App.css';

function Dashboard() {
  const [dataFromFlask, setDataFromFlask] = useState([]);
  const [averageRating, setAverageRating] = useState(0);
  const [interviewCount, setInterviewCount] = useState(0);
  const [loading, setLoading] = useState(true);

  // Fetch dashboard data from Flask backend
  useEffect(() => {
    async function fetchDashboardData() {
      try {
        const response = await fetch('/dashboard_data'); // You may need to create this API in Flask to return JSON
        const result = await response.json();

        setDataFromFlask(result.data);
        setAverageRating(result.average_rating);
        setInterviewCount(result.interview_count);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      } finally {
        setLoading(false);
      }
    }

    fetchDashboardData();
  }, []);

  // Plotly charts rendering function
  useEffect(() => {
    if (!loading && dataFromFlask.length > 0) {
      // Skill Breakdown Horizontal Bar Chart
      const selectedUser = dataFromFlask[0];
      const skills = ['Technical Knowledge', 'Communication', 'Problem Solving', 'Time Management'];
      const ratings = [
        selectedUser.technical_rating || 0,
        selectedUser.communication_rating || 0,
        selectedUser.problem_solving_rating || 0,
        selectedUser.time_management_rating || 0,
      ];

      const skillTrace = {
        x: ratings,
        y: skills,
        type: 'bar',
        orientation: 'h',
        marker: {
          color: ['#A0CED9', '#FFB5A7', '#CBAACB', '#FFDAC1'],
          line: { color: '#999', width: 1 },
        },
        hoverinfo: 'x+y',
      };

      const skillLayout = {
        title: `Skill Breakdown - User: ${selectedUser.user_id}`,
        xaxis: { range: [0, 10], title: 'Rating (/10)', gridcolor: '#e6e6e6' },
        yaxis: { automargin: true },
        margin: { l: 150, r: 30, t: 50, b: 40 },
        height: 400,
        paper_bgcolor: '#fcfcfc',
        plot_bgcolor: '#fcfcfc',
      };

      Plotly.newPlot('skill-breakdown', [skillTrace], skillLayout, { responsive: true });

      // Performance Trend Line Chart
      const performanceTrace = {
        x: dataFromFlask.map((_, i) => i + 1),
        y: dataFromFlask.map((u) => u.total_rating || 0),
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#FFB5A7', width: 3 },
        marker: { size: 8, color: '#A0CED9' },
        name: 'Total Rating',
      };

      const performanceLayout = {
        title: 'Performance Trend (Total Rating)',
        xaxis: { title: 'Interview Order', gridcolor: '#e6e6e6' },
        yaxis: { title: 'Total Rating (/10)', range: [0, 10], gridcolor: '#e6e6e6' },
        height: 400,
        paper_bgcolor: '#fcfcfc',
        plot_bgcolor: '#fcfcfc',
        margin: { t: 50, l: 50, r: 40, b: 60 },
      };

      Plotly.newPlot('performance-trend', [performanceTrace], performanceLayout, { responsive: true });

      // Radar Chart
      const radarTraces = dataFromFlask.map((user, index) => ({
        type: 'scatterpolar',
        r: [
          user.technical_rating || 0,
          user.communication_rating || 0,
          user.problem_solving_rating || 0,
          user.time_management_rating || 0,
        ],
        theta: skills,
        fill: 'toself',
        name: `User ${user.user_id}`,
        opacity: 0.55,
        line: { color: ['#A0CED9', '#FFB5A7', '#CBAACB', '#FFDAC1'][index % 4], width: 2 },
        marker: { color: ['#A0CED9', '#FFB5A7', '#CBAACB', '#FFDAC1'][index % 4] },
      }));

      const radarLayout = {
        title: 'Skill Comparison Radar Chart',
        polar: {
          radialaxis: {
            visible: true,
            range: [0, 10],
            gridcolor: '#ccc',
            tickfont: { size: 12 },
          },
          angularaxis: {
            tickfont: { size: 12 },
            gridcolor: '#ccc',
          },
        },
        showlegend: true,
        height: 500,
        paper_bgcolor: '#fcfcfc',
        plot_bgcolor: '#fcfcfc',
        margin: { t: 60, l: 40, r: 40, b: 40 },
      };

      Plotly.newPlot('radar-chart', radarTraces, radarLayout, { responsive: true });
    }
  }, [loading, dataFromFlask]);

  if (loading) return <div>Loading dashboard data...</div>;

  if (!dataFromFlask.length) return <div>No data available.</div>;

  return (
    <div style={{ padding: 20 }}>
      <h1>Interview Ratings Dashboard</h1>

      {/* KPI Cards */}
      <div style={{ display: 'flex', gap: 30, marginTop: 20, marginBottom: 30 }}>
        <div
          style={{
            flex: 1,
            background: '#ffffff',
            borderRadius: 12,
            boxShadow: '0 4px 10px rgba(0,0,0,0.05)',
            padding: 20,
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: 28, color: '#2196F3' }}>📈</div>
          <div style={{ fontSize: 32, fontWeight: 'bold' }}>{averageRating}</div>
          <div style={{ color: '#666' }}>Average Rating</div>
        </div>
        <div
          style={{
            flex: 1,
            background: '#ffffff',
            borderRadius: 12,
            boxShadow: '0 4px 10px rgba(0,0,0,0.05)',
            padding: 20,
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: 28, color: '#2196F3' }}>✅</div>
          <div style={{ fontSize: 32, fontWeight: 'bold' }}>{interviewCount}</div>
          <div style={{ color: '#666' }}>Completed Interviews</div>
        </div>
      </div>

      {/* Charts Section */}
      <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
        <div id="skill-breakdown" style={{ flex: 1, minWidth: 400, height: 400 }}></div>
        <div id="performance-trend" style={{ flex: 1, minWidth: 400, height: 400 }}></div>
      </div>

      <div style={{ marginTop: 30 }}>
        <div id="radar-chart" style={{ height: 500 }}></div>
      </div>
    </div>
  );
}

export default Dashboard;
