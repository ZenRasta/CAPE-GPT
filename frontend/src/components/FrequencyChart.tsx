// frontend/src/components/FrequencyChart.tsx
import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { IFrequencyChartProps } from '../types';

const FrequencyChart: React.FC<IFrequencyChartProps> = ({ data, summary }) => {
  // If no data, return message
  if (!data || data.length === 0) {
    return (
      <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.50' }}>
        <Typography variant="body2" color="text.secondary">
          Not enough data to generate frequency analysis.
        </Typography>
      </Paper>
    );
  }

  // Sort data by year for proper display
  const sortedData = [...data].sort((a, b) => a.year - b.year);

  return (
    <Box sx={{ width: '100%' }}>
      <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.50', mb: 2 }}>
        <Typography variant="body2">{summary}</Typography>
      </Paper>

      <Box sx={{ height: 300, width: '100%' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={sortedData}
            margin={{ top: 5, right: 20, left: 10, bottom: 25 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="year"
              angle={-45}
              textAnchor="end"
              height={50}
              tickMargin={10}
              label={{ value: 'Year', position: 'bottom', offset: 0 }}
            />
            <YAxis
              label={{ value: 'Frequency', angle: -90, position: 'insideLeft' }}
              allowDecimals={false}
            />
            <Tooltip
              formatter={(value) => [`${value} question(s)`, 'Frequency']}
              labelFormatter={(year) => `Year: ${year}`}
            />
            <Bar dataKey="count" name="Frequency" fill="#8884d8" barSize={35} />
          </BarChart>
        </ResponsiveContainer>
      </Box>
    </Box>
  );
};

export default FrequencyChart;
