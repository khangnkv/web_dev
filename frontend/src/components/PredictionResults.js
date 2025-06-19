import React from 'react';
import { Typography, Box, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@mui/material';

const PredictionResults = ({ predictions }) => {
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Prediction Results
      </Typography>
      
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Row</TableCell>
              <TableCell>Prediction</TableCell>
              <TableCell>Confidence</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {predictions.map((prediction, index) => (
              <TableRow key={index}>
                <TableCell>{index + 1}</TableCell>
                <TableCell>{prediction}</TableCell>
                <TableCell>
                  {typeof prediction === 'object' && prediction.probability
                    ? `${(prediction.probability * 100).toFixed(2)}%`
                    : 'N/A'}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default PredictionResults; 