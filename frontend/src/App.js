import React, { useState, useRef, useEffect } from 'react';
import { 
  Container, 
  Box, 
  Typography, 
  Paper,
  CircularProgress,
  TableContainer,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Button,
  ButtonGroup,
  Select,
  MenuItem,
  InputLabel,
  FormControl
} from '@mui/material';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import FileUpload from './components/FileUpload';
import PredictionResults from './components/PredictionResults';
import { uploadFile, getPredictions } from './services/api';

function App() {
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [dataPreview, setDataPreview] = useState(null);
  const [previewType, setPreviewType] = useState('head'); // 'head' or 'tail'
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedFilename, setUploadedFilename] = useState(null);
  const [selectedProjectId, setSelectedProjectId] = useState('');
  const [projectIdOptions, setProjectIdOptions] = useState([]);

  const handleFileUpload = async (file) => {
    try {
      setLoading(true);
      setUploadedFile(file);
      const formData = new FormData();
      formData.append('file', file);
      formData.append('preview_type', previewType);
      const response = await uploadFile(formData);
      setDataPreview(response.data.data_preview);
      setUploadedFilename(response.data.filename);
      // Extract unique projectid values for dropdown
      const projectidCol = response.data.data_preview['projectid'];
      if (projectidCol) {
        const uniqueIds = Array.from(new Set(Object.values(projectidCol)));
        setProjectIdOptions(uniqueIds);
        setSelectedProjectId('');
      } else {
        setProjectIdOptions([]);
        setSelectedProjectId('');
      }
      // Get predictions
      const predictionResponse = await getPredictions({
        filename: response.data.filename
      });
      setPredictions(predictionResponse.data.predictions);
      toast.success('File processed successfully!');
    } catch (error) {
      toast.error(error.response?.data?.error || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  // When previewType changes, re-request preview from backend if file is uploaded
  useEffect(() => {
    const fetchPreview = async () => {
      if (!uploadedFilename) return;
      setLoading(true);
      const formData = new FormData();
      formData.append('filename', uploadedFilename);
      formData.append('preview_type', previewType);
      try {
        const response = await uploadFile(formData);
        setDataPreview(response.data.data_preview);
        // Extract unique projectid values for dropdown
        const projectidCol = response.data.data_preview['projectid'];
        if (projectidCol) {
          const uniqueIds = Array.from(new Set(Object.values(projectidCol)));
          setProjectIdOptions(uniqueIds);
        } else {
          setProjectIdOptions([]);
        }
      } catch (error) {
        toast.error(error.response?.data?.error || 'An error occurred');
      } finally {
        setLoading(false);
      }
    };
    // Only fetch preview if file is already uploaded
    if (uploadedFilename) {
      fetchPreview();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewType]);

  // Helper to get head or tail rows from dataPreview, filtered by projectid if selected
  const getPreviewRows = () => {
    if (!dataPreview || !Array.isArray(dataPreview)) return [];
    let rows = dataPreview;
    // If a projectid is selected, filter rows
    if (selectedProjectId) {
      rows = rows.filter(row => row['projectid'] === selectedProjectId);
    }
    if (previewType === 'head') {
      return rows.slice(0, 5);
    } else {
      // Only show last 5 rows, but if there are less than 5 rows, show all
      return rows.slice(Math.max(rows.length - 5, 0));
    }
  };

  // Helper to get columns from dataPreview
  const getPreviewColumns = () => {
    if (!dataPreview || !Array.isArray(dataPreview) || dataPreview.length === 0) return [];
    return Object.keys(dataPreview[0]);
  };

  // Helper to check if there are missing values in the preview
  const hasMissingValues = () => {
    const rows = getPreviewRows();
    return rows.some(row => getPreviewColumns().some(col => row[col] === null || row[col] === undefined || row[col] === ''));
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          ML Prediction App
        </Typography>
        
        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <FileUpload onFileUpload={handleFileUpload} />
        </Paper>

        {loading && (
          <Box display="flex" justifyContent="center" my={4}>
            <CircularProgress />
          </Box>
        )}

        {dataPreview && (
          <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Data Preview
            </Typography>
            <ButtonGroup sx={{ mb: 2 }}>
              <Button
                variant={previewType === 'head' ? 'contained' : 'outlined'}
                onClick={() => setPreviewType('head')}
                disabled={previewType === 'head'}
              >
                Head (first 5)
              </Button>
              <Button
                variant={previewType === 'tail' ? 'contained' : 'outlined'}
                onClick={() => setPreviewType('tail')}
                disabled={previewType === 'tail'}
              >
                Tail (last 5)
              </Button>
            </ButtonGroup>
            {projectIdOptions.length > 0 && (
              <FormControl sx={{ mb: 2, minWidth: 200 }} size="small">
                <InputLabel id="projectid-select-label">Project ID</InputLabel>
                <Select
                  labelId="projectid-select-label"
                  value={selectedProjectId}
                  label="Project ID"
                  onChange={e => setSelectedProjectId(e.target.value)}
                >
                  <MenuItem value=""><em>All</em></MenuItem>
                  {projectIdOptions.map((id) => (
                    <MenuItem key={id} value={id}>{id}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}
            {hasMissingValues() && (
              <Typography variant="body2" color="warning.main" sx={{ mb: 1 }}>
                Warning: Some values are missing in the preview and are shown as <em style={{color:'#888'}}>N/A</em>.
              </Typography>
            )}
            <Box sx={{ overflow: 'auto' }}>
              <TableContainer component={Paper}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      {getPreviewColumns().map((col) => (
                        <TableCell key={col}>{col}</TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {getPreviewRows().map((row, rowIdx) => (
                      <TableRow key={rowIdx}>
                        {getPreviewColumns().map((col, colIdx) => (
                          <TableCell key={colIdx}>
                            {row[col] === null || row[col] === undefined || row[col] === '' ? <em style={{color:'#888'}}>N/A</em> : row[col]}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          </Paper>
        )}

        {predictions && (
          <Paper elevation={3} sx={{ p: 3 }}>
            <PredictionResults predictions={predictions} />
          </Paper>
        )}
      </Box>
      <ToastContainer />
    </Container>
  );
}

export default App; 