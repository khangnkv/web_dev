# ML Prediction Web Application

This web application allows users to upload CSV files and get predictions from a machine learning model.

## Project Structure
```
.
├── backend/
│   ├── app.py
│   ├── model/
│   │   └── predictor.py
│   └── utils/
│       └── data_processor.py
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   └── App.js
│   └── package.json
└── requirements.txt
```

## Setup Instructions

### Backend Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask server:
   ```bash
   cd backend
   python app.py
   ```

### Frontend Setup
1. Install Node.js dependencies:
   ```bash
   cd frontend
   npm install
   ```
2. Start the development server:
   ```bash
   npm start
   ```

## Features
- CSV file upload
- Data validation
- ML model prediction
- Results display
- Error handling

## API Endpoints
- POST /api/upload: Upload CSV file
- POST /api/predict: Get predictions
- GET /api/health: Health check endpoint 