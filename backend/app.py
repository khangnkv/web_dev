from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from werkzeug.utils import secure_filename
from model.predictor import MLPredictor
from utils.data_processor import DataProcessor

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simple in-memory cache for uploaded files and processed DataFrames
file_cache = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    preview_type = request.form.get('preview_type', 'head')
    filename = None
    filepath = None
    processed_data = None

    # If file is uploaded, save and process it
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                # Determine file type and process accordingly
                ext = filename.rsplit('.', 1)[1].lower()
                if ext == 'csv':
                    data = pd.read_csv(filepath)
                elif ext in ['xlsx', 'xls']:
                    data = pd.read_excel(filepath)
                else:
                    return jsonify({"error": "Unsupported file type"}), 400
                # Use DataProcessor for further processing if needed
                data_processor = DataProcessor(filepath)
                data_processor.data = data  # Set loaded data directly
                processed_data = data_processor.process(skip_load=True)
                # Cache the DataFrame
                file_cache[filename] = processed_data
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "Invalid file type"}), 400
    else:
        # No file uploaded, try to use cached DataFrame
        filename = request.form.get('filename')
        if not filename or filename not in file_cache:
            return jsonify({"error": "No file uploaded or cached"}), 400
        processed_data = file_cache[filename]

    # Return the correct preview
    try:
        MAX_PREVIEW_ROWS = 100
        if preview_type == 'tail':
            preview_df = processed_data.tail(min(5, MAX_PREVIEW_ROWS))
        else:
            preview_df = processed_data.head(min(5, MAX_PREVIEW_ROWS))
        # Convert to row-oriented format for frontend compatibility
        preview_rows = preview_df.to_dict(orient='records')
        return jsonify({
            "message": "File uploaded and processed successfully",
            "filename": filename,
            "data_preview": preview_rows
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'filename' not in data:
            return jsonify({"error": "No data provided"}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], data['filename'])
        
        # Process data and make predictions
        data_processor = DataProcessor(filepath)
        processed_data = data_processor.process()
        
        predictor = MLPredictor()
        predictions = predictor.predict(processed_data)
        
        return jsonify({
            "predictions": predictions.tolist(),
            "message": "Predictions generated successfully"
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 