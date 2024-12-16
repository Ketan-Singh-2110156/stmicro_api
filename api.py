from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
def load_trained_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

model_path = 'cnn_model_tf'
lstm_model_path = 'lstm_model_tf'  # Adjust the path as per your setup
trained_model = load_trained_model(model_path)
lstm_model = load_trained_model(lstm_model_path)

@app.route('/conv', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.json
        if 'input_data' not in data:
            return jsonify({'error': 'Input data is missing'}), 400

        input_data = np.array(data['input_data'])
        
        # Ensure the input shape matches the model's requirement
        if len(input_data.shape) != 2 or input_data.shape[1] != 14:
            return jsonify({'error': 'Invalid input shape. Expected (n_samples, 14)'}), 400

        input_data = input_data.reshape(-1, 14, 1)  # Reshape for Conv1D model
        
        # Make predictions
        predictions = trained_model.predict(input_data)
        predictions = predictions.tolist()  # Convert numpy array to list for JSON serialization
        
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/lstm', methods=['POST'])
def lstmpredict():
    try:
        # Get input data
        data = request.json
        if 'input_data' not in data:
            return jsonify({'error': 'Input data is missing'}), 400

        input_data = np.array(data['input_data'])
        
        # Ensure the input shape matches the model's requirement
        if len(input_data.shape) != 2 or input_data.shape[1] != 14:
            return jsonify({'error': 'Invalid input shape. Expected (n_samples, 14)'}), 400

        input_data = input_data.reshape(-1, 14, 1)  # Reshape for Conv1D model
        
        # Make predictions
        predictions = lstm_model.predict(input_data)
        predictions = predictions.tolist()  # Convert numpy array to list for JSON serialization
        
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
