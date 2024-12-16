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

# Model paths
model_path = 'cnn_model_tf'
lstm_model_path = 'lstm_model_tf'
trained_model = load_trained_model(model_path)
lstm_model = load_trained_model(lstm_model_path)

@app.route('/conv', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'input_data' not in data:
            return jsonify({'error': 'Input data is missing'}), 400

        input_data = np.array(data['input_data'])

        if len(input_data.shape) != 2 or input_data.shape[1] != 14:
            return jsonify({'error': 'Invalid input shape. Expected (n_samples, 14)'}), 400

        input_data = input_data.reshape(-1, 14, 1)
        predictions = trained_model.predict(input_data).tolist()

        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/lstm', methods=['POST'])
def lstmpredict():
    try:
        data = request.json
        if 'input_data' not in data:
            return jsonify({'error': 'Input data is missing'}), 400

        input_data = np.array(data['input_data'])

        if len(input_data.shape) != 2 or input_data.shape[1] != 14:
            return jsonify({'error': 'Invalid input shape. Expected (n_samples, 14)'}), 400

        input_data = input_data.reshape(-1, 14, 1)
        predictions = lstm_model.predict(input_data).tolist()

        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Use the PORT environment variable provided by Render

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
