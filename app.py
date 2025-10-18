from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the model and scaler
model_path = "model.pkl"
scaler_path = "scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Column mapping for 30 features
# Replace with your actual column order from training data
columns = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
           'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
           'V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        V2 = float(data['V2'])
        V4 = float(data['V4'])
        V11 = float(data['V11'])
        Amount = float(data['Amount'])

        # Create 30-length input array filled with zeros
        x_input = np.zeros(30)

        # Fill the 4 inputs in the correct positions
        x_input[1] = V2    # V2 index
        x_input[3] = V4    # V4 index
        x_input[10] = V11  # V11 index
        x_input[28] = Amount  # Amount index

        # Scale and predict
        x_scaled = scaler.transform([x_input])
        pred = model.predict(x_scaled)[0]

        result_text = "Not Fraud" if pred == 0 else "Fraud"

        return jsonify({'prediction': int(pred), 'result': result_text})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
