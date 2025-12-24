from flask import Flask, request, jsonify, render_template
import pickle,joblib,os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

# Absolute paths (important on Windows)
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#MODEL_DIR = os.path.join(BASE_DIR, "models")
#
#ridge_model = joblib.load(os.path.join(MODEL_DIR, "ridge.pkl"))
#standard_scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

ridge_model = pickle.load(open('models/ridge.pickle', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pickle', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_data_point():
    if request.method == 'POST':
        # Process input and predict
        temperature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('WS'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))
        
        input_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
        input_scaled = standard_scaler.transform(input_data)
        result = ridge_model.predict(input_scaled)
        return render_template('home.html', result=result[0])
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")