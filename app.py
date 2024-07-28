import os
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('fish_weight_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    prediction_text = f'Predicted Fish Weight: {prediction[0]:.2f}'

    return render_template('result.html',
                           prediction_text=prediction_text,
                           Length1=features[0],
                           Length2=features[1],
                           Length3=features[2],
                           Height=features[3],
                           Width=features[4])

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
