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
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return render_template('index.html', prediction_text=f'Predicted Weight: {prediction:.2f} grams')

if __name__ == "__main__":
    app.run()

# from flask import Flask # we imported the Flask class. An instance of this class will be our WSGI application
# app = Flask(__name__) #creating the Flask class object
#
# @app.route('/') #We then use the route() decorator to tell Flask what URL should trigger our function.
# def hello_world(): #function
#     return "<p>Hello, World!</p>"
#
# if __name__ == "__main__": # It Allows You to Execute Code When the File Runs as a Script
#     app.run()

# from flask import Flask, redirect, url_for, request, render_template
#
# app = Flask(__name__)
#
# @app.route('/')
# def index():
#    return render_template('index.html')
#
# if __name__ == '__main__':
#    app.run(debug = True)