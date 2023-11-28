import numpy as np
from flask import Flask, request, render_template
import pickle

# Create an app object using Flask
app = Flask(__name__, static_folder='static')

# Load the model
model = pickle.load(open('model/nb_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    features_to_float = [float(feature) for feature in request.form.values()]
    
    # Converts the features into numpy array to be fed to the model
    features = [np.array(features_to_float)]
    pred = model.predict(features)
    level_pred = pred[0]

    return render_template('index.html', prediction=level_pred)


if __name__ == '__main__':
    app.run(debug=True)
