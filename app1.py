# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:19:50 2020

@author: Admin
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
lm1 = pickle.load(open('lm1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = lm1.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The predicted house price is $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)