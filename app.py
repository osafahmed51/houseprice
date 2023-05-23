#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

pickle_in = open('model15.pkl', 'rb')
classifier = pickle.load(pickle_in)

def predict_price(location, sqft, bath, bhk):
    x = np.zeros(244)
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    return classifier.predict([x])[0]

@app.route('/predict', methods=['POST'])
def predict():
    location = request.json['location']
    sqft = float(request.json['sqft'])
    bath = float(request.json['bath'])
    bhk = float(request.json['bhk'])

    result = predict_price(location, sqft, bath, bhk)
    response = {'predicted_price': result}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False)





