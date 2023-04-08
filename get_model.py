import json
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import numpy as np
import math

model = pickle.load(open('model2.pkl', 'rb'))


def getAge():

    data1 = json["data1"]
    input = (data1, 2, 0, 6, 6, 1, 1, 70, 0, 0,0, 11, 0, 1)  # row 1 data values

    input_data_arr = np.asarray(input)
    print(input_data_arr)

    input_data_reshaped = input_data_arr.reshape(1, -1)
    print(input_data_reshaped)
    prediction = model.predict(input_data_reshaped)
    ans = math.floor(prediction)


    return ans


getAge()
