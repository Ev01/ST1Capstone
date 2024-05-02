from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

from ml_preprocess import generate_prediction


app = Flask(__name__)

@app.route('/prediction_api', methods=["GET"])
def prediction_api():
    try:
        # Getting the paramters from API call
        bedrooms = float(request.args.get('bedrooms'))
        bathrooms=float(request.args.get('bathrooms'))
        accommodates=float(request.args.get('accommodates'))

        # Calling the funtion to get predictions
        prediction_from_api = generate_prediction(
                                                    bedrooms=bedrooms,
                                                    bathrooms=bathrooms,
                                                    accommodates=accommodates
                                                )

        return prediction_from_api

    except Exception as e:
        return('Something is not right!:'+str(e))


if __name__ =="__main__":
    # Hosting the API in localhost
    app.run(host='127.0.0.1', port=9000, threaded=True, debug=True, use_reloader=False)
    # Interrupt kernel to stop the API
    '''
    Sample URL to call the API
    Copy and paste below URL in the web browser
    [3, 4, 2]
    http://127.0.0.1:9000/prediction_api?bedrooms=3&accommodates=4&bathrooms=2
    '''