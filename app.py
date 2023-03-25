from flask import Flask
from flask import request
import pickle
import logging
import sys
def predict(X_in, w, b):
    y = b + X_in[0]*w[0]+ X_in[1]*w[1] + X_in[2]*w[2] + X_in[3]*w[3]
    return y

print(__name__)
app = Flask(__name__)
api_mlparams=pickle.load(open("mlparams", 'rb'))

logging.info(api_mlparams)

@app.route('/infer')
def infer(inputs):
    W = api_mlparams.coef_
    b = api_mlparams.intercept_

    X_in=np.ndarray(shape=(1,4),buffer=np.array(inputs))
    print("W",W)
    print("b",b)
    print("X_in",X_in)
    return {'y':predict(X_in, W, b)}

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False)
