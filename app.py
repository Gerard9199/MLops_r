from flask import Flask
from flask import request
import pickle
import logging
import sys
def predict(x_1,x_2,x_3,x_4,w,b):
    y = b + x_1*w[0]+ x_2*w[1] + x_3*w[2] + x_4*w[3]
    return y

print(__name__)
app = Flask(__name__)
api_mlparams=pickle.load(open("mlparams", 'rb'))

logging.info(api_mlparams)

@app.route('/infer')
def infer():
    W = api_mlparams.coef_
    b = api_mlparams.intercept_

    reqX_1 = request.args.get('x_1')
    reqX_2 = request.args.get('x_2')
    reqX_3 = request.args.get('x_1')
    reqX_4 = request.args.get('x_2')
    x_1 = float(reqX_1)
    x_2 = float(reqX_2)
    x_3 = float(reqX_3)
    x_4 = float(reqX_4)
    
    return {'y':predict(x_1,x_2,x_3,x_4,W,b)}

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False)
