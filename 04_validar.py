import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd


def function_testing(train, test, alpha, w, iteraciones=10):
    from math import isclose
    
    for i in range(iteraciones):
        #y=m+x_1*w_1+x_2*w_2+...+x_n*w_n
        a = (alpha+train.iloc[i].lat*w[0]+train.iloc[i].long*w[1] + train.iloc[i].depth*w[2]+train.iloc[i].stations*w[3])
        b = test.iloc[i]
        if isclose(a, b, abs_tol=0.4) == True:
            print('La predicción tiende al resultado original')
        else:
            print('La predicción difiere por más de 0.4 decimales')

ml_param_validate=pickle.load(open("mlparams", 'rb'))
df_quakes_test=pd.read_csv('test.csv')
X_test=df_quakes_test[['lat','long','depth','stations']]
y_test=df_quakes_test['mag']
reg_test=ml_param_validate.fit(X_test,y_test)

w = reg_test.coef_
a = reg_test.intercept_

function_testing(train=X_test, test=y_test, alpha=a, w=w, iteraciones=10)

R_2_test=reg_test.score(X_test,y_test)
print(R_2_test)
