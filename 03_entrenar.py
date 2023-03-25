import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import f
import pickle

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

df_quakes=pd.read_csv('train.csv')
X_quakes=df_quakes[['lat','long','depth','stations']]
y_quakes=df_quakes['mag']
reg = LinearRegression().fit(X_quakes, y_quakes)
quakes_weights=reg.coef_
print(quakes_weights)
print(reg.intercept_) #y=m+bx

R_2=reg.score(X_quakes, y_quakes)
print(R_2)

function_testing(train=X_quakes, test=y_quakes, alpha=reg.intercept_, w=quakes_weights, iteraciones=10)

df_quakes_validate=pd.read_csv('validate.csv')
X_quakes_validate=df_quakes_validate[['lat','long','depth','stations']]
y_quakes_validate=df_quakes_validate['mag']

reg_validate = LinearRegression().fit(X_quakes_validate, y_quakes_validate)
reg_validate.score(X_quakes_validate, y_quakes_validate)
function_testing(train=X_quakes_validate, test=y_quakes_validate, alpha=reg_validate.intercept_, w=reg_validate.coef_, iteraciones=15)

for i in X_quakes_validate.columns:
    print(f'Correlación de la Magnitud respecto a {i}')
    print('Correlación Pearson: ', df_quakes_validate['mag'].corr(df_quakes_validate[i], method='pearson'))
    if df_quakes_validate['mag'].corr(df_quakes_validate[i], method='pearson')>0.8: print('Variable esta correlacionada')
    else: pass
    print('Correlación spearman: ', df_quakes_validate['mag'].corr(df_quakes_validate[i], method='spearman'))
    if df_quakes_validate['mag'].corr(df_quakes_validate[i], method='spearman')>0.8: print('Variable esta correlacionada')
    else: pass
    print('Correlación kendall: ', df_quakes_validate['mag'].corr(df_quakes_validate[i], method='kendall'))
    if df_quakes_validate['mag'].corr(df_quakes_validate[i], method='kendall')>0.8: print('Variable esta correlacionada')
    else: pass
    print('--------------------------')
    
    
n = len(df_quakes_validate)
k = len(df_quakes_validate.columns)
r2 = reg_validate.score(X_quakes_validate, y_quakes_validate)
a = 0.01
v2=(n-(k+1))


F_a = f.isf(a, k, v2)
F = (r2 / k) / ((1-r2)/v2)

if F>F_a == True: print('En zona de rechazo') 
else: print('No se rechaza')

pickle.dump(reg_validate,open("mlparams", 'wb') )
