import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

df=pd.read_csv('data_medical.csv')
x=df.iloc[:,:-1].astype(float).values
y=df.iloc[:,-1].astype(float).values

def build_poly_features(X):
    m, n = X.shape
    features = []
    for i in range(n):
        features.append(X[:, i])
    for i in range(n):
        features.append(X[:, i] ** 2)
    for i, j in combinations(range(n), 2):
        features.append(X[:, i] * X[:, j])
    return np.column_stack(features)

x_poly=build_poly_features(x)

def x_normalisation(x):
    x_mean=np.mean(x,axis=0)
    x_sd=np.std(x,axis=0)
    x_norm=(x-x_mean)/x_sd
    return x_norm,x_mean,x_sd

def y_normalisation(y):
    y_mean=np.mean(y)
    y_sd=np.std(y)
    y_norm=(y-y_mean)/y_sd
    return y_norm,y_mean,y_sd

x_norm,x_mean,x_sd=x_normalisation(x_poly)
y_norm,y_mean,y_sd=y_normalisation(y)

#find w(weight) and b(bias)
w=np.zeros(x_norm.shape[1])
b=0.0

def cost_function(x,y,w,b):
    error=np.dot(x,w)+b-y
    return (np.sum(error**2))/(2*x.shape[0])

def update_derivatives(x,y,w,b):
    m,n=x.shape
    dj_dw=np.zeros(n)
    dj_db=0.0
    for i in range(m):
        y_cap=np.dot(w,x[i])+b
        error=y_cap-y[i]
        for j in range(n):
            dj_dw[j]+=error*x[i][j]
        dj_db+=error
    dj_dw,dj_db=dj_dw/m,dj_db/m
    return dj_dw,dj_db

def update_values(x,y,w,b,a,iterations):
    cost_history=[]
    for i in range(iterations):
        dj_dw,dj_db=update_derivatives(x,y,w,b)
        w-=a*dj_dw
        b-=a*dj_db
        cost=cost_function(x,y,w,b)
        cost_history.append(cost)
    return w,b,cost_history

a=0.03;iterations=80000
w,b,cost_history=update_values(x_norm,y_norm,w,b,a,iterations)

def x_input_normalize(x_poly, mean, sd):
    return (x_poly - mean) / sd

def polynomial_transform_single(x_input):
    x1, x2, x3 = x_input
    return np.array([
        x1, x2, x3,      # linear
        x1**2, x2**2,x3**2,    # squared
        x1*x2, x1*x3, x2*x3  # interactions
    ])

#input
user_input=[]
prediction=[]
while True:
    user_in=input("carbs in take, exercise(hours) and sleep(hours) or -1 to exit: ")
    if(user_in=="-1"):
        break
    x_input=np.array(list(map(float,user_in.split())))
    x_poly_input=polynomial_transform_single(x_input)
    x_input_norm=x_input_normalize(x_poly_input,x_mean,x_sd)
    y_predict_norm=np.dot(x_input_norm,w)+b
    y_predict=y_predict_norm*y_sd+y_mean
    user_input.append(x_input)
    prediction.append(y_predict)

print(user_input)
print(prediction)

plt.plot(range(iterations),cost_history,'b-')
plt.show()