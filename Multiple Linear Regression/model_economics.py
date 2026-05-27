import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import csv

# target=[]
# features=[]
# with open(r'data_economics.csv') as file:
#     readrow=csv.reader(file)
#     next(readrow)
#     for row in readrow:
#         features.append([float (row[i] for i in range(5))])
#         target.append(float(row[5]))
# x=np.array(features)
# y=np.array(target)

df=pd.read_csv('data_economics.csv')
X=df.iloc[:,:-1].astype(float).values
y=df.iloc[:,-1].astype(float).values

#normalize X,y
def normalize_X(X):
    X_mean=np.mean(X,axis=0) #for considering columns only
    X_sd=np.std(X,axis=0,ddof=1)
    X_norm=(X-X_mean)/X_sd
    return X_norm,X_mean,X_sd

def normalize_y(y):
    y_mean=np.mean(y)
    y_sd=np.std(y)
    y_norm=(y-y_mean)/y_sd
    return y_norm,y_mean,y_sd

X_norm,X_mean,X_sd=normalize_X(X)
y_norm,y_mean,y_sd=normalize_y(y)

#finding w(weight) and b(bias)
w=np.zeros(X.shape[1])
b=0.0
a=0.001; iterations=5000

def cost_function(X,y,w,b):
    m=len(y)
    cost=np.sum(((np.dot(X,w)+b)-y)**2)
    return cost/(2*m)

def update_derivatives(X,y,w,b):
    m,n=X.shape
    dj_dw=np.zeros(n)
    dj_db=0.0
    for i in range(m):
        y_cap=np.dot(X[i],w)+b
        error=y_cap-y[i]
        for j in range(n):
            dj_dw+=error*X[i][j]
        dj_db+=error
    return dj_dw/m,dj_db/m

def update_values(X,y,w,b,a,iterations):
    cost_history=[]
    for i in range(iterations):
        dj_dw,dj_db=update_derivatives(X,y,w,b)
        w-=a*dj_dw
        b-=a*dj_db
        cost=cost_function(X,y,w,b)
        cost_history.append(cost)
    return w,b,cost_history

w,b,cost_history=update_values(X_norm,y_norm,w,b,a,iterations)

#input
user_input=[]
predicted=[]

def x_input_normalize(x_input,X_mean,X_sd):
    return (x_input-X_mean)/X_sd


while True:
    x_input=input("Enter the values of area, number of rooms, age and location score space separated or -1 to exit: ")
    if(x_input=="-1"):
        break
    else:
        x_input=np.array(list(map(float,x_input.split())))
        x_input_norm=x_input_normalize(x_input,X_mean,X_sd)
        y_predicted_norm=np.dot(x_input_norm,w)+b
        y_predicted=(y_predicted_norm*y_sd)+y_mean
        user_input.append(x_input)
        predicted.append(y_predicted)

print(user_input)
print(predicted)

#checking the value of a / alpha (learning rate)
# plt.plot(range(iterations),cost_history,'b-')
# plt.grid(True)
# plt.xlabel('Iterations')
# plt.ylabel('Cost Function')
# plt.title('Learning Curve')
# plt.show()