import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# features=[]
# target=[]
# with open(r'data_house.csv') as file:
#     readrow=csv.reader(file) 
#     next(readrow)
#     for row in readrow:
#         features.append([float(row[0]),float(row[1]),float(row[2]),float(row[3])])
#         target.append(float(row[4]))
# x=np.array(features)
# y=np.array(target)

df=pd.read_csv('data_house.csv')
X=df.iloc[:,:-1].astype(float).values
y=df.iloc[:,-1].astype(float).values

#normalize x,y 
#usually there is no need to normalize y, but in some cases the high value can cause an issue while finding gradient descent
def normalize_X(x):
    x_mean=np.mean(x,axis=0) # axis=0 for considering data column-wise
    x_sd=np.std(x,axis=0,ddof=1)#ddof (degrees of freedom) - 
    x_norm=(x-x_mean)/x_sd
    return x_norm,x_mean,x_sd

def normalize_y(y):
    y_mean=np.mean(y)
    y_sd=np.std(y)
    y_norm=(y-y_mean)/y_sd
    return y_norm,y_sd,y_mean

X_norm,X_mean,X_sd=normalize_X(X)
y_norm,y_sd,y_mean=normalize_y(y)

#finding w(weight) and b(bias)
w=np.zeros(X.shape[1]) #indicates we are considering columns (features) and not samples
b=0.0

def cost_function(X,y,w,b):
    m=len(y)
    y_cap=np.dot(X,w)+b
    j_function=np.sum((y_cap-y)**2)
    return j_function/(2*m)

def update_derivative(X,y,w,b):
    m,n=X.shape
    dj_dw=np.zeros(n)
    dj_db=0.0
    for i in range(m):
        y_cap=np.dot(X[i],w)+b
        error=y_cap-y[i]
        for j in range(n):
            dj_dw[j]+=error*X[i][j]
        dj_db+=error
    return dj_dw/m,dj_db/m

def gradient_descent(X,y,w,b,a,iterations):
    cost_history=[]
    for i in range(iterations):
        dj_dw,dj_db=update_derivative(X,y,w,b)
        w-=a*dj_dw
        b-=a*dj_db
        cost=cost_function(X,y,w,b)
        cost_history.append(cost)
    return w,b,cost_history

a=0.0001
iterations=10000
w,b,cost_hist=gradient_descent(X_norm,y_norm,w,b,a,iterations)

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

#plot b/w iterations and cost 
#To determine better learning rate (alpha / a(in this case))
plt.plot(range(iterations),cost_hist,'b-')
plt.xlabel('Iteration')
plt.ylabel('Cost J(w, b)')
plt.title('Training Curve')
plt.grid(True)
plt.show()

    