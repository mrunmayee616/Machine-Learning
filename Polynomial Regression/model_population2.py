import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import csv

# x=[]
# y=[]
# with open(r'data_population.csv') as file:
#     reader=csv.reader(file)
#     next(reader)
#     for row in reader:
#         x.append(row[0])
#         y.append(row[1])

df=pd.read_csv('data_population.csv')
x=df.iloc[:,0].astype(float).values
y=df.iloc[:,1].astype(float).values

# m=len(x)
center_year=np.mean(x)
x_centered=x-center_year

y_log=np.log(y)

#building polynomial features
def build_polynomial(x,degree):
    return np.vstack([x**i for i in range(1,degree+1)]).T
degree=2
x_poly=build_polynomial(x_centered,degree)

#normalize x and y
def normalize_x(x):
    x_mean=np.mean(x,axis=0)
    x_sd=np.std(x,axis=0)
    x_norm=(x-x_mean)/x_sd
    return x_norm,x_mean,x_sd

def normalize_y(y):
    y_mean=np.mean(y)
    y_sd=np.std(y)
    y_norm=(y-y_mean)/y_sd
    return y_norm,y_mean,y_sd

x_norm,x_mean,x_sd=normalize_x(x_poly)
y_norm,y_mean,y_sd=normalize_y(y_log)

#finding w(weights) and b(bias)
w=np.zeros(degree)
b=0.0

def update_derivative(x,y,w,b):
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
    for i in range(iterations):
        dj_dw,dj_db=update_derivative(x,y,w,b)
        w-=a*dj_dw
        b-=a*dj_db
    return w,b

a=0.01;iterations=20000
w,b=update_values(x_norm,y_norm,w,b,a,iterations)

#inputs
user_input=[]
prediction=[]
while True:
    x_input=float(input("Enter year (-1 to exit): "))
    if(x_input==-1):
        break
    x_input_centered=x_input-center_year
    x_input_year=np.array([x_input_centered**i for i in range(1,degree+1)])
    x_input_norm=(x_input_year-x_mean)/x_sd
    y_prediction_norm=np.dot(x_input_norm,w)+b
    y_pred = y_prediction_norm * y_sd + y_mean
    y_pred=np.exp(y_pred)
    user_input.append(x_input)
    prediction.append(y_pred)

print(user_input)
print(prediction)

#visualization
x_line = np.linspace(x.min(), x.max()+5, 200)
x_line_centered = x_line - center_year
x_line_poly = build_polynomial(x_line_centered, degree)

x_line_norm = (x_line_poly - x_mean) / x_sd

y_line_norm = np.dot(x_line_norm, w) + b
y_line_log = y_line_norm * y_sd + y_mean

y_line = np.exp(y_line_log)

plt.scatter(x, y, label="Data")
plt.plot(x_line, y_line, 'r-', label=f"Polynomial Fit (degree {degree})")
plt.scatter(user_input,prediction,c='green',marker='s',label='Your predicted value')
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.show()