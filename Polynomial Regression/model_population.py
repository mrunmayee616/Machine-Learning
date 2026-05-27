import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import csv

# x=[]
# y=[]
# with open(r'data_population.csv') as file:
#     reader=csv.reader(file)
#     next(file)
#     for row in reader:
#         x.append(row[0])
#         y.append(row[1])

df=pd.read_csv('data_population.csv')
x=df.iloc[:,0].astype(float).values
y=df.iloc[:,1].astype(float).values

def x_normalize(x):
    x_mean=np.mean(x)
    x_sd=np.std(x)
    x_norm=(x-x_mean)/x_sd
    return x_norm,x_mean,x_sd

def y_normalize(y):
    y_mean=np.mean(y)
    y_sd=np.std(y)
    y_norm=(y-y_mean)/y_sd
    return y_norm,y_mean,y_sd

x_norm,x_mean,x_sd=x_normalize(x)
y_norm,y_mean,y_sd=y_normalize(y)

#initialize
w=0.0 ; b=0.0
a=0.0001 ; iteration=100000

def cost_function(x,y,w,b):
    m=len(x)
    error=(w*x+b-y)**2
    return np.sum(error)/(2*m)

#update w,b
def update_derivative(x,y,w,b):
    m=len(x)
    dj_dw=0.0
    dj_db=0.0
    for i in range(m):
        error=(w*x[i]+b)-y[i]
        dj_dw+=error*x[i]
        dj_db+=error
    dj_dw,dj_db=dj_dw/m,dj_db/m
    return dj_dw,dj_db

def update_values(x,y,w,b,a,iterations):
    cost_history=[]
    for i in range(iterations):
        dj_dw,dj_db=update_derivative(x,y,w,b)
        w-=a*dj_dw
        b-=a*dj_db
        cost=cost_function(x,y,w,b)
        cost_history.append(cost)
    return w,b,cost_history

w,b,cost_history=update_values(x_norm,y_norm,w,b,a,iteration)

#input
user_input=[]
prediction=[]

while(True):
    year=float(input("Enter year (-1 to exit): "))
    if(year==-1):
        break
    x_input_norm=(year-x_mean)/x_sd
    y_predicted_norm=w*x_input_norm+b
    y_predicted=(y_predicted_norm*y_sd)+y_mean
    user_input.append(year)
    prediction.append(y_predicted)

print(user_input)
print(prediction)


#visualization
plt.scatter(x,y,c='blue',label='Actual Population') #original dataset

x_line=[min(x),max(x)]
x_line_norm=[(xi-x_mean)/x_sd for xi in x_line]
y_line = [w * xi + b for xi in x_line_norm]
plt.plot(x_line,y_line,'r-',label=f'Regression: {w:.2f}*x +{b:.2f}')

if user_input:
    plt.scatter(user_input,prediction,c='green',marker='s',s=100,label='Your Predictions')

plt.xlabel('Years')
plt.ylabel('Population')
plt.title('Population Prediction')
plt.legend()
plt.grid(True)

plt.show()

# plt.plot(range(iteration),cost_history,'b-')
# plt.show()