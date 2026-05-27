import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('data_marks.csv')
x=df.iloc[:,:-1].astype(float).values
y=df.iloc[:,-1].astype(float).values

def x_normalize(x):
    x_mean=np.mean(x,axis=0)
    x_sd=np.std(x,axis=0)
    x_norm=(x-x_mean)/x_sd
    return x_norm,x_mean,x_sd

x_norm,x_mean,x_sd=x_normalize(x)

w=np.zeros(x.shape[1])
b=0.0

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost_function(X, y, w, b):
    m = len(y)
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    cost = -(1/m) * np.sum(
        y*np.log(y_hat + 1e-10) +
        (1-y)*np.log(1-y_hat + 1e-10)
    )
    return cost

def update_derivatives(x,y,w,b):
    m,n=x.shape
    dj_dw=np.zeros(x.shape[1])
    dj_db=0.0
    for i in range(m):
        y_cap=sigmoid(np.dot(w,x[i])+b)
        error=y_cap-y[i]
        for j in range(n):
            dj_dw[j]+=error*x[i][j]
        dj_db+=error
    return dj_dw/m,dj_db/m

def update_values(x,y,w,b,a,iterations):
    cost_history=[]
    for i in range(iterations):
        dj_dw,dj_db=update_derivatives(x,y,w,b)
        w-=a*dj_dw
        b-=a*dj_db
        cost_history.append(cost_function(x,y,w,b))
    return w,b,cost_history

a=0.01;iterations=5000
w,b,cost_history=update_values(x_norm,y,w,b,a,iterations)

#inputs
user_input=[]
prediction=[]

def x_input_normalize(x):
    return (x-x_mean)/x_sd

while True:
    val=input("Enter TestScore(out of 100),cgpa, number of projects:")
    if(val=="-1"):
        break
    x_input=np.array(list(map(float,val.split())))
    x_input_norm=x_input_normalize(x_input)
    val_predicted=sigmoid(np.dot(w,x_input_norm)+b)
    user_input.append(x_input)
    if(val_predicted>=0.5):
        prediction.append(1)
    else:
        prediction.append(0)

print(user_input)
print(prediction)

#plotting graph
plt.figure(figsize=(6,4))

#original dataset
accepted = y == 1
rejected = y == 0

plt.scatter(x[accepted,0], x[accepted,1], 
            label="Admitted (original)", color="green", marker="o", facecolors='none')

plt.scatter(x[rejected,0], x[rejected,1], 
            label="Rejected (original)", color="red", marker="x")

# plotting user_input
user_input = np.array(user_input)  # convert list to array

if len(user_input) > 0:
    # We must plot the first 2 features only (CGPA, Test Score)
    plt.scatter(user_input[:,0], user_input[:,1],
                color="green", marker="D", s=80,
                label="User Inputs")

#decision boundary

cgpa_range = np.linspace(min(x[:,0]), max(x[:,0]))
test_range = np.linspace(min(x[:,1]), max(x[:,1]))

cgpa_grid, test_grid = np.meshgrid(cgpa_range, test_range)

cgpa_n = (cgpa_grid - x_mean[0]) / x_sd[0]
test_n = (test_grid - x_mean[1]) / x_sd[1]

Z = sigmoid(w[0]*cgpa_n + w[1]*test_n + b)

plt.contour(cgpa_grid, test_grid, Z, levels=[0.5], colors='blue')

plt.xlabel("Test Score")
plt.ylabel("CGPA")
plt.title("Decision Boundary with Original Data + User Inputs")
plt.legend()
plt.grid(True)
plt.show()
