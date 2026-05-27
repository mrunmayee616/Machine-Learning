import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.DataFrame({
    'Years':[1,2,3,4,5],
    'Salary':[30000,35000,40000,45000,50000]
})
x=data['Years'].values
y=data['Salary'].values
x_mean=np.mean(x)
x_sd=np.std(x)
x_norm=[(xi-x_mean)/x_sd for xi in x]
w=0.0
b=0.0
a=0.01
def derivative(x_norm,y,w,b):
    m=len(x_norm)
    dj_dw=0.0
    dj_db=0.0
    for i in range(m):
        dj_dw+=(w*x_norm[i]+b-y[i])*x_norm[i]
        dj_db+=(w*x_norm[i]+b-y[i])
    dj_dw,dj_db=dj_dw/m,dj_db/m
    return dj_dw,dj_db
def update_values(x_norm,y,w,b,a,iterations):
    for i in range(iterations):
        dj_dw,dj_db=derivative(x_norm,y,w,b)
        w-=a*dj_dw
        b-=a*dj_db
    return w,b
    
w,b=update_values(x_norm,y,w,b,a,1000)

def cost(x_norm,y,w,b):
    m=len(x_norm)
    cost_val=0.0
    for i in range(m):
        cost_val+=(w*x_norm[i]+b-y[i])**2
    cost_val=cost_val/(2*m)
    return cost_val

user_input=[]
prediction=[]
while True:
    x_input=float(input("Enter years (-1 to exit): "))
    if(x_input==-1):
        break
    try:
        x_input_norm=(x_input-x_mean)/x_sd
        y_predicted=w*x_input_norm+b
        user_input.append(x_input)
        prediction.append(y_predicted)
    except:
        print("Enter a valid value")
for i in range(len(user_input)):
    print(user_input[i]," -> ",prediction[i])

#graph plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

#Left subplot: Line + user predictions 
ax1.scatter(x, y, c='blue', label='Actual Salary')
x_line = np.linspace(min(x), max(x), 100)
x_line_norm = (x_line - x_mean)/x_sd
y_line = w*x_line_norm + b
ax1.plot(x_line, y_line, 'r-', label='Regression Line')
if user_input:
    ax1.scatter(user_input, prediction, c='green', marker='s', s=80, label='Your Prediction')
ax1.set_xlabel('Years of Experience')
ax1.set_ylabel('Salary')
ax1.set_title('Salary Prediction')
ax1.legend()
ax1.grid(True)

#Right subplot: Contour plot of cost function 
W = np.linspace(-20000, 20000, 200)
B = np.linspace(0, 60000, 200)
W_grid, B_grid = np.meshgrid(W, B)
Z = np.zeros_like(W_grid)

for i in range(W_grid.shape[0]):
    for j in range(W_grid.shape[1]):
        Z[i, j] = cost(x_norm, y, W_grid[i, j], B_grid[i, j])

cp = ax2.contour(W_grid, B_grid, Z, levels=20)
ax2.clabel(cp)
ax2.set_title("Contour Plot of Cost Function J(w,b)")
ax2.set_xlabel("Weight (w)")
ax2.set_ylabel("Bias (b)")

plt.tight_layout()
plt.show()
