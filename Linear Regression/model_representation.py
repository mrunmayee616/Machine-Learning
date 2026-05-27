import pandas as pd
import matplotlib.pyplot as plt
import csv
#plt.style.use('./deeplearning.mplstyle') #it is used to provide style to the graphs
#initialize an empty list
x,y=[],[]
try:
    with open(r'data.csv') as file:
        readrow=csv.reader(file)
        next(readrow)
        for row in readrow:
            x.append(float(row[0]))
            y.append(float(row[1]))

    # before normalizing 
    # plt.scatter(x,y)
    # after normalizing square footage
    #square footage is normalized as we want price in real - life unit
    x_mean=sum(x)/len(x)
    x_sd=(sum([(xi-x_mean)**2 for xi in x])/len(x))**0.5
    x_norm=[(xi-x_mean)/x_sd for xi in x]
    # plt.scatter(x_norm,y)
except FileNotFoundError:
    print("Enter data manually: ")
    x=list(map(float,input("Enter space separated area in square feet: ").split()))
    y=list(map(float,input("Enter space separated price: ").split()))
w=0.0
b=0.0
def cost_function(x_norm,y,w,b):
    m=len(x_norm)
    total_error=0
    for i in range(m):
        total_error+=((w*x_norm[i]+b-y[i])**2)
    total_error=total_error/(2*m)
    return total_error

def derivative(x_norm,y,w,b):
    m=len(x_norm)
    dJ_dw,dJ_db=0.0,0.0
    for i in range(m):
        error=(w*x_norm[i]+b-y[i])
        dJ_dw+=error*x_norm[i]
        dJ_db+=error
    dJ_dw,dJ_db=dJ_dw/m,dJ_db/m
    return dJ_dw,dJ_db

def update_value(x_norm,y,w,b,a,iterations):
    for i in range(iterations):
        dj_dw,dj_db=derivative(x_norm,y,w,b)
        w-=a*dj_dw
        b-=a*dj_db
    return w,b

a=0.01
w,b=update_value(x_norm,y,w,b,a,1000)
user_input=[]
predictions=[]
print("Enter square feet -> price to be determined: ")
while True:
    x_input=float(input())
    if(x_input==-1):
        break
    try:
        x_input_norm=(x_input-x_mean)/x_sd
        user_input.append(x_input)
        y_predicted=w*x_input_norm+b
        predictions.append(y_predicted)
    except:
        print("Enter proper value")
# Plot original dataset
plt.scatter(x, y, c = 'blue', label = 'Actual Prices')

# Plot regression line
x_line = [min(x), max(x)]
x_line_norm = [(xi - x_mean) / x_sd for xi in x_line]
y_line = [w * xi + b for xi in x_line_norm]
plt.plot(x_line, y_line, 'r-', label = f'Regression: ${w:.2f}*x + ${b:.2f}')
# Plot user predictions (if any)
if user_input:
    plt.scatter(user_input, predictions, c = 'green', marker = 's', s = 100, label = 'Your Predictions')

# Formatting
plt.xlabel('Square Footage')
plt.ylabel('Price ($)')
plt.title('House Price Prediction')
plt.legend()
plt.grid(True)
plt.show()