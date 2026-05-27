import csv
x,y=[],[]
try:
    with open(r'data.csv') as file:
        readrow=csv.reader(file)
        next(readrow)
        for row in readrow:
            x.append(float(row[0]))
            y.append(float(row[1]))
    x_mean=sum(x)/len(x)
    x_sd=((sum([(xi-x_mean)**2 for xi in x]))/len(x))**0.5
    x_norm=[(xi-x_mean)/x_sd for xi in x]
except FileNotFoundError:
    print("Enter values manually: ")
    x=list(map(float,input("Enter space separated area in square feet: ")))
    y=list(map(float,input("Enter space separated prices:")))
    x_mean=sum(x)/len(x)
    x_sd=((sum([(xi-x_mean)**2 for xi in x]))/len(x))**0.5
    x_norm=[(xi-x_mean)/x_sd for xi in x]
w=0.0
b=0.0
def derivative(x_norm,y,w,b):
    m=len(x_norm)
    dj_dw,dj_db=0.0,0.0
    for i in range(m):
        error=w*x_norm[i]+b-y[i]
        dj_dw+=error*x_norm[i]
        dj_db+=error
    dj_dw,dj_db=dj_dw/m,dj_db/m
    return dj_dw,dj_db
def update_value(x_norm,y,w,b,a,iteration):
    for i in range(iteration):
        dj_dw,dj_db=derivative(x_norm,y,w,b)
        w-=a*dj_dw
        b-=a*dj_db
    return w,b

a=0.01
w,b=update_value(x_norm,y,w,b,a,1000)
user_input=[]
prediction=[]
print("Enter the area in square feet: ")
while True:
    x_input=float(input())
    if (x_input==-1):
        break
    try:
        x_input_norm=(x_input-x_mean)/x_sd
        user_input.append(x_input)
        y_predicted=w*x_input_norm+b
        prediction.append(y_predicted)
    except:
        print("Enter proper value!!")
for i in range(len(user_input)):
    print(user_input[i]," -> ",prediction[i])