import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

# Hours studied
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)#scikit learn expects input x as a 2-D array
y = np.array([35, 40, 50, 60, 65])# Marks obtained

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("MSE is: ",mean_squared_error(y_test,y_pred))
print(y_pred)

#convert it into 2-D array
x_input=3
y_pred=model.predict(np.array([[x_input]]))
print(y_pred)