#import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

#step2: create sample data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([35,40,48,55,60,68,72,80])

#step3: spliting test, train data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#step4: Train the mode
model = LinearRegression()
model.fit(X_train,y_train)

#step5: Make predictions
predictions=model.predict(X_test)
print("R2 score", r2_score(y_test, predictions))
print("MAE", mean_absolute_error(y_test,predictions))
print("Slope(m): ", model.coef_[0])
print("Intercept(c): ", model.intercept_)
print("Training Dataset: ",y_test)
print("Prediction: ",predictions)
