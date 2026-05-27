import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.DataFrame({
    'age':[22, 25, 47, 52, 46, 56, 55],
    'income':[20000, 25000, 50000, 60000, 52000, 65000, 62000],
    'buy':[0, 0, 1, 1, 1, 1, 1]
})

X=df[['age','income']]
y=df['buy']

#split testing data and training data
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
#stratify guarantees both the cases appear in test as well as training data sets. 

#XGBoost
model = XGBClassifier(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.1,
    eval_metric='logloss'
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(x_test, "->", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))