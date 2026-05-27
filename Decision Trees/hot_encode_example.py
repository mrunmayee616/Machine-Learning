#using pandas
import pandas as pd

df=pd.DataFrame({
    'city':['Mumbai','Chennai','Delhi','Mumbai']
})
encoded=pd.get_dummies(df,drop_first=True)
print(encoded)

#using scikit learn
# from sklearn.preprocessing import OneHotEncoder
# encoder=OneHotEncoder(sparse_output=False)
# encoded=encoder.fit_transform(df[['city']])

# print(encoded)