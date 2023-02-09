#machine learning model

#Data Preprocessing
#importing the libraries
import numpy as np
import pandas as pd
import pickle

#load the dataset
dataset = pd.read_csv('c_w_r.csv')

print(dataset.dtypes)
#handling missing data
#print(dataset.isnull().sum())

'''# converting 'Weight' from float to int
dataset['WATER_REQUIREMENT'] = dataset['WATER_REQUIREMENT'].astype(int)
#print(dataset.dtypes)'''

print(dataset.dtypes)

#label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
dataset['CROP_TYPE'] = LE.fit_transform(dataset.CROP_TYPE)
st = LE.classes_
print(sorted(st))
dataset['SOIL_TYPE'] = LE.fit_transform(dataset.SOIL_TYPE)
st = LE.classes_
print(sorted(st))
dataset['REGION'] = LE.fit_transform(dataset.REGION)
st = LE.classes_
print(sorted(st))
dataset['TEMPERATURE'] = LE.fit_transform(dataset.TEMPERATURE)
st = LE.classes_
print(sorted(st))
dataset['WEATHER_CONDITION'] = LE.fit_transform(dataset.WEATHER_CONDITION)
st = LE.classes_
print(sorted(st))


#seperate into independent and dependent variables
#print(dataset.columns)
x = dataset.iloc[:,:-1]
y = dataset.loc[:,'WATER_REQUIREMENT']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#importing random forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

#model training
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

#accuracy
print("model score:",model.score(x_test,y_test)*100)

#RMS Score method for accuracy
from sklearn.metrics import r2_score
r1 = r2_score(y_test,y_pred)
print("R2 score :",r1*100)

'''marker="*", s=100, alpha=0.6, 
marker="x", s=50, alpha=0.8, '''


import matplotlib.pyplot as plt
plt.scatter(y_test,range(0,len(y_test)), color = 'red', label="Actual" )
plt.scatter(y_pred,range(0,len(y_pred)), color = 'green' , label="Predicted")
plt.title("Irrigation Level")
plt.xlabel("fertilizers")
plt.ylabel("data points")
plt.legend(['Actual','predicted'],loc="upper right")
plt.show()


CROP_TYPE	SOIL_TYPE	REGION	TEMPERATURE	WEATHER_CONDITION	WATER_REQUIREMENT
input("Enter crop type : ",c)
input("Enter soil type : ", s)
input("Enter Region : ". r)
input("Enter Temperature : ", t)
input("Enter weather condition : ", w)
input("Enter water requirement : ", wr)

l=[0,0,0,0,0]
arr=np.array(l)
arr=arr.reshape(1,-1)
print(model.predict(arr))

pickle.dump(model, open('c_w_r.pkl', 'wb'))
c_w_r = pickle.load(open('c_w_r.pkl', 'rb'))
