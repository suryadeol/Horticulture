#machine learning model

#Data Preprocessing
#importing the libraries
import numpy as np
import pandas as pd
import pickle

#load the dataset
dataset = pd.read_csv('Fertilizer_Prediction.csv')

#print(dataset.isnull().sum())
print(dataset.dtypes)
#label encoding 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
dataset['Soil_Type'] = LE.fit_transform(dataset.Soil_Type)
st = LE.classes_
print(sorted(st))
dataset['Crop_Type'] = LE.fit_transform(dataset.Crop_Type)
ct = LE.classes_
print(sorted(ct))
dataset['Fertilizer_Name'] = LE.fit_transform(dataset.Fertilizer_Name)
fn = LE.classes_
print(sorted(fn))

#divide into dependent and independent variables
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

#dividing into training and testing the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#model creation
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

#training the model
model.fit(x_train,y_train)


#testing the model
y_pred = model.predict(x_test)
print(y_pred)
print(type(y_pred))


#accuracy
print("model score:",model.score(x_test,y_test)*100)



#data visualization
import matplotlib.pyplot as plt
plt.scatter(y_test,range(0,len(y_test)),label="Actual")
plt.scatter(y_pred,range(0,len(y_pred)),label="Predicted")
plt.title("Fertilizer Type")
plt.xlabel('Fertilizer name.')
plt.ylabel('Ingridients level')
plt.legend(loc='upper left')
plt.show()



#plt.scatter(y_pred,range(0,len(y_pred)))
#plt.show()

pickle.dump(model, open('fertilizers.pkl', 'wb'))
fertilizers = pickle.load(open('fertilizers.pkl', 'rb'))



