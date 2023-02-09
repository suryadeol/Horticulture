#machine learning model

#Data Preprocessing
#importing the libraries
import numpy as np
import pandas as pd

#load the dataset
dataset = pd.read_csv('crop_production_ap.csv')

print(dataset.dtypes)
#handling missing data
dataset.dropna(inplace=True)
dataset.drop(['Crop_Year'],axis=1, inplace = True)

#label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
dataset['State_Name'] = LE.fit_transform(dataset.State_Name)
sn = LE.classes_
print(sorted(sn))
dataset['District_Name'] = LE.fit_transform(dataset.District_Name)
dn = LE.classes_
print(sorted(dn))
dataset['Season'] = LE.fit_transform(dataset.Season)
s = LE.classes_
print(sorted(s))
dataset['Crop'] = LE.fit_transform(dataset.Crop)
c = LE.classes_
print(sorted(c))

#seperate into independent and dependent variables
#print(dataset.columns)
x = dataset.iloc[:,0:5]
y = dataset.loc[:,'Production']

#dividing into train and test data
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
print("R2 score : ",r1*100)

#data visualization
import matplotlib.pyplot as plt
plt.scatter(y_test,range(0,len(y_test)), color = "red")
#plt.show()

print("type(y_pred)",type(y_pred))
print("type(y_test)",type(y_test))

plt.scatter(y_pred,range(0,len(y_pred)), color = 'green')
plt.title("Crop Yeild ")
plt.xlabel('Hectares')
plt.ylabel('Tones')
plt.legend(['y_test','y_pred'])
plt.show()

#arr=np.array(l)
#arr=arr.reshape(1,-1)
#print("output:",model.predict(arr))

#creating or dumping model into pickle file
import pickle
pickle.dump(model, open('crop_yeild.pkl', 'wb'))
crop_yeild = pickle.load(open('crop_yeild.pkl', 'rb'))
