from pandas import *
from numpy import *
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

crop=read_csv("crop_recommendation.csv")

crop.rename(columns={'label':'Crop','N':'Nitrogen','P':'Phosporous','K':'Potassium'},inplace=True)
#print(crop.head())  #displaying dataset
#crop.describe()  #all statistical measures
#print(crop.isnull().sum()) #finding missing values
#crop.count() #count total number of values in each columns
crop['Crop'].replace('jute','groundnut',inplace=True)
crop['Crop'].replace('apple','drychillies',inplace=True)
crop['Crop'].replace('muskmelon','onion',inplace=True)
crop['Crop'].replace('watermelon','ragi',inplace=True)
crop['Crop'].replace('lentil','sunflower',inplace=True)
crop['Crop'].replace('blackgram','tobacco',inplace=True)
crop['Crop'].replace('mungbean','sugarcane',inplace=True)
crop['Crop'].replace('mothbeans','sweetpoatato',inplace=True)
crop['Crop'].replace('pigeonpeas','tumeric',inplace=True)
crop['Crop'].replace('kidneybeans','cashewnut',inplace=True)
crop['Crop'].replace('chickpea','bananan',inplace=True)
crop['Crop'].replace('coffee','tomato',inplace=True)
crop['Crop'].value_counts()
crop.dropna(inplace=True)  #removing empty values rows
#print(crop.isnull().sum()) #counting null values
#crop.isnull().values.any() #checking entire dataset for null values
#finding correlation using heatmap
sns.heatmap(crop.corr(), annot =True)
plt.title('Correlation Matrix')

from sklearn.preprocessing import LabelEncoder 
ob=LabelEncoder()
crop['new']=ob.fit_transform(crop['Crop'])
crop.head()
m=crop['new'].unique()#observing numerical data converted from categorical data
print(m.shape)

#sepearting dependent & independent variables
x=crop.iloc[:,0:7]
y=crop.loc[:,'new']
print(x.head())
print(y.head())
#crop.sort_index(axis=0)
#divided into train and test data
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.25,random_state=0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
#importing random forest
from sklearn.ensemble import RandomForestClassifier as Rf
rf=Rf()
rf.fit(x_train,y_train)

#predicting
y_pred=rf.predict(x_test)
print(y_pred.shape)


# In[13]:


#accuracy measure
from sklearn.metrics import accuracy_score 
print(accuracy_score(y_test,y_pred))


# In[23]:


"""v=[]
for i in range(7):
    v.append(float(input()))
m=array(v)
m=m.reshape(1,-1)
y_pred=rf.predict(m)
print(y_pred)"""


# In[15]:

import matplotlib.pyplot as plt
plt.scatter(y_pred,range(0,len(y_pred)),color='green')
plt.scatter(y_test,range(0,len(y_test)),color='red')
plt.title("Crop Type")
plt.xlabel('Crop Name')
plt.ylabel('Ingridients level')
plt.legend(['Actual','predicted'],loc="upper right")
plt.show()


# In[16]:


print(ob.classes_) #total avalibale lables


# In[17]:


crop['new'].sort_values().unique() #total new Labeled values


# In[18]:


df=DataFrame({'col1':['banana','bananan' ,'cashewnut' ,'coconut' ,'cotton', 'drychillies' ,'grapes',
 'groundnut', 'maize' ,'mango' ,'onion', 'orange', 'papaya', 'pigeonpeas',
 'pomegranate' ,'ragi', 'rice' ,'sugarcane' ,'sunflower' ,'sweetpoatato',
 'tobacco' ,'tomato']})
#print(df.to_string())


# In[19]:


#df['main']=ob.fit_transform(df['col1'])
#print(df.to_string())


# In[20]:


#print(crop['Crop'].value_counts())


# In[21]:


#print(crop.to_string())


# In[ ]:

pickle.dump(rf, open('crop_prediction.pkl', 'wb'))
crop_prediction = pickle.load(open('crop_prediction.pkl', 'rb'))
