import pandas as pd
import matplotlib.pyplot as plt
import pickle
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import iplot
import plotly.express as px

import itertools
from itertools import permutations

df3=pd.read_csv("crop_recommendation_updated.csv")

x = df3.iloc[:, :-2]
y = df3.iloc[:, [7,8]]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
'''plt.scatter(y_test,range(0,len(y_test)), color = 'red')
plt.show()'''

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df3['label']=LE.fit_transform(df3['label'])
c1 = LE.classes_
print(len(c1))
df3['crop']=LE.fit_transform(df3['crop'])
c2 = LE.classes_
print(len(c2))

#print(len(list(zip(c1,c2))))


from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
classifier = MultiOutputClassifier(knn)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)


p = y_pred[: , 0]
q = y_pred[: , 1]
r = y_test['label']
s = y_test['crop']


#print(q)
#print(type(q))
#print(p)
#print(type(p))
#print(y_pred)

'''print("y_pred.shape",y_pred.shape)
print("type(p_pred)",type(y_pred))
y_pred = y_pred.reshape(-1,1

print("y_pred.shape",y_pred.shape)
print("type(p_pred)",type(y_pred))'''


#scatter plot for project
import matplotlib.pyplot as plt

fig = plt.figure()
fig.set_size_inches(100,100)

'''y_test['label']=LE.fit_transform(y_test['label'])
y_test['crop']=LE.fit_transform(y_test['crop'])

y_pred[0]=LE.fit_transform(y_pred[0])
y_pred[1]=LE.fit_transform(y_pred[1])'''
#y_test.plot()

#plt.style.use('seaborn')
plt.scatter(p,r)
plt.scatter(q,s)
#plt.figure(figsize=(50, 45))
#plt.scatter(p,range(0,len(p)), color = 'red')
#plt.scatter(q,range(0,len(q)), color = 'red')
#plt.scatter(r,range(0,len(r)), color = 'green')
#plt.scatter(s,range(0,len(s)), color = 'green')
#plt.ylim([-2,50])
#plt.rcParams['figure.figsize'] = [50, 50]
plt.xticks(rotation = 90)

#plt.legend(['y_test','y_pred'])
plt.show()

'''g = go.Scatter(x=p,y=r,mode='markers',marker={'color':'magenta'})
layout = go.Layout(title='mixed crop prediction')
fig = go.Figure(data=g,layout=layout)
iplot(fig)
fig'''

#fig = px.scatter(p,r)
#fig.show(renderer=('vscode'))


from numpy import *
a=array([12,2,3,4.5,2,4,6])
a=a.reshape(1,-1)
print(classifier.predict(a))


import numpy as np
print( classifier.score(x_train,np.array(y_train)) * 100)


pickle.dump(classifier, open('crop_mixed.pkl', 'wb'))
crop_mixed = pickle.load(open('crop_mixed.pkl', 'rb'))

