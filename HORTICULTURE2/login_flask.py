import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings

df = pd.read_csv("login_details.csv")
#df = np.array(df)
'''print(df)
enc = LabelEncoder()
df['USER_NAME'] = enc.fit_transform(df.USER_NAME)
print(df)
df.to_csv("index_updated.csv")
#df = np.array(df)'''
x = df.iloc[1:, 1:-1].values
y = df.iloc[1:, -1].values
X_train, Y_train, X_test, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
Y_train = sc.fit_transform(Y_train)
log = LogisticRegression()
log.fit(X_train, Y_train)
pickle.dump(log, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))