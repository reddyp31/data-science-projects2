import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

dataset = pd.read_csv("CollegeDataset/Dataset.csv", usecols=['rank', 'gender', 'caste', 'region', 'branch', 'college'], nrows=2000)
dataset.fillna(0, inplace = True)
print(dataset)

encoder = []
columns = ['gender', 'caste', 'region','branch', 'college']

print(np.unique(dataset['gender'].tolist()))
print(np.unique(dataset['caste']).tolist())
print(np.unique(dataset['region']).tolist())
print(np.unique(dataset['branch']).tolist())

for i in range(len(columns)):
    le = LabelEncoder()
    dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))
    encoder.append(le)

dataset = dataset.values
X = dataset[:,0:dataset.shape[1]-1]
Y = dataset[:,dataset.shape[1]-1]

X = sc.fit_transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

print(X)
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1)

cls = RandomForestClassifier()
cls.fit(X_train, y_train)
predict = cls.predict(X_test) 
a = accuracy_score(y_test,predict)*100
print(a)

testData = [71654, 'F', 'BC_B', 'OU', 'PHARM - D (M.P.C. STREAM)']
temp = []
temp.append(testData)
temp = np.asarray(temp)
print(temp.shape)

df = pd.DataFrame(temp, columns=['rank', 'gender', 'caste', 'region', 'branch'])
for i in range(len(encoder)-1):
    df[columns[i]] = pd.Series(encoder[i].transform(df[columns[i]].astype(str)))
    
df = df.values
df = sc.transform(df)
predict = cls.predict(df)
print(predict)
print(encoder[4].inverse_transform(predict))





            

