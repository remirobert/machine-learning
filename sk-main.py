import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



def train_data(arrays, y):
    cv = model_selection.ShuffleSplit()
    train, test = next(cv.split(X=arrays[0], y=y))
    return list(chain.from_iterable((safe_indexing(a, train),
                                     safe_indexing(a, test)) for a in arrays))


train = pd.read_csv("train.csv")
features = train.columns[1:]
X = train[features]
y = train['label']

user_train = pd.read_csv("input.csv")
user_features = user_train.columns[1:]
user_X = user_train[user_features]
user_y = user_train['label']

print(type(X))
print(X.size)

X_train, X_test = model_selection.train_test_split(X/255.,y,test_size=20000,random_state=0)
y_train, y_test = model_selection.train_test_split(user_X/255.,user_y,test_size=user_X.size,random_state=0)

print(y_test)

print(y_train)

print("***************************************************************")

print(user_y)

print("===============================================================")

print(type(y_train))
print(y_train)

print("...............................................................")
print(y_test)

clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("pred : ", y_pred_rf)
print("random forest accuracy: ",acc_rf)