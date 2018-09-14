# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 12:40:43 2018

@author: 003632
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers import Input, Dense
from keras.optimizers import SGD
import keras.optimizers

from sklearn.preprocessing import Imputer

import numpy as np
import pandas as pd

print("1")

ham = pd.read_csv("Churn_Modelling2.csv") 
veri = pd.read_csv("Churn_Modelling2.csv") 

print("2")

##veri.replace('?', -99999, inplace=True) 

#veri.drop(['id'], axis=1) 
##veriyeni = veri.drop(['1000025'],axis=1) 


##imp = Imputer(missing_values=-99999, strategy="mean",axis=0) 

##veriyeni = imp.fit_transform(veriyeni) 

X = veri.iloc[:,2:13].values
y = veri.iloc[:,13].values 
print("3")

# Importing the Keras libraries and package
# Sequential module - initialize neural network
# Dense - layers 

model = Sequential()
model.add(Dense(10,input_dim=11))

model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
print("4")

optimizer = keras.optimizers.SGD(lr=0.1)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_0 = LabelEncoder()
X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0])
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])

print("5")

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print("6")
"""
lr (learn rate) ile epochs gezinme sayısında ters orantı vardır
"""
model.compile(optimizer=optimizer, loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
print("7")
model.fit(X,y, epochs=100,batch_size=32,validation_split=0.20)
print("8")

# Save Model
model.save("models/ChurnAnalysis_Deep Learning.h5")
#tahmin = np.array([5,5,5,8,10,8,7,3]).reshape(1,8)
#print(model.predict_classes(tahmin))

# Load Model. Daha önce eğitilmiş olan model. Veriseti yada model parametreleri değişirse
model = load_model("models/ChurnAnalysis_Deep Learning.h5")

# Predicting 
y_pred = model.predict(X_test)
y_pred = y_pred.astype(int)
print(X_test)
print("9")
print(y_test)
print("10")
print(y_pred)
print("11")

#Saving predictions
np.savetxt("y_test.csv", y_test, delimiter="|")
np.savetxt("y_pred.csv", y_pred, delimiter="|")




