import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data collection and analysis"""

#loading the diabetes dataset into the pandas Dataframe

diabetes_dataset = pd.read_csv('/Users/vanshsaxena/Documents/Machine Learning Models/Diabetes Prediction/Data/diabetes.csv')


diabetes_dataset['Outcome'].value_counts()

"""
0 --> No diabetes
1 --> Diabetes
"""

diabetes_dataset.groupby('Outcome').mean()

#separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

"""DATA STANDARDIZATION"""

scaler = StandardScaler()

scaler.fit(X.values)
standardized_data = scaler.transform(X.values)

X = standardized_data
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

"""TRAINING THE MODEL"""

classifier = svm.SVC(kernel = 'linear')

#training the Support Vector Machine Classifier
classifier.fit(X_train, Y_train)

"""MODEL EVALUATION"""

#accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('accuracy on trained data:',training_data_accuracy)

#accuracy score on the testing data
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('accuracy on testing data: ', testing_data_accuracy)

"""MAKING A PREDICTIVE SYSTEM"""

user_input = input("Enter values as comma-separated (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age): ")

input_data = tuple(map(float, user_input.split(",")))

#changing the input data into NumpyArray
input_data_as_numpyarray = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpyarray.reshape(1,-1)

#standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
