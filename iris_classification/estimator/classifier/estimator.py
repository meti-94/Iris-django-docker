# Essentials
import pandas as pd 
import numpy as np 
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('Iris.csv')
X = df.iloc[:, 1:-1].to_numpy()
y = df['Species'].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Split it to training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Initialize the estimator and train it
model = RandomForestClassifier().fit(X_train, y_train)

# Test set score evaluation
score = model.score(X_test, y_test)
print("Score on test set: ", score)

# Save the model
filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))


