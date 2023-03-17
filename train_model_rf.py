import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

features = []
labels = []
# loop over range 2015 to 2020
for year in range(2015, 2021):
    # Read features from csv
    # column names: 'AGE', 'SEX', 'ARREMS', 'TEMPF', 'PULSE', 'BPSYS', 'BPDIAS', 'RESPR','POPCT', 'RFV1', 'NOCHRON', 'TOTCHRON'
    X_tmp = pd.read_csv(f'features_{year}.csv')

    # Read labels from csv
    y_tmp = pd.read_csv(f'labels_{year}.csv')

    features.append(X_tmp)
    labels.append(y_tmp)

X = pd.concat(features)
y = pd.concat(labels)

# print dimensions of X and y
print(X.shape)
print(y.shape)


# Define preprocessing steps for numerical and categorical features
numerical_features = ['AGE', 'TEMPF', 'PULSE', 'BPSYS', 'BPDIAS', 'RESPR', 'POPCT', 'ARREMS']
categorical_features = ['SEX', 'RFV1', 'TOTCHRON']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train = preprocessor.fit_transform(X_train).toarray()
X_test = preprocessor.transform(X_test).toarray()

# Calculate the number of instances in each class
n_positive = np.sum(y_train)
n_negative = len(y_train) - n_positive

# Compute the class weights
n_negative = np.count_nonzero(y_train == 0)
n_positive = np.count_nonzero(y_train == 1)
class_weights = {
    0: 1 / n_negative,
    1: 1 / n_positive
}

# Reshape y_train into a 1D array
y_train = np.ravel(y_train)

# Instantiate the random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=500, random_state=42, class_weight=class_weights)

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the performance of the model
print(confusion_matrix(y_test, y_pred))
