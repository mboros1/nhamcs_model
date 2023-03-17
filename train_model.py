import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint


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

# Create the neural network
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation='sigmoid'))  # Use 'softmax' for multi-class classification

# Load the saved weights into the model
# model.load_weights('model_weights.h5')

# Calculate the number of instances in each class
n_positive = np.sum(y_train)
n_negative = len(y_train) - n_positive

# Compute the class weights
class_weights = {
    0: 1 / n_negative,
    1: 1 / n_positive
}

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Use 'categorical_crossentropy' for multi-class classification

# Set up the ModelCheckpoint callback
checkpoint_filepath = 'model_weights.h5'
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, save_best_only=True, monitor='val_accuracy', mode='min', verbose=1)

# Call the 'caffeinate' command to prevent the Mac from going to sleep
os.system('caffeinate i -s &')

# Train the model
model.fit(X_train, y_train, epochs=500, batch_size=64, validation_split=0.1, class_weight=class_weights, callbacks=[checkpoint_callback])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# Confusion matrix
y_pred = model.predict(X_test)

# 50% threshold
y_pred_50 = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred_50)
print("50% threshold")
print(cm)
print("")

# 75% threshold
y_pred_75 = (y_pred > 0.75)
cm = confusion_matrix(y_test, y_pred_75)
print("75% threshold")
print(cm)
print("")

# 90% threshold
y_pred_90 = (y_pred > 0.90)
cm = confusion_matrix(y_test, y_pred_90)
print("90% threshold")
print(cm)
print("")

# kill the caffeinate process
os.system('killall caffeinate')
