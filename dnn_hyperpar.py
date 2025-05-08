import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from sklearn.model_selection import GridSearchCV # to optimize DNN hyper-parameters
from scikeras.wrappers import KerasRegressor

import pandas as pd
import numpy as np
import os
import argparse

# Create argument parser
parser = argparse.ArgumentParser(description="Hyperparameter tuning for DNN")
# Add arguments
parser.add_argument("--npzfile",    type=str,               help="npz file path",                   default='/Users/saranabili/Desktop/jobHunts/chem/files/Xy.npz')
parser.add_argument("--HD",         type=int,   nargs="+",  help="tune DNN hidden units",           default=[1024])
parser.add_argument("--NHD",          type=int,   nargs="+",  help="tune number of hidden layers",  default=[17])
parser.add_argument("--epochs",     type=int,   nargs="+",  help="tune epochs",                     default=[50])
parser.add_argument("--BS",         type=int,   nargs="+",  help="tune batch size",                 default=[256])
# Parse the arguments
args = parser.parse_args()

# load X_train and y_train npz files:
X_train = np.load(args.npzfile)['X']
y_train = np.load(args.npzfile)['y']

# optimize
adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0003)
# 1. Define the model architecture function
def build_model(hidden_units=64, layer_num=18, activation='relu'):
    # initiate the model
    model = Sequential()
    model.add(Dense(hidden_units, activation=activation, input_shape=[X_train.shape[1]]))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Hidden layers (loop-based architecture)
    for _ in range(layer_num):
        model.add(Dense(hidden_units, activation=activation))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

    # Output layer
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer=adam, loss='mae')
    return model

# 2. Wrap the model using KerasRegressor
model = KerasRegressor(model=build_model, epochs=50, batch_size=256, verbose=0)

# 3. Define the correct hyperparameter grid
param_grid = {
    'model__hidden_units':  args.HD,  # hidden layers
    'model__layer_num':     args.NHD,  # Number of hidden layers
    'epochs':               args.epochs,
    'batch_size':           args.BS
}

# 4. Run GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# 5. Print best model results
print("Best Score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# 6. Save best model hyperparameters result to a txt file
with open(os.getcwd()+'/files/hypertune.txt') as f:
     print("Best Score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))