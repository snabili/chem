import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers

# Linear regression and random forest packages
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# General packages
import numpy as np
import pandas as pd
import os, re
from collections import OrderedDict, defaultdict
from flatten_dict import flatten
import pickle

# material project datasets

from scikeras.wrappers import KerasRegressor


def classify_stability(energy_above_hull, threshold=0.02):
    """
    Classify materials into 0 (stable) and 1 (unstable).
    
    Args:
        energy_above_hull (array-like): Energy above hull values (eV/atom).
        threshold (float): Stability threshold (eV/atom). Default is 0.05.
        
    Returns:
        np.ndarray: Array of 0 (stable) and 1 (unstable).
    """
    energy_above_hull = np.array(energy_above_hull)
    labels = np.where(energy_above_hull <= threshold, 0, 1)
    return labels

# read CSV with all features including energy_above_hull column
df = pd.read_csv("/Users/saranabili/Desktop/jobHunts/chem/files/final_material_data.csv")

# Apply the classification
df['stability_label'] = classify_stability(df['energy_above_hull'])

y = df['stability_label'].values
excluded = ['energy_above_hull','volume','nsites','is_stable','formula_pretty','total_magnetization','stability_label','composition','structure','material_id']
included = [c for c in df.columns if c not in excluded]
X = df[included].values

# save pandas dataframe to use for feature importance plot
df[included].to_csv('files/pd_feature.csv')


scaler = StandardScaler() # normalized features to mean = 0 and std = 1
X = scaler.fit_transform(X)

X_train, X_temp,y_train, y_temp = train_test_split(X,       y,      test_size=0.3, random_state=8911123)
X_val,   X_test,y_val,   y_test = train_test_split(X_temp,  y_temp, test_size=0.5, random_state=8911123)

# save test, train, val for plotting purposes:
data_dict = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'X_test': X_test,
    'y_val': y_val,
    'y_test': y_test
}
np.savez('files/data_splits.npz', **data_dict)

# read hyperparamters from dnn_hyperpar.py output file
neurons = 64
act='relu'
epochs = 100
batch = 128
layer_num = 4
# Optimizer with learning rate
adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0003)
loss = 'binary_crossentropy'


lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

# Define model
dnn_model = Sequential()

# Input layer
dnn_model.add(layers.Dense(neurons, activation=act, input_shape=[X.shape[1]]))
dnn_model.add(layers.Dropout(0.1))
dnn_model.add(layers.BatchNormalization())

# Hidden layers (using a loop)
for _ in range(layer_num):  # 4 hidden layers
    dnn_model.add(layers.Dense(neurons, activation=act))
    dnn_model.add(layers.Dropout(0.1))
    dnn_model.add(layers.BatchNormalization())

# Output layer
#dnn_model.add(layers.Dense(1,activation='relu'))
dnn_model.add(layers.Dense(1,activation='sigmoid'))

#Compile the model
dnn_model.compile(
    optimizer=adam,
    loss=loss,
    metrics=['accuracy']
)
print(dnn_model.summary())

early_stopping = keras.callbacks.EarlyStopping(
    patience = 5,
    min_delta = 0.001,
    restore_best_weights = True
)

from sklearn.utils import class_weight
# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

history_dnn = dnn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch,
    epochs=epochs,
    callbacks=[early_stopping, lr_scheduler],
    class_weight=class_weights
)

# Save dnn trained classification file
dnn_model.save("/Users/saranabili/Desktop/jobHunts/chem/files/dnn_classification.h5")

# Save history as a pickle file
with open("/Users/saranabili/Desktop/jobHunts/chem/files/dnn_history.pkl", "wb") as f:
    pickle.dump(history_dnn.history, f)
