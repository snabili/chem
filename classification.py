# DNN modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers

# data preparation for MLs
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris


# Logistic regression and random forest packages
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# General modules
import numpy as np
import pandas as pd
import os, re
from collections import OrderedDict, defaultdict
import pickle
import joblib


import general as gen
path = os.path.join(os.getcwd() , 'files')
logger = gen.setup_logging(path + "/classification.txt")


# load split data
data_dict = np.load(path + '/npz_datasplits.npz', allow_pickle=True)
X_train,    X_train_scaled, y_train = data_dict['X_train'], data_dict['X_train_scaled'],data_dict['y_train']
X_val,      X_val_scaled,   y_val   = data_dict['X_val'],   data_dict['X_val_scaled'],  data_dict['y_val']
X_test,     X_test_scaled,  y_test  = data_dict['X_test'],  data_dict['X_test_scaled'], data_dict['y_test']

# *************************************************************************************
# *********************** Extract Hyperparameter values from **************************
# ***********************    the out of classification_hyperpars.py *******************
# *************************************************************************************

def extract_best_scores(filename):
    pattern = r"Best Score:\s+([0-9.]+)\s+using\s+(\{.*?\})"
    results = []
    with open(filename, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                score = float(match.group(1))
                params = eval(match.group(2))  # use `ast.literal_eval()` if safety is a concern
                results.append((score, params))
    return results

ML_dir = path + '/MLHypertune_pars/'
dnn_txtfile =  ML_dir + 'DNN_hypertune.txt'
rf_txtfile  =  ML_dir + 'RF_hypertune.txt'
lr_txtfile  =  ML_dir + 'LR_hypertune.txt'

dnn_entries = extract_best_scores(dnn_txtfile)
rf_entries  = extract_best_scores(rf_txtfile)
lr_entries  = extract_best_scores(lr_txtfile)

Params = defaultdict()
ml_entries = {'DNN':dnn_entries,'RF':rf_entries,'LR':lr_entries}
for key,entry in ml_entries.items():
    Score=[]
    Params[key] = {}
    for i, (score, params) in enumerate(entry):
        Score.append(score)
        Params[key][score] = params


# *************************************************************************************
# *********************** Deep Neural Net *********************************************
# *************************************************************************************

# read hyperparamters from classification_hyperpars.py output file
dnn_params = Params['DNN'][max(Params['DNN'])]
neurons    = dnn_params['model__hidden_units']
epochs     = dnn_params['epochs']
batch      = dnn_params['batch_size']
layer_num  = dnn_params['model__layer_num']

act='relu'
# Optimizer with learning rate
adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0003)
loss = 'binary_crossentropy'

# learning rate scheduler; lowers the learning rate when the validation loss plateaus
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

# Define model
DNN_model = Sequential()
# Input layer
DNN_model.add(layers.Dense(neurons, activation=act, input_shape=[X_train_scaled.shape[1]]))
DNN_model.add(layers.Dropout(0.1))
DNN_model.add(layers.BatchNormalization())

# Hidden layers (using a loop)
for _ in range(layer_num):  # 2 hidden layers
    DNN_model.add(layers.Dense(neurons, activation=act))
    DNN_model.add(layers.Dropout(0.3)) # typical dropout for moderate to strong regularization
    DNN_model.add(layers.BatchNormalization())

# Output layer
DNN_model.add(layers.Dense(1,activation='sigmoid'))

#Compile the model
DNN_model.compile(
    optimizer=adam,
    loss=loss,
    metrics=['accuracy']
)
logger.info(DNN_model.summary())

# to avoid overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience = 5,
    min_delta = 0.001,
    restore_best_weights = True
)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

history_dnn = DNN_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    batch_size=batch,
    epochs=epochs,
    callbacks=[early_stopping, lr_scheduler],
    #callbacks=[lr_scheduler],
    class_weight=class_weights
)

data = load_iris()
y_DNN_pred = DNN_model.predict(X_test_scaled)
labels = np.unique(y_test)

final_val = defaultdict(list)
for s in ['val_loss','loss','val_accuracy','accuracy']:
    final_val[s].append(history_dnn.history[s][-1])
logger.info({k: f"{v[0]:.4f}" for k, v in final_val.items()})

# Save dnn trained classification + history files
DNN_model.save(path + '/dnn_classification_allelements.h5')
with open(path + "/dnn_history_allelements.pkl", "wb") as f:
    pickle.dump(history_dnn.history, f)

# *****************************************************************************************
# *********************** Logistic Regression *********************************************
# *****************************************************************************************

# read hyperparamters from classification_hyperpars.py output file
lr_params = Params['LR'][max(Params['LR'])]
C = lr_params['C']
max_iter = lr_params['max_iter']
penalty = lr_params['penalty']

LR_model = LogisticRegression(C=C, penalty=penalty, solver='saga', max_iter=max_iter, class_weight='balanced') # hypertuned
LR_model.fit(X_train_scaled, y_train)
y_LR_pred = LR_model.predict(X_test_scaled)
labels = np.unique(y_test)
logger.info(classification_report(y_test, y_LR_pred, target_names=data.target_names[labels]))
joblib.dump(LR_model, path + "/LogisticRegression_model.joblib")

# *************************************************************************************
# *********************** Random Forest ***********************************************
# *************************************************************************************
# read hyperparamters from classification_hyperpars.py output file
rf_params = Params['RF'][max(Params['RF'])]
n_estimator       = rf_params['calibrated_rf__base_estimator__n_estimators']
min_samples_split = rf_params['calibrated_rf__base_estimator__min_samples_split']
max_depth         = rf_params['calibrated_rf__base_estimator__max_depth']

RF_model = RandomForestClassifier(class_weight='balanced', n_estimators=n_estimator, min_samples_split=min_samples_split, max_depth=max_depth, random_state=42 + 2)
RF_model.fit(X_train, y_train) # Random Forests is scale-invariant --> normalization doesn’t help
#data = load_iris()
y_RF_pred = RF_model.predict(X_test)
labels = np.unique(y_test)
logger.info(classification_report(y_test, y_RF_pred, target_names=data.target_names[labels]))
joblib.dump(RF_model, path + "/RandomForest_model.joblib")






