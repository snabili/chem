# general modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # --> heat maps
import pickle # --> to access history data
import os
import joblib
import warnings

# DNN modules
import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.metrics import roc_curve, auc
import xgboost as xgb # --> feature importance
# Linear regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set logging level and format
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)
logger.setLevel(logging.ERROR) # supress prints

# **********************************************************************************************************
# ********* load data and model ****************************************************************************
# **********************************************************************************************************

# load scaled features to make roc curves for DNN and LogisticRegression
data_dict = np.load('files/data_splits.npz')
X_train, X_train_scaled, y_train = data_dict['X_train'], data_dict['X_train_scaled'],data_dict['y_train'] 
X_test,  X_test_scaled,  y_test  = data_dict['X_test'],  data_dict['X_test_scaled'], data_dict['y_test']
X_val,   X_val_scaled,   y_val   = data_dict['X_val'],   data_dict['X_val_scaled'],  data_dict['y_val']
y_true = y_test

# Load trained model
path_dir = os.getcwd()
dnn_model = load_model(os.path.join(path_dir, 'files','dnn_classification_allelements.h5'))
LR_model = joblib.load(os.path.joint(path_dir, 'files','models','LogisticRegression_model.joblib'))
RF_model = joblib.load(os.path.joint(path_dir, 'files','models','RandomForest_model.joblib'))
# Load history from pickle file
with open(os.path.join(path_dir, 'files','dnn_history_allelements.pkl'),'rb') as f:
    history_dnn = pickle.load(f)

# Access loss, accuracy
logger.info(history_dnn.keys())  # Check available metrics

def plot_roc(fpr, tpr, auc_val, label, filename):
    plt.figure()
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_val:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# **********************************************************************************************************
# ********* Plot DNN loss, accuracy + feature importance & correlation matrix ******************************
#***********************************************************************************************************

# Plot training and validation loss + accuracy
plt.plot(history_dnn['loss'], label='Training Loss')
plt.plot(history_dnn['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.tight_layout()
plt.savefig('files/results/plots/dnn/loss_trainVal_allelements.pdf')
plt.close()

plt.plot(history_dnn['accuracy'], label='Training Accuracy')
plt.plot(history_dnn['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.tight_layout()
plt.savefig('files/results/plots/dnn/accuracy_trainVal_allelements.pdf')
plt.close()

# plot ROC curves
# Get models
y_pred_proba_dnn = dnn_model.predict(X_test_scaled).flatten()
y_pred_proba_LR = LR_model.predict_proba(X_test_scaled)[:, 1]
y_pred_proba_RF = RF_model.predict_proba(X_test)[:, 1]  # Prob for positive class
dnn_fpr, dnn_tpr, _ = roc_curve(y_true, y_pred_proba_dnn)
lr_fpr, lr_tpr, _ = roc_curve(y_true, y_pred_proba_LR)
rf_fpr, rf_tpr, _ = roc_curve(y_true, y_pred_proba_RF)
dnn_roc_auc = auc(dnn_fpr, dnn_tpr)
lr_roc_auc = auc(lr_fpr, lr_tpr)
rf_roc_auc = auc(rf_fpr, rf_tpr)

plot_roc(dnn_fpr, dnn_tpr, dnn_roc_auc, "DNN", "files/results/ML_ROC/DNN_ROC_allelements.pdf")
plot_roc(lr_fpr,  lr_tpr,  lr_roc_auc,  "LR",  "files/results/ML_ROC/LR_ROC_allelements.pdf")
plot_roc(rf_fpr,  rf_tpr,  rf_roc_auc,  "RF",  "files/results/ML_ROC/RF_ROC_allelements.pdf")

# plot feature importance
df = pd.read_csv("files/pd_feature.csv")
feature_names = df.columns # Replace with actual names
xgbmodel = xgb.XGBClassifier()
xgbmodel.fit(X_train_scaled,y_train)
importance = xgbmodel.get_booster().get_score(importance_type='gain')
# Convert to DataFrame
importance_df = pd.DataFrame.from_dict(importance, orient='index', columns=['importance'])
logger.info('importance_df = ',importance_df)
importance_df.index.name = 'feature'
importance_df.reset_index(inplace=True)
# Sort values
importance_df.sort_values(by='importance', ascending=True, inplace=True)
importance_df['feature'] = importance_df['feature'].apply(lambda x: feature_names[int(x[1:])])
sns.barplot(
    x='feature', 
    y='importance', 
    data=importance_df.sort_values(by='importance', ascending=False),
    palette='coolwarm'
)
plt.xticks(rotation=90)
plt.axhline(y=0.01,linewidth=5)
plt.yscale('log')
plt.tight_layout()
plt.savefig('files/results/feature_importance_allelements.pdf')
plt.close()

# plot correlation matrix
corr_matrix = df.iloc[:, 1:14].corr() # restricting features to plot all but compositions
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": 0.75})
plt.title("Feature Correlation Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('files/results/correlation_matrix_allelements.pdf')
plt.close()


# ROC curves: LR, RF, DNN
plt.figure(figsize=(6, 5))
plt.plot(lr_fpr, lr_tpr, label=f"LogisticRegression (AUC = {lr_roc_auc:.2f})")
plt.plot(rf_fpr, rf_tpr, label=f"RandomForest (AUC = {rf_roc_auc:.2f})")
plt.plot(dnn_fpr, dnn_tpr, label=f"DNN (AUC = {dnn_roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: LR, RF, DNN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('files/results/ML_ROC/AllML_ROC_allelements.pdf')