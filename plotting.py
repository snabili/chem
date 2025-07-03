# general modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns # --> heat maps
import pickle # --> to access dnn history
import joblib
import os

# DNN modules
import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.metrics import roc_curve, auc
import xgboost as xgb # --> feature importance
# Linear regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

import general as gen

path = os.path.join(os.getcwd(), 'files')
logger = gen.setup_logging(path + "/plotting.txt")
gen.set_matplotlib_fontsizes() # set up plt fontsize

# **********************************************************************************************************
# ********* load data and model ****************************************************************************
# **********************************************************************************************************

# load scaled features to make roc curves for DNN and LogisticRegression
data_dict = np.load(path + '/npz_datasplits.npz')
X_train, X_train_scaled, y_train = data_dict['X_train'], data_dict['X_train_scaled'],data_dict['y_train'] 
X_test,  X_test_scaled,  y_test  = data_dict['X_test'],  data_dict['X_test_scaled'], data_dict['y_test']
X_val,   X_val_scaled,   y_val   = data_dict['X_val'],   data_dict['X_val_scaled'],  data_dict['y_val']
y_true = y_test

df_splits = pd.read_csv(path + '/df_datasplit.csv')
Xpd_ebh_train_scaled = df_splits[df_splits['split'] == 'train_ebh_scaled'].drop(columns=['split','label'])
ypd_ebh_train        = df_splits[df_splits['split'] == 'train_ebh_label']['label']

print('*'*50, len(Xpd_ebh_train_scaled['band_gap']),len(ypd_ebh_train))

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
# Load history from pickle file
with open(path + '/dnn_history_allelements.pkl','rb') as f:
    history_dnn = pickle.load(f)
# Access loss, accuracy
logger.info(history_dnn.keys())  # Check available metrics
save_path = os.path.join(path, 'results', 'plots', 'classification')
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# ********* training and validation loss + accuracy plots *********
metrics=['loss', 'accuracy']
titles=['Loss', 'Accuracy']
ylabel_map={'loss': 'Loss', 'accuracy': 'Accuracy'}
n = len(metrics)
plt.figure(figsize=(6 * n, 4))
for i, metric in enumerate(metrics, 1):
    train_metric = history_dnn.get(metric)
    val_metric = history_dnn.get(f"val_{metric}")
    if train_metric is None or val_metric is None:
        continue
    plt.subplot(1, n, i)
    plt.plot(train_metric, label='Training')
    plt.plot(val_metric, label='Validation')
    plt.xlabel('Epochs')
    ylabel = ylabel_map.get(metric, metric.capitalize()) if ylabel_map else metric.capitalize()
    plt.ylabel(ylabel)
    title = 'DNN ' + ylabel_map[metric] + ' vs Epochs'
    plt.title(title)
    plt.legend()
    plt.tight_layout()
plt.savefig(save_path + '/dnn_trainVal.pdf')
plt.close()

# ********* plot ROC curves *********
# Load trained model
dnn_model = load_model(path + '/dnn_classification_allelements.h5')
LR_model = joblib.load(path + '/models/LogisticRegression_model.joblib')
RF_model = joblib.load(path + '/models/RandomForest_model.joblib')

# Get models
y_pred_proba_dnn = dnn_model.predict(X_test_scaled).flatten() # keras sequential 
y_pred_proba_LR = LR_model.predict_proba(X_test_scaled)[:, 1]
y_pred_proba_RF = RF_model.predict_proba(X_test)[:, 1]  # Prob for positive class
dnn_fpr, dnn_tpr, _ = roc_curve(y_true, y_pred_proba_dnn)
lr_fpr, lr_tpr, _ = roc_curve(y_true, y_pred_proba_LR)
rf_fpr, rf_tpr, _ = roc_curve(y_true, y_pred_proba_RF)
dnn_roc_auc = auc(dnn_fpr, dnn_tpr)
lr_roc_auc = auc(lr_fpr, lr_tpr)
rf_roc_auc = auc(rf_fpr, rf_tpr)

ML_path = path + '/results/ML_ROC'
plot_roc(dnn_fpr, dnn_tpr, dnn_roc_auc, "DNN", ML_path + '/DNN_ROC_allelements.pdf')
plot_roc(lr_fpr,  lr_tpr,  lr_roc_auc,  "LR",  ML_path + '/LR_ROC_allelements.pdf')
plot_roc(rf_fpr,  rf_tpr,  rf_roc_auc,  "RF",  ML_path + '/RF_ROC_allelements.pdf')


# ********* energy above the hull vs features *********
feature_list = list(Xpd_ebh_train_scaled.columns[:9]) + list(Xpd_ebh_train_scaled.columns[-4:])
for feat in feature_list:
    fig, ax = plt.subplots(figsize=(6, 5))  # Individual figure for each feature
    h = ax.hist2d(np.squeeze(Xpd_ebh_train_scaled[feat]),np.squeeze(ypd_ebh_train),bins=50,cmap='viridis',norm=LogNorm())
    plt.colorbar(h[3], ax=ax)

    ax.set_xlabel(feat,fontsize=16)
    ax.set_ylabel("EBH",fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    filename = save_path + f"/{feat}.pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)  # Free memory
    logger.info(f"Saved {filename}")


# ********* plot feature importance *********
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