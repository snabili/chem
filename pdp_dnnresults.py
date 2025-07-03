import numpy as np
import pandas as pd
import os
import joblib 
import matplotlib.pyplot as plt
from datetime import datetime

from tensorflow.keras.models import load_model
from scikeras.wrappers import KerasClassifier # keras optimizers are not picklable --> it can't be sent to subprocesses using joblib's backend

from sklearn.inspection import partial_dependence

import general as gen

path = os.path.join(os.getcwd(), 'files')
logger = gen.setup_logging(path + "/plotting.txt")
gen.set_matplotlib_fontsizes() # set up plt fontsize

# define the base for binary classification
def classify_stability(energy_above_hull, threshold=0.05):
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

def compute_pdp(feat):
    return feat, partial_dependence(
        wrapped_model,
        Xpd_train_scaled,
        features=[feat],
        kind='average',
        grid_resolution=100,
    )

# Load dnn model and datasplit CSV
dnn_model = load_model(path + '/dnn_classification_allelements.h5')
df_splits = pd.read_csv("files/df_datasplit.csv")

# load scaled trained features and labels
Xpd_train_scaled = df_splits[df_splits['split'] == 'train_scaled'].drop(columns=['split','label'])
ypd_train = df_splits[df_splits['split'] == 'train_label']['label']


logger.info(f"start wrapping model with keras:  {datetime.now().time().strftime('%H:%M:%S')}")
wrapped_model = KerasClassifier(model=dnn_model, epochs=50, batch_size=32)
wrapped_model.fit(Xpd_train_scaled, ypd_train)

logger.info(f"start partial dependence:  {datetime.now().time().strftime('%H:%M:%S')}")

feature_list = list(Xpd_train_scaled.columns[:9]) + list(Xpd_train_scaled.columns[-4:])
pdp_results_average = {}
for feat in feature_list:
    feat_name, result = compute_pdp(feat)
    pdp_results_average[feat_name] = result
logger.info(f"done partial dependence:  {datetime.now().time().strftime('%H:%M:%S')}")

# Save to disk
joblib.dump(pdp_results_average, path + '/pdp_results_average.pkl')  
np.savez(path + '/pdp_results_average.npz', **pdp_results_average) # use np.savez for numpy format

# ****************************************** Plots: pdp vs feat + feat hist ***********************************
data = np.load(path + '/pdp_results_average.npz', allow_pickle=True)
feature_list = data.files
#data = pdp_results_average
# Create subplots
num_features = len(data)
rows = int(np.ceil(num_features / 5))
fig, axes = plt.subplots(nrows=rows, ncols=5, figsize=(12, 2 * rows))
axes = axes.flatten()

if len(feature_list) == 1:
    axes = [axes]  # Ensure iterable
for i, feature_key in enumerate(data.keys()):
    logger.info(feature_key)
    pdp_entry = data[feature_key].item()
    x_vals = np.squeeze(pdp_entry['grid_values'])
    y_vals = np.squeeze(pdp_entry['average'])

    ax1 = axes[i]
    ax1.plot(x_vals, y_vals, color='blue', label='PDP')
    ax1.set_ylabel('Average Partial Dependence', color='blue',fontsize=12)
    ax1.set_xlabel(feature_key,fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_yscale('log')

    # Add histogram on second y-axis
    ax2 = ax1.twinx()
    ax2.hist(Xpd_train_scaled[feature_key], bins=30, color='gray', alpha=0.3, label='Data Distribution')
    ax2.set_ylabel('Frequency', color='gray',fontsize=12)
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_yscale('log')
    for j in range(len(feature_list), len(axes)):
        axes[j].set_visible(False)

fig.suptitle('Average Partial Dependence Plots with Feature Distributions', fontsize=16, x=1.0, y=2.0)
fig.tight_layout(rect=[0, 0, 2, 1.97])
plt.savefig(path + '/pdpaverage_vs_features.pdf', dpi=300,bbox_inches='tight')
