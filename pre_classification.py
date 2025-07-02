from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import numpy as np
import pandas as pd
from datetime import datetime
import os
path = os.path.join(os.getcwd() + 'files')

import logging
# Set logging level and format; logging.info go directly to pdp_results_log.txt
logging.basicConfig(
    level=logging.INFO,
    #logger.setLevel(logging.ERROR), # supress prints
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(path + "/pdp_results_log.txt"),  # Log file
        logging.StreamHandler()                      # Optional to show in console
    ]
)
logger = logging.getLogger(__name__)

# suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

# **************************** Read Dataframe features ****************************
logger.info('Read features: ')
df = pd.read_csv(path + '/df_combined.csv')
excluded = ['energy_above_hull','volume','nsites','is_stable','formula_pretty','total_magnetization','stability_label','composition',
            'structure','material_id','all_elements',
            'Unnamed: 0','Unnamed: 0.1','Unnamed: 0.3','Unnamed: 78','Unnamed: 0.2', # remove 1st pandas columns from combining struct + bond + other feats
            'He','Ne','Ar','Kr','Xe'] # removing noble gas

included = [c for c in df.columns if c not in excluded]
df['stability_label'] = classify_stability(df['energy_above_hull'])

# ******************* split train, test, validation in pandas format *******************
logger.info(f'Split train, validation and test in pandas format: {datetime.now().time().strftime('%H:%M:%S')}')
Xpd_train, Xpd_temp, ypd_train, ypd_temp = train_test_split( 
    df[included], df['stability_label'], 
    test_size=0.2,         # 20% of data goes into temporary set
    random_state=123,      # ensures reproducibility
    shuffle=True           # shuffles before splitting
)

Xpd_val, Xpd_test, ypd_val, ypd_test = train_test_split( 
    df[included], df['stability_label'], 
    test_size=0.5,         # 20% of data goes into temporary set
    random_state=123 + 1,  # ensures reproducibility
    shuffle=True           # shuffles before splitting
)

# normalize train, test, val
Xpd_train_scaled = pd.DataFrame(scaler.fit_transform(Xpd_train),columns=Xpd_train.columns,  index=Xpd_train.index)
Xpd_val_scaled   = pd.DataFrame(scaler.transform(Xpd_val),      columns=Xpd_val.columns,    index=Xpd_val.index)
Xpd_test_scaled  = pd.DataFrame(scaler.transform(Xpd_test),     columns=Xpd_test.columns,   index=Xpd_test.index)

df_split = [Xpd_train, Xpd_train_scaled, ypd_train, Xpd_val, Xpd_val_scaled, ypd_val, Xpd_test, Xpd_test_scaled, ypd_test]
combined_df = pd.concat(df_split, ignore_index=True)
combined_df.to_csv(path + "/df_datasplit.csv", index=False)

# ******************* split train, test, validation in numpy format *******************
logger.info(f'Split train, validation and test in pandas format: {datetime.now().time().strftime('%H:%M:%S')}')
X = df[included].values
X_train, X_temp,y_train, y_temp = train_test_split(X,       y,      test_size=0.2, random_state=123,     shuffle=True)
X_val,   X_test,y_val,   y_test = train_test_split(X_temp,  y_temp, test_size=0.5, random_state=123 + 1, shuffle=True)
# normalize features
X_train_scaled, X_val_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

# save test, train, val for plotting purposes:
data_dict = {
    'X_train':          X_train,
    'X_train_scaled':   X_train_scaled,
    'y_train':          y_train,
    'X_val':            X_val,
    'X_val_scaled':     X_val_scaled,
    'y_val':            y_val,
    'X_test':           X_test,
    'X_test_scaled':    X_test_scaled,
    'y_test':           y_test
}
np.savez(path + '/npz_datasplits.npz', **data_dict)
logger.info(f'Done!!!: {datetime.now().time().strftime('%H:%M:%S')}')