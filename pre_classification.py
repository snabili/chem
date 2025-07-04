from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import numpy as np
import pandas as pd
from datetime import datetime
import os
import general as gen


path = os.path.join(os.getcwd() , 'files')
logger = gen.setup_logging(path + "/preclassification.txt",False)

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
logger.info(f"Split train, validation and test in pandas format: {datetime.now().time().strftime('%H:%M:%S')}")
Xpd_train, Xpd_temp, ypd_train, ypd_temp = train_test_split( 
    df[included], df['stability_label'], 
    test_size=0.2,         # 20% of data goes into temporary set
    random_state=123,      # ensures reproducibility
    shuffle=True           # shuffles before splitting
)

Xpd_ebh_train, Xpd_ebh_temp, ypd_ebh_train, ypd_ebh_temp = train_test_split( 
    df[included], df['energy_above_hull'], 
    test_size=0.2,         # 20% of data goes into temporary set
    random_state=123,      # ensures reproducibility
    shuffle=True           # shuffles before splitting
)


Xpd_val, Xpd_test, ypd_val, ypd_test = train_test_split( 
    df[included], df['stability_label'], 
    test_size=0.5,         # 50% of data goes to val and test
    random_state=123 + 1,  # ensures reproducibility
    shuffle=True           # shuffles before splitting
)

# normalize train, test, val
Xpd_train_scaled        = pd.DataFrame(scaler.fit_transform(Xpd_train),     columns=Xpd_train.columns,      index=Xpd_train.index)
Xpd_ebh_train_scaled    = pd.DataFrame(scaler.fit_transform(Xpd_ebh_train), columns=Xpd_ebh_train.columns,  index=Xpd_ebh_train.index) # to plot ebh vs feat for train data
Xpd_val_scaled          = pd.DataFrame(scaler.transform(Xpd_val),           columns=Xpd_val.columns,        index=Xpd_val.index)
Xpd_test_scaled         = pd.DataFrame(scaler.transform(Xpd_test),          columns=Xpd_test.columns,       index=Xpd_test.index)

# Add identifier columns
Xpd_train['split']              = 'train'
Xpd_train_scaled['split']       = 'train_scaled'
Xpd_ebh_train['split']          = 'train_ebh'
Xpd_ebh_train_scaled['split']   = 'train_ebh_scaled'
Xpd_val['split']                = 'val'
Xpd_val_scaled['split']         = 'val_scaled'
Xpd_test['split']               = 'test'
Xpd_test_scaled['split']        = 'test_scaled'

ypd_train       = ypd_train.to_frame(name='label')
ypd_train['split']      = 'train_label'
ypd_ebh_train   = ypd_ebh_train.to_frame(name='label')
ypd_ebh_train['split']  = 'train_ebh_label'
ypd_val         = ypd_val.to_frame(name='label')
ypd_val['split']        = 'val_label'
ypd_test        = ypd_test.to_frame(name='label')
ypd_test['split']       = 'test_label'

# Combine all into a single DataFrame
df_splits = pd.concat([
    Xpd_train,      Xpd_train_scaled,
    Xpd_ebh_train,  Xpd_ebh_train_scaled,
    Xpd_val,        Xpd_val_scaled,
    Xpd_test,       Xpd_test_scaled,
    ypd_train,      ypd_ebh_train,
    ypd_val,        ypd_test
])

# Move split column to front
cols = df_splits.columns.tolist()
cols.insert(0, cols.pop(cols.index('split')))
df_splits = df_splits[cols]

# Save to CSV
output_path = os.path.join(path, 'df_datasplit.csv')
df_splits.to_csv(output_path, index=False)

# ******************* split train, test, validation in numpy format *******************
logger.info(f"Split train, validation and test in pandas format: {datetime.now().time().strftime('%H:%M:%S')}")
X = df[included].values
y = df['stability_label'].values
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
logger.info(f"Done!!!: {datetime.now().time().strftime('%H:%M:%S')}")