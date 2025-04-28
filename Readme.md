General Information:
The scope of this project is to train Machine Learning (ML) algorithm with chemical compound, following the idea of: https://www.nature.com/articles/s41598-018-35934-y paper. The ML used is the supervised learning regression type, to predict the "energy above the hull" which is the formation energy from elemental composition with Silicon as the primary element.
The ML models used in this study are Linear regression, Random Forest, DeepNeuralNet (DNN) and XGBoost.

The dataset are downloaded from https://next-gen.materialsproject.org/ using Application Programming Interface (API). 
ML frameworks used are Tensoflow for DNN and Scikit-Learn for random forest and linear regression. 

Technical details:
The code is based on python jupyter notebook, using python3.11.
Conda is used to set the environmnet. 

Required packages:

conda activate -n chemenv
pip install tensorflow==2.12.0
pip install scikit-learn
pip install numpy==1.24.3 
pip install pandas
pip install matplotlib
pip install scipy
pip install mp-api
pip install xgboost
pip install seaborn

Features used (from material project datasets) are: 'nelements', 'density', 'energy_per_atom', 'formation_energy_per_atom','band_gap', 'cbm', 'vbm', 'vpa', 'magmom_pa'. The last two features ('vpa','magmom_pa') are computed from three original descriptors: 'volume', 'total_magnetization' and 'nsites', where the two former are normalized by the last descriptor.