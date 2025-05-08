# Machine Learning for Material properties

## General Information
This project trains **Machine Learning (ML) models** using chemical compound data, inspired by the publication:  
[ElemNet](https://www.nature.com/articles/s41598-018-35934-y)

The goal is to **predict the "energy above the hull"** (formation energy from elemental composition) using *Silicon* as the primary element. This project could be extended for future elements.

### Dataset Source:
The dataset is obtained via the **Next-Gen Materials Project API**:[Materials Project (MPR)](https://next-gen.materialsproject.org/). To pull dataset, download (free) api from MPR.


### ML Models Used:
- **Linear Regression**
- **Random Forest**
- **Deep Neural Networks (DNN)**
- **XGBoost**


### ML Frameworks:
- **TensorFlow** for DNN  
- **Scikit-Learn** for Random Forest & Linear Regression

### ML features: 

nelements, density, energy_per_atom, formation_energy_per_atom, band_gap, cbm, vbm, vpa, magmom_pa.
The last two features (vpa,magmom_pa) are computed from three original descriptors: volume, total_magnetization and nsites, the two former are normalized by the last descriptor.

### DNN hyperparameters:
- Slightly modified the hyperparameters of [ElemNet](https://www.nature.com/articles/s41598-018-35934-y) paper.  
- Split datasets to train, validation and test. 

---

## Technical Details
- *Python Version*: `Python 3.10`
- *Notebook*: Jupyter Notebook
- *Environment Manager*: Conda  

### Setting up the Environment:
Required packages to set the environment

```bash
# Create & activate environment
conda create -n chemenv
conda activate chemenv

# Install required packages:
pip install tensorflow==2.12.0
pip install scikit-learn
pip install numpy==1.24.3 
pip install pandas
pip install matplotlib
pip install scipy
pip install mp-api
pip install xgboost
pip install seaborn

```
The above packages are add to this `mlenv.yml` file. To set up the environment run:

``` conda env create -f env_ml.yml  ```

### Tunning DNN hyper-parameters

Dry run the hyperparametrization code:

``` python -m py_compile dnn_hyperpar.py \
    --HD 256 512 1024 \
    --NHD 10 17 \
    --BS 256 512 ```

To hyperparameterize number of nodes in hidden layers(HD), number of DNN layers (NHD), and batch size (BS):

``` python dnn_hyperpar.py \
    --npzfile \path\to\npzfile \
    --HD 256 512 1024 \
    --NHD 10 17 \
    --BS 256 512 ```
---

### Todo:
- Train ML classification by categorizing stable vs unstable element based on element above hull limit