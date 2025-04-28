# Machine Learning for Material properties

## General Information
This project trains **Machine Learning (ML) models** using chemical compound data, inspired by the publication:  
[Nature Paper](https://www.nature.com/articles/s41598-018-35934-y)

The goal is to **predict the "energy above the hull"** (formation energy from elemental composition) using *Silicon* as the primary element. This project could be extended for future elements.

### ML Models Used:
- **Linear Regression**
- **Random Forest**
- **Deep Neural Networks (DNN)**
- **XGBoost**

### Dataset Source:
The dataset is obtained via the **Next-Gen Materials Project API**:[Materials Project](https://next-gen.materialsproject.org/)

### ML Frameworks:
- **TensorFlow** for DNN  
- **Scikit-Learn** for Random Forest & Linear Regression  

---

## Technical Details
- *Python Version*: `Python 3.11`
- *Notebook*: Jupyter Notebook
- *Environment Manager*: Conda  

### Setting up the Environment:
Run the following commands to set up the required dependencies:

```bash
# Create & activate environment
conda create -n chemenv
conda activate chemenv

# Install required packages
pip install tensorflow==2.12.0
pip install scikit-learn
pip install numpy==1.24.3 
pip install pandas
pip install matplotlib
pip install scipy
pip install mp-api
pip install xgboost
pip install seaborn

### ML train features used: 

nelements, density, energy_per_atom, formation_energy_per_atom, band_gap, cbm, vbm, vpa, magmom_pa.
The last two features (vpa,magmom_pa) are computed from three original descriptors: volume, total_magnetization and nsites, the two former are normalized by the last descriptor.