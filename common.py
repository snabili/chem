# MPR access + pymatgen
from mp_api.client import MPRester
from pymatgen.core.structure import Structure # to load data structure from json file
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN  # faster way to get bond structure w.r.t VoronoiNN, CrystalNN
from pymatgen.core import Composition # to extract atomic fraction

# General packages
import pandas as pd
import numpy as np
from tqdm import tqdm
import traceback
from datetime import datetime
import json
import argparse as arg
from multiprocessing import Pool, cpu_count  # multi-processing
from joblib import Parallel, delayed # to avoid bad structure delay
import os
os.environ["OMP_NUM_THREADS"] = "1"

# costume project modules:
import creds
import general as gen 

# Suppress annoying warnings
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
ALL_ELEMENTS = None

import logging
# Set logging level and format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)
logger.setLevel(logging.ERROR) # supress prints
# ==========================================================================================
#  Functions:
#     Bond structure, atomic fraction, remove bad element from queue
# ==========================================================================================
def process_material(args):
    oxidation = 0
    mid, structure = args
    if structure is None or structure.num_sites > 100:
        print(f"Warning: No structure for {mid}, skipping...")
        return None
    try:
        if not structure.is_ordered:
            structure = structure.get_ordered_structure()
        # Cache oxidation states if possible
        if not structure.site_properties.get("oxidation_states"):
            logger.info('oxidation not available')
            oxidation += 1
            structure.add_oxidation_state_by_guess()
        sg = StructureGraph.from_local_env_strategy(structure, MinimumDistanceNN(cutoff=5))  # cutoff x d_min
        bond_lengths = []
        bond_types = set()
        
        for i, j, attr in sg.graph.edges(data=True):
            site_i = structure[i]
            site_j = structure[j]
            bond_types.add("-".join(sorted([site_i.specie.symbol, site_j.specie.symbol])))
            bond_lengths.append(attr.get("weight", site_i.distance(site_j)))  # Use precomputed weight if available

        if not bond_lengths:
            return None
        
        mean_bond_length = np.mean(bond_lengths)
        std_bond_length = np.std(bond_lengths)
        return {
            "material_id": mid,
            "num_bonds": len(bond_lengths),
            "mean_bond_length": mean_bond_length,
            "std_bond_length": std_bond_length,
            "unique_bond_types": len(bond_types)
        }
    
    except Exception as e:
        logger.info(f"Error processing {mid}: {e}")
        return None
    logger.info('structures without oxidation data = ', oxidation)

logger.info("Get atomic fractions")
def get_atomic_fractions(comp, elements_list):
    fractions = dict.fromkeys(elements_list, 0.0)
    total = sum(comp.get_el_amt_dict().values())
    for el, amt in comp.get_el_amt_dict().items():
        if el in fractions:
            fractions[el] = amt / total
    return fractions

def safe_process_material(args):
    try:
        return process_material(args)
    except Exception as e:
        logger.info(f"Failed to process {args[0]}: {e}")
        return None
    
def atomic_fraction_row(row):
        comp = Composition(row['composition'])
        fractions = get_atomic_fractions(comp, ALL_ELEMENTS)
        fractions['material_id'] = row['material_id']
        return fractions

scripter = gen.Scripter()

@scripter
def data_acquisition():
    fields = [
        "material_id",
        "formation_energy_per_atom",
        "band_gap",
        "energy_per_atom",
        "total_magnetization",
        "volume",
        "density",
        "energy_above_hull",
        "is_stable",
        "nelements",
        "nsites",
        "vbm",
        "cbm",
        "composition",  # needed for atomic fractions
        "structure", # needed for material structure
        "formula_pretty",  # (we will drop later)
        "vbm"
    ]
    logger.info(f"start getting material properties:  {datetime.now().time().strftime('%H:%M:%S')}")
    # Basic material properties
    logger.info("Querying material summaries...")
    api_key = creds.api_key
    mpr = MPRester(api_key)  
    docs = mpr.materials.summary.search(
        fields=fields,
        formation_energy=(-20, 5),
    )
    docs_list, all_elements, seen_formula, structures_dict = [], set(), set(), {}
    for doc in docs:
        if (doc.structure is None or pd.isna(doc.cbm) or pd.isna(doc.vbm) or doc.nsites > 100 or doc.volume > 2000): 
            continue # # Skip entries with bad/missing data
        # remove duplicate formulas and save structure_dict
        formula = getattr(doc, 'formula_pretty', None)
        if formula in seen_formula: continue
        seen_formula.add(formula)
        structures_dict[doc.material_id] = doc.structure.as_dict()

        doc_data = {field: getattr(doc, field, None) for field in fields}
        docs_list.append(doc_data)    
        # Collect all elements composition
        comp = Composition(doc.composition)
        all_elements.update(comp.get_el_amt_dict().keys())
        
    all_elements = sorted(list(all_elements))
    logger.info(f"done getting material properties:  {datetime.now().time().strftime('%H:%M:%S')}")

    # JSON serialization of Structure objects
    with open('files/structures_dict_modified.json', 'w') as f:
        json.dump(structures_dict, f)

    logger.info('DONE WITH STRUCTURE_DICT!!!')
    # transform to pandas dataframe
    df = pd.DataFrame(docs_list)
    logger.info('DF columns: ', df.columns)
    df.drop_duplicates('formula_pretty', keep='first', inplace=True) # filter duplicate polymorph per composition
    logger.info(f"structure dict done:  {datetime.now().time().strftime('%H:%M:%S')}")

    # Additional features
    df['vpa'] = df['volume'] / df['nsites']
    df['magmom_pa'] = df['total_magnetization'] / df['nsites']
    df['all_elements'] = pd.DataFrame(all_elements)
    df.to_csv('files/df.csv',index=False)
    logger.info(f"finished adding more features at:  {datetime.now().time().strftime('%H:%M:%S')}")
    

@scripter
def atomic_fraction():
    global ALL_ELEMENTS
    df = pd.read_csv('files/df.csv', low_memory=False)    
    records = df.to_dict("records")
    ALL_ELEMENTS = df['all_elemets']
    fraction_records = []
    for record in records:
        fraction = atomic_fraction_row(record)
        fraction_records.append(fraction)
    logger.info(f"done atomic fraction at:  {datetime.now().time().strftime('%H:%M:%S')}")
    df_fractions = pd.DataFrame(fraction_records)
    df_fractions.to_csv('files/df_fractions.csv',index=False)

@scripter
def bond_structure():
    logger.info('opening structure json file:  ')
    with open('files/structures_dict_modified.json', 'r') as f:
        structures_dict_raw = json.load(f)

    structures_dict = {
        mid: Structure.from_dict(data)
        for mid, data in structures_dict_raw.items()
    }
    material_ids = list(structures_dict.keys())
    args = [(mid, structures_dict[mid]) for mid in material_ids if mid in structures_dict]
    logger.info("Calculating bond statistics with multiprocessing and joblib...")
    n_jobs = min(8, cpu_count())
    logger.info('number of jobs: ',n_jobs)  
    results = Parallel(n_jobs=n_jobs, verbose=10, batch_size="auto")(
        delayed(safe_process_material)(arg) for arg in args[39000:42000]
    )
    results = [r for r in results if r is not None]
    logger.info(f"done with bond statistics at:  {datetime.now().time().strftime('%H:%M:%S')}")
    df_bond_stats = pd.DataFrame(results)
    df_bond_stats.to_csv('files/df_bond_stats_from39000to42000.csv')

@scripter
def merge_df_file():
    df_bond_struct = pd.read_csv('files/df_bond_stats.csv')
    df_atomic_frac = pd.read_csv('files/df_fractions.csv')
    df_features    = pd.read_csv('files/df.csv')
    df_combined = pd.merge(df_features, df_atomic_frac, on='material_id', how='inner')
    df_combined = pd.merge(df_combined, df_bond_struct, on='material_id', how='inner')
    df_combined.to_csv('files/df_combined.csv')

@scripter
def merge_df_structure():    
    file_path = '/Users/saranabili/Desktop/jobHunts/chem/files/complete_struct/'
    file_path = os.path.join(os.getcwd(), 'files', 'complete_struct')
    full_paths = [os.path.join(file_path, f) for f in os.listdir(file_path)]
    df_list = [pd.read_csv(path) for path in full_paths]
    df_merged_struct = pd.concat(df_list, ignore_index=True)
    df_merged_struct.to_csv(os.path.join(os.getcwd(),'files','df_bond_stats.csv'))

    
if __name__ == '__main__':
    scripter.run()
