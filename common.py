# MPR access + pymatgen
from mp_api.client import MPRester
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN  # faster way to get bond structure; other more precise methods: VoronoiNN, CrystalNN
from pymatgen.core import Composition # to extract atomic fraction

# General packages
import pandas as pd
import numpy as np
from tqdm import tqdm
import traceback
import time
import argparse as arg
from multiprocessing import Pool, cpu_count  # multi-processing
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Chem project modules:
import creds
import general as gen 

from multiprocessing import Pool, cpu_count  # multi-processing
from joblib import Parallel, delayed # to avoid bad structure delay

# Suppress annoying warnings
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
ALL_ELEMENTS = None
# ==========================================================================================
#  Functions:
#     Bond structure + atomic fraction + remove stuck element from the queue
# ==========================================================================================
def process_material(args):
    mid, structure = args
    try:
        if structure is None or structure.num_sites > 100:
            print(f"Warning: No structure for {mid}, skipping...")
            return None
        structure.add_oxidation_state_by_guess()
        #cnn = CrystalNN()
        #cnn = VoronoiNN(cutoff=6, allow_pathological=False)
        cnn = MinimumDistanceNN(cutoff=3)
        sg = StructureGraph.from_local_env_strategy(structure, cnn)
        bond_types = []
        bond_lengths = []
        for edge in sg.graph.edges(data=True):
            i, j, bond_info = edge
            site_i = structure[i]
            site_j = structure[j]
            bond_type = "-".join(sorted([site_i.specie.symbol, site_j.specie.symbol]))
            bond_length = site_i.distance(site_j)
            bond_types.append(bond_type)
            bond_lengths.append(bond_length)
        num_bonds = len(bond_lengths)
        mean_bond_length = sum(bond_lengths) / num_bonds if num_bonds > 0 else None
        std_bond_length = (sum([(x - mean_bond_length)**2 for x in bond_lengths]) / num_bonds)**0.5 if num_bonds > 0 else None
        unique_bond_types = len(set(bond_types))
        return {
            "material_id": mid,
            "num_bonds": num_bonds,
            "mean_bond_length": mean_bond_length,
            "std_bond_length": std_bond_length,
            "unique_bond_types": unique_bond_types
        }
    except ValueError as ve:
        # Special catch for Voronoi errors
        print(f"Skipping {mid} due to MinimumDistanceNN error: {ve}")
        return None
    except Exception:
        print(f"Failed {mid} for unknown reason")
        traceback.print_exc()
        return None


print("Get atomic fractions")
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
        print(f"Failed to process {args[0]}: {e}")
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
        "vbm","cbm"
    ]
    # Basic material properties
    print("Querying material summaries...")
    api_key = creds.api_key
    mpr = MPRester(api_key)  
    docs = mpr.materials.summary.search(
        fields=fields,
        formation_energy=(-20, 5),
    )
    docs_list, all_elements = [], set()
    start = time.time()
    for doc in docs:
        if (doc.structure is None or pd.isna(doc.cbm) or pd.isna(doc.vbm) or doc.nsites > 100 or doc.volume > 2000): 
            continue # # Skip entries with bad/missing data
        doc_data = {field: getattr(doc, field, None) for field in fields}
        docs_list.append(doc_data)    
        # Collect all elements composition
        comp = Composition(doc.composition)
        all_elements.update(comp.get_el_amt_dict().keys())
    all_elements = sorted(list(all_elements))
    et1 = time.time()
    print('elapsed time: ', (et1 - start) / 60,' minutes')
    structures_dict = {doc.material_id: doc.structure for doc in docs}
    np.savez('files/structures_dict.npz', structures_dict)

    # transform to pandas dataframe
    df = pd.DataFrame(docs_list)
    print('what is insede DF: ', df.columns)
    df.drop_duplicates('formula_pretty', keep='first', inplace=True) # filter duplicate polymorph per composition
    et2 = time.time()
    print('elapsed time to structure dict: ', (et2 - et1) / 60,' minutes')
    # Additional features
    df['vpa'] = df['volume'] / df['nsites']
    df['magmom_pa'] = df['total_magnetization'] / df['nsites']
    df['all_elements'] = pd.DataFrame(all_elements)
    df.to_csv('files/df.csv',index=False)
    et3 = time.time()
    print('elapsed time to add more features: ', (et2 - et3) / 60,' minutes')
    

@scripter
def atomic_fraction():
    global ALL_ELEMENTS
    df = pd.read_csv('files/df.csv', low_memory=False)
    #all_elements = np.load('all_elements.npz')
    n_jobs = min(2, cpu_count())
    print('number of jobs: ',n_jobs)   
    
    records = df.to_dict("records")
    ALL_ELEMENTS = df['all_elemets']

    start = time.time()

    fraction_records = []
    for record in records:
        fraction = atomic_fraction_row(record)
        fraction_records.append(fraction)

    elapsed_time = (time.time()-start) / 60
    print('elapsed_time = ',elapsed_time, ' minutes')
    df_fractions = pd.DataFrame(fraction_records)
    df_fractions.to_csv('files/df_fractions.csv',index=False)

@scripter
def bond_structure():
    structures_dict = np.load('files/structures_dict.npz')
    material_ids = list(structures_dict.keys())
    args = [(mid, structures_dict[mid]) for mid in material_ids if mid in structures_dict]
    print("Calculating bond statistics with multiprocessing and joblib...")
    #n_jobs = min(8, cpu_count())   
    results = Parallel(n_jobs=n_jobs, verbose=10, batch_size="auto")(
        delayed(safe_process_material)(arg) for arg in args
    )
    results = [r for r in results if r is not None]

    print("bond statistics done in: ", time.time() - time_elapsed, ' seconds')
    df_bond_stats = pd.DataFrame(results)
    df_bond_stats.to_csv('files/df_bond_stats.csv')


if __name__ == '__main__':
    scripter.run()