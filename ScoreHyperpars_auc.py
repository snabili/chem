import re,os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

import logging

# Set logging level and format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

dnn_txtfile = os.path.join(os.getcwd(), 'files','MLHypertune_pars', 'DNN_hypertune.txt')

grouped_scores = defaultdict(lambda: defaultdict(list))


with open(dnn_txtfile,"r") as file: # hypertuned based on AUC
   for line in file:
        match = re.search(
            r"Best Score:\s+([0-9.]+)\s*using\s*(\{.*?\})", line
        )
        if match:
            score  = float(match.group(1))
            params = eval(match.group(2))
            bs = params['batch_size']
            ep = params['epochs']
            hu = params['model__hidden_units']
            nl = params['model__layer_num']
            key = (f"BS={bs}", f"HU={hu}")
            grouped_scores[key][f"NL={nl}"].append(float(score))
            logger.info(f"{key} → CV Scores: {score}")

group_labels = []
nl_labels = []
mean_scores = []
std_scores = []

for i, ((bs_label, hu_label), nl_dict) in enumerate(grouped_scores.items()):
    group_labels.append(f"{bs_label}, {hu_label}")
    nl_keys = sorted(nl_dict.keys(), key=lambda x: int(x.split('=')[1]))
    if not nl_labels:
        nl_labels = nl_keys  # Set only once, assuming all groups share the same NLs
    mean_scores.append([nl_dict[nl][0] for nl in nl_keys])


# Plotting parameters
num_groups = len(group_labels)
num_nls = len(nl_labels)
x = np.arange(num_groups)
group_width = 1.0
bar_width = group_width / num_nls

# --------------------------------------
# STEP 3: Plot grouped bar chart
# --------------------------------------

plt.figure(figsize=(12, 8))
for i, nl in enumerate(nl_labels):
    bar_positions = x - group_width/2 + i * bar_width + bar_width / 2
    heights = [mean_scores[group_idx][i] for group_idx in range(num_groups)]
    plt.bar(bar_positions, heights, width=bar_width, label=nl)

plt.xticks(ticks=x, rotation=30, ha="right",fontsize=14, labels=group_labels)
plt.yticks(fontsize=14)
plt.ylabel("Mean CV Score",fontsize=16)
plt.ylim(0.78,0.84)
plt.title("Mean Cross-Validation Scores vs DNN Hyperparameters",fontsize=20)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(fontsize=16,ncol=3)
plt.tight_layout()
plt.savefig('files/hyperparameters_auc_cvscores.pdf')
