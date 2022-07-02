import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

sns.set(style='whitegrid')

# Load log data per model
# PyPHM: p_entailment, A_conf, I_conf, E_conf, O_conf, T_conf, D_conf
# mReasoner: epsilon, lambda, omega, sigma

# Individual analysis
params_mreasoner = []
params_phm = []

for fname in os.listdir('../data'):
    if not fname.endswith('.json'):
        continue
    
    with open('../data/' + fname) as fh:
        param_dict = json.load(fh)
        
        # Extract mReasoner
        for subj, data in param_dict['mReasoner'].items():
            p_epsilon = data['epsilon']
            p_lambda = data['lambda']
            p_omega = data['omega']
            p_sigma = data['sigma']

            params_mreasoner.append({
                'model': 'mReasoner',
                'condition': "classic" if "classic" in fname else "genQuant",
                'id': subj,
                'epsilon': p_epsilon,
                'lambda': p_lambda,
                'omega': p_omega,
                'sigma': p_sigma
            })
        
        # Extract PHM
        for subj, data in param_dict['PHM-Indiv'].items():
            p_p_entailment = data['p_entailment']
            p_A_conf = data['A_conf']
            p_T_conf = data['T_conf']
            p_D_conf = data['D_conf']
            p_I_conf = data['I_conf']
            p_E_conf = data['E_conf']
            p_O_conf = data['O_conf']

            params_phm.append({
                'model': 'PHM',
                'condition': "classic" if "classic" in fname else "genQuant",
                'id': subj,
                'p_entailment': p_p_entailment,
                'A_conf': p_A_conf,
                'T_conf': np.nan if "classic" in fname else p_T_conf,
                'D_conf': np.nan if "classic" in fname else p_D_conf,
                'I_conf': p_I_conf,
                'E_conf': p_E_conf,
                'O_conf': p_O_conf
            })
        
# Convert to dataframes
df_params_mreasoner = pd.DataFrame(params_mreasoner)
df_params_phm = pd.DataFrame(params_phm)


# Visualize distribution
pnames_mreasoner = ['epsilon', 'lambda', 'omega', 'sigma']
greek_mreasoner = ['$\epsilon$', '$\lambda$', '$\omega$', '$\sigma$']
pnames_phm = ['p_entailment', 'A_conf', 'T_conf', 'D_conf', 'I_conf', 'E_conf', 'O_conf']

hue_order = ['classic', 'genQuant']
plot_width = 9
plot_height = 2

# mReasoner
fig, axs = plt.subplots(1, 4, figsize=(plot_width, plot_height))

for idx, pname in enumerate(pnames_mreasoner):
    sns.kdeplot(x=pname, hue='condition', data=df_params_mreasoner, hue_order=hue_order, ax=axs[idx], legend=False, common_norm=False)
    if idx > 0:
        axs[idx].set_ylabel('')
    if pnames_mreasoner[idx] == "lambda":
        axs[idx].set_xlim([0, 10])
    axs[idx].set_xlabel(greek_mreasoner[idx])
    

custom_lines = [Line2D([0], [0], color="C0", lw=4),
                Line2D([0], [0], color="C1", lw=4)]

fig.legend(custom_lines, ['Classic', 'GenQuant'], loc='center', bbox_to_anchor=(0.5, 0.9), ncol=2, frameon=False)
plt.tight_layout(rect=(0,0,1,0.9))
plt.savefig('param_mreasoner.pdf')
plt.show()

# PHM
fig, axs = plt.subplots(1, len(pnames_phm), figsize=(plot_width, plot_height))

for idx, pname in enumerate(pnames_phm):
    pdf = df_params_phm[['condition', pname]]
    
    plot_data = []
    for condition, condition_df in pdf.groupby('condition'):
        keys, cnts = np.unique(condition_df[pname], return_counts=True)
        cnts = cnts.astype('float')
        cnts /= cnts.sum()
        
        addendum = [{'condition': condition, 'label': x, 'value': y} for x, y in zip(keys, cnts)]
        plot_data.extend(addendum)
    
    df_plot = pd.DataFrame(plot_data)
    sns.barplot(x='label', y='value', hue='condition', data=df_plot, hue_order=hue_order, ax=axs[idx])
    
    axs[idx].get_legend().remove()
    axs[idx].set_xlabel(pname)
    axs[idx].set_ylim(0, 1)
    axs[idx].set_xticklabels([0,1])
    if idx > 0:
        axs[idx].set_ylabel('')
        axs[idx].set_yticklabels([])
    else:
        axs[idx].set_ylabel('Proportion')
    

plt.tight_layout(rect=(0,0,1,0.9))
plt.tight_layout(rect=(0,0,1,0.9))
plt.savefig('param_phm.pdf')
plt.show()