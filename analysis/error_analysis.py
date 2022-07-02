import pandas as pd
import numpy as np
import ccobra

from matplotlib import pyplot as plt
import seaborn as sns

RESPONSES = []
for _quant in ['A', 'T', 'D', 'I', 'E', 'O']:
    for _direction in ['ac', 'ca']:
        RESPONSES.append(_quant + _direction)
RESPONSES.append('NVC')

def analyse_errors(file, model, genquant=True):
    direction_and_nvc = 0
    total = 0
    df = pd.read_csv(file)
    model_df = df[df["model"] == model]
    
    wrong_responses = {}
    
    for _, row in model_df.iterrows():
        item = ccobra.Item(row["id"], row["domain"], row["task"], row["response_type"], row["choices"], row["sequence"])
        syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
        
        enc_task = syl.encoded_task
        if genquant:
            if "T" not in enc_task and "D" not in enc_task:
                continue
        else:
            if "T" in enc_task or "D" in enc_task:
                continue

        enc_pred = syl.encode_response(row["prediction"].split(";"))
        enc_truth = syl.encode_response(row["truth"].split(";"))
        
        if enc_pred != enc_truth:
            total += 1
            
            if enc_pred == "NVC" or enc_truth == "NVC":
                direction_and_nvc += 1
            elif enc_pred[0] == enc_truth[0]:
                direction_and_nvc += 1
            
            key = (enc_pred, enc_truth)
            if key not in wrong_responses:
                wrong_responses[key] = 0
            wrong_responses[key] += 1
    print(model, direction_and_nvc, total, direction_and_nvc/total, genquant)
    return sorted(wrong_responses.items(), key=lambda x: x[1], reverse=True)

def get_error_matrix(wrong_responses):
    result = np.zeros((13,13))
    for elem in wrong_responses:
        key, errors = elem
        pred_idx = RESPONSES.index(key[0])
        truth_idx = RESPONSES.index(key[1])
        result[pred_idx, truth_idx] = errors
    
    return result

def plot_errors(error_matrix_phm, error_matrix_mreas, total_max=0, cat="genquant"):
    sns.set(style='whitegrid', palette='colorblind')
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    matrix_max = max(error_matrix_phm.max(), error_matrix_mreas.max())
    total_max = max(total_max, matrix_max)

    # plot PHM
    sns.heatmap(error_matrix_phm,
                ax=axs[0],
                vmin=0,
                vmax=total_max,
                cmap='Blues',
                cbar=False,
                linewidths=0.5,
                linecolor='#0000001F')

    #axs[0].set_xlabel("Truth")
    axs[0].set_xticklabels(RESPONSES, fontweight="bold")

    axs[0].set_yticks(np.arange(len(RESPONSES)) + 0.5, fontweight="bold")
    axs[0].set_yticklabels(RESPONSES, rotation=0, fontweight="bold")
    axs[0].set_title("PHM")
    
    # plot mReasoner
    sns.heatmap(error_matrix_mreas,
                ax=axs[1],
                vmin=0,
                vmax=total_max,
                cmap='Blues',
                cbar=False,
                linewidths=0.5,
                linecolor='#0000001F')

    axs[1].set_ylabel("")

    axs[1].set_xticklabels(RESPONSES, fontweight="bold")

    axs[1].set_yticks(np.arange(len(RESPONSES)) + 0.5, fontweight="bold")
    axs[1].set_yticklabels(RESPONSES, rotation=0, fontweight="bold")
    axs[1].set_title("mReasoner")

    fig.supxlabel("Truth", y=0.05)
    fig.supylabel("Prediction")
    plt.tight_layout()
    plt.savefig("errors_on_{}.pdf".format(cat))
    plt.show()


wrong_responses_phm_genquant = analyse_errors("../data/results_full.csv", "PHM-Indiv", genquant=True)
wrong_responses_mreas_genquant = analyse_errors("../data/results_full.csv", "mReasoner", genquant=True)

error_matrix_phm_genquant = get_error_matrix(wrong_responses_phm_genquant)
error_matrix_mreas_genquant = get_error_matrix(wrong_responses_mreas_genquant)

total_max_genquant = max(error_matrix_phm_genquant.max(), error_matrix_mreas_genquant.max())

wrong_responses_phm_classic = analyse_errors("../data/results_full.csv", "PHM-Indiv", genquant=False)
wrong_responses_mreas_classic = analyse_errors("../data/results_full.csv", "mReasoner", genquant=False)

error_matrix_phm_classic = get_error_matrix(wrong_responses_phm_classic)
error_matrix_mreas_classic = get_error_matrix(wrong_responses_mreas_classic)

total_max_classic = max(error_matrix_phm_classic.max(), error_matrix_mreas_classic.max())

total_max = max(total_max_genquant, total_max_classic)
plot_errors(error_matrix_phm_genquant, error_matrix_mreas_genquant, total_max, cat="genquant")
plot_errors(error_matrix_phm_classic, error_matrix_mreas_classic, total_max, cat="classic")
