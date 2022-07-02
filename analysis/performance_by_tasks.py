import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import ccobra

fname = "../data/results_full.csv"

result_df = pd.read_csv(fname)
result_df["model"] = result_df["model"].apply(lambda x: x.replace("PHM-Indiv", "PHM"))

results = []

for _, row in result_df.iterrows():
    item = ccobra.Item(row["id"], row["domain"], row["task"], row["response_type"], row["choices"], row["sequence"])
    syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
    
    enc_task = syl.encoded_task
    if "T" not in enc_task and "D" not in enc_task:
        results.append({
            "type" : "Classic",
            "model" : row["model"],
            "id" : row["id"],
            "task" : row["task"],
            "score_response" : row["score_response"]
        })
    else:
        results.append({
            "type" : "Generalized",
            "model" : row["model"],
            "id" : row["id"],
            "task" : row["task"],
            "score_response" : row["score_response"]
        })

result_df = pd.DataFrame(results)

result_df = result_df.groupby(
        ['model', 'id', 'type'], as_index=False)['score_response'].agg('mean')


order_df = result_df.groupby(['model'], as_index=False)['score_response'].agg('mean')
order = order_df.sort_values('score_response')['model']

sns.set(style="whitegrid")


plt.figure(figsize=(7, 4))
point = [0.3, 0.6, 0.8]
box = [0.3, 0.6, 0.8, 0.5]

ax = sns.stripplot(x="model", y="score_response", hue="type", data=result_df, order=order,
                dodge=True, linewidth=0.5, size=4, zorder=1) 

ax = sns.boxplot(x="model", y="score_response", hue="type", data=result_df, order=order,
            showcaps=False, boxprops={"alpha": 0.6, "zorder":10},
            showfliers=False, whiskerprops={"zorder":10}, linewidth=1,
            zorder=10, showmeans=True, meanprops={
                       "markerfacecolor":"#FFFFFFA0", 
                       "markeredgecolor":"black",
                      "markersize":"8"}
            )

ax.set_ylim([0, 1])
ax.set_xlabel("")
ax.set_ylabel("Accuracy")
custom_patch = [Patch(facecolor="C0", label="Classic Syllogisms"),
                Patch(facecolor="C1", label="GenQuant")]

ax.legend(custom_patch, ['Classic', 'GenQuant'], loc='center', bbox_to_anchor=(0.21, 0.95), ncol=2, frameon=True)

plt.tight_layout(rect=(0,0,1,1))
plt.savefig("genquant_vs_classic_boxplot.pdf")
plt.show()