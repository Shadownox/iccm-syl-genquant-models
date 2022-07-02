import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ccobra
from scipy.stats import mannwhitneyu

def evaluate_performance(data):
    fname = "../data/results_{}.csv".format(data)

    result_df = pd.read_csv(fname)
    
    result_df["model"] = result_df["model"].apply(lambda x: x.replace("PHM-Indiv", "PHM"))
    
    subj_df = result_df.groupby(
        ['model', 'id'], as_index=False)['score_response'].agg('mean')
    
    mfa_scores = subj_df[subj_df["model"] == "MFA"]["score_response"].values
    mr_scores = subj_df[subj_df["model"] == "mReasoner"]["score_response"].values
    phm_scores = subj_df[subj_df["model"] == "PHM"]["score_response"].values
    U1, p1 = mannwhitneyu(mfa_scores, mr_scores, method="exact")
    U2, p2 = mannwhitneyu(mfa_scores, phm_scores, method="exact")
    
    print("Statistics:")
    print("MR: U={}, p={}".format(U1, p1))
    print("PHM: U={}, p={}".format(U2, p2))

    
    order_df = subj_df.groupby(['model'], as_index=False)['score_response'].agg('mean')
    order = order_df.sort_values('score_response')['model']
    
    print(subj_df.head())
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 4))
    point = [0.3, 0.6, 0.8]
    box = [0.3, 0.6, 0.8, 0.5]
    
    sns.stripplot(x="model", y="score_response", data=subj_df, order=order,
                dodge=True, linewidth=0.5, size=4, edgecolor=[0.3,0.3,0.3], color=point, zorder=1)
    
    ax = sns.boxplot(x="model", y="score_response", data=subj_df, order=order,
                    showcaps=False,boxprops={'facecolor': box, "zorder":10},
                    showfliers=False,whiskerprops={"zorder":10}, linewidth=1, color="black",
                    zorder=10, showmeans=True, meanprops={
                       "markerfacecolor":"#FFFFFFF0", 
                       "markeredgecolor":"black",
                      "markersize":"8"})
    
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.xlabel('')
    ax.set_ylabel('Accuracy', size=16)
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    
    plt.savefig('performance_boxplots_{}.pdf'.format(data))
    
    plt.show()


evaluate_performance("full")