# evaluation_run.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
from evaluation import (
    precision_at_k,
    mean_average_precision,
    ndcg_at_k
)


# Load qrels
def load_qrels(path="data/qrels.xlsx"):
    df = pd.read_excel(path)
    qrels = {}
    for _, row in df.iterrows():
        if row["relevant"] == 1:
            qrels.setdefault(str(int(row["query_id"])), []).append(int(row["doc_id"]))
    return qrels

qrels = load_qrels()


# ranked results

queries_df = pd.read_excel("data/queries.xlsx")  
df_runs = pd.read_excel("data/qrels.xlsx")       

results = {}
for model in df_runs["model"].unique():
    model_runs = {}
    for qid in df_runs["query_id"].unique():
        ranked_docs = (
            df_runs[(df_runs["model"] == model) & (df_runs["query_id"] == qid)]
            .sort_values("rank")["doc_id"]
            .tolist()
        )
        model_runs[qid] = ranked_docs
    results[model] = model_runs


# Evaluation
def evaluate_group(run, query_ids, k=5):
    p, n = [], []
    for qid in query_ids:
        docs = run.get(qid, [])
        rel = qrels.get(str(qid), [])
        p.append(precision_at_k(docs, rel, k))
        n.append(ndcg_at_k(docs, rel, k))
    return np.mean(p), np.mean(n), mean_average_precision({qid: run[qid] for qid in query_ids}, qrels)


# Evaluate each model
query_types = queries_df.set_index("query_id")["query_type"].to_dict()

for model, run in results.items():
    print(f"\n=== {model} ===")

    # Separate query_ids by type
    explicit_qids = [qid for qid, t in query_types.items() if t == "explicit"]
    implicit_qids = [qid for qid, t in query_types.items() if t == "implicit"]
    all_qids = list(run.keys())

    # Evaluate
    p_e, n_e, map_e = evaluate_group(run, explicit_qids)
    p_i,  n_i, map_i = evaluate_group(run, implicit_qids)
    p_all, n_all, map_all = evaluate_group(run, all_qids)

    print("Explicit emotion queries:")
    print(f"P@5     {p_e:.3f}")
    print(f"nDCG@5  {n_e:.3f}")
    print(f"MAP     {map_e:.3f}")

    print("Implicit emotion queries:")
    print(f"P@5     {p_i:.3f}")
    print(f"nDCG@5  {n_i:.3f}")
    print(f"MAP     {map_i:.3f}")

    print("Overall:")
    print(f"P@5     {p_all:.3f}")
    print(f"nDCG@5  {n_all:.3f}")
    print(f"MAP     {map_all:.3f}")


output_dir = "plots"
os.makedirs(output_dir, exist_ok=True) 


metrics_list = []

for model, run in results.items():
    explicit_qids = [qid for qid, t in query_types.items() if t == "explicit"]
    implicit_qids = [qid for qid, t in query_types.items() if t == "implicit"]
    all_qids = list(run.keys())

    groups = {
        "explicit": explicit_qids,
        "implicit": implicit_qids,
        "overall": all_qids
    }

    for qtype, qids in groups.items():
        p, n, map_score = evaluate_group(run, qids)
        metrics_list.append({
            "model": model,
            "query_type": qtype,
            "P@5": p,
            #"R@5": r,
            "nDCG@5": n,
            "MAP": map_score
        })

df_metrics = pd.DataFrame(metrics_list)
df_melted = df_metrics.melt(
    id_vars=["model", "query_type"],
    value_vars=["P@5","nDCG@5","MAP"],
    var_name="metric",
    value_name="score"
)


for qtype in df_melted["query_type"].unique():
    plt.figure(figsize=(10,6))
    subset = df_melted[df_melted["query_type"] == qtype]
    ax = sns.barplot(data=subset, x="model", y="score", hue="metric")

    # 
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=9,
                    xytext=(0, 3),
                    textcoords='offset points')

    plt.title(f"Metrics for {qtype} queries")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.ylim(0,1)
    plt.legend(title="Metric")
    plt.tight_layout()

    # Save the plot
    filename = os.path.join(output_dir, f"{qtype}_metrics.png")
    plt.savefig(filename, dpi=300)  
    plt.show()


