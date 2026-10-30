import os
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import date

bar_props = {
    "joint": {"color": mcolors.TABLEAU_COLORS["tab:blue"]},
    "independent": {"color": mcolors.TABLEAU_COLORS["tab:orange"]},
    "mmvm,0.01,exp,1000": {"color": mcolors.TABLEAU_COLORS["tab:green"]},
    "mmvm,0.1,exp,1000": {"color": mcolors.TABLEAU_COLORS["tab:red"]},
    "mmvm,1.0,exp,1000": {"color": mcolors.TABLEAU_COLORS["tab:purple"]},
    "mmvm,10.0,exp,1000": {"color": mcolors.TABLEAU_COLORS["tab:brown"]},
    "mmvm,0.01,linear,1000": {"color": mcolors.TABLEAU_COLORS["tab:green"]},
    "mmvm,0.1,linear,1000": {"color": mcolors.TABLEAU_COLORS["tab:red"]},
    "mmvm,1.0,linear,1000": {"color": mcolors.TABLEAU_COLORS["tab:purple"]},
    "mmvm,10.0,linear,1000": {"color": mcolors.TABLEAU_COLORS["tab:brown"]},
}

use_local_wandb = True
if use_local_wandb:
    os.environ["HTTP_PROXY"] = "socks5h://localhost:10080"
    wandb.login(host="http://wandb-vogtlab.leomed.ethz.ch:19319")
# wandb.login(host=os.getenv("WANDB_LOCAL_URL"))
api = wandb.Api()
# Project is specified by <entity/project-name>
runs = api.runs("suttetho/mmmae_polymnist")


summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)


runs_df = pd.concat(
    [
        pd.DataFrame(summary_list),
        pd.DataFrame(
            [
                {
                    k: v
                    for k, v in x.items()
                    if k != "log"
                    and k != "model"
                    and k != "dataset"
                    and k != "training"
                }
                for x in config_list
            ]
        ),
        pd.DataFrame(
            [{f"dataset.{k}": v for k, v in x["dataset"].items()} for x in config_list]
        ),
        pd.DataFrame(
            [
                {f"training.{k}": v for k, v in x["training"].items()}
                for x in config_list
            ]
        ),
        pd.DataFrame(
            [{f"model.{k}": v for k, v in x["model"].items()} for x in config_list]
        ),
        pd.DataFrame(
            [{f"log.{k}": v for k, v in x["log"].items()} for x in config_list]
        ),
        pd.DataFrame(name_list),
    ],
    axis=1,
)

DIR_OUT = "/home/thomas/polybox/PhD/projects/multimodality/mimic-cxr/plots"
if not os.path.exists(DIR_OUT):
    os.makedirs(DIR_OUT)
datasets = ["PM_translated75", "PM_translated_50", "PM_translated_50_ms"]
datasets = ["PM_translated_50"]
num_views = 3
modalities = ["m" + str(m) for m in range(0, num_views)]
model_names = ["independent", "joint", "mmvm"]
wandb_group = "20241202"
n_epochs_training = 1000
n_epochs_warmup = 100
patch_size = 4
beta_values = [1.0]
share_weights = False
beta_annealing_functions = ["linear", "sigmoid", "cos", "exp"]
# beta_annealing_functions = ["linear", "exp"]
n_epochs_beta_annealing = [1000]
label_information = "shared"

agg_df = runs_df.copy()
agg_df = agg_df.loc[agg_df["training.n_epochs"] == n_epochs_training]
agg_df = agg_df.loc[agg_df["training.n_epochs_warmup"] == n_epochs_warmup]
agg_df = agg_df.loc[agg_df["model.patch_size"] == patch_size]
agg_df = agg_df.loc[agg_df["model.share_weights"] == share_weights]

if label_information == "ms":
    list_str_ds = [f"downstream/{label_information}/m{k}" for k in range(num_views)]
else:
    list_str_ds = [
        f"downstream/{label_information}/unimodal/m{k}" for k in range(num_views)
    ]

results = {}
num_res = 0
for dataset in datasets:
    df_ds = agg_df.loc[agg_df["dataset.name"] == dataset]
    print(f"dataset df: {df_ds.shape}")
    results_dataset = {}
    for m_idx, m_str in enumerate(model_names):
        if m_str == "mmvm":
            df_m = df_ds.loc[df_ds["model.name"] == m_str]
            for beta in beta_values:
                df_m_b = df_m.loc[df_m["model.beta"] == beta]
                for b_annealing in beta_annealing_functions:
                    df_m_b_a = df_m_b.loc[df_m_b["model.beta_annealing"] == b_annealing]
                    print(df_m_b_a.shape)
                    for n_e_annealing in n_epochs_beta_annealing:
                        df_m_b_a_e = df_m_b_a.loc[
                            df_m_b_a["model.beta_n_epochs_annealing"] == n_e_annealing
                        ]
                        print(df_m_b_a_e.shape)
                        print(df_m_b_a_e[list_str_ds])
                        if df_m_b_a_e.shape[0] > 0:
                            vals_m = df_m_b_a_e[list_str_ds].mean(axis=1).values[0]
                            str_m = f"{m_str},{beta},{b_annealing},{n_e_annealing}"
                            results_dataset[str_m] = vals_m
                            num_res += 1
        else:
            df_m = df_ds.loc[df_ds["model.name"] == m_str]
            if df_m.shape[0] > 0:
                vals_m = df_m[list_str_ds].mean(axis=1).values[0]
                results_dataset[m_str] = vals_m
                num_res += 1
    results[dataset] = results_dataset

today = date.today()
date = today.strftime("%Y%m%d")
fn_plot = os.path.join(
    DIR_OUT,
    f"plot_mm_mae_{datasets[0]}_{date}_patchsize{patch_size}_{label_information}.png",
)
indices = np.arange(len(results))
print(indices)
fig, ax = plt.subplots(figsize=(9, 6))
for idx_ds, dataset in enumerate(datasets):
    res_ds = results[dataset]
    print(res_ds)
    num_bars = len(res_ds)
    b_w = 1 / (num_bars + 1)
    print(b_w)
    idx_res = 0
    for key, value in res_ds.items():
        ax.bar(
            idx_ds + (idx_res - num_bars / 2) * b_w,
            height=value,
            width=b_w,
            label=key,
            # color=bar_props[key]["color"],
        )
        idx_res += 1
ax.set_axisbelow(True)
ax.grid(color="lightgray", linestyle="dashed", linewidth=0.5)
ax.set_ylim(0.0, 1.0)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Dataset")
ax.set_xticks(indices, list(results.keys()))
ax.legend(
    bbox_to_anchor=(1, 0.5),
    loc="center left",
    # mode="expand",
    # borderaxespad=0,
    ncol=1,
)
plt.draw()
print(fn_plot)
fig.savefig(
    fn_plot,
    bbox_inches="tight",
)

if use_local_wandb:
    os.unsetenv("HTTP_PROXY")
    wandb.login(host="https://api.wandb.ai")
