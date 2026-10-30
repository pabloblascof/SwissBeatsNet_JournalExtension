import os
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

use_local_wandb = True
if use_local_wandb:
    os.environ["HTTP_PROXY"] = "socks5h://localhost:10080"
    wandb.login(host="http://wandb-vogtlab.leomed.ethz.ch:19319")
# wandb.login(host=os.getenv("WANDB_LOCAL_URL"))
api = wandb.Api()
# Project is specified by <entity/project-name>
runs = api.runs("suttetho/mvvae_polymnist_20240116")

# plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
mpl.rcParams.update(
    {
        "text.latex.preamble": "\n".join(
            [
                #     r"\usepackage{mathpazo}",
                r"\usepackage[OT1]{fontenc}",
                r"\usepackage[utf8]{inputenc}",
                r"\usepackage{amsmath,amssymb,amsfonts,mathrsfs,bm}",
            ]
        ),
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "text.usetex": True,
        # "font.serif": "Palatino",
        # "font.sans-serif": "sans-serif",
        #    "font.size": 11,
        #    "axes.titlesize": 14,
        #    "axes.labelsize": 10,
        #    "xtick.labelsize": 8,
        #    "ytick.labelsize": 8,
        #    "legend.fontsize": 10,
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "legend.framealpha": 1,
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.01,
        "figure.figsize": (6, 3.5),
        "axes.xmargin": 0,
        "lines.markersize": 5,
        "lines.linewidth": 1.5,
    }
)


def export_legend(legend, filename="legend.png", expand=[-5, -5, 5, 5]):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def generate_plot(
    df_mean,
    df_std,
    str_metric_x,
    str_metric_y,
    str_filename,
    str_xlabel,
    str_ylabel,
    str_model,
    model_names,
    xlim,
    fn_legend=None,
):
    fig, ax = plt.subplots()
    for model, model_name_vis in model_names.items():
        print(model)
        x_vals_mean = df_mean.loc[df_mean[str_model] == model, str_metric_x].values
        y_vals_mean = df_mean.loc[df_mean[str_model] == model, str_metric_y].values
        print(x_vals_mean)
        print(y_vals_mean)
        # x_vals_std = df_std.loc[df_std["model.name"] == model, str_metric_x].values
        # y_vals_std = df_std.loc[df_std["model.name"] == model, str_metric_y].values

        res = ax.scatter(
            x_vals_mean,
            y_vals_mean,
            s=75,
            label=model_name_vis,
            alpha=0.9,
            edgecolors="none",
        )
        ax.quiver(
            x_vals_mean[:-1],
            y_vals_mean[:-1],
            x_vals_mean[1:] - x_vals_mean[:-1],
            y_vals_mean[1:] - y_vals_mean[:-1],
            scale_units="xy",
            linewidth=0.05,
            angles="xy",
            scale=1,
            color=res.get_facecolor()[0],
            # label=model,
        )
    ax.grid(True)
    ax.set_ylim([0, 1])
    ax.set_xlim(xlim)
    ax.set_ylabel(str_ylabel)
    ax.set_xlabel(str_xlabel)
    plt.draw()
    print(str_filename)
    fig.savefig(
        str_filename,
        bbox_inches="tight",
    )
    if fn_legend is not None:
        ncols_l = 3
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            bbox_to_anchor=(0.5, 1.05),
            ncol=ncols_l,
            loc="lower center",
        )
        export_legend(legend, fn_legend)
    plt.close()


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
                    if k != "log" and k != "model" and k != "dataset"
                }
                for x in config_list
            ]
        ),
        pd.DataFrame(
            [{f"dataset.{k}": v for k, v in x["dataset"].items()} for x in config_list]
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


DIR_OUT = "/home/thomas/polybox2/PhD/projects/research_stay/plots"
# DIR_OUT = "/cluster/work/vogtlab/Group/suttetho/multimodality/experiments/mvvae/results"
if not os.path.exists(DIR_OUT):
    os.makedirs(DIR_OUT)

aggregation_method = "avg"
n_epochs_threshold = 300
beta_values = [0.25, 0.5, 1.0, 2.0, 4.0]
beta_threshold = 0.0
latent_dim = 512
dataset = "PM_translated75"
num_views = 3
modalities = ["m" + str(m) for m in range(0, num_views)]
lr = 5e-4
batch_size = 256
alpha_annealing = True
model_names = ["unimodal", "joint", "mixedprior"]
model_names_vis = {
    "unimodal": "independent",
    "joint": "aggregated",
    "mixedprior": "MM-VAMP",
}
wandb_group = "20240123"
dataset_names = ["PM_vanilla", "PM_rotated", "PM_translated75"]
dataset_names = ["PM_translated75"]

agg_df = runs_df.copy()
# agg_df = agg_df.loc[agg_df["model.beta"] == beta]
agg_df = agg_df.loc[agg_df["model.lr"] == lr]
agg_df = agg_df.loc[agg_df["model.latent_dim"] == latent_dim]
agg_df = agg_df.loc[agg_df["model.alpha_annealing"] == alpha_annealing]
agg_df = agg_df.loc[agg_df["log.wandb_group"] == wandb_group]
agg_df = agg_df.loc[agg_df["model.aggregation"] == aggregation_method]
agg_df = agg_df.loc[agg_df["model.beta"] >= beta_threshold]
agg_df = agg_df.loc[agg_df["epoch"] >= n_epochs_threshold]

col_names = [
    "model.name",
    "model.aggregation",
    "dataset.name",
    "model.beta",
    "model.seed",
]
list_str_coh = []
list_str_ds_agg = []
list_str_ds_uni = []
for m_key in modalities:
    for m_tilde_key in modalities:
        str_coh = "val/coherence/" + m_key + "_to_" + m_tilde_key
        col_names.append(str_coh)
        list_str_coh.append(str_coh)
    # str_lhood = "val/likelihood/" + m_key
    # col_names.append(str_lhood)
    str_ds_agg = "val/downstream/aggregated/" + m_key
    col_names.append(str_ds_agg)
    list_str_ds_agg.append(str_ds_agg)
    str_ds_uni = "val/downstream/unimodal/" + m_key
    col_names.append(str_ds_uni)
    list_str_ds_uni.append(str_ds_uni)
str_cond_rec = "val/condition_generation/avg_rec_loss"
str_rec = "val/loss/avg_rec_loss_epoch"
col_names.append(str_rec)
col_names.append(str_cond_rec)

dataset_df = agg_df.loc[agg_df["dataset.name"] == dataset]

single_plots_datasets = True
if single_plots_datasets:
    coh_df = dataset_df[col_names]
    mean_coh = coh_df[list_str_coh].mean(axis=1)
    coh_df["coherence"] = mean_coh
    coh_mean = coh_df.groupby(
        ["model.name", "model.aggregation", "dataset.name", "model.beta"]
    ).mean()
    coh_std = coh_df.groupby(
        ["model.name", "model.aggregation", "dataset.name", "model.beta"]
    ).std()
    coh_mean = coh_mean.reset_index()
    coh_std = coh_std.reset_index()
    coh_mean_sorted = coh_mean.sort_values(by=["model.beta"])
    coh_std_sorted = coh_std.sort_values(by=["model.beta"])

    str_fn_legend = os.path.join(DIR_OUT, "legend_coherence_recloss.png")
    str_filename_coh = os.path.join(
        DIR_OUT, "plot_coherence_recloss_" + str(dataset) + ".png"
    )
    generate_plot(
        coh_mean_sorted,
        coh_std_sorted,
        str_rec,
        "coherence",
        str_filename_coh,
        "Reconstruction Error",
        "Accuracy",
        "model.name",
        model_names_vis,
        xlim=[3800, 2900],
        fn_legend=str_fn_legend,
    )
    str_filename_coh_condrec = os.path.join(
        DIR_OUT, "plot_coherence_condrecloss_" + str(dataset) + ".png"
    )
    generate_plot(
        coh_mean_sorted,
        coh_std_sorted,
        str_cond_rec,
        "coherence",
        str_filename_coh_condrec,
        "Conditional Reconstruction Error",
        "Accuracy",
        "model.name",
        model_names_vis,
        xlim=[6500, 2900],
    )

    # generate plots for latent representations (aggregated)
    ds_df = dataset_df[col_names]
    mean_lr = ds_df[list_str_ds_agg].mean(axis=1)
    ds_df["downstream_lr_agg"] = mean_lr

    ds_mean = ds_df.groupby(
        ["model.name", "model.aggregation", "dataset.name", "model.beta"]
    ).mean()
    ds_std = ds_df.groupby(
        ["model.name", "model.aggregation", "dataset.name", "model.beta"]
    ).std()
    ds_mean = ds_mean.reset_index()
    ds_std = ds_std.reset_index()
    ds_mean_sorted = ds_mean.sort_values(by=["model.beta"])
    ds_std_sorted = ds_std.sort_values(by=["model.beta"])

    str_filename_lr = os.path.join(
        DIR_OUT, "plot_downstream_agg_recloss_" + str(dataset) + ".png"
    )
    generate_plot(
        ds_mean_sorted,
        ds_std_sorted,
        str_rec,
        "downstream_lr_agg",
        str_filename_lr,
        "Reconstruction Error",
        "Accuracy",
        "model.name",
        model_names_vis,
        xlim=[3800, 2900],
    )
    str_filename_lr = os.path.join(
        DIR_OUT, "plot_downstream_agg_recloss_" + str(dataset) + ".png"
    )
    generate_plot(
        ds_mean_sorted,
        ds_std_sorted,
        str_rec,
        "downstream_lr_agg",
        str_filename_lr,
        "Reconstruction Error",
        "Accuracy",
        "model.name",
        model_names_vis,
        xlim=[4000, 2900],
    )

    # generate plot conditional generation and unimodal representations/coherence
    ds_uni_df = dataset_df[col_names]
    mean_uni_lr = ds_uni_df[list_str_ds_uni].mean(axis=1)
    ds_uni_df["downstream_lr_uni"] = mean_uni_lr
    ds_uni_mean = ds_uni_df.groupby(
        ["model.name", "model.aggregation", "dataset.name", "model.beta"]
    ).mean()
    ds_uni_std = ds_uni_df.groupby(
        ["model.name", "model.aggregation", "dataset.name", "model.beta"]
    ).std()
    ds_uni_mean = ds_uni_mean.reset_index()
    ds_uni_std = ds_uni_std.reset_index()
    ds_uni_mean_sorted = ds_uni_mean.sort_values(by=["model.name", "model.beta"])
    ds_uni_std_sorted = ds_uni_std.sort_values(by=["model.name", "model.beta"])
    ds_uni_mean_sorted.to_csv(
        os.path.join(DIR_OUT, "results_downstream_uni_" + str(dataset) + ".csv")
    )
    str_filename_lr_uni_condrec = os.path.join(
        DIR_OUT, "plot_downstream_uni_condrecloss_" + str(dataset) + ".png"
    )
    generate_plot(
        ds_uni_mean_sorted,
        ds_uni_std_sorted,
        str_cond_rec,
        "downstream_lr_uni",
        str_filename_lr_uni_condrec,
        "Conditional Reconstruction Error",
        "Accuracy",
        "model.name",
        model_names_vis,
        xlim=[6500, 2900],
    )
    str_filename_lr_uni_rec = os.path.join(
        DIR_OUT, "plot_downstream_uni_recloss_" + str(dataset) + ".png"
    )
    generate_plot(
        ds_uni_mean_sorted,
        ds_uni_std_sorted,
        str_rec,
        "downstream_lr_uni",
        str_filename_lr_uni_rec,
        "Reconstruction Error",
        "Accuracy",
        "model.name",
        model_names_vis,
        xlim=[3800, 2900],
    )

# generate plot downstream task performance vs conditional generation accuracy
for dataset in dataset_names:
    ds_df = dataset_df[col_names]
    mean_lr = ds_df[list_str_ds_agg].mean(axis=1)
    mean_coh = ds_df[list_str_coh].mean(axis=1)
    ds_df["coherence"] = mean_coh
    ds_df["downstream_lr_agg"] = mean_lr

    ds_mean = ds_df.groupby(
        ["model.name", "model.aggregation", "dataset.name", "model.beta"]
    ).mean()
    ds_std = ds_df.groupby(
        ["model.name", "model.aggregation", "dataset.name", "model.beta"]
    ).std()
    ds_mean = ds_mean.reset_index()
    ds_std = ds_std.reset_index()
    ds_mean_sorted = ds_mean.sort_values(by=["model.beta"])
    ds_std_sorted = ds_std.sort_values(by=["model.beta"])
    str_filename_ds_coh = os.path.join(DIR_OUT, "plot_ds_coh_" + str(dataset) + ".png")
    generate_plot(
        ds_mean_sorted,
        ds_std_sorted,
        "downstream_lr_agg",
        "coherence",
        str_filename_ds_coh,
        "Latent Representation (Acc)",
        "Accuracy",
        "model.name",
        model_names_vis,
        xlim=[0, 1],
    )

# plot joint models with different aggregation functions ("avg", "moe", "poe")
n_epochs_threshold = 50
beta_values = [0.25, 0.5, 1.0, 2.0, 4.0]
latent_dim = 512
dataset = "PM_translated75"
num_views = 3
modalities = ["m" + str(m) for m in range(0, num_views)]
lr = 5e-4
batch_size = 256
alpha_annealing = True
model_names = ["joint"]
agg_names_vis = {
    "avg": "Avg",
    "moe": "MoE",
    "poe": "PoE",
}
wandb_group = "20240123"
dataset_names = ["PM_translated75"]

agg_df = runs_df.copy()
# agg_df = agg_df.loc[agg_df["model.beta"] == beta]
agg_df = agg_df.loc[agg_df["model.lr"] == lr]
agg_df = agg_df.loc[agg_df["model.latent_dim"] == latent_dim]
agg_df = agg_df.loc[agg_df["model.alpha_annealing"] == alpha_annealing]
agg_df = agg_df.loc[agg_df["log.wandb_group"] == wandb_group]
agg_df = agg_df.loc[agg_df["model.name"] == "joint"]
# agg_df = agg_df.loc[agg_df["model.aggregation"] == aggregation_method]
agg_df = agg_df.loc[agg_df["epoch"] >= n_epochs_threshold]
dataset_df = agg_df.loc[agg_df["dataset.name"] == dataset]

ds_uni_df = dataset_df[col_names]

mean_uni_lr = ds_uni_df[list_str_ds_uni].mean(axis=1)
ds_uni_df["downstream_lr_uni"] = mean_uni_lr
ds_uni_mean = ds_uni_df.groupby(
    [
        "model.name",
        "model.aggregation",
        "dataset.name",
        "model.beta",
    ]
).mean()
ds_uni_std = ds_uni_df.groupby(
    [
        "model.name",
        "model.aggregation",
        "dataset.name",
        "model.beta",
    ]
).std()
ds_uni_mean = ds_uni_mean.reset_index()
ds_uni_std = ds_uni_std.reset_index()
ds_uni_mean_sorted = ds_uni_mean.sort_values(
    by=["model.name", "model.aggregation", "model.beta"]
)
ds_uni_std_sorted = ds_uni_std.sort_values(
    by=["model.name", "model.aggregation", "model.beta"]
)
ds_uni_mean_sorted.to_csv(
    os.path.join(DIR_OUT, "results_downstream_uni_" + str(dataset) + "_aggregation.csv")
)
str_filename_lr_uni_condrec = os.path.join(
    DIR_OUT, "plot_downstream_uni_condrecloss_" + str(dataset) + "_aggregation.png"
)
str_fn_legend_aggregation = os.path.join(
    DIR_OUT, "legend_downstream_uni_condrecloss_" + str(dataset) + "_aggregation.png"
)
generate_plot(
    ds_uni_mean_sorted,
    ds_uni_std_sorted,
    str_cond_rec,
    "downstream_lr_uni",
    str_filename_lr_uni_condrec,
    "Conditional Reconstruction Error",
    "Accuracy",
    "model.aggregation",
    agg_names_vis,
    xlim=[8500, 2900],
    fn_legend=str_fn_legend_aggregation,
)
str_filename_lr_uni_rec = os.path.join(
    DIR_OUT, "plot_downstream_uni_recloss_" + str(dataset) + "_aggregation.png"
)
generate_plot(
    ds_uni_mean_sorted,
    ds_uni_std_sorted,
    str_rec,
    "downstream_lr_uni",
    str_filename_lr_uni_rec,
    "Reconstruction Error",
    "Accuracy",
    "model.aggregation",
    agg_names_vis,
    xlim=[6000, 2900],
    fn_legend=str_fn_legend_aggregation,
)


if use_local_wandb:
    os.unsetenv("HTTP_PROXY")
    wandb.login(host="https://api.wandb.ai")
