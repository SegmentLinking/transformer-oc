import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from itertools import cycle

import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

# Helper function to add CMS-style labels
def add_cms_labels(ax, lumi=None):
    """Add CMS labels and preliminary text to axis"""
    cms_kwargs = {
        "ax": ax,
        "text": "Preliminary",
        "data": False,
        "rlabel": r"Phase-2 (14 TeV)",
    }
    if lumi is not None:
        cms_kwargs["lumi"] = lumi

    hep.cms.label(**cms_kwargs)
    ax.text(0.95, 0.95, r"$t\bar{t}$, $\langle\mu\rangle=200$", transform=ax.transAxes, fontsize=20, verticalalignment='top', horizontalalignment='right')

log_dir = "../output/lightning_logs/version_6/"

event_accumulator = EventAccumulator(log_dir)
event_accumulator.Reload()

def extract_scalars(event_accumulator, tag):
    events = event_accumulator.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]

    return pd.DataFrame({"step": steps, tag: values})


def smooth_data(df, window_size=20):
    smoothed_data = []
    for i in range(0, len(df), window_size):
        window = df.iloc[i:i+window_size]
        if len(window) > 0:
            mean_step = window["step"].mean()
            mean_value = window.iloc[:, 1].mean()
            smoothed_data.append({"step": mean_step, window.columns[1]: mean_value})
    
    return pd.DataFrame(smoothed_data)

# get all scalar tags
tags = event_accumulator.Tags()["scalars"]

loss_tags = ["loss", "loss_attractive", "loss_repulsive", "loss_beta"]

title_tags = ["Total Loss", "Attractive Loss", "Repulsive Loss", "Beta Loss"]

train_loss_df_list = [extract_scalars(event_accumulator, "train_" + tag + "_epoch") for tag in loss_tags]
val_loss_df_list = [extract_scalars(event_accumulator, "val_" + tag) for tag in loss_tags]

fig, axs = plt.subplots(2, 2, figsize=(20, 16))
axs = axs.flatten()

markers = cycle(['o', 's', '^', 'd', 'x'])
linestyles = cycle(['-', '--', '-.', ':'])
colors = cycle(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray'])

for i in range(len(train_loss_df_list)):
    train_df = train_loss_df_list[i]
    val_df = val_loss_df_list[i]
    
    # Add initial point to validation with same y-value as first training point
    first_train_value = train_df[train_df.columns[1]].iloc[0]
    val_with_init = pd.concat([
        pd.DataFrame({"step": [0], val_df.columns[1]: [first_train_value]}),
        val_df
    ], ignore_index=True)
    
    axs[i].plot(train_df["step"], train_df[train_df.columns[1]], label="Training", marker='o', linestyle='-', linewidth=2.5, color='#1f77b4')
    axs[i].plot(val_with_init["step"], val_with_init[val_with_init.columns[1]], label="Validation", marker='s', linestyle='--', linewidth=2.5, color='#ff7f0e')
    
    axs[i].set_xlabel("Steps", fontsize=24)
    axs[i].set_ylabel(title_tags[i], fontsize=24)
    axs[i].legend(fontsize=18, loc='upper right', bbox_to_anchor=(0.95, 0.88))
    axs[i].set_yscale("log")
    axs[i].grid(True, alpha=0.3, linestyle='--')
    add_cms_labels(axs[i])

plt.tight_layout()

plt.savefig(log_dir + "loss.png")

# Plot gradient norm
grad_norm_tag = "grad_norm"
train_grad_norm_df = extract_scalars(event_accumulator, grad_norm_tag)

fig_grad, ax_grad = plt.subplots(figsize=(12, 9))
    
smoothed_grad_norm_df = smooth_data(train_grad_norm_df, window_size=50)
ax_grad.plot(smoothed_grad_norm_df["step"], smoothed_grad_norm_df[smoothed_grad_norm_df.columns[1]], marker='o', linestyle='-', linewidth=2.5, color='#2ca02c')
ax_grad.set_xlabel("Steps", fontsize=16)
ax_grad.set_ylabel("Gradient Norm", fontsize=20)
ax_grad.set_yscale("log")
ax_grad.grid(True, alpha=0.3, linestyle='--')
add_cms_labels(ax_grad)
plt.tight_layout()
plt.savefig(log_dir + "grad_norm.png")


