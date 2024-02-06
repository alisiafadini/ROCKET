import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import re
from itertools import islice
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


sns.set_context("notebook", font_scale=1.8)
plt.rcParams["axes.linewidth"] = 3.0
plt.rcParams["axes.edgecolor"] = "slategrey"
plt.rcParams["xtick.color"] = "slategrey"
plt.rcParams["ytick.color"] = "slategrey"
plt.rcParams["axes.labelcolor"] = "slategrey"


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    # Required arguments
    parser.add_argument(
        "-root",
        "--file_root",
        required=True,
        help=("File path for directory to data"),
    )

    parser.add_argument(
        "-prot",
        "--protein_type",
        required=True,
        help=("Type of protein (prion, lyso, or abl)"),
    )

    return parser.parse_args()


def main():
    # Parse commandline arguments
    args = parse_arguments()
    path = args.file_root
    iterations_len = int(re.search(r"it(\d+)", path).group(1))

    #####################################
    # Load LLG per iteration
    LLGs = np.load("{path}/LLG_it.npy".format(path=path))

    # Load mean plddt per iteration
    plddt_per_it = np.load("{path}/mean_it_plddt.npy".format(path=path))

    # Load MSE loss matrix
    MSE_loss_matrix = np.load("{path}/MSE_loss_it.npy".format(path=path))

    # Load mean plddt per residue
    plddt_per_residue = np.load("{path}/mean_plddt_res.npy".format(path=path))

    # Load MSA changes per iteration matrix
    MSA_changes_matrix = np.load("{path}/MSA_changes_it.npy".format(path=path))

    # Load residue shifts
    residue_numbers = np.load("{path}/residue_numbers.npy".format(path=path))
    mean_perresidue_tostart = np.load(
        "{path}/meanshift_perresidue_tostart.npy".format(path=path)
    )
    mean_perresidue = np.load("{path}/meanshift_perresidue.npy".format(path=path))

    # Ranges of interest
    protein_to_labels = {
        "prion": [
            "N-terminus",
            "Loop",
            "Small Helix",
            "Hinge",
            "Control",
            "C-terminus Helix",
        ],
        "lyso": [
            "Control",
            "Small Beta Sheet",
            "Beta+Loop",
            "Short Helix",
            "Helices Loop",
            "Long Helix",
        ],
        "abl": [
            "Gly Loop",
            "Beta+Loop",
            "Loop Connection",
            "Beta+Loop+Beta",
            "DFG Loop",
            "Control",
        ],
    }

    title_labels = protein_to_labels[args.protein_type]

    with open("{path}/pseudoB_lineouts_data.pkl".format(path=path), "rb") as file:
        pseudob_data = pickle.load(file)
    #######################################

    def shared_y(ax, x, y1, y2, colors, labels, y3=None):
        ax2 = ax.twinx()

        ax.plot(x, y1, color=colors[0], linewidth=4)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1], color=colors[0])
        ax.tick_params(axis="y", labelcolor=colors[0])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        ax2.plot(x, y2, color=colors[1], linewidth=4, linestyle="--")
        if y3 is not None:
            ax2.plot(x, y3, color=colors[-1], linewidth=4, linestyle="-")
        ax2.set_ylabel(labels[-1], color=colors[1])
        ax2.tick_params(axis="y", labelcolor=colors[1])
        ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    fig = plt.figure(layout="constrained", figsize=(30, 14))
    subfigs = fig.subfigures(
        2, 3, wspace=0.1, width_ratios=[1.3, 1, 1], height_ratios=[1, 1]
    )

    # Subfig [0,0]
    tl_ax = subfigs[0, 0].subplots(1, 1)
    shared_y(
        tl_ax,
        range(iterations_len),
        LLGs,
        plddt_per_it,
        ["mediumblue", "cornflowerblue"],
        ["Iteration", "-LLG", "Mean pLDDT (Over Residues)"],
    )

    # Subfig [0,1]
    # Plot lineouts for each range in the first column
    tr_ax = subfigs[0, 1].subplots(2, 1, sharex=True)
    for i, (range_, data_b) in enumerate(islice(pseudob_data.items(), 2)):
        iterations, mean_b_factors = zip(*data_b)
        average_lineout = np.mean(MSE_loss_matrix[:, range_[0] : range_[1]], axis=1)
        shared_y(
            tr_ax[i],
            np.arange(iterations_len),
            average_lineout,
            mean_b_factors,
            ["slategrey", "darkslategrey"],
            ["", "", ""],
        )
        tr_ax[i].grid(True)
        tr_ax[i].set_ylabel("{start}-{end}".format(start=range_[0], end=range_[1]))
    tr_ax[0].set_title(title_labels[0], color="darkslategrey")
    tr_ax[1].set_title(title_labels[1], color="darkslategrey")
    tr_ax[-1].set_xlabel("Iteration")

    # Subfig [0,2]
    tr_ax = subfigs[0, 2].subplots(2, 1, sharex=True)
    for i, (range_, data_b) in enumerate(islice(pseudob_data.items(), 2, 4)):
        iterations, mean_b_factors = zip(*data_b)
        average_lineout = np.mean(MSE_loss_matrix[:, range_[0] : range_[1]], axis=1)
        shared_y(
            tr_ax[i],
            np.arange(iterations_len),
            average_lineout,
            mean_b_factors,
            ["slategrey", "darkslategrey"],
            ["", "", ""],
        )
        tr_ax[i].grid(True)
        tr_ax[i].set_ylabel("{start}-{end}".format(start=range_[0], end=range_[1]))
    tr_ax[0].set_title(title_labels[2], color="darkslategrey")
    tr_ax[1].set_title(title_labels[3], color="darkslategrey")
    tr_ax[-1].set_xlabel("Iteration")

    # Subfig [1,2]
    tr_ax = subfigs[1, 2].subplots(2, 1, sharex=True)
    for i, (range_, data_b) in enumerate(islice(pseudob_data.items(), 4, 6)):
        iterations, mean_b_factors = zip(*data_b)
        average_lineout = np.mean(MSE_loss_matrix[:, range_[0] : range_[1]], axis=1)
        shared_y(
            tr_ax[i],
            np.arange(iterations_len),
            average_lineout,
            mean_b_factors,
            ["slategrey", "darkslategrey"],
            ["", "", ""],
        )
        tr_ax[i].grid(True)
        tr_ax[i].set_ylabel("{start}-{end}".format(start=range_[0], end=range_[1]))
    tr_ax[0].set_title(title_labels[4], color="darkslategrey")
    tr_ax[1].set_title(title_labels[5], color="darkslategrey")
    tr_ax[-1].set_xlabel("Iteration")

    # Subfig [1,0]
    bl_ax = subfigs[1, 0].subplots(1, 1)
    shared_y(
        bl_ax,
        residue_numbers,
        plddt_per_residue,
        mean_perresidue_tostart,
        ["darkslategrey", "silver"],
        [
            "Residue",
            "Mean pLDDT (Over Iterations)",
            "Mean Residue Shift",
        ],
        y3=mean_perresidue,
    )

    # Subfig [1,1]
    # br_ax = subfigs[1, 1].subplots(1, 1)
    # map = br_ax.imshow(MSE_loss_matrix.T, cmap="viridis", vmax=2.5)
    # br_ax.set_xlabel("Iteration")
    # br_ax.set_ylabel("Residue Calpha")
    # subfigs[1, 1].colorbar(
    #    map,
    #    fraction=1e-2,
    #    pad=0.05,
    #   shrink=0.9,
    #    ax=br_ax,
    #    label="MSE to True Position ($\mathrm{\AA}$)",
    # )

    fig.suptitle("{}".format(path))
    plt.show()
    fig.savefig("{path}/results.pdf".format(path=path))

    ## MSE loss and MSA changes colormaps figure

    # Create a 2x1 subplot grid
    fig, subfigs = plt.subplots(2, 1, figsize=(30, 14))

    # Plot the top heatmap (MSE_loss_matrix)
    top_map = subfigs[0].imshow(MSE_loss_matrix.T, cmap="viridis")
    subfigs[0].set_xlabel("Iteration")
    subfigs[0].set_ylabel("Residue Calpha")
    subfigs[0].set_title("MSE Loss")

    # Add colorbar for the top heatmap
    top_colorbar = fig.colorbar(
        top_map,
        fraction=0.05,
        pad=0.05,
        shrink=0.9,
        ax=subfigs[0],
        label="MSE to True Position ($\mathrm{\AA}$)",
    )

    # Plot the bottom heatmap (MSA_changes_matrix)
    bottom_map = subfigs[1].imshow(MSA_changes_matrix.T, cmap="viridis")
    subfigs[1].set_xlabel("Iteration")
    subfigs[1].set_ylabel("Residue Calpha")
    subfigs[1].set_title("MSA Changes")

    # Add colorbar for the bottom heatmap
    bottom_colorbar = fig.colorbar(
        bottom_map,
        fraction=0.05,
        pad=0.05,
        shrink=0.9,
        ax=subfigs[1],
        label="Change in mean of MSA representation",
    )

    fig.suptitle("{}".format(path))
    plt.tight_layout()
    plt.show()
    fig.savefig("{path}/results_heatmaps.pdf".format(path=path))


if __name__ == "__main__":
    main()
