"""
@File    :   main.py
@Time    :   2024/08/29 02:07:25
@Author  :   Nikola Milicevic 
@Version :   1.0
@Contact :   nikola260896@gmail.com
@License :   (C)Copyright 2024, Nikola Milicevic
@Desc    :   None
"""

import datetime
import logging
import sys
import random
import time
import os
import warnings
import subprocess
import multiprocessing as mp
import argparse as ap
from collections import Counter
from itertools import repeat
import warnings

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

import ctopt

random.seed(3)


def plot_spatial(
    adata, annotation, ax: plt.Axes, spot_size: float, palette=None, title: str = ""
):
    """
    Scatter plot in spatial coordinates.

    Parameters:
        - adata (AnnData): Annotated data object which represents the sample
        - annotation (str): adata.obs column used for grouping
        - ax (Axes): Axes object used for plotting
        - spot_size (int): Size of the dot that represents a cell. We are passing it as a diameter of the spot, while
                the plotting library uses radius therefore it is multiplied by 0.5
        - palette (dict): Dictionary that represents a mapping between annotation categories and colors
        - title (str): Title of the figure

    """
    palette = sns.color_palette("coolwarm", as_cmap=True)
    s = spot_size * 0.5
    data = adata
    ax = sns.scatterplot(
        data=data.obs,
        hue=annotation,
        x=data.obsm["spatial"][:, 0],
        y=data.obsm["spatial"][:, 1],
        ax=ax,
        s=s,
        linewidth=0,
        palette=(
            palette
            if ("float" in str(type(adata.obs[annotation].iloc[0])).lower())
            else None
        ),
        marker=".",
    )
    ax.invert_yaxis()
    ax.set(yticklabels=[], xticklabels=[], title=title)
    ax.tick_params(bottom=False, left=False)
    ax.set_aspect("equal")
    sns.despine(bottom=True, left=True, ax=ax)


def main(args):
    start = time.time()

    adata_sc = sc.read_h5ad(args.sc_path)
    adata_st = sc.read_h5ad(args.st_path)
    adata_sc.var_names_make_unique()
    adata_st.var_names_make_unique()
    adata_sc.obs_names_make_unique()
    adata_st.obs_names_make_unique()

    # place cell spatial coordinates in .obsm['spatial']
    # coordinates are expected in 'spatial', 'X_spatial', and 'spatial_stereoseq'
    if "X_spatial" in adata_st.obsm:
        adata_st.obsm["spatial"] = adata_st.obsm["X_spatial"].copy()
    elif "spatial_stereoseq" in adata_st.obsm:
        adata_st.obsm["spatial"] = np.array(adata_st.obsm["spatial_stereoseq"].copy())
    elif "spatial" in adata_st.obsm:
        pass
    else:
        warnings.warn(
            'Spatial coordinates not found. Labels expected in: \
                .obsm["spatial"] or\n \
                .obsm["X_spatial"] or\n \
                .obsm["spatial_stereoseq"]'
        )

    # Calculate marker genes  TODO: Add to separate function in preprocessing.py
    start_marker = time.time()
    adata_sc.layers["counts"] = adata_sc.X.copy()  # Used in contrastive learning
    if "rank_genes_groups" not in adata_sc.uns:
        if adata_sc.X.min() >= 0:  # If already logaritmized skip
            sc.pp.normalize_total(adata_sc, target_sum=1e4)
            sc.pp.log1p(adata_sc)
        sc.tl.rank_genes_groups(adata_sc, groupby=args.annotation, use_raw=False)
    else:
        logger.info(f"***d Using precalculated marker genes in input h5ad.")

    markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[
        0 : args.num_markers, :
    ]

    markers = list(np.unique(markers_df.melt().value.values))

    markers_intersect = list(set(markers).intersection(adata_st.var.index))
    logger.info(
        f"Using {len(markers_intersect)} unique single cell marker genes that exist in ST dataset ({args.num_markers} per cell type)"
    )
    end_marker = time.time()
    marker_time = np.round(end_marker - start_marker, 3)
    logger.info(f"Calculation of marker genes took {marker_time:.2f}")

    kwargs = dict(
        sc_path=args.sc_path,
        st_path=args.st_path,
        adata_sc=adata_sc[:, markers_intersect],
        adata_st=adata_st[:, markers_intersect],
        annotation_sc=args.annotation,
        batch_size=args.batch_size,
        epochs=args.epochs,
        embedding_dim=args.emb_dim,
        encoder_depth=args.enc_depth,
        classifier_depth=args.class_depth,
        filename=filename,
        augmentation_perc=args.augmentation_perc,
        wandb_key=args.wandb_key,
        n_views=args.n_views,
        temperature=args.temperature,
        annotation_st=args.annotation_st
    )

    # ctopt.preprocess(adata_sc)
    # ctopt.preprocess(adata_st)

    df_probabilities, predictions = ctopt.contrastive_process(**kwargs)
    # contr process da vraca df_probabilities i predictions
    adata_st.obsm["probabilities_contrastive"] = df_probabilities
    adata_st.obs["ctopt"] = predictions
    # Write CSV and H5AD  TODO: Add to separate function in core/util.py
    adata_st.obs.index.name = "cell_id"
    adata_st.obs["ctopt"].to_csv(
        os.path.basename(args.st_path).replace(".h5ad", f"ctopt.csv")
    )
    adata_st.write_h5ad(os.path.basename(args.st_path).replace(".h5ad", f"_ctopt.h5ad"))

    # if "spatial" in adata_st.obsm_keys():  # TODO: Add to separate function in core/util.py
    #     fig, axs = plt.subplots(1, 2, figsize=(14, 14))
    #     plot_spatial(
    #         adata_st, annotation=f"ctopt", spot_size=50, ax=axs[0], title="Cell types"
    #     )
    #     plot_spatial(
    #         adata_st,
    #         annotation=f"ctopt_confidence",
    #         spot_size=50,
    #         ax=axs[1],
    #         title="Confidence map",
    #     )
    #     plt.savefig(
    #         os.path.basename(args.st_path).replace(
    #             ".h5ad", f"_ctopt.png"
    #         ),
    #         dpi=120,
    #         bbox_inches="tight",
    #     )

    end = time.time()
    logger.info(f"Total execution time: {(end - start):.2f}s")


if __name__ == "__main__":
    if not os.path.exists("logs"):
        os.makedirs("logs")

    parser = ap.ArgumentParser(
        description="A script that performs reference based cell type annotation."
    )
    parser.add_argument(
        "--sc_path", help="A single cell reference dataset", type=str, required=True
    )
    parser.add_argument(
        "--st_path", help="A spatially resolved dataset", type=str, required=True
    )
    parser.add_argument(
        "-a",
        "--annotation",
        help="Annotation label for cell types",
        type=str,
        required=True,
        default="cell_subclass",
    )
    parser.add_argument(
        "-at",
        "--annotation_st",
        help="Annotation label for cell types",
        type=str,
        required=True,
        default="cell_subclass",
    )
    parser.add_argument(
        "--wandb_key", help="Wandb key for loss monitoring", type=str, required=True
    )
    parser.add_argument(
        "--num_markers",
        help="Number of marker genes per cell type. Defaults to 300.",
        type=int,
        required=False,
        default=100,
    )
    parser.add_argument(
        "--batch_size",
        help="Contrastive: Number of samples in the batch. Defaults to 512.",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--n_views",
        help="Contrastive: Number of views of one sample.",
        type=int,
        required=False,
        default=2,
    )
    parser.add_argument(
        "--epochs",
        help="Contrastive: Number of epochs to train deep encoder. Default is 50.",
        type=int,
        required=False,
        default=100,
    )
    parser.add_argument(
        "--emb_dim",
        help="Contrastive: Dimension of the output embeddings. Default is 256.",
        type=int,
        required=False,
        default=128,
    )
    parser.add_argument(
        "--enc_depth",
        help="Contrastive: Number of layers in the encoder MLP. Default is 4.",
        type=int,
        required=False,
        default=4,
    )
    parser.add_argument(
        "--class_depth",
        help="Contrastive: Number of layers in the classifier MLP. Default is 0 - logistic regression.",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--augmentation_perc",
        help="Contrastive: Percentage for the augmentation of scRNA reference data. \
            If not provided it will be calculated automatically. Defaults to None.",
        type=float,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--temperature",
        help="Contrastive: temperature for loss",
        type=float,
        required=False,
        default=0.07,
    )
    parser.add_argument("-l", "--log_mem", action="store_true", default=False)
    parser.add_argument(
        "-v",
        "--verbose",
        help="Enable logging by specifying --verbose",
        action="store_const",
        const=logging.INFO,
        default=logging.WARNING,
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        level=args.verbose,
    )
    logger = logging.getLogger(__name__)

    filename = None
    if args.verbose != logging.WARNING:
        timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
        filename = os.path.basename(args.st_path).replace(".h5ad", "")
        filename = f"logs/{filename}_{timestamp}.log"
        file_handler = logging.FileHandler(filename)
        logger.addHandler(file_handler)

    if args.log_mem:
        mem_logger_fname = os.path.basename(args.st_path).replace(
            ".h5ad", "_cpu_gpu_memlog.csv"
        )
        if os.path.isfile(mem_logger_fname):
            os.remove(mem_logger_fname)

        logger_pid = subprocess.Popen(
            [
                "python",
                "ctopt/log_gpu_cpu_stats.py",
                mem_logger_fname,
            ]
        )
        logger.info("Started logging compute resource utilisation")

    main(args=args)

    if args.log_mem:
        # End the background process logging the CPU and GPU utilisation.
        logger_pid.terminate()
        print("Terminated the compute utilization logger background process")

        # read cpu and gpu memory utilization
        logger_df = pd.read_csv(mem_logger_fname)

        max_cpu_mem = logger_df.loc[:, "RAM"].max()
        max_gpu_mem = logger_df.loc[:, "GPU 0"].max()

        logger.info(
            f"Peak RAM Usage: {max_cpu_mem} MiB\nPeak GPU Usage: {max_gpu_mem} MiB\n"
        )
