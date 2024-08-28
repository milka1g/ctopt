import math
import numpy as np
import torch
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr, target_lr):
    """Adjust learning rate according to warm-up strategy."""
    if epoch < warmup_epochs:
        lr = base_lr + (target_lr - base_lr) * (epoch / warmup_epochs)
    else:
        lr = target_lr
    print("calculated lr: ", lr)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (
            args.warm_epochs * total_batches
        )
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def plot_embeddings_tsne(ce, X, y, figname="tsne.png"):
    embs = ce.get_embeddings(X)
    embs = embs.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embs)
    df = pd.DataFrame({"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "label": y})

    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="x", y="y", hue="label", data=df, palette="tab10", alpha=0.8)
    plt.title("t-SNE Plot of Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Cell types", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Save the plot to a file
    plt.savefig(figname, bbox_inches="tight")
    plt.close()
