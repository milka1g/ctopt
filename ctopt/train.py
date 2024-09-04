from pathlib import Path
import datetime
import logging
from typing import Optional
import time
import random
import os
import socket


import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd
import scipy
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
import wandb

from ctopt.contrastive_augmentation import augment_data
from ctopt.preprocessing import preprocess
from ctopt.losses import SupConLoss
from ctopt.datasets import ContrastiveDataset, EmbDataset
from ctopt.models import MLP, DeepEnc
from ctopt.utils import adjust_learning_rate, plot_embeddings_tsne, EarlyStopper, plot_embeddings_umap


timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
logger = logging.getLogger(__name__)


def init_wandb(model_folder="weights", name="", model=None):
    wandb_output_dir = os.path.join(model_folder, "wandb_home")
    Path(wandb_output_dir).mkdir(parents=True, exist_ok=True)
    wandb.require("core")
    run = wandb.init(
        project="contrastive_lr",
        notes=socket.gethostname(),
        name=f"{name}_contrastive_lr_{timestamp}",
        group="classifier",
        dir=wandb_output_dir,
        job_type="training",
        reinit=True,
    )
    if model:
        wandb.watch(model, log="all")

    return run


class CombinedLoss(nn.Module):
    def __init__(self, temperature=1, contrastive_weight=0.75, class_weights=None):
        """Loss for contrastive learning using cosine distance as similarity metric.

        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = torch.tensor(temperature).to(self.device)
        self.contrastive_weight = torch.tensor(contrastive_weight).to(self.device)
        self.class_weights = class_weights

    def _forward_2(self, z_i, z_j):
        """Compute loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)

        # First, one needs to apply an L2 normalization to the features
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # compute similarity between the sample's embeddings
        z = torch.cat([z_i, z_j], dim=0)
        similarity = cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (
            (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool))
            .float()
            .to(self.device)
        )
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss

    def forward(self, z_i, z_j, z_k, anchor_target=None, log_reg=None):
        positive_loss = self._forward_2(z_i, z_j)
        negative_loss = 0
        if torch.is_tensor(z_k):
            negative_loss = 1
        else:
            for neg in z_k:
                negative_loss += self._forward_2(z_i, neg)
        contrastive_loss = positive_loss / negative_loss

        if anchor_target is None:
            log_loss = 0
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            log_loss = criterion(log_reg, anchor_target)
        log_reg_loss = contrastive_loss * self.contrastive_weight + log_loss * (
            1 - self.contrastive_weight
        )
        # return a tuple representing both contrastive and classification loss
        return (
            contrastive_loss * self.contrastive_weight,
            log_loss * (1 - self.contrastive_weight),
        )


class ContrastiveEncoder:
    def __init__(
        self,
        num_classes,
        num_of_negatives=2,
        batch_size=512,
        epochs=50,
        emb_dim=16,
        encoder_depth=4,
        classifier_depth=2,
        contrastive_only_perc=0.3,
        contrastive_weight=0.8,
        freeze_encoder=True,
        out_dim=1,
        freeze_hidden=True,
        verbose=0,
        class_weights=None,
        n_views=2,
        temperature=0.07,
    ):
        """Implementation of Contrastive learning encoder with logistic regression.
        It is done by minimizing the contrastive loss of a sample and its positive and negative view.

            Args:
                num_of_negatives (int, optional): number of random negative samples picket randomply from endpoint group
                                                  oposite to anchor.Defaults to 2.
                batch_size (int, optional): number of samples in the batch. Defaults to 512.
                epochs (int, optional): number of to train deep encoder. Defaults to 50.
                emb_dim (int, optional): Dimension of the output embeddings. Defaults to 16.
                encoder_depth (int, optional): Number of layers in the input MLP layer. Defaults to 4.
                classifier_depth (int, optional): Number of layers in the classifier MLP layer. Defaults to 2.
                contrastive_only_perc (float [0, 1], optional): % of epochs to use only contrastive loss for training
                contrastive_weight (float [0, 1], optional): Weight of contrastive loss  when combined in cross-entropy
                                                             loss of logistic regression with formula:
                                                             L_total = cw * L_contrastive + (1-cw) * L_classification
                freeze_encoder (Boolean, optional): Freeze weights of input layer (Multy-Layer Perceptron) after
                                                    contrastive_only_perc of epochs.
                freeze_hidden (Boolean, optional): Freeze weights of hidden layer (Multy-Layer Perceptron) after
                                                   contrastive_only_perc of epochs.
                out_dim (int, optional): Dimension of the output result. Defaults to 1.
        """
        self.num_of_negatives = num_of_negatives
        self.batch_size = batch_size
        self.epochs = epochs
        self.emb_dim = emb_dim
        self.encoder_depth = encoder_depth
        self.classifier_depth = classifier_depth
        self.out_dim = out_dim
        self.contrastive_only_perc = contrastive_only_perc
        self.contrastive_weight = contrastive_weight
        self.freeze_encoder = freeze_encoder
        self.freeze_hidden = freeze_hidden
        self.verbose = verbose
        self.class_weights = class_weights
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_views = n_views
        self.temperature = temperature
        self.num_classes = num_classes

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Instantiate and train DeepEnc that will try to minimize the loss between anchor and positive view
        (random sample from same endpoint group like anchor) and maximize the loss between anchor and negative view(s).

        Args:
            X (Pandas Dataframe): Rows are samples, columns are features
            y (Array): Binary targets for all samples in data
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.data = X_train
        self.y = y_train

        self.contr_ds_train = ContrastiveDataset(
            X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train,
            y_train,
            n_views=self.n_views,
        )
        self.contr_ds_val = ContrastiveDataset(
            X_val.to_numpy() if isinstance(X_val, pd.DataFrame) else X_val,
            y_val,
            n_views=self.n_views,
        )

        head_ds_train = ContrastiveDataset(
            X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train,
            y_train,
            n_views=1,
        )
        head_ds_val = ContrastiveDataset(
            X_val.to_numpy() if isinstance(X_val, pd.DataFrame) else X_val,
            y_val,
            n_views=1,
        )

        train_loader = DataLoader(
            self.contr_ds_train,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=3,
            shuffle=True,
        )

        val_loader = DataLoader(
            self.contr_ds_val,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=3,
        )

        head_train_loader = DataLoader(
            head_ds_train,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=3,
            shuffle=True,
        )

        head_val_loader = DataLoader(
            head_ds_val,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=3,
            shuffle=True,
        )

        self.model = DeepEnc(
            input_dim=self.contr_ds_train.shape[1],
            emb_dim=self.emb_dim,
            encoder_depth=self.encoder_depth,
            num_classes=self.num_classes,
            classifier_depth=self.classifier_depth,
            head_type='mlp'
        ).to(self.device)


        optimizer = Adam(self.model.encoder.parameters(), lr=0.0001, weight_decay=1e-2)

        # optimizer = SGD(
        #     self.model.parameters(),
        #     lr=0.1,
        #     momentum=0.9,
        #     weight_decay=1e-4,
        # )

        combined_loss = SupConLoss(
            temperature=self.temperature, base_temperature=self.temperature
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=3, factor=0.5
        )
        early_stopper = EarlyStopper(patience=5)
        # warmup_epochs = 5
        # base_lr = 0.0001
        # target_lr = 0.001
        print("Starting training...")
        # training encoder
        for epoch in range(1, self.epochs + 1):
            start = time.time()
            self.model.train()
            train_loss_contrastive = 0.0

            # if epoch < warmup_epochs:
            #     adjust_learning_rate(
            #         optimizer, epoch, warmup_epochs, base_lr, target_lr
            #     )
            for idx, (cells, labels) in enumerate(train_loader):
                cells = torch.cat([c for c in cells], dim=0)
                cells = cells.to(self.device)

                bsz = labels.shape[0]

                # warm-up learning rate
                # warmup_learning_rate(optimizer, epoch, idx, len(train_loader), optimizer)

                features = self.model(cells)
                feats = torch.split(features, [bsz] * self.n_views, dim=0)
                features = torch.cat([f.unsqueeze(1) for f in feats], dim=1)

                loss = combined_loss(features, labels)

                train_loss_contrastive += loss.item()

                optimizer.zero_grad()

                loss.backward()

                # update model weights
                optimizer.step()

            loss = train_loss_contrastive / len(train_loader)
            wandb.log({"Encoder training loss": loss})

            # validation
            self.model.eval()
            val_loss_contrastive = 0.0
            val_loss_classification = 0.0
            with torch.no_grad():
                for idx, (cells, labels) in enumerate(val_loader):
                    cells = torch.cat([c for c in cells], dim=0)
                    cells = cells.to(self.device)

                    bsz = labels.shape[0]

                    features = self.model(cells)
                    feats = torch.split(features, [bsz] * self.n_views, dim=0)
                    features = torch.cat([f.unsqueeze(1) for f in feats], dim=1)

                    val_loss_contrastive += combined_loss(features, labels).item()

            loss = val_loss_contrastive / len(val_loader)
            wandb.log({"Encoder validation loss": loss})
            if early_stopper.early_stop(loss):
                logger.warning(f"Early stopping in epoch {epoch}...")
                break
            # if epoch > warmup_epochs:
            scheduler.step(loss)

            end = time.time()
            elapsed = end - start
            print(f"Epoch {epoch} took {elapsed:.2f}s; lr = ", scheduler._last_lr)

        # training classifier head
        early_stopper.reset()
        self.model.freeze_encoder_weights()
        ce_loss = nn.CrossEntropyLoss()  # Assuming classification task
        optimizer = Adam(self.model.head.parameters(), lr=0.0001, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=3, factor=0.2
        )
        # If the encoder is frozen, set it to evaluation mode to prevent batchnorm from updating running
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            acc = 0.0
            # self.model.head.train()
            # self.model.encoder.eval()
            running_loss = 0.0
            # training
            for idx, (cells, labels) in enumerate(head_train_loader):
                cells = cells.to(self.device)
                labels = labels.to(self.device)
                # warm-up learning rate
                # warmup_learning_rate(optimizer, epoch, idx, len(train_loader), optimizer)

                outputs = self.model(cells, classify=True)
                _, predicted = torch.max(outputs, 1)
                # feats = torch.split(features, [bsz] * self.n_views, dim=0)
                # features = torch.cat([f.unsqueeze(1) for f in feats], dim=1)

                loss = ce_loss(outputs, labels)
                acc_score = accuracy_score(
                    labels.cpu().numpy(), predicted.cpu().numpy()
                )
                acc += acc_score

                running_loss += loss.item()

                optimizer.zero_grad()

                loss.backward()

                # update model weights
                optimizer.step()

            loss = running_loss / len(head_train_loader)
            acc = acc / len(head_train_loader)
            wandb.log({"Classifier head training loss": loss})
            wandb.log({"Classifier head training accuracy": acc})

            # validation
            # self.model.head.eval()
            self.model.eval()
            head_val_loss = 0.0
            acc = 0.0
            with torch.no_grad():
                for idx, (cells, labels) in enumerate(head_val_loader):
                    cells = cells.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(cells, classify=True)
                    _, predicted = torch.max(outputs, 1)

                    loss = ce_loss(outputs, labels)
                    head_val_loss += loss.item()
                    acc_score = accuracy_score(
                        labels.cpu().numpy(), predicted.cpu().numpy()
                    )
                    acc += acc_score

            loss = head_val_loss / len(head_val_loader)
            acc = acc / len(head_val_loader)
            if early_stopper.early_stop(loss):
                logger.warning(
                    f"Early stopping classification head in epoch {epoch}..."
                )
                break
            scheduler.step(loss)
            wandb.log({"Classifier head validation loss": loss})
            wandb.log({"Classifier head validation accuracy": acc})
            print(f"Epoch classifier: {epoch}")

        print("finished")
        return self

    def load_model(self, path: str):
        """Creates DeepEncoder and loads it from path

        Args:
            path (str): path to the .pt file that has model params
        """
        self.model = DeepEnc(
            input_dim=self.contr_ds_train.shape[1],
            emb_dim=self.emb_dim,
            out_dim=self.out_dim,
            encoder_depth=self.encoder_depth,
            classifier_depth=self.classifier_depth,
        )
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def _transform_or_predict(self, get_emb_func, X, y=None):
        """Perform encoding on previously trained Contrastive encoder. If data is not provided it will process the
        same data used for fit()
        """

        dataset = EmbDataset(X)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                embeddings.append(get_emb_func(batch))
        embeddings = torch.cat(embeddings)
        return embeddings

    def predict(self, X: np.ndarray):
        preds = self._transform_or_predict(self.model.get_log_reg, X)
        preds = torch.argmax(preds, dim=1)
        return preds.cpu().numpy()

    def get_embeddings(self, X: np.ndarray):
        embs = self._transform_or_predict(self.model.get_embeddings, X)
        return embs

    def predict_proba(self, X: np.ndarray):
        preds = self._transform_or_predict(self.model.get_log_reg, X)
        return preds.cpu().numpy()

    def transform(self, X: np.ndarray):
        return self._transform_or_predict(self.model.get_embeddings, X)

    def set_params(self, **params):
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                pass

        return self

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def contrastive_process(
    sc_path: str,
    st_path: str,
    adata_sc: ad.AnnData,
    adata_st: ad.AnnData,
    annotation_sc: str,
    batch_size: int,
    epochs: int,
    embedding_dim: int,
    encoder_depth: int,
    classifier_depth: int,
    filename: str,
    augmentation_perc: float,
    wandb_key: str,
    n_views: int,
    temperature: float,
    annotation_st: str,
):
    if filename:
        file_handler = logging.FileHandler(filename)
        logger.addHandler(file_handler)

    wandb.login(key=wandb_key)
    fix_seed(0)
    adata_sc.X = adata_sc.layers["counts"]

    adata_sc = augment_data(
        adata_sc,
        adata_st,
        annotation=annotation_sc,
        percentage=augmentation_perc,
        logger=logger,
    )

    print("adata shape after aug:", adata_sc.shape)
    sc.pp.normalize_total(adata_sc, target_sum=1e6)
    sc.pp.log1p(adata_sc)
    # preprocess(adata_sc)

    # perform preprocessing like removing all 0 vectors, normalization and scaling

    X = adata_sc.X.toarray()
    logger.info("Input ready...")

    y = adata_sc.obs[annotation_sc]
    le = LabelEncoder()
    y_le = le.fit_transform(y)
    logger.info("Labels ready...")

    ce = ContrastiveEncoder(
        out_dim=len(le.classes_),
        batch_size=batch_size,
        epochs=epochs,
        emb_dim=embedding_dim,
        encoder_depth=encoder_depth,
        classifier_depth=classifier_depth,
        n_views=n_views,
        temperature=temperature,
        num_classes=len(le.classes_)
    )

    run = init_wandb(name=os.path.basename(st_path).replace(".h5ad", ""))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_le, test_size=0.1, random_state=42, stratify=y_le
    )
    logger.info(
        f"""Fitting a model with: \n - embedding dim: {embedding_dim} \n - encoder depth: {encoder_depth} \n - classifier depth: {classifier_depth} \n - epochs: {epochs} \n - n_views: {n_views} \n temp: {temperature} \n"""
    )
    ce.fit(X_train, y_train)

    plot_embeddings_umap(
        ce, X_test, le.inverse_transform(y_test), os.path.basename(sc_path).replace(".h5ad", "_tsne.png")
    )

    y_pred = ce.predict(X_test)

    y_true = y_test
    acc = accuracy_score(le.inverse_transform(y_true), le.inverse_transform(y_pred))
    f1 = f1_score(
        le.inverse_transform(y_true), le.inverse_transform(y_pred), average="macro"
    )

    logger.info("-------------Test data------------")
    logger.info(f"Accuracy: {acc}")
    logger.info(f"F1 macro score: {f1}")
    logger.info("----------------------------------")

    logger.info("-------------ST prediction------------")
    adata_st.var_names_make_unique()
    if not scipy.sparse.issparse(adata_st.X):
        adata_st.X = scipy.sparse.csr_matrix(adata_st.X)
        logger.info(f"Converted gene exp matrix of ST to csr_matrix")
    if annotation_st != "NaN":
        plot_embeddings_umap(
            ce, adata_st.X.toarray(), adata_st.obs[annotation_st].values, os.path.basename(st_path).replace(".h5ad", "_ST_tsne.png")
        )
    else:
        plot_embeddings_umap(
            ce, adata_st.X.toarray(), figname=os.path.basename(st_path).replace(".h5ad", "_ST_tsne.png")
        )
    y_pred = ce.predict(adata_st.X.toarray())
    adata_st.obs["contrastive"] = le.inverse_transform(y_pred)
    adata_st.obs.index.name = "cell_id"
    probabilities = ce.predict_proba(adata_st.X.toarray())
    adata_st.obs["entropy"] = entropy(probabilities, axis=1)
    df_probabilities = pd.DataFrame(
        data=probabilities, columns=le.classes_, index=adata_st.obs.index
    )
    wandb.unwatch()
    run.finish()
    return (df_probabilities, adata_st.obs["contrastive"])
