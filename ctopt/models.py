"""
@File    :   models.py
@Time    :   2024/08/27 15:40:19
@Author  :   Nikola Milicevic
@Version :   1.0
@Contact :   nikolamilicevic@genomics.cn
@Desc    :   None
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.1):
        layers = []
        in_dim = input_dim
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2.")

        step = (input_dim - hidden_dim) // (n_layers - 1)
        hidden_sizes = [input_dim - i * step for i in range(1, n_layers)]
        print("Hidden sizes: ", hidden_sizes)

        for hidden_dim_size in hidden_sizes:
            layers.append(torch.nn.Linear(in_dim, hidden_dim_size))
            # added batch norm before activation func.: https://arxiv.org/abs/1502.03167
            layers.append(torch.nn.BatchNorm1d(hidden_dim_size))
            layers.append(nn.LeakyReLU())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim_size

        layers.append(torch.nn.Linear(in_dim, hidden_dim))
        super().__init__(*layers)


class DeepEnc(nn.Module):
    def __init__(
        self, input_dim, emb_dim, num_classes, encoder_depth=4, classifier_depth=2, head_type="mlp"
    ):
        """Implementation of deep encoder consisted of: 1. Input layer MLP, 2. Hidden layer MLP 3. Linear.
        Args:
            input_dim (int): size of the inputs
            emb_dim (int): dimension of the embedding space
            encoder_depth (int, optional): number of layers of the encoder MLP. Defaults to 4.
        """
        super().__init__()
        self.encoder = MLP(input_dim, emb_dim, encoder_depth, dropout=0.4)

        if head_type == "linear":
            self.head = nn.Linear(emb_dim, num_classes)
        elif head_type == "mlp":
            # self.head = nn.Sequential(
            #     nn.Linear(emb_dim, emb_dim),
            #     nn.LeakyReLU(),
            #     nn.Linear(emb_dim, num_classes),
            # )
            self.head = MLP(emb_dim, num_classes, classifier_depth, dropout=0.1)
        else:
            raise NotImplementedError(f"Not supported head type: {head_type}")
        # initialize weights
        self.encoder.apply(self._init_weights)
        self.head.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # torch.nn.init.xavier_uniform_(module.weight)
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.01)

    def forward(self, x, classify=False):
        embeddings = self.encoder(x)
        if classify:
            return self.head(F.normalize(embeddings, dim=1))
        embeddings = F.normalize(embeddings, dim=1)
        return embeddings

    def get_embeddings(self, input_data):
        embeddings = self.encoder(input_data)
        norm = F.normalize(embeddings, dim=1)
        return norm

    def get_log_reg(self, input_data):
        embeddings = self.get_embeddings(input_data)
        log_prediction = torch.softmax(self.head(embeddings), dim=1)
        # log_prediction = torch.softmax(embeddings, dim=1)
        return log_prediction

    def freeze_encoder_weights(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
