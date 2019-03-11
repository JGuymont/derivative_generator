import time
import torch
import torch.nn as nn

from .regressor import BaseRegressor


class MLP(BaseRegressor):

    def __init__(self, embedding_dims, n_continuous_features, hidden_dim, output_dim, 
                 loss='mse', optimizer='adam', lr=0.001, n_epochs=10, device='cpu', **kargs):
        super(MLP, self).__init__(loss=loss, optimizer=optimizer, lr=lr, **kargs)

        self.n_epochs = n_epochs
        self.device = device

        self.n_continuous_features = n_continuous_features
        self.n_categorical_features = len(embedding_dims)
        self.hidden_dim = hidden_dim

        # Embedding layers
        self.embedding_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in embedding_dims])

        # Compute new input dimension
        self.n_embeddings = sum([y for _, y in embedding_dims])
        self.input_dim = self.n_continuous_features + self.n_embeddings

        self.normalize = nn.BatchNorm1d(n_continuous_features)

        self.input2hidden = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.hidden2output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

        self.optimizer = self._get_optimizer(optimizer, lr)

    def forward(self, X_num, X_cat):

        X_cat = [emb_layer(X_cat[:, i]) for i, emb_layer in enumerate(self.embedding_layers)]
        X_cat = torch.cat(X_cat, 1)

        X_num = self.normalize(X_num)

        X = torch.cat([X_cat, X_num], 1)

        hidden = self.input2hidden(X)
        output = self.hidden2output(hidden)
        return output
