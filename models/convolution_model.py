import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import Model
from .word_embedder import WordEmbedder

class CNN(nn.Module):

    kernel_size = 6
    num_filters = 32

    def __init__(self, embed_dim: int, max_words: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_words = max_words

        self.conv1 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            dtype=torch.float64
        )

        self.fc1_dim_in = (self.max_words - self.kernel_size + 1) * self.num_filters
        fc1_dim_out = 100

        self.fc1 = nn.Linear(self.fc1_dim_in, fc1_dim_out, dtype=torch.float64)
        self.fc2 = nn.Linear(fc1_dim_out, 1, dtype=torch.float64)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = x.view(-1, self.fc1_dim_in)
        x = F.leaky_relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x)) * 2 - 1
        return x

class ConvolutionModel(Model):

    def __init__(self, device: str = 'cpu'):
        self.embedder = WordEmbedder('data/embeddings/glove.twitter.27B.200d.txt')
        self.device = device
        self.network = None
        self.max_words = 64

    def train(self, data: DataLoader, iterations=10000, batch_size=200, lr=1e-3):

        X = np.array(data['tweet'])
        Y = np.array(data['sent'])

        self.network = CNN(self.embedder.dimension, self.max_words)

        self.network.train()
        self.network.to(self.device)

        print(f'Training on device: {self.device}.')

        optimiser = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-7)
        bce = nn.BCELoss()

        for i in range(iterations):

            samples = torch.randint(len(X), (batch_size,))
            embedded = self.embedder.embed_dataset((X[samples]), self.max_words)
            x = torch.from_numpy(np.array(embedded)).to(self.device)
            y = torch.from_numpy(Y[samples]).to(self.device)

            optimiser.zero_grad()
            result = torch.flatten(self.network(x))

            loss = bce(
                torch.clamp(((result + 1.0) / 2.0), 0.0, 1.0).double(),
                torch.clamp((y + 1.0) / 2.0, 0.0, 1.0).double()
            )

            loss.backward()
            optimiser.step()

            print(f'Iteration: {i}/{iterations}, Cross-entropy training loss: {loss}.')

    def predict(self, data: DataLoader) -> list[int]:

        self.network.eval()
        self.network.to('cpu')

        embedded = self.embedder.embed_dataset(data['tweet'], self.max_words)
        X = torch.from_numpy(np.array(embedded))
        Y = torch.flatten(self.network(X)).detach().numpy()

        return [np.sign(y) for y in Y]