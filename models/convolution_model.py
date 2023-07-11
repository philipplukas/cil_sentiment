import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import Model
from .word_embedder import WordEmbedder

class CNN(nn.Module):

    def __init__(self, embed_dim: int, max_words: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_words = max_words

        self.conv1 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=6,
            kernel_size=4,
            dtype=torch.float64
        )
        self.pool1 = nn.MaxPool1d(
            kernel_size=2
        )
        self.fc1 = nn.Linear((6 * (self.max_words - 3)), 100, dtype=torch.float64)
        self.fc1_fudge = nn.Linear(embed_dim * max_words, 100, dtype=torch.float64)
        self.fc2 = nn.Linear(100, 1, dtype=torch.float64)

    def forward(self, x):
        x = x.view(-1, self.embed_dim * self.max_words)
        #x = F.leaky_relu(self.conv1(x))
        #x = self.pool1(x)
        #x = x.view(-1, (6 * (self.max_words - 3)))
        #x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc1_fudge(x))
        x = self.fc2(x)
        return x

class ConvolutionModel(Model):

    def __init__(self, device: str = 'cpu'):
        self.embedder = WordEmbedder('data/embeddings/glove.twitter.27B.25d.txt')
        self.device = device
        self.network = None

    def train(self, data: DataLoader, iterations=30000, batch_size=100, lr=1e-1):

        X = np.array(self.embedder.embed_dataset(data['tweet']))
        Y = np.array(data['sent'])

        self.network = CNN(self.embedder.dimension, len(X[0][0]))

        self.network.train()
        self.network.to(self.device)

        print(f'Training on device: {self.device}.')

        optimiser = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-7)

        for i in range(iterations):

            samples = torch.randint(len(X), (batch_size,))
            x = torch.from_numpy(X[samples]).to(self.device)
            y = torch.from_numpy(Y[samples]).to(self.device)

            optimiser.zero_grad()
            loss = torch.abs(self.network(x) - y).mean()
            loss.backward()
            optimiser.step()

            print(f'Iteration: {i}/{iterations}, Training loss: {loss}.')

        self.network.eval()
        self.network.to('cpu')

    def evaluate(self, data: DataLoader) -> float:
        Y = self.predict(data)
        return len([Y for y0, y1 in zip(data['sent'], Y) if y0 == y1]) / len(Y)

    def predict(self, data: DataLoader) -> list[int]:
        X = torch.from_numpy(data['tweet'])
        return self.network(X).detach().numpy()