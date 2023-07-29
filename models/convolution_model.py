import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import Model
from .word_embedder import WordEmbedder

class CNN(nn.Module):
    """
    A PyTorch neural network architecture intended for use in sentiment classification.
    Features a single 1D convolution layer followed by two fully connected layers.
    """

    # The length of kernel used in the convolution layer.
    kernel_size = 7
    # The number of unique filters produced by the convolution layer.
    num_filters = 32

    def __init__(self, embed_dim: int, max_words: int):
        """
        @param embed_dim: The dimensionality of the word embeddings.
        @param max_words: The (padded) number of words in each sentence.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_words = max_words

        self.conv1 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            dtype=torch.float64
        )

        # Calculate the dimensionality of the first fully-connected hidden layer.
        self.fc_dim = self.max_words * self.num_filters

        self.fc1 = nn.Linear(self.fc_dim, 100, dtype=torch.float64)
        self.fc2 = nn.Linear(100, 1, dtype=torch.float64)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        # Flatten results from all filters into a single vector.
        x = x.view(-1, self.fc_dim)
        x = F.leaky_relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x

class ConvolutionModel(Model):
    """
    A model for classifying the sentiment (positive/negative) of a Tweet.
    Uses ADAM to train a neural network featuring a convolution layer.
    """

    def __init__(self, device: str = 'cpu'):
        self.embedder = WordEmbedder('data/embeddings/glove.twitter.27B.200d.txt')
        self.device = device
        self.max_words = 64
        self.network = CNN(self.embedder.dimension, self.max_words)

    def train(self, data: DataLoader, iterations=10, batch_size=50, lr=1e-3):
        """
        @param data: The labelled training data for training the network.
        @param iterations: The total number of iterations (not number of epochs).
        @param batch_size: The number of training examples used in each iteration.
        @param lr: The learning rate at which the NN is trained.
        """

        X = np.array(data['tweet'])
        Y = np.array(data['sent'])

        self.network.train()
        self.network.to(self.device)

        print(f'Training on device: {self.device}.')

        optimiser = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-7)

        for i in range(iterations):
            loss = self.step(X, Y, batch_size, optimiser)
            print(f'Iteration: {i}/{iterations}, Cross-entropy training loss: {loss}.')

    def step(self, X, Y, batch_size, optimiser):
        """
        Perform a single step of stochastic gradient descent.
        @param X: The entire set of sentences used for training.
        @param Y: The corresponding labels for X.
        @param batch_size: The number of training examples used in each iteration.
        @param optimiser: The gradient-descent-based PyTorch optimiser.
        """

        # Select a random subset of the data for this iteration.
        samples = torch.randint(len(X), (batch_size,))

        # Embed this subset of the data into a latent space.
        border = (self.network.kernel_size - 1) // 2
        embedded = self.embedder.embed_dataset(X[samples], self.max_words, border)

        # Load training examples on the GPU.
        x = torch.from_numpy(np.array(embedded)).to(self.device)
        y = torch.from_numpy(Y[samples]).to(self.device)

        # Perform the forward pass.
        optimiser.zero_grad()
        result = torch.flatten(self.network(x))

        # Calculate binary cross-entropy loss between predictions and the actual categories.
        loss = nn.BCELoss()(
            # Rescale values to be between 0 and 1 (instead of -1 and 1).
            # This is required by PyTorch's inbuilt BCELoss function.
            torch.clamp(((result + 1.0) / 2.0), 0.0, 1.0).double(),
            torch.clamp((y + 1.0) / 2.0, 0.0, 1.0).double()
        )

        # Perform the backward pass.
        loss.backward()
        optimiser.step()

        return loss

    def predict(self, data: DataLoader, batch_size=100) -> list[int]:
        self.network.eval()
        self.network.to('cpu')
        return [y
            for i in range(0, len(data['tweet']), batch_size)
            for y in self.predict_batch(data['tweet'][i:i+batch_size])]

    def predict_batch(self, X):
        # Embed the sentences into a latent space.
        border = (self.network.kernel_size - 1) // 2
        embedded = self.embedder.embed_dataset(X, self.max_words, border)
        x = torch.from_numpy(np.array(embedded))
        # Use the trained NN to make sentiment predictions.
        y = torch.flatten(self.network(x)).detach().numpy()
        # Use only the sign of the output to make categorical predictions.
        return [np.sign(yi) for yi in y]

    def reset(self):
        self.network = CNN(self.embedder.dimension, self.max_words)

    def save(self, file: str):
        torch.save(self.network.state_dict(), file)

    def load(self, file: str):
        self.network.load_state_dict(torch.load(file))