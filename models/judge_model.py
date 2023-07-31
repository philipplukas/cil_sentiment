import pickle
import numpy as np
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

from .model import Model, partition_dataset

class JudgeModel(Model):
    """
    A model for combining other models,
    and choosing one in each individual case based on word count vectors.
    """

    cv = CountVectorizer()
    clf = None

    def __init__(self, models: list[Model]):
        """
        @param models: The collection of untrained underlying models to invoke.
        """
        self.models = models

    def train(self, data: DataLoader, p=0.05):

        # Train the underlying models and the judge on different datasets.
        judge_data, model_data = partition_dataset(data, p)

        # Train the underlying models.
        for model in self.models:
            model.train(model_data)

        X, Y = judge_data['tweet'], judge_data['sent']
        counts = self.cv.fit_transform(X)

        # Predict each sample according to each underlying model.
        predictions = np.transpose([model.predict({'tweet': X, 'sent': Y})
            for model in self.models])

        # Determine which models were correct in which cases.
        scores = np.transpose([[1 if p == y else 0 for p in pred]
            for pred, y in zip(predictions, Y)])

        # Learn to predict which model is correct based off word count vectors.
        self.clf = [SGDClassifier(loss='hinge')
            .fit(counts, s)
            for s in scores]

    def predict(self, data: DataLoader):

        counts = self.cv.transform(data['tweet'])
        scores = np.transpose([clf.predict(counts) for clf in self.clf])

        # Predict each sample according to each underlying model.
        predictions = np.transpose([model.predict(data) for model in self.models])

        # Use word count vectors to determine which model to use in each case.
        return [max(zip(pred, scores), key=lambda x: x[1])[0]
            for pred, scores in zip(predictions, scores)]

    def reset(self):
        for model in self.models:
            model.reset()

    def save(self, file: str):
        with open(f"{file}-cv", "wb") as f:
            pickle.dump(self.cv, f)
        with open(f"{file}-clf", "wb") as f:
            pickle.dump(self.clf, f)
        for i, model in enumerate(self.models):
            model.save(f"{file}:{i}")

    def load(self, file: str):
        with open(f"{file}-cv", "rb") as f:
            self.cv = pickle.load(f)
        with open(f"{file}-clf", "rb") as f:
            self.clf = pickle.load(f)
        for i, model in enumerate(self.models):
            model.load(f"{file}-{i}.pt")