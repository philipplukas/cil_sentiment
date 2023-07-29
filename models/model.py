from typing import List, final
from abc import ABC
from abc import abstractmethod
import numpy as np
from torch.utils.data import DataLoader

class Model(ABC):
    """ Abstract class for our model """

    @abstractmethod
    def train(self, train_data: DataLoader):
        """
        Training model using given pytorch dataloader,
        logging training progress as side effect.
        """
        pass

    def evaluate(self, data: DataLoader) -> float:
        """
        Evaluating model using test data,
        return accuracy as result,
        Logging of intermediate results as side effect.
        """
        Y = self.predict(data)
        return len([Y for y0, y1 in zip(data['sent'], Y) if y0 * y1 > 0]) / len(Y)

    def predict(self, test_data: DataLoader) -> List[int]:
        """
        Predicting sentiment,
        returning list of sentiment containing -1 or 1,
        Logging of intermediate results as side effect.
        """
        pass

    def reset(self):
        """
        Reset the model to undo all training.
        """
        pass

    def save(self, file: str):
        """
        Save the current state of the model to a file.
        Used when the model needs to be reused in a separate runtime without retraining.
        @param file: The name of the file to save the model weights to.
        """
        pass

    def load(self, file: str):
        """
        Load a previous state of the model from a file.
        Used when the model needs to be reused in a separate runtime without retraining.
        @param file: The name of the file to load the model weights from.
        """
        pass

    @final
    def train_and_evaluate(self, data: DataLoader, p: float = 0.02) -> float:
        """
        Train the model based on a portion of the given dataset,
        reserving another portion for subsequent testing.
        @param data: The data used for model training and validation.
        @param p: The portion of datapoints reserved for the validation set.
        @return: The validation accuracy of the model.
        """
        test_data, train_data = partition_dataset(data, p)
        self.train(train_data)
        return self.evaluate(test_data)

    @final
    def train_and_track(self, data: DataLoader, n: int = 10, p: float = 0.01) -> list[float]:
        """
        Train the model based on a portion of the given dataset,
        reserving another potion for ongoing testing.
        Performs multiple rounds of training.
        After each round of training, reports the validation accuracy at that point.
        @param data: The data used for model training and validation.
        @param n: The number of training iterations to undergo.
        @param p: The portion of datapoints reserved for the validation set.
        @return: The validation accuracy after each training iteration.
        """
        test_data, train_data = partition_dataset(data, p)
        return [(self.train(train_data), self.evaluate(test_data))[1]
            for _ in range(n)]

    @final
    def cross_validate(self, data: DataLoader, k: int = 5, p: float = 0.05) -> float:
        """
        Evaluate the model using k-fold cross-validation.
        @param data: The data used for model training and validation.
        @param k: The number of times to train the model on different data.
        @param p: The portion of datapoints reserved for the validation set.
        @return: The validation accuracy of the model.
        """
        # Average the results from k calls to train_and_evaluate,
        # where each training attempt should use a different
        # (but possibly overlapping) portion of the dataset.
        return np.mean([
            (self.train_and_evaluate(data, p), self.reset())[0]
            for _ in range(k)])

def partition_dataset(data: DataLoader, p: float = 0.1) -> (DataLoader, DataLoader):
    """
    Partition a dataset into two parts at random.
    @param data: The data to partition.
    @param p: The portion of datapoints reserved for the validation set.
    @return: A tuple of the training and validation sets respectively.
    """
    test_ids = np.random.randint(0, len(data), int(p * len(data)))
    train_ids = np.setdiff1d(range(len(data)), test_ids)
    return data[test_ids], data[train_ids]