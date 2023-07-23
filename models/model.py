from typing import List, final
from abc import ABC
from abc import abstractmethod

import numpy as np
from torch.utils.data import DataLoader


""" Abstract class for our model """
class Model(ABC):

    """ 
    Training model using given pytorch dataloader, 
    logging training progress as side effect. 
    """
    @abstractmethod
    def train(self, train_data: DataLoader):
        pass

    """ 
    Evaluating model using test data, 
    return accuracy as result, 
    Logging of intermediate results as side effect.
    """
    def evaluate(self, data: DataLoader) -> float:
        Y = self.predict(data)
        return len([Y for y0, y1 in zip(data['sent'], Y) if y0 * y1 > 0]) / len(Y)

    """ 
    Predicting sentiment, 
    returning list of sentiment containing -1 or 1, 
    Logging of intermediate results as side effect.
    """
    def predict(self, test_data: DataLoader) -> List[int]:
        pass

    """
    Save the current state of the model to a file.
    Used when the model needs to be reused in a separate runtime without retraining.
    @param file: The name of the file to save the model weights to.
    """
    def save(self, file: str):
        pass

    """
    Load a previous state of the model from a file.
    Used when the model needs to be reused in a separate runtime without retraining.
    @param file: The name of the file to load the model weights from.
    """
    def load(self, file: str):
        pass

    @final
    def train_and_evaluate(self, data: DataLoader, p: float = 0.05) -> float:
        """
        Train the model based on a portion of the given dataset,
        reserving another portion for subsequent testing.
        @param data: The data used for model training and validation.
        @param p: The portion of data points reserved for the validation set.
        @return: The validation accuracy of the model.
        """
        test_data, train_data = partition_dataset(data, p)
        self.train(train_data)
        return self.evaluate(test_data)


    @final
    def cross_validate(self, data: DataLoader, k: int = 5, p: float = 0.05) -> float:
        """
        Evaluate the model using k-fold cross-validation.
        @param data: The data used for model training and validation.
        @param k: The number of times to train the model on different data.
        @param p: The portion of data points reserved for the validation set.
        @return: The validation accuracy of the model.
        """
        # Average the results from k calls to train_and_evaluate,
        # where each training attempt should use a different
        # (but possibly overlapping) portion of the dataset.
        return np.mean([self.train_and_evaluate(data, p) for _ in range(k)])

def partition_dataset(data: DataLoader, p: float = 0.1) -> (DataLoader, DataLoader):
    """
    Partition a dataset into two parts at random.
    @param data: The data to partition.
    @param p: The portion of data points reserved for the validation set.
    @return: A tuple of the training and validation sets respectively.
    """
    test_ids = np.random.randint(0, len(data), int(p * len(data)))
    train_ids = np.setdiff1d(range(len(data)), test_ids)
    return data[test_ids], data[train_ids]