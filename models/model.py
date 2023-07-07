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
    @abstractmethod
    def evaluate(self, test_data: DataLoader) -> float:
        pass

    """ 
    Predicting sentiment, 
    returning list of sentiment containing -1 or 1, 
    Logging of intermediate results as side effect.
    """
    def predict(self, test_data: DataLoader) -> List[int]:
        pass

    @final
    def train_and_evaluate(self, data: DataLoader, p: float = 0.1) -> float:
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
    def cross_validate(self, data: DataLoader, k: int = 5, p: float = 0.1) -> float:
        """
        Evaluate the model using k-fold cross-validation.
        @param data: The data used for model training and validation.
        @param k: The number of times to train the model on different data (default: 5).
        @param p: The portion of data points reserved for the validation set (default: 0.1).
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
    @param p: The portion of data points reserved for the validation set (default: 0.1).
    @return: A tuple of the training and validation sets respectively.
    """
    test_ids = np.random.randint(0, len(data), int(p * len(data)))
    train_ids = np.setdiff1d(range(len(data)), test_ids)
    return data[test_ids], data[train_ids]