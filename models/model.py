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
    def train(train_data: DataLoader):
        pass

    """ 
    Evaluating model using test data, 
    return accruacy as result, 
    Logging of interemediate results as side effect.
    """
    @abstractmethod
    def evaluate(test_data: DataLoader) -> float:
        pass

    """ 
    Predicting sentiment, 
    returning list of sentiment containing -1 or 1, 
    Logging of interemediate results as side effect.
    """
    def predict(test_data: DataLoader) -> List[int]:
        pass

    """
    Train the model based on a portion of the given dataset,
    reserving another portion for subsequent testing.
    @param: data The data used for model training and validation.
    @param: p The portion of data points reserved for the validation set (default: 0.1).
    @return: The validation accuracy of the model.
    """
    @final
    def train_and_evaluate(self, data: DataLoader, p: float = 0.1) -> float:
        test_data, train_data = partition_dataset(data, p)
        self.train(train_data)
        return self.evaluate(test_data)

    """
    
    @param: data The data used for model training and validation.
    @param: k The number of times to train the model on different data (default: 5).
    @param: p The portion of data points reserved for the validation set (default: 0.1).
    @return: The validation accuracy of the model.
    """
    @final
    def cross_validate(self, data: DataLoader, k: int = 5, p: float = 0.1) -> float:
        # Average the results from k calls to train_and_evaluate,
        # where each training attempt should use a different
        # (but possibly overlapping) portion of the dataset.
        return np.mean(self.train_and_evaluate(data, p) for _ in range(k))

"""
Partition a dataset into two parts at random.
@param: data The data to partition.
@param: p The portion of data points reserved for the validation set (default: 0.1).
@return A tuple of the training and validation sets respectively.
"""
def partition_dataset(data: DataLoader, p: float = 0.1) -> (DataLoader, DataLoader):
    test_ids = np.random.randint(0, len(data), p * len(data))
    train_ids = np.setdiff1d(range(len(data)), test_ids)
    return data[test_ids], data[train_ids]