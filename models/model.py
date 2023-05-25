from abc import ABC
from abc import abstractmethod

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
    def evaluate(test_data: DataLoader, accuracy: float):
        pass
