from abc import ABC
from abc import abstractmethod

from torch.utils.data import DataLoader

from typing import List

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

    """ 
    Predicting sentiment, 
    returning list of sentiment containing -1 or 1, 
    Logging of interemediate results as side effect.
    """
    def predict(test_data: DataLoader) -> List[int]:
        pass

    """
    Every implementation should make use of a config propery whic is a dictionary containing all
    the parameters in the mdoel which can be chhanged. For example, the learning rate, train/test split ration etc.
    Also, those parameters will be shown int frontend.
    """
    @property
    @abstractmethod
    def parameters(self) -> dict:
        ...


