from torch.utils.data import DataLoader

from .model import Model

class DummyModel(Model):

    def train(self, data: DataLoader):
        pass

    def predict(self, data: DataLoader) -> list[int]:
        return [1] * len(data['tweet'])