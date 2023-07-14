from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

from .model import Model

class BagOfWords(Model):
    cv = CountVectorizer()
    clf = None

    def train(self, data: DataLoader):
        counts = self.cv.fit_transform(data['tweet'])
        self.clf = SGDClassifier(loss='hinge')\
            .fit(counts, data['sent'])

    def predict(self, data: DataLoader) -> list[int]:
        counts = self.cv.transform(data['tweet'])
        return self.clf.predict(counts)