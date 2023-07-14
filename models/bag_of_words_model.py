from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

from .model import Model

class BagOfWords(Model):
    """
    A model for classifying the sentiment (positive/negative) of a Tweet.
    Counts the number of occurrences of each word in each tweet, and performs a linear regression.
    """

    # Mechanism for counting the number of occurrences of each word in a sentence.
    cv = CountVectorizer()
    # Classifier for performing linear regression on count vectors.
    clf = None

    def train(self, data: DataLoader):
        counts = self.cv.fit_transform(data['tweet'])
        self.clf = SGDClassifier(loss='hinge')\
            .fit(counts, data['sent'])

    def predict(self, data: DataLoader) -> list[int]:
        counts = self.cv.transform(data['tweet'])
        return self.clf.predict(counts)