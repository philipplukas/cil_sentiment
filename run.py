
from data.data_set import TweetData
from models.transformers.pretrained_sentiment import RobertaBaseSentiment
from torch.utils.data import DataLoader

data = TweetData("train_sample")
model = RobertaBaseSentiment()
loader = DataLoader(data, batch_size=None)

accuracy = model.evaluate(loader)
print("Accuracy of zero-shot classification: {}".format(accuracy))

