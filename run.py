
from data.data_set import TweetData, ResultData
from models.transformers.pretrained_sentiment import RobertaBaseSentiment
from torch.utils.data import DataLoader

import torch
import logging


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = TweetData("debug_train")
model = RobertaBaseSentiment(device)

# For inference, no batching required.
loader = DataLoader(data, batch_size=None)

accuracy = model.evaluate(loader)
print("Accuracy of zero-shot classification: {}".format(accuracy))

#results = model.predict(loader)
#ResultData(results).store("twitterbert-pretrained")
