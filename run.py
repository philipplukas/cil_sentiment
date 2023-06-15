
from data.data_set import TweetData, ResultData
from models.transformers.pretrained_sentiment import RobertaBaseSentiment
from models.transformers.finetuned_sentiment import RobertaBaseFinetuned
from torch.utils.data import DataLoader

import torch
import logging

from datasets import Dataset

MODEL = "roberta-finetuned"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if MODEL == "roberta-zero_shot":

    data = TweetData("debug_train")
    model = RobertaBaseSentiment(device)

    # For inference, no batching required.
    loader = DataLoader(data, batch_size=None)

    accuracy = model.evaluate(loader)
    print("Accuracy of zero-shot classification: {}".format(accuracy))

    #results = model.predict(loader)
    #ResultData(results).store("twitterbert-pretrained")

elif MODEL == "roberta-finetuned":

    data = TweetData("debug_train")
    model = RobertaBaseFinetuned(device)

    # Turn it into a huggingface dataset object to access more powerful data manipulation methods.
    dataset = Dataset.from_list(data)
    
    # Finetune model
    model.train(dataset)

    # For inference, no batching required.
    # loader = DataLoader(data, batch_size=None)

    #accuracy = model.evaluate(loader)
    #print("Accuracy of zero-shot classification: {}".format(accuracy))
