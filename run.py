
from data.data_set import TweetData, ResultData
from models.transformers.pretrained_sentiment import RobertaBaseSentiment
from models.transformers.finetuned_sentiment import RobertaBaseFinetuned
from torch.utils.data import DataLoader

import torch
import logging

# Import the W&B Python Library 
import wandb

from datasets import Dataset

# For now, log all levels
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

MODEL = "roberta-finetuned"
USE_WANDB = True

if USE_WANDB:

  # 1. Start a W&B Run
  run = wandb.init(
    project="cil-sentiment",
    notes="My first experiment",
    tags=["test"]
  )

  # 2. Capture a dictionary of hyperparameters
  wandb.config = {
    "train_test_ratio": 0.001
  }


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if MODEL == "roberta-zero_shot":

    data = TweetData("debug_train")
    model = RobertaBaseSentiment(wandb.config, device)

    # For inference, no batching required.
    loader = DataLoader(data, batch_size=None)

    accuracy = model.evaluate(loader)
    print("Accuracy of zero-shot classification: {}".format(accuracy))

    #results = model.predict(loader)
    #ResultData(results).store("twitterbert-pretrained")

elif MODEL == "roberta-finetuned":

    data = TweetData("train_sample")
    model = RobertaBaseFinetuned(wandb.config, device)

    # Turn it into a huggingface dataset object to access more powerful data manipulation methods.
    dataset = Dataset.from_list(data)
    
    # Finetune model
    model.train(dataset, wandb.log)

    # For inference, no batching required.
    # loader = DataLoader(data, batch_size=None)

    #accuracy = model.evaluate(loader)
    #print("Accuracy of zero-shot classification: {}".format(accuracy))
