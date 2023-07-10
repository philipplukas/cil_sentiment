import random

import numpy as np

from data.data_set import TweetData, ResultData
from models.bag_of_words_model import BagOfWords
from models.transformers.pretrained_sentiment import RobertaBaseTweetSentiment, RobertaBaseSentiment
from models.transformers.finetuned_sentiment import RobertaBaseTweetFinetuned, RobertaBaseFinetuned
from torch.utils.data import DataLoader
import nltk

import torch
import logging

# Import the W&B Python Library 
import wandb

from datasets import Dataset

# Log info level as well
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

MODEL = "bag-of-words"
PRETRAINED_ON_TWEETS = True
USE_WANDB = False

if USE_WANDB:

  wandb.login()

  # 1. Start a W&B Run
  run = wandb.init(
    project="cil-sentiment",
    notes=MODEL
    #tags=["test"]
  )

  # 2. Capture a dictionary of hyperparameters
  wandb.config = {
    "train_test_ratio": 0.001
  }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure consistent seed for reproducibility.
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

nltk.download('wordnet')


if MODEL == "roberta-zero_shot":

    data = TweetData("eval_train")

    if PRETRAINED_ON_TWEETS:
      model = RobertaBaseTweetSentiment(wandb.config, device)
    else:
      model = RobertaBaseSentiment(wandb.config, device)

    # For inference, no batching required.
    loader = DataLoader(data, batch_size=None)

    accuracy, confusion_matrix = model.evaluate(loader)
    print("Accuracy of zero-shot classification: {}".format(accuracy))
    wandb.run.summary["accuracy"] = accuracy
    wandb.run.summary["confusion_matrix"] = confusion_matrix

    test_data = TweetData("test")
    loader = DataLoader(test_data, batch_size=None)

    logging.info("Start prediction on test dataset")
    results = model.predict(loader)
    logging.info("Finished prediction step")

    ResultData(results).store("tweetbert-pretrained")

    #results_artifact = wandb.Artifact('pretrained_results', type='result_file')
    #results_artifact.add_file(file_path)

    #logging.info("log artifact")
    #wandb.log_artifact(results_artifact)

elif MODEL == "roberta-finetuned":

    data = TweetData("train_sample")

    if PRETRAINED_ON_TWEETS:
      model = RobertaBaseTweetFinetuned(wandb.config, device)
    else:
      model = RobertaBaseFinetuned(wandb.config, device)
       

    # Turn it into a huggingface dataset object to access more powerful data manipulation methods.
    dataset = Dataset.from_list(data)
    
    # Finetune model
    model.train(dataset, wandb.log)

    # For inference, no batching required.
    # loader = DataLoader(data, batch_size=None)
    test_data = TweetData("test")
    loader = DataLoader(test_data, batch_size=None)

    logging.info("Start prediction on test dataset")
    results = model.predict(loader)
    logging.info("Finished prediction step")
 
    ResultData(results).store("tweetbert-finetuned")

    #accuracy = model.evaluate(loader)
    #print("Accuracy of zero-shot classification: {}".format(accuracy))

elif MODEL == "bag-of-words":

    data = TweetData("train_sample")
    model = BagOfWords()
    accuracy = model.cross_validate(data)
    print(f"Accuracy: {accuracy*100:.2F}%")