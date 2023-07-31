import os
import random

import numpy as np

from data.data_set import TweetData, ResultData
from models.bag_of_words_model import BagOfWords
from models.convolution_model import ConvolutionModel
from models.judge_model import JudgeModel
from models.pretrained_sentiment import RobertaBaseTweetSentiment, RobertaBaseSentiment
from models.finetuned_sentiment import RobertaBaseTweetFinetuned, RobertaBaseFinetuned
from torch.utils.data import DataLoader

import torch
import logging

# Import the W&B Python Library 
import wandb

from datasets import Dataset

# Log info level as well
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

MODEL = "convolution"
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

else:
    os.environ["WANDB_DISABLED"] = "true"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Ensure consistent seed for reproducibility.
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# EXPERIMENT 1: Pre-trained transformer
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


# EXPERIMENT 2: Fine-tuned transformer
elif MODEL == "roberta-finetuned":

    data = TweetData("train_sample")
    data = data[range(len(data))]

    if PRETRAINED_ON_TWEETS:
      model = RobertaBaseTweetFinetuned(wandb.config if USE_WANDB else None, device)
    else:
      model = RobertaBaseFinetuned(wandb.config, device)
    
    # Finetune model
    model.train(data, wandb.log if USE_WANDB else None)

    test_data = TweetData("test")
    test_data = test_data[range(len(test_data))]
    results = model.predict(test_data)
    ResultData(results).store("tweetbert-finetuned")

    #accuracy = model.evaluate(loader)
    #print("Accuracy of zero-shot classification: {}".format(accuracy))
    model.save("./data/models/tweetbert-finetuned.save")


# EXPERIMENT 3: Bag of words
elif MODEL == "bag-of-words":

    data = TweetData("train_sample")
    data = data[range(len(data))]

    model = BagOfWords()
    accuracy = model.cross_validate(data)
    print(f"Accuracy: {accuracy*100:.2F}%")

    test_data = TweetData("test")
    test_data = test_data[range(len(test_data))]
    results = model.predict(test_data)
    ResultData(results).store("bag-of-words")


# EXPERIMENT 5: CNN
elif MODEL == "convolution":

    data = TweetData("train_full")
    data = data[range(len(data))]

    model = ConvolutionModel(device)
    accuracy = model.train_and_track(data)
    model.save("data/cnn_weights.pt")
    print([f"{a * 100:.2F}%" for a in accuracy])

    test_data = TweetData("test")
    test_data = test_data[range(len(test_data))]
    results = model.predict(test_data)
    ResultData(results).store("convolution")


# EXPERIMENT 6: Judge w/ transformer & CNN
elif MODEL == "judge":

    data = TweetData("train_full")
    data = data[range(len(data))]
   
    model = JudgeModel([
        RobertaBaseTweetFinetuned(None, device),
        ConvolutionModel(device)
    ])

    accuracy = model.train_and_evaluate(data)
    model.save("data/judge")
    print(f"Accuracy: {accuracy * 100:.2F}%")

    test_data = TweetData("test")
    test_data = test_data[range(len(test_data))]
    results = model.predict(test_data)
    ResultData(results).store("judge")

