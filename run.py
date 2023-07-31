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

MODEL = "judge"
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
    model.save("./data/models/tweetbert-finetuned.save")

elif MODEL == "bag-of-words":

    data = TweetData("train_sample")
    model = BagOfWords()
    accuracy = model.cross_validate(data)
    print(f"Accuracy: {accuracy*100:.2F}%")

elif MODEL == "convolution":

    data = TweetData("train_full")
    model = ConvolutionModel(device)
    accuracy = model.train_and_track(data)
    model.save("data/cnn_weights.pt")
    print([f"{a * 100:.2F}%" for a in accuracy])


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

