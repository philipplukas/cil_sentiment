import random

import numpy as np

from data.data_set import TweetData, ResultData
from models.bag_of_words_model import BagOfWords
from models.convolution_model import ConvolutionModel
from models.transformers.pretrained_sentiment import RobertaBaseTweetSentiment, RobertaBaseSentiment
from models.transformers.finetuned_sentiment import RobertaBaseTweetFinetuned, RobertaBaseFinetuned
from torch.utils.data import DataLoader
import nltk

from transformers import AutoModelForSequenceClassification

import torch
import logging

# Import the W&B Python Library 
import wandb

from datasets import Dataset

import pickle

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

    data = TweetData("train_full", convolution_mode=True)
    model = ConvolutionModel(device)
    accuracy = model.train_and_evaluate(data)
    model.save("data/cnn_weights.pt")
    print(f"Accuracy: {accuracy * 100:.2F}%")


elif MODEL == "judge":
   
  cnn_model = ConvolutionModel(device)
  cnn_model.load('./data/models/cnn_weights.pt')

  trans_model = RobertaBaseTweetFinetuned(wandb.config, device)
  trans_model.load("./data/models/tweetbert-finetuned.save")

  print("Succesfull with loading")

  data = TweetData("train_sample", convolution_mode=True)
  data = data[list(range(0,10000))]
  original_data  = data.copy()
  data_trans = TweetData("train_sample")
  loader = DataLoader(data_trans, batch_size=None)

  #output_cnn = cnn_model.predict(data)
  #with open("./data/models/cnn_results", mode='wb') as fp:
  #   pickle.dump(output_cnn, fp)

  with open("./data/models/cnn_results", mode='rb') as fp:
    output_cnn = pickle.load(fp)

  #output_trans = trans_model.predict(data_trans, num_elem=10000, test_mode=False)

  with open("./data/models/trans_results", mode='rb') as fp:
    output_trans = pickle.load(fp)
  #for i in range(len(data['tweet'])):

  # Calculate whether transformers model or cnn model was correct.
  data['sent'][:10000]

  for i in range(10000):
     if data['sent'][i] == output_trans[i]:
             data['sent'][i] = output_trans[i]

     else:
             data['sent'][i] = output_cnn[i]

  eval_data = {}
  eval_data['tweet'] = data['tweet'][9000:10000]
  eval_data['sent'] = data['sent'][9000:10000]

  data['tweet'] = data['tweet'][:9000]
  data['sent'] = data['sent'][:9000]

  
  
  model = BagOfWords()
  model.train(data, bag_of_words_data=original_data)

  accuracy = model.evaluate(eval_data)
  print(f"Accuracy: {accuracy*100:.2F}%")