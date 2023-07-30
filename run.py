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
    accuracy = model.train_and_track(data)
    model.save("data/cnn_weights.pt")
    print([f"{a * 100:.2F}%" for a in accuracy])


elif MODEL == "judge":

  train_size = 199000
   
  cnn_model = ConvolutionModel(device)
  cnn_model.load('./data/models/cnn_weights.pt')

  trans_model = RobertaBaseTweetFinetuned(wandb.config, device)
  trans_model.load("./data/models/tweetbert-finetuned.save")

  print("Succesfull with loading")

  data = TweetData("train_sample", convolution_mode=True)
  original_data  = data[list(range(0,len(data)))]
  tweetbert_test_data = islice(TweetData("train_sample"),train_size, train_size+1000)

  data = {}
  data["tweet"] = original_data['tweet'][:train_size]
  data["sent"] = original_data['sent'][:train_size]
  #data = data[:10000]
  #original_data  = data.copy()
  data_trans = TweetData("train_sample")
  loader = DataLoader(data_trans, batch_size=None)

  #output_cnn = cnn_model.predict(data)
  #with open("./data/models/cnn_results", mode='wb') as fp:
  #   pickle.dump(output_cnn, fp)

  with open("./data/models/cnn_results", mode='rb') as fp:
    output_cnn = pickle.load(fp)

  #output_trans = trans_model.predict(data_trans, num_elem=200000, test_mode=False)

  #with open("./data/models/trans_results", mode='wb') as fp:
  #   pickle.dump(output_trans, fp)

  with open("./data/models/trans_results", mode='rb') as fp:
      output_trans = pickle.load(fp)
  #for i in range(len(data['tweet'])):

  # Calculate whether transformers model or cnn model was correct.
  #data['sent'][:10000]

  train_data = {}
  train_data['tweet'] = []
  train_data['sent'] = []

  for i in range(train_size):
     if data['sent'][i] == output_trans[i]:
             train_data['tweet'].append(data['tweet'][i])
             train_data['sent'].append(1)

     elif data['sent'][i] == output_cnn[i]:
             train_data['tweet'].append(data['tweet'][i])
             train_data['sent'].append(-1)
     else:
             continue

  eval_data = {}
  eval_data['tweet'] = train_data['tweet'][-1000:]
  eval_data['sent'] = train_data['sent'][-1000:]

  test_data = {}
  test_data['tweet'] = original_data['tweet'][train_size:train_size+1000]
  test_data['sent'] = original_data['sent'][train_size:train_size+1000]

  data['tweet'] = train_data['tweet'][:-1000]
  data['sent'] = train_data['sent'][:-1000]
  
  model = BagOfWords()
  model.train(data, bag_of_words_data=original_data,alpha=10)

  accuracy = model.evaluate(eval_data)
  print(f"Accuracy: {accuracy*100:.2F}%")

  results_trans = trans_model.predict(tweetbert_test_data, test_mode=False)
  results_cnn = cnn_model.predict(test_data)
  model_results = model.predict(test_data)
  results = []
  # Calcualt test accuracy
  for i in range(0,1000):
      #res = model.predict(data['tweet'][i])
      res = model_results[i]
      if res == 1:
          results.append(results_trans[i])
      else:
          results.append(results_cnn[i])

  print("So far soo good")
  test_accuracy = len([1 for y0, y1 in zip(test_data['sent'], results) if y0 * y1 > 0]) / len(results)

  print(test_accuracy)
