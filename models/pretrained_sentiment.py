# Code taken from the example from the following site
# The only thing we changed is the source data for prediction.
# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

from models.model import Model

from torch.utils.data import DataLoader

from typing import List, Tuple
import logging

from sklearn.metrics import confusion_matrix

class RobertaBaseTweetSentiment(Model):

    # Preprocess text (username and link placeholders)
    def preprocess(self, text):
        new_text = []
    
    
        for t in text.split(" "):
            # t = '@user' if t.startswith('@') and len(t) > 1 else t
            # Change this since <user> is used in soruce data but @user by this model.
            t = '@user' if t == '<user>' else t
            t = 'http' if t == '<url>' else t
            new_text.append(t)
        return " ".join(new_text)

    # Tasks:
    # emoji, emotion, hate, irony, offensive, sentiment
    # stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

    def label_to_idx(self, label: str): 
        translate_labels = { "negative": 0, "positive": 2 }
        return translate_labels[label]
    

    def __init__(self, config, device='cpu'):
        task='sentiment'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.device = device
        self.config = config

        # download label mapping
        # self.labels=[]
        # mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        # with urllib.request.urlopen(mapping_link) as f:
        #     html = f.read().decode('utf-8').split("\n")
        #    csvreader = csv.reader(html, delimiter='\t')
        # self.labels = [row[1] for row in csvreader if len(row) > 1]

        # PT
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(self.device)
        #self.model.save_pretrained(MODEL)

    
    """
    Since we are performing zero-shot, no training is required.
    """
    def train(self):
        pass


    def evaluate(self, test_data: DataLoader):

        correct = 0
        total = 0

        y_true = []
        y_pred = []

        for idx, data_point in enumerate(test_data):

            if (idx % 1000 == 0):
                logging.info("Step {}".format(idx))


            text = data_point["tweet"]

            try:
                text = self.preprocess(text)
            except AttributeError: 
                # If something goes wrong (for example for NaN values ), ignore
                logging.warning("'{}' Couldn't be preprocssed and will be ignored")
                continue

            encoded_input = self.tokenizer(text, return_tensors='pt').to(self.device)
            output = self.model(**encoded_input)
            scores = output[0][0].detach().to('cpu').numpy()
            scores = softmax(scores)

            # More negative than positive, excluding neutral as an option
            if scores[self.label_to_idx("negative")] > scores[self.label_to_idx("positive")]:
                predicted_sent = -1
            else:
                predicted_sent = 1

            if predicted_sent == data_point["sent"]:
                correct += 1

            y_true.append(data_point["sent"])
            y_pred.append(predicted_sent)

            total += 1

            #ranking = np.argsort(scores)
            #ranking = ranking[::-1]
            #for i in range(scores.shape[0]):
            #    l = self.labels[ranking[i]]
            #    s = scores[ranking[i]]
            #    print(f"{i+1}) {l} {np.round(float(s), 4)}")


        return correct/total, confusion_matrix(y_true, y_pred)
    
    def predict(self, test_data: DataLoader) -> List[int]:

        # First element is index and second element sentiment
        results : List[Tuple[int, int]] = []

        for idx, data_point in enumerate(test_data):

            if (idx % 1000 == 0):
                logging.info("Step {}".format(idx))

            text = data_point["tweet"]

            try:
                text = self.preprocess(text)
            except AttributeError: 
                # If something goes wrong (for example for NaN values ), ignore
                logging.warning("'{}' Couldn't be preprocssed and will be ignored".format(data_point["tweet"]))
                continue

            encoded_input = self.tokenizer(text, return_tensors='pt').to(self.device)
            output = self.model(**encoded_input)
            scores = output[0][0].detach().to('cpu').numpy()
            scores = softmax(scores)

            # More negative than positive, excluding neutral as an option
            if scores[self.label_to_idx("negative")] > scores[self.label_to_idx("positive")]:
                predicted_sent = -1
            else:
                predicted_sent = 1


            results.append((data_point["id"], predicted_sent))
        
        return results
    

class RobertaBaseSentiment(RobertaBaseTweetSentiment):

    def __init__(self, config, device='cpu'):
        super().__init__(config, device)

        task='sentiment'
        MODEL = f"siebert/sentiment-roberta-large-english"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.device = device
        self.config = config

        # download label mapping
        # self.labels=[]
        # mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        # with urllib.request.urlopen(mapping_link) as f:
        #     html = f.read().decode('utf-8').split("\n")
        #    csvreader = csv.reader(html, delimiter='\t')
        # self.labels = [row[1] for row in csvreader if len(row) > 1]

        # PT
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(self.device)
        #self.model.save_pretrained(MODEL)

    def label_to_idx(self, label: str): 
        translate_labels = { "negative": 0, "positive": 1 }
        return translate_labels[label]