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

from ..model import Model

from torch.utils.data import DataLoader

from typing import List, Tuple
import logging

class RobertaBaseSentiment(Model):

    # Preprocess text (username and link placeholders)
    def preprocess(self, text):
        new_text = []
    
    
        for t in text.split(" "):
            # t = '@user' if t.startswith('@') and len(t) > 1 else t
            # Change this since <user> is used in soruce data but @user by this model.
            t = '@user' if t == '<user>' else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    # Tasks:
    # emoji, emotion, hate, irony, offensive, sentiment
    # stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

    def __init__(self, device='cpu'):
        task='sentiment'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.device = device

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

        for idx, data_point in enumerate(test_data):
            text = data_point["tweet"]

            try:
                text = self.preprocess(text)
            except AttributeError: 
                # If something goes wrong (for example for NaN values ), ignore
                logging.warning("'{}' Couldn't be preprocssed and will be ignored")
                continue

            encoded_input = self.tokenizer(text, return_tensors='pt').to(self.device)
            output = self.model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)

            # More negative than positive, excluding neutral as an option
            if scores[0] > scores[2]:
                predicted_sent = -1
            else:
                predicted_sent = 1

            if predicted_sent == data_point["sent"]:
                correct += 1

            total += 1

            #ranking = np.argsort(scores)
            #ranking = ranking[::-1]
            #for i in range(scores.shape[0]):
            #    l = self.labels[ranking[i]]
            #    s = scores[ranking[i]]
            #    print(f"{i+1}) {l} {np.round(float(s), 4)}")

        return correct/total
    
    def predict(self, test_data: DataLoader) -> List[int]:

        # First element is index and second element sentiment
        results : List[Tuple[int, int]] = []

        for idx, data_point in enumerate(test_data):

            text = data_point["tweet"]

            try:
                text = self.preprocess(text)
            except AttributeError: 
                # If something goes wrong (for example for NaN values ), ignore
                logging.warning("'{}' Couldn't be preprocssed and will be ignored".format(data_point["tweet"]))
                continue

            encoded_input = self.tokenizer(text, return_tensors='pt')
            output = self.model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)

            # More negative than positive, excluding neutral as an option
            if scores[0] > scores[2]:
                predicted_sent = -1
            else:
                predicted_sent = 1


            results.append((data_point["id"], predicted_sent))
        
        return results