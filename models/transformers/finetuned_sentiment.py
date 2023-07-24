# Code taken from the example from the following site
# The only thing we changed is the source data for prediction.
# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, TrainerCallback

from sklearn.model_selection import train_test_split
import evaluate

import numpy as np
from scipy.special import softmax
import csv
import urllib.request

from ..model import Model

from torch.utils.data import DataLoader,Dataset

from sklearn.metrics import confusion_matrix

from typing import List, Tuple
import logging

class RobertaBaseTweetFinetuned(Model):

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

    def __init__(self, config, device='cpu'):
        task='sentiment'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

        self.model = MODEL

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

        self.config = config

    def label_to_idx(self, label: str): 
        translate_labels = { "negative": 0, "positive": 2 }
        return translate_labels[label]
    
    """
    Since we are performing zero-shot, no training is required.
    """
    def train(self, train_dataset: Dataset, log_callback):
        # Don't pin memory to avoid error message
        training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="steps", dataloader_pin_memory=False, eval_steps=5, load_best_model_at_end=True, per_device_train_batch_size=1, gradient_accumulation_steps=8, max_steps=200)

        metric = evaluate.load("accuracy")

        # Preprocess data
        train_dataset = train_dataset.map(lambda e: self.tokenizer(e['tweet'], truncation=True, padding='max_length', max_length=512), batched=True)

        # Rename columns to match the names in the huggingface documentation
        train_dataset = train_dataset.rename_column("sent", "label")

        def replace_label(e):
            mapping = {-1:self.label_to_idx("negative"), 1: self.label_to_idx("positive")}
            e['label'] = mapping[e['label']]
            return e
        # Replace labels -1,1 with model specific labels
        train_dataset = train_dataset.map(replace_label, batched=False)

    
        # Change to tensors and put them on the appropriate device
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'], device=self.device)


        train_dataset = train_dataset.train_test_split(test_size=self.config['train_test_ratio'])


        # Callback to compute metrics for huggingface Trainer class
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
        

        # Callback for logs, passing it onto wandb
        class EvalCallback(TrainerCallback):
            def on_eval(self, args, state, control, metrics=None, **kwargs):
                if state.is_local_process_zero:
                    log_callback(metrics)

        # Without pin_memory=False, there will be an error
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset["train"],
            eval_dataset=train_dataset["test"],
            compute_metrics=compute_metrics,
            callbacks=[EvalCallback()]
        )

        trainer.train()

        # Set to eval mode
        self.model.eval()

    def evaluate(self, test_data: DataLoader):

        correct = 0
        total = 0

        y_true = []
        y_pred = []

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
            if scores[self.label_to_idx("negative")] > scores[self.label_to_idx("positive")]:
                predicted_sent = -1
            else:
                predicted_sent = 1

            if predicted_sent == data_point["sent"]:
                correct += 1

            total += 1

            y_true.append(data_point["sent"])
            y_pred.append(predicted_sent)

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

            text = data_point["tweet"]

            try:
                text = self.preprocess(text)
            except AttributeError: 
                # If something goes wrong (for example for NaN values ), ignore
                logging.warning("'{}' Couldn't be preprocssed and will be ignored".format(data_point["tweet"]))
                continue

            encoded_input = self.tokenizer(text, return_tensors='pt').to(device=self.device)
            output = self.model(**encoded_input)
            scores = output[0][0].detach().to(device="cpu").numpy()
            scores = softmax(scores)

            # More negative than positive, excluding neutral as an option
            if scores[self.label_to_idx("negative")] > scores[self.label_to_idx("positive")]:
                predicted_sent = -1
            else:
                predicted_sent = 1


            results.append((data_point["id"], predicted_sent))
        
        return results
    
    def save(self, file_path: str):
        self.model.save_pretrained(file_path)

class RobertaBaseFinetuned(RobertaBaseTweetFinetuned):

    def __init__(self, config, device='cpu'):
        super().__init__(config, device)

        task='sentiment'
        MODEL = f"siebert/sentiment-roberta-large-english"
        self.model = MODEL

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