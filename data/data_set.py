from torch.utils.data import Dataset
from sklearn.utils import shuffle

from functools import partial

import pandas as pd
import os
import csv

from typing import List, Tuple, Union

from time import strftime


FILE_BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "twitter-datasets")
RESULTS_BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")

SPLIT_NAMES = {
    "test": {"test": "test_data.txt"},
    "train_full": {"pos": "train_pos_full.txt", "neg": "train_neg_full.txt"},
    "train_sample": {"pos": "train_pos.txt", "neg": "train_neg.txt"},
    "debug": {"test": "debug_data.txt"},
    "debug_train": {"pos": "debug_pos.txt", "neg": "debug_neg.txt"},
    "eval_train": {"pos": "train_pos_eval.txt", "neg": "train_neg_evalgit.txt"}
}
DELIMITER = " "

"""
If in test mode we will parse and pass the Id along
to remain consistent when returning the results later.
"""
def parse_csv(file_path: str) -> Tuple[List[int], List[str]]:

    ids: List[int] = []
    tweets: List[str] = []

    with open(file_path) as fp:
        for line in fp.readlines():
            line = line.rstrip()

            parts = line.split(',')
            ids.append(parts[0])
            tweets.append(','.join(parts[1:]))

    return ids, tweets

def parse_tweets(file_path: str) -> List[str]:

    tweets: List[str] = []

    with open(file_path) as fp:
        for line in fp.readlines():
            line = line.rstrip()

            tweets.append(line)

    return tweets


#class TrainDataElement(TypedDict):
#    tweet: str
#    sentiment: int

#class TestDataElement(TypedDict):
#    tweet: str



class TweetData(Dataset):

    def __init__(self, split_name: str) -> None:

        if split_name not in SPLIT_NAMES:
            raise ValueError("Split doesn't exist")

        self.test_mode = False
        if "test" in split_name:
            self.test_mode = True


        # Set up pandas table
        # Depending on whether we act on testing data for prediction,
        # Generate different tables
        if self.test_mode:

            file_name = SPLIT_NAMES[split_name]["test"]
            full_path = os.path.join(FILE_BASE, file_name)
        
            ids, tweets = parse_csv(full_path)
            new_table = pd.DataFrame({"id": ids, "tweet":tweets}).set_index("id")
            self.pd_table = new_table


        else:

            file_name_neg = SPLIT_NAMES[split_name]["neg"]
            file_name_pos = SPLIT_NAMES[split_name]["pos"]

            full_path_neg = os.path.join(FILE_BASE, file_name_neg)
            full_path_pos = os.path.join(FILE_BASE, file_name_pos)

            tweets_neg = parse_tweets(full_path_neg)
            tweets_pos = parse_tweets(full_path_pos)
            
            pd_tables = []

            table_neg = pd.DataFrame({"tweet":tweets_neg})
            table_neg["sent"] = -1
            table_pos = pd.DataFrame({"tweet":tweets_pos})
            table_pos["sent"] = 1

            self.pd_table = pd.concat([table_neg, table_pos])
            self.pd_table = shuffle(self.pd_table)

            # Reset indices 
            self.pd_table.reset_index(inplace=True, drop=True) 


    def __len__(self) -> int:
        return self.pd_table.shape[0]
    

    def __getitem__(self, index: int | List[int]) -> dict:

        index = [index] if index is int else index

        if any(i >= len(self) for i in index):
            raise IndexError

        data_element = self.pd_table.iloc[index]
        if self.test_mode:        
            return {'id': int(data_element.name), 'tweet': data_element['tweet']}
        
        # Training data in this case
        else:
            # Tweet is in the first column, sentiment in the second
            return {'tweet': data_element['tweet'], 'sent': data_element['sent']}
            

class ResultData():

    def __init__(self, results: List[Tuple[int, int]]) -> None:
    
        self.index, self.sentiments = list(zip(*results))
        # Starting from 1, but pandas default starts from 0
        # Therefore we custom generate the indices here.


        # self.index = [idx+1 for idx in self.index]

        self.table = pd.DataFrame({"Id": self.index, "Prediction":self.sentiments}).set_index("Id")



    """
    Returns path to new file
    """
    def store(self, result_name: str, time_suffix: bool = True):
        
        if time_suffix:
            suffix = strftime("%m-%d_%H-%M-%S")
            result_name += "_" + suffix

        result_name += ".csv"


        full_path = os.path.join(RESULTS_BASE, result_name)
        self.table.to_csv(full_path)

        return full_path
