from torch.utils.data import Dataset
from sklearn.utils import shuffle

from functools import partial

import pandas as pd
import os

#from typing import Union
#from typing import TypedDict


FILE_BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "twitter-datasets")
SPLIT_NAMES = {
    "test": ["test_data.txt"],
    "train_full": ["train_neg_full.txt", "train_pos_full.txt"],
    "train_sample": ["train_neg.txt", "train_pos.txt"]
}
DELIMITER = " "



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

        file_names = SPLIT_NAMES[split_name]
        full_paths = [os.path.join(FILE_BASE, file_name) for file_name in file_names]
        
        pd_tables = []
        for full_path in full_paths:
            if self.test_mode:
                new_table = pd.read_csv(full_path, names=['tweet'], dtype={'tweet': str})
            else:
                new_table = pd.read_csv(full_path, names=['tweet', 'sent'], dtype={'tweet': str, 'sent': int})
            pd_tables.append(new_table)

        self.pd_table = pd.concat(pd_tables)
        self.pd_table = shuffle(self.pd_table)

        # Reset indices 
        self.pd_table.reset_index(inplace=True, drop=True) 


    def __len__(self) -> int:
        return self.pd_table.shape[0]
    

    def __getitem__(self, index: int) -> dict: #Union[TestDataElement, TrainDataElement]:
        data_element = self.pd_table.iloc[index]
        if self.test_mode:
            return {'tweet': data_element['tweet']}#TestDataElement(data_element[0])
        
        else:
             # Tweet is in the first column, sentiment in the second
            return {'tweet': data_element['tweet'], 'sent': data_element['sent']}#TrainDataElement(data_element[0], data_element[1])