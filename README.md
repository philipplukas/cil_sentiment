
# Overview

This project is about sentiment prediction on tweet data.
We have implemented various methods to achieve this task.
The main file for running the models is ```run.py```.
To run any of the models simply change the variable ```MODEL``` in this file.
Then you can simply run the model training/evaluation using ```python3 run.py```
Further information can be fond in the comments of this file.

See detailed instructions about this project here: https://docs.google.com/document/d/1D2T5ckhOxnW8C3Xq01GUNaQXMo6jbXkU19qFZ_SoswU/edit?usp=sharing

General Information about the semester project can be found here: https://docs.google.com/document/d/1kXMPYBRJYzMQNceVUpsaQBSMec8kuaAHiIRGQuG9DQA/edit?usp=sharing

## Note for the Judge model option

Put the following files in the ```data/models``` directory to make it work
https://drive.google.com/drive/folders/1Bar-I8oxDV5-ahozwOdtOugB79S8Mu5q?usp=sharing

## Note for convolution model option

This needs an additional embedding file which is too big to store directly in Git Hub.
However, it can be downloaded from the following site:
https://www.kaggle.com/datasets/fullmetal26/glovetwitter27b100dtxt/discussion

After the download, simply create a subdirectory ```embeddings``` in the ```data``` directory
and put the unzipped embedding file there.

## Directory structure

- ``` models ```
  In this directory, the code for all the model implementations remains.

- ``` data ```
  Here, the class handling the data can be found.Moreoverr, all the results are stored in the ```results``` subdirectory



## Dataset Description
### File descriptions

train_pos.txt and train_neg.txt - a small set of training tweets for each of the two classes. (Dataset available in the zip file, see link below)

train_pos_full.txt and train_neg_full.txt - a complete set of training tweets for each of the two classes, about 1M tweets per class. (Dataset available in the zip file, see link below)

test_data.txt - the test set, that is the tweets for which you have to predict the sentiment label.

sampleSubmission.csv - a sample submission file in the correct format, note that each test tweet is numbered. (submission of predictions: -1 = negative prediction, 1 = positive prediction)

Note that all tweets have been tokenized already, so that the words and punctuation are properly separated by a whitespace.

Link for the larger datasets:
http://www.da.inf.ethz.ch/files/twitter-datasets.zip

### Twitter  Datasets

Download the tweet datasets from here:
http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip


The dataset should have the following files:
- sample_submission.csv
- train_neg.txt :  a subset of negative training samples
- train_pos.txt: a subset of positive training samples
- test_data.txt:
- train_neg_full.txt: the full negative training samples
- train_pos_full.txt: the full positive training samples

#### Build the Co-occurence Matrix

To build a co-occurence matrix, run the following commands.  (Remember to put the data files
in the correct locations)

Note that the cooc.py script takes a few minutes to run, and displays the number of tweets processed.

- build_vocab.sh
- cut_vocab.sh
- python3 pickle_vocab.py
- python3 cooc.py

####  Template for Glove Question

Your task is to fill in the SGD updates to the template
glove_template.py

Once you tested your system on the small set of 10% of all tweets, we suggest you run on the full datasets train_pos_full.txt, train_neg_full.txt
