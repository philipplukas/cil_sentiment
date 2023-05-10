
# Overview

See detailed instructions about this project here: https://docs.google.com/document/d/1D2T5ckhOxnW8C3Xq01GUNaQXMo6jbXkU19qFZ_SoswU/edit?usp=sharing

General Information about the semester project can be found here: https://docs.google.com/document/d/1kXMPYBRJYzMQNceVUpsaQBSMec8kuaAHiIRGQuG9DQA/edit?usp=sharing
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
