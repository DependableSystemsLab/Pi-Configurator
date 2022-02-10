# Load the libraries
import warnings
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
import tracemalloc
import time
from time import process_time
import csv
import sys
import os
warnings.filterwarnings('ignore')


################## Input param ###################
filename = str(sys.argv[1])  # '010.csv'
model = str(sys.argv[2])  # 'Tfidf'
difference_limit = int(sys.argv[3])
range_limit = int(sys.argv[4])
name = filename+"_"+model+"_"+str(difference_limit)+"_"+str(range_limit)

data_limit = 1000

##################### Time count start ########################
tracemalloc.start()
t1_start = process_time()
start = time.time()

# importing the training data
imdb_data = pd.read_csv(filename)
# Spliting the training dataset
norm_train_reviews = imdb_data.review[:data_limit]
norm_test_reviews = imdb_data.review[data_limit:]
# train dataset
train_reviews = imdb_data.review[:data_limit]
train_sentiments = imdb_data.sentiment[:data_limit]
# test dataset
test_reviews = imdb_data.review[data_limit:]
test_sentiments = imdb_data.sentiment[data_limit:]


def tokenizer(model, difference_limit, range_limit):

    if model == 'bag':
        ############### Bags of words model ###################
        ############### It is used to convert text documents to numerical vectors or bag of words. ################
        # Count vectorizer for bag of words
        cv = CountVectorizer(min_df=0, max_df=difference_limit,
                             binary=False, ngram_range=(1, range_limit))
        # transformed train reviews
        tv_train_reviews = cv.fit_transform(norm_train_reviews)
        # transformed test reviews
        tv_test_reviews = cv.transform(norm_test_reviews)
    else:
        ############ Term Frequency-Inverse Document Frequency model (TFIDF) ########################
        ############ It is used to convert text documents to matrix of tfidf features. ##############

        # Tfidf vectorizer
        tv = TfidfVectorizer(min_df=0, max_df=difference_limit,
                             use_idf=True, ngram_range=(1, range_limit))
        # transformed train reviews
        tv_train_reviews = tv.fit_transform(norm_train_reviews)
        # transformed test reviews
        tv_test_reviews = tv.transform(norm_test_reviews)

    sparse.save_npz(name+"_"+"train_reviews.npz", tv_train_reviews)
    sparse.save_npz(name+"_"+"test_reviews.npz", tv_test_reviews)


tokenizer(model, difference_limit, range_limit)


############ Stop the stopwatch / counter ################
t1_stop = process_time()
_, first_peak = tracemalloc.get_traced_memory()

passed = round((t1_stop-t1_start), 2)
elapsed = round((time.time() - start), 2)
memoryUSed = round((first_peak/10**6), 2)

with open("Pi0WStat.csv", "a", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([name, passed, elapsed, memoryUSed])
    
os._exit(0)
