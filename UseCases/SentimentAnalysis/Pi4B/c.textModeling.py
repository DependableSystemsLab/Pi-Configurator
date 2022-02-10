import warnings
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tracemalloc
import time
from time import process_time
from scipy import sparse
import csv
import os
import sys
warnings.filterwarnings('ignore')

# wallcolockX12.6 cpucycleX26 accuracy 0.66

################## Input param ###################
filename = str(sys.argv[1])  # '001.csv'
cleanername = str(sys.argv[2])  # 'tfidf'
arg1 = str(sys.argv[3])  # '1'
arg2 = str(sys.argv[4])  # '1'
model = str(sys.argv[5])  # 'sgd'
if model == 'sgd':
    param = float(sys.argv[6])
else:
    param = int(sys.argv[6])  # 20

train_filename = filename+'_'+cleanername + \
    '_'+arg1+'_'+arg2+'_train_reviews.npz'
test_filename = filename+'_'+cleanername+'_'+arg1+'_'+arg2+'_test_reviews.npz'
# name = train_filename[:15]+"_"+model+"_"+str(param)
name = filename+'_'+cleanername+'_'+arg1+'_'+arg2+'_'+model+'_'+str(param)
data_limit = 1000
##################### Time count start ########################
tracemalloc.start()
t1_start = process_time()
start = time.time()

# importing the training data
imdb_data = pd.read_csv(filename)
tv_train_reviews = sparse.load_npz(train_filename)
tv_test_reviews = sparse.load_npz(test_filename)
# Labeling the sentiment text
lb = LabelBinarizer()
# transformed sentiment data
sentiment_data = lb.fit_transform(imdb_data['sentiment'])
# Split the sentiment tdata
train_sentiments = sentiment_data[:data_limit]
test_sentiments = sentiment_data[data_limit:]


################## Modelling the dataset ##################

def train_model(model, param):
    if model == "logis":
        lr = LogisticRegression(
            penalty='l2', max_iter=500, C=param, random_state=42)
        lr.fit(tv_train_reviews, train_sentiments)
        lr_tfidf_predict = lr.predict(tv_test_reviews)
        accuracy = accuracy_score(test_sentiments, lr_tfidf_predict)

    elif model == "sgd":
        # training the linear SGD
        SGD = SGDClassifier(loss='hinge', alpha=param,
                            max_iter=500, random_state=42)
        # fitting the SGD
        SGD.fit(tv_train_reviews, train_sentiments)
        # Predicting the model
        SGD_tfidf_predict = SGD.predict(tv_test_reviews)
        # Accuracy score
        accuracy = accuracy_score(test_sentiments, SGD_tfidf_predict)

    elif model == "knn":
        # training the linear knn
        knn = KNeighborsClassifier(n_neighbors=param)
        # fitting the knn
        knn.fit(tv_train_reviews, train_sentiments)
        # Predicting the model
        knn_tfidf_predict = knn.predict(tv_test_reviews)
        # Accuracy score
        accuracy = accuracy_score(test_sentiments, knn_tfidf_predict)

    return accuracy


accuracy = train_model(model, param)

############ Stop the stopwatch / counter ################
t1_stop = process_time()
_, first_peak = tracemalloc.get_traced_memory()

passed = round((t1_stop-t1_start), 2)
elapsed = round((time.time() - start), 2)
memoryUSed = round((first_peak/10**6), 2)

with open("Pi4Stat.csv", "a", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([name, passed, elapsed, memoryUSed, accuracy])

os._exit(0)
