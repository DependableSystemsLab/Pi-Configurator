from unicodedata import name
import warnings
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
import tracemalloc
import time
from time import process_time
import csv
import sys
import os
warnings.filterwarnings('ignore')

# wallcolockX12.6 cpucycleX26 accuracy 0.66

################## Input param ###################
denoise = int(sys.argv[1])
stopWord = int(sys.argv[2])
lemmatize = int(sys.argv[3])
name = str(denoise)+str(stopWord)+str(lemmatize)
##################### Time count start ########################
tracemalloc.start()
t1_start = process_time()
start = time.time()

####################### importing the training data ##################################
imdb_data = pd.read_csv('IMDB.csv')


############################## Removing the noisy text #####################

def denoise_text(text, remove_digits=True):
    # Removing the html strips
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    # Removing the square brackets
    text = re.sub('\[[^]]*\]', '', text)
    # removing special characters
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)

    return text

######################### Removing stopwords ##############################


def remove_stopwords(text, is_lower_case=False):
    tokenizer = ToktokTokenizer()
    # Setting English stopwords
    stopword_list = nltk.corpus.stopwords.words('english')
    # removing the stopwords
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [
            token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [
            token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# ####################### Stemming the text #################################
# def simple_stemmer(text):
#     ps = nltk.stem.PorterStemmer()
#     text = ' '.join([ps.stem(word) for word in text.split()])
#     return text


# # Apply function on review column
# imdb_data['review'] = imdb_data['review'].apply(simple_stemmer)


####################### Lemmatizing the text #################################
def simple_lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text


####################### Calling functions #################################
def preProcess(denoise, stopWord, lemmatize):
    if denoise:
        imdb_data['review'] = imdb_data['review'].apply(denoise_text)
    if stopWord:
        imdb_data['review'] = imdb_data['review'].apply(remove_stopwords)
    if lemmatize:
        imdb_data['review'] = imdb_data['review'].apply(simple_lemmatizer)
    imdb_data.to_csv(name+".csv", sep=',', index=False)
    print(name)


preProcess(denoise, stopWord, lemmatize)


############ Stop the stopwatch / counter ################
t1_stop = process_time()
_, first_peak = tracemalloc.get_traced_memory()

passed = round((t1_stop-t1_start), 2)
elapsed = round((time.time() - start), 2)
memoryUSed = round((first_peak/10**6), 2)


with open("AzureStat.csv", "a", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([name, passed, elapsed, memoryUSed])

os._exit(0)
