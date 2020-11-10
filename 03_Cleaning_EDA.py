import os
import pandas as pd
import numpy as np
import time
import textwrap
from matplotlib import pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import re
from textblob import TextBlob
import nltk
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# %% Functions
def clean_texts(text_col):
    start_time = time.time()
    print("Cleaning {} texts...".format(len(data)))

    # define stop words
    stop_words = set(stopwords.words('english'))
    stop_words.update(('wwwsnbchsnbsnbchzurich', 'press', 'relationspo', 'box', 'zurichtelephone',))

    # clean on tweet level
    text_col = text_col.apply(lambda x: re.sub(r'', '', x))  #
    text_col = text_col.apply(lambda x: re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x))  # rm links
    text_col = text_col.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', '', x))  # remove special characters
    text_col = text_col.apply(lambda x: re.sub('[^A-Za-z ]+', '', x))  # remove numbers
    text_col = text_col.apply(lambda x: x.lower())  # convert to lower

    # Clean at word level
    tokens = text_col.apply(lambda x: [w for w in word_tokenize(x)])  # splits text_col into tokens / words.
    tokens = tokens.apply(lambda x: [w for w in x if w not in stop_words])  # remove stopwords
    tokens = tokens.apply(lambda x: [WordNetLemmatizer().lemmatize(t) for t in x])  # lemmatize tokens
    text_col = tokens.apply(lambda x: ' '.join(x))  # lemmatize tokens

    text_col = text_col.apply(lambda x: re.sub(r"\b[a-zA-Z]\b", "", x))  # remove all single

    print("Cleaning done --- runtime {} s ---".format(int(round(time.time() - start_time, 0))))

    return text_col


def get_sentiment(text_col):
    start_time = time.time()
    print("Calculating sentiment of {} texts...".format(len(data)))

    polarity = text_col.apply(lambda x: TextBlob(x).sentiment.polarity)
    subjectivity = text_col.apply(lambda x: TextBlob(x).sentiment.subjectivity)

    print("Calculation finished  --- runtime {} s ---".format(int(round(time.time() - start_time, 0))))

    return polarity, subjectivity


def plot_missing_values(data):
    sns.heatmap(data.isnull(), cbar=False)
    plt.title("Missing values in tweets data frame")
    plt.show()


def plot_wordcloud(text_col):
    text = ' '.join(text_col)
    # stop_words = ["covid19", "coronavirus"]

    wordcloud = WordCloud(width=2000, height=1000, collocations=False).generate(text)

    plt.figure(figsize=(18, 10))
    plt.title("Top Words")
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def plot_sentiment_dist(sent_col):
    sns.distplot(sent_col)
    plt.title("Distribution of sentiment")
    plt.show()


# %% Main
def main():
    data_dir = "data/"
    data = pd.read_excel(data_dir + 'articles_raw_gen2020-11-07.xlsx', index_col=0)

    data['text_clean'] = clean_texts(data['text'])
    data['polarity'], data['sent'] = get_sentiment(data['text_clean'])

    plot_missing_values(data)
    pd.Series(' '.join(data['text']).split()).value_counts()[:10].plot.bar()
    pd.Series(' '.join(data['text_clean']).split()).value_counts()[:10].plot.bar()
    plot_wordcloud(data['text_clean'])
    plot_sentiment_dist(data['sent'])
    print(data.text[0])
    print(data.text_clean)


# %% Run file
if __name__ == '__main__':
    main()
