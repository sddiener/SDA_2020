import os
import pandas as pd
import numpy as np
import time
import textwrap
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import re
from textblob import TextBlob
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# %% Functions
def clean_texts(text_col):
    start_time = time.time()
    print("Cleaning {} texts...".format(len(text_col)))

    # define stop words
    stop_words = set(stopwords.words('english'))
    stop_words.update(('wwwsnbchsnbsnbchzurich', 'press', 'relationspo', 'box', 'zurichtelephone',
                       'suisse', 'swiss', 'schweizerische', 'svizzera', 'national', 'nationale', 'naziunala',
                       'nazionale', 'bank', 'banca', 'nationalbankbanque', 'pcommunicationspo', 'ch', 'suissebanca',
                       'svizzerabanca', 'svizraswiss', 'release', 'svizrapress', 'communicationssnbchberne'))

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
    print("Calculating sentiment of {} texts...".format(len(text_col)))

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

def plot_sentiment_dev(sent_col, date_col):
    dates = matplotlib.dates.date2num(date_col)
    matplotlib.pyplot.plot_date(dates, sent_col)
    plt.title("Sentiment Score over time")
    plt.show()


# %% Main
def main():
    # Cleaning
    data_dir = "data/"
    df = pd.read_excel(data_dir + 'articles_raw_gen2020-11-07.xlsx', index_col=0)
    df['date'] = pd.to_datetime(df['date'])

    df['text_clean'] = clean_texts(df['text'])
    df['polarity'], df['sent'] = get_sentiment(df['text_clean'])

    df = df[df['sent'] != 0]

    df = df.groupby('filename').agg({'date': 'first',
                                     'polarity': 'mean',
                                     'sent': 'mean',
                                     'text': lambda x: ''.join(x),
                                     'text_clean': lambda x: ','.join(x)})

    # EDA
    print(df.text_clean[90])
    plot_missing_values(df)
    pd.Series(' '.join(df['text']).split()).value_counts()[:10].plot.bar()
    pd.Series(' '.join(df['text_clean']).split()).value_counts()[:10].plot.bar()
    plot_wordcloud(df['text_clean'])
    plot_sentiment_dist(df['sent'])
    plot_sentiment_dev(df['sent'], df['date'])




# %% Run file
if __name__ == '__main__':
    main()
