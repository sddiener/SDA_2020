# Sentiment Analysis of Central Bank statement

## Repository structure

### 01_SNB_PDF_Webscaper
This script scrapes the monetary policy assessment pdf files of the Swiss National Bank (see [Link](https://www.snb.ch/en/iabout/monpol/id/monpol_current) from the period 2000 to 2020 and saves them in the data folder.

### 02_PDF_Reader
This script reads the pdf files from data folder with the package pytesseract and gives as output an excel file with articles_raw.xlsx

### 03_Cleaning_EDA
This scipt cleans the unstrucuted text data and performs an LDA analysis.

10 most common words used in the monetary policy assessment
![plot1](plots/most_common_words.png)

Wordcloud
![plot2](plots/wordcloud.png)

