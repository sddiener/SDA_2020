# Sentiment Analysis of Central Bank statement

## Repository structure

### 01_SNB_PDF_Webscaper
This script scrapes the [monetary policy assessment](https://www.snb.ch/en/iabout/monpol/id/monpol_current) pdf files of the Swiss National Bank from the period 2000 to 2020 and saves them in the data folder.

### 02_PDF_Reader
This script reads the pdf files from data folder with the package pytesseract and gives as output an excel file with articles_raw.xlsx

### 03_Cleaning_EDA
This scipt cleans the unstrucuted text data and performs an LDA analysis.

10 most common words used in the monetary policy assessment
![plot1](plots/most_common_words.png)

Wordcloud
![plot2](plots/wordcloud.png)


![html](LDA_visualization.html)


### 04_Sentiment
Count of positve vs. negative words with fed dictionary
![plot3](plots/count_words.png)

### 05_Quant_Data

Count of positve vs. negative words with fed dictionary
![plot4](plots/economic_varibales.png)

Count of positve vs. negative words with fed dictionary
![plot5](plots/model_results.png)