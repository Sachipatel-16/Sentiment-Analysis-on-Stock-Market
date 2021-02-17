# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:07:40 2021

@author: yashk
"""


from turtle import pd
from urllib.request import urlopen, Request


import matplotlib
from bs4 import BeautifulSoup
import os
import pandas as pds
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'
# NLTK VADER for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

finwiz_URL = 'https://finviz.com/quote.ashx?t='

news_tables = {}
tickers = ['AMZN', 'TSLA', 'GOOG', 'WNW', 'ARTL', 'CLEU']

for ticker in tickers:
    url = finwiz_URL + ticker
    request = Request(url = url, headers= {'user-agent': 'SEASTOM/0.0.1'})
    response  = urlopen(request)

    # reading the content into HTML file
    html = BeautifulSoup(response,"html.parser")

    # finding the news-table in the bBeautifulSoup and loading it into 'news-table'
    news_table = html.find(id='news-table')

    # Adding table to our dictionary
    news_tables[ticker] = news_table

    # reading one single headline from Amazon
    amzn = news_tables['AMZN']

    # get all the rows tagged in HTMLas <tr> into amzn_tr
    amzn_tr = amzn.findAll('tr')

    for i, table_row in enumerate(amzn_tr):
        # Read the text of the element 'a' into 'link_text'
        a_text = table_row.a.text
        # Read the text of the element 'td' into 'data_text'
        td_text = table_row.td.text
        # Print the contents of 'link_text' and 'data_text'
        #   print(a_text)
        #print(td_text)
        # Exit after printing 4 rows of data
        #if i == 3:
         #   break
parsed_news = []

# Parsing through the scrapped news
for file_name, new_table in news_tables.items():
    # Iterate through all the tags in 'news_table'
    for x in news_table.findAll('tr'):
        # read the text from each tr tag into text
        # get text from a only
        text = x.a.get_text()
        # Split the text in td tag into a list
        date_scrap = x.td.text.split()
        # if the length of scrapped  data is 1 then load 'time' as the only element

        if len(date_scrap) == 1:
            time = date_scrap[0]

        # else load 'date' as the element and 'time' as second
        else:
            date = date_scrap[0]
            time = date_scrap[1]

        # Extract the ticker from the file name, get the string up to the 1st '_'
        ticker = file_name.split('_')[0]

        # Append ticker, date, time and headline as a list to the 'parsed_news' list
        parsed_news.append([ticker, date, time, text])

parsed_news

# Instantiate the sentiment intensity analyzer
vader = SentimentIntensityAnalyzer()

# Set column names
columns = ['ticker', 'date', 'time', 'headline']

# Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
parsed_and_scored_news = pds.DataFrame(parsed_news, columns=columns)

# Iterate through the headlines and get the polarity scores using vader
scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()

# Convert the 'scores' list of dicts into a DataFrame
scores_df = pds.DataFrame(scores)

# Join the DataFrames of the news and the list of dicts
parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')

# Convert the date column from string to datetime
parsed_and_scored_news['date'] = pds.to_datetime(parsed_and_scored_news.date).dt.date

parsed_and_scored_news.head()


plt.rcParams['figure.figsize'] = [10, 6]

# Group by date and ticker columns from scored_news and calculate the mean
mean_scores = parsed_and_scored_news.groupby(['ticker','date']).mean()

# Unstack the column ticker
mean_scores = mean_scores.unstack()

# Get the cross-section of compound in the 'columns' axis
mean_scores = mean_scores.xs('compound', axis="columns").transpose()

# Plot a bar chart with pandas
mean_scores.plot(kind = 'bar')
plt.grid()