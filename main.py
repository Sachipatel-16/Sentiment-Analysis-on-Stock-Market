# Sentiment Analysis of Stockmarket

# Importing the necessary libraries
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import streamlit_wordcloud as wordcloud
import re

###### FOR SENTIMENT MODEL#######

st.write(""" 
#**SASTOM - Sentiment Analysis of Stock Market**
""")

image = Image.open("C:/Users/priya/Downloads/Projectimage.jpg")
st.image(image, use_column_width = True)

# Creating sidebar
st.sidebar.header('User Input')

# Function to create user input
def get_input():
    start_date = st.sidebar.text_input("Start Date", "2018-01-02")
    end_date = st.sidebar.text_input("Start Date", "2018-03-16")
    stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
    return start_date, end_date, stock_symbol

# Function to get the company name
def get_company_name(symbol):
    if(symbol == 'AAPL'):
        return 'Apple'
    elif(symbol == 'AXP'):
        return 'American Express'
    elif(symbol == 'BA'):
        return 'Boeing'
    else:
        'None'

# Function to get the dataset

def get_data(symbol,start,end):

    # Loading data
    if symbol.upper() == 'AAPL':
        df = pd.read_csv('C:/Users/priya/Downloads/Data/AAPL.csv')
    elif symbol.upper() == 'AXP':
        df = pd.read_csv('C:/Users/priya/Downloads/Data/AXP.csv')
    elif symbol.upper() == 'BA':
        df = pd.read_csv('C:/Users/priya/Downloads/Data/BA.csv')
    else:
        df = pd.DataFrame(columns= ['Date','Close','Open', 'Volume', 'Adj Close', 'High', 'Low'])

    # Getting the date Range
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    start_row = 0
    end_row = 0

    #start date from the top of the data and going down to see is users start date is <= the date in the dataset
    for i in range (0, len(df)):
        if start <= pd.to_datetime(df['Date'][i]):
            start_row = i
            break

    # end date from the bottom of the dataset and going up to see if users end date is >= the date in the dataset
    for j in range (0, len(df)):
        if end >= pd.to_datetime(df['Date'][len(df)-1-j]):
            end_row = len(df) -1 -j
            break

    # Set index to be the date
    df = df.set_index(pd.DatetimeIndex(df['Date'].values))
    return df.iloc[start_row:end_row + 1, :]

#get user input
start, end, symbol = get_input()

# Get Data
df = get_data(symbol, start, end)

# get company name
company_name = get_company_name(symbol.upper())

# Display the close price
st.header(company_name+" Close Price\n")
st.line_chart(df['Close'])

# Display the Volume
st.header(company_name+" Volume\n")
st.line_chart(df['Volume'])

# Get statistics on data
st.header('Data Statistics')
st.write(df.describe())

df1=pd.read_csv('Data/Final_Data.csv', encoding = "ISO-8859-1")


# Cleaning the data
def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r'https?:\/\/\s+', '', text)

    return text

df1['headlines']= df1['headlines'].apply(cleanTxt)

# Calculating the polarity score
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

df1['Polarity']=df1['headlines'].apply(getPolarity)
st.set_option('deprecation.showPyplotGlobalUse', False)
#Plot the WordCloud
allWord = ''.join([twts for twts in df1['headlines']])
#allWord = open(df1['headlines']).read()
#pre_process = ' '.join([word for word in allWord.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
wordCloud = WordCloud(width = 500, height=300, random_state = 21, max_font_size = 119).generate(allWord)
#return_obj = wordcloud.visualize(allWord, per_word_coloring = False)
#wc = WordCloud(stopwords=STOPWORDS, background_color='#000', height=550, width=550).generate(pre_process)
plt.imshow(wordCloud)
plt.axis('off')
#plt.show()
st.pyplot()




#Predict the Sentiment using polarity score
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df1['Analysis']=df1['Polarity'].apply(getAnalysis)


#Convert the dataframe into Excelsheet
df1.to_excel(r'Data/Result_Data.xlsx' , index = False)

# Get the percentage of positive headlines
pheadlines = df1[df1.Analysis == 'Positive']
pheadlines = pheadlines['headlines']

round( (pheadlines.shape[0] / df1.shape[0])* 100, 1)

#Get the percentage of negative headlines
nheadlines = df1[df1.Analysis == 'Negative']
nheadlines = nheadlines['headlines']
round( (nheadlines.shape[0] / df1.shape[0])* 100, 1)


#Show the value counts

df1['Analysis'].value_counts()

#plot and visualize the counts
#plt.title('Sentiment Analysis')
#plt.xlabel('Sentiment')
#plt.ylabel('Counts')
#df1['Analysis'].value_counts().plot(kind = 'bar')
#plt.show()
@st.cache
def load_data(nrows):
    data=pd.read_csv('Data/Result_Data.csv', nrows=nrows, encoding="ISO-8859-1")
    return data
df1=load_data(50)
st.subheader('Sentiment Analysis')
st.write(df1)
st.bar_chart(df1['Polarity'])


