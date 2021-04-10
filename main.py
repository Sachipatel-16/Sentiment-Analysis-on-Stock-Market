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

image = Image.open("C:/Users/priya/PycharmProjects/StockPrediction/PIC4.jpg")
st.image(image, use_column_width = True)

# Creating sidebar
st.sidebar.header('User Input')

# Function to create user input
def get_input():
    start_date = st.sidebar.text_input("Start Date", "2018-01-02")
    end_date = st.sidebar.text_input("End Date", "2018-03-16")
    #stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
    option = st.sidebar.selectbox("Select any ticker from the list.", ('AAPl - Apple',
                                                                       'AXP - American Express',
                                                                       'BA - Boeing',
                                                                       'CAT - Caterpillar Inc',
                                                                       'CSCO - Cisco Systems Inc',
                                                                       'CVX - Chevron Group',
                                                                       'DIS - Walt Disney Company',
                                                                       'DWDP - Dowdupont',
                                                                       'GE - Genral Electric Company',
                                                                       'GS - Goldman Sachs Group',
                                                                       'HD - Home Depot',
                                                                       'IBM - International Business Machines',
                                                                       'INTC - Intel Corp',
                                                                       'JNJ - Jhonson & Jhonson',
                                                                       'JPM - JP Morgan Chase & Company',
                                                                       'KO - Coca-cola Company',
                                                                       'MCD - McDonald\'s Corp',
                                                                       'MMM - 3M Company',
                                                                       'MRK - Merk & Company',
                                                                       'MSFT - Microsoft Corp',
                                                                       'NKE- Nike Inc',
                                                                       'PFE - Pfizer Inc',
                                                                       'PG - Procter & Gamble Company',
                                                                       'TRV - The Travelers Company',
                                                                       'UNH - Unitedhealth Group Inc',
                                                                       'UTX - United Technologies Corp',
                                                                       'V - Visa Inc',
                                                                       'VZ - Verizon Communication Inc',
                                                                       'WMT - Wal-Mart Stores',
                                                                       'XOM - Exxon Mobil Corp'))
    option = option.split(" ", 1)
    option = option[0]
    return start_date, end_date, option #, stock_symbol

# Function to get the company name
def get_company_name(option):
    if(option == 'AAPL'):
        return 'Apple'
    elif(option == 'AXP'):
        return 'American Express'
    elif(option == 'BA'):
        return 'Boeing'
    elif (option == 'CAT'):
        return 'Caterpillar Inc'
    elif (option == 'CSCO'):
        return 'Cisco Systems Inc'
    elif (option == 'CVX'):
        return 'Chevron Group'
    elif (option == 'DIS'):
        return 'Walt Disney Company'
    elif (option == 'DWDP'):
        return 'Dowdupont'
    elif (option == 'GE'):
        return 'Genral Electric Company'
    elif (option == 'GS'):
        return 'Goldman Sachs Group'
    elif (option == 'HD'):
        return 'Home Depot'
    elif (option == 'IBM'):
        return 'International Business Machines'
    elif (option == 'INTC'):
        return 'Intel Corp'
    elif (option == 'JNJ'):
        return 'Jhonson & Jhonson'
    elif (option == 'JPM'):
        return 'JP Morgan Chase & Company'
    elif (option == 'KO'):
        return 'Coca-cola Company'
    elif (option == 'MCD'):
        return 'McDonald\'s Corp'
    elif (option == 'MMM'):
        return '3M Company'
    elif (option == 'MRK'):
        return 'Merk & Company'
    elif (option == 'MSFT'):
        return 'Microsoft Corp'
    elif (option == 'NKE'):
        return 'Nike Inc'
    elif (option == 'PFE'):
        return 'Pfizer Inc'
    elif (option == 'PG'):
        return 'Procter & Gamble Company'
    elif (option == 'TRV'):
        return 'The Travelers Company'
    elif (option == 'UNH'):
        return 'Unitedhealth Group Inc'
    elif (option == 'UTX'):
        return 'United Technologies Corp'
    elif (option == 'V'):
        return 'Visa Inc'
    elif (option == 'VZ'):
        return 'Verizon Communication Inc'
    elif (option == 'WMT'):
        return 'Wal-Mart Stores'
    elif (option == 'XOM'):
        return 'Exxon Mobil Corp'

    else:
        'None'

# Function to get the dataset

def get_data(option,start,end):

    # Loading data
    if option.upper() == 'AAPL':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/AAPL.csv')
    elif option.upper() == 'AXP':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/AXP.csv')
    elif option.upper() == 'BA':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/BA.csv')
    elif option.upper() == 'CAT':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/CAT.csv')
    elif option.upper() == 'CSCO':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/CSCO.csv')
    elif option.upper() == 'CVX':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/CVX.csv')
    elif option.upper() == 'DIS':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/DIS.csv')
    elif option.upper() == 'DWDP':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/DWDP.csv')
    elif option.upper() == 'GE':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/GE.csv')
    elif option.upper() == 'GS':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/GS.csv')
    elif option.upper() == 'HD':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/HD.csv')
    elif option.upper() == 'IBM':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/IBM.csv')
    elif option.upper() == 'INTC':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/INTC.csv')
    elif option.upper() == 'JNJ':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/JNJ.csv')
    elif option.upper() == 'JPM':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/JPM.csv')
    elif option.upper() == 'KO':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/KO.csv')
    elif option.upper() == 'MCD':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/MCD.csv')
    elif option.upper() == 'MMM':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/MMM.csv')
    elif option.upper() == 'MRK':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/MRK.csv')
    elif option.upper() == 'MSFT':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/MSFT.csv')
    elif option.upper() == 'NKE':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/NKE.csv')
    elif option.upper() == 'PFE':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/PFE.csv')
    elif option.upper() == 'PG':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/PG.csv')
    elif option.upper() == 'TRV':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/TRV.csv')
    elif option.upper() == 'UNH':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/UNH.csv')
    elif option.upper() == 'UTX':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/UTX.csv')
    elif option.upper() == 'V':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/V.csv')
    elif option.upper() == 'VZ':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/VZ.csv')
    elif option.upper() == 'WMT':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/WMT.csv')
    elif option.upper() == 'XOM':
        df = pd.read_csv('C:/Users/priya/PycharmProjects/StockPrediction/Data/XOM.csv')
    else:
        df = pd.DataFrame(columns=['Date', 'Close', 'Open', 'Volume', 'Adj Close', 'High', 'Low'])

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


