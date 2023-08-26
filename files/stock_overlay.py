import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import time

# Nutzer darf ausschen welche Aktien er Analysieren möchte oder sich viszualisieren möchte


tickers = st.multiselect(
    'What are your Stocks',
    ['AAPL', 'BYDDF', 'EONGY', 'LNVGF', 'NIO', 'PLUN.F', 'TSLA', 'TKA.DE', 'XIACF'])

button_for_the_download = st.button('Download Data for the Stocks')

if button_for_the_download:
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.02)
    my_bar.progress(percent_complete + 1, text=progress_text)
    data = pd.DataFrame()

    end_date = datetime.today()
    start_date = end_date - timedelta(days=2 * 365)
    
    for ticker in tickers:
        # stock_data = yf.download(ticker, start=start_date, end=end_date)
        # for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        #     data[(ticker, col)] = stock_data[col]
        # stock_data['Formatted Date'] = stock_data.index.strftime('%Y-%m')
        # data[(ticker, 'Formatted Date')] = stock_data['Formatted Date']
        data = pd.read_csv('stock_data.csv')   
           
st.dataframe(data)






### mit einer csv es versuchen da versuche begrenzt sind 




