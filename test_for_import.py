import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# Nutzer darf ausschen welche Aktien er Analysieren möchte oder sich viszualisieren möchte


options = st.multiselect(
    'What are your Stocks',
    ['AAPL', 'BYDDF', 'EONGY', 'LNVGF', 'NIO', 'PLUN.F', 'TSLA', 'TKA.DE', 'XIACF'])

button_for_the_download = st.button('Download Data for the Stocks')

if button_for_the_download:
    tickers = options
    data = pd.DataFrame()

    end_date = datetime.today()
    start_date = end_date - timedelta(days=2 * 365)
    
    for option in tickers:
        stock_data = yf.download(option, start=start_date, end=end_date)
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            data[(option, col)] = stock_data[col]
        stock_data['Formatted Date'] = stock_data.index.strftime('%Y-%m')
        data[(option, 'Formatted Date')] = stock_data['Formatted Date']

st.dataframe(data)






### mit einer csv es versuchen da versuche begrenzt sind 




