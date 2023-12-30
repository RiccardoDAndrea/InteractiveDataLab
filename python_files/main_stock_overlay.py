import streamlit as st
from streamlit_lottie import st_lottie
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import time
import requests

def load_lottieurl(url:str):
    """ 
    The follwing function request a url from the homepage
    lottie files if status is 200 he will return
    instand we can use this func to implement lottie files for 
    our Homepage
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

### START OF THE WEBPAGE ### 

st.title('Stock Dashboard') 
no_data_avaible_female = load_lottieurl('https://lottie.host/70333dae-5d9d-4887-ac38-25dcbfe23e80/3TCrl817lO.json')
stock_options = st.multiselect(
    'What are your Stocks',
    options = ['AAPL', 'BYDDF', 'EONGY', 'LNVGF', 'NIO', 'PLUN.F', 'TSLA', 'TKA.DE', 'XIACF'])

if stock_options:
    button_for_the_download = st.button('Download Data for the Stocks')
    
    start_date, end_date = st.columns((1, 2))
    
    with start_date:
        start_date_input = st.date_input("Start")
    with end_date:
        end_date_input = st.date_input("Last day")
    
    if button_for_the_download:
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        close_df = pd.DataFrame()
        for stock_option in stock_options:
            data = yf.download(stock_option, start=start_date_input, end=end_date_input)
            if 'Close' in data.columns:
                close_df[stock_option] = data['Close']
        if not close_df.empty:
            close_df.reset_index(inplace=True)  # Hinzufügen der ausgewählten Startdatum-Spalte
            close_df['Date'] = pd.to_datetime(close_df['Date']).dt.date  # Hier wird nur das Datum extrahiert
            st.dataframe(close_df, hide_index=True)
        else:
            st.warning("No data available.")



if len(stock_options) == 0:
    st.markdown("**you have nothing entered**")        
    no_data_found = load_lottieurl('https://lottie.host/70333dae-5d9d-4887-ac38-25dcbfe23e80/3TCrl817lO.json')   
    st_lottie( no_data_found,
                quality='high',
                width=650,
                height=400)
        










