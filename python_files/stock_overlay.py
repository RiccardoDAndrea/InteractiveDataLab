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

no_data_avaible_female = load_lottieurl('https://lottie.host/70333dae-5d9d-4887-ac38-25dcbfe23e80/3TCrl817lO.json')
stock_options = st.multiselect(
    'What are your Stocks',
    options = ['AAPL', 'BYDDF', 'EONGY', 'LNVGF', 'NIO', 'PLUN.F', 'TSLA', 'TKA.DE', 'XIACF'])

if stock_options is not None:
    button_for_the_download = st.button('Download Data for the Stocks')
    if button_for_the_download:
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        data = pd.DataFrame()  # Sie erstellen ein leeres DataFrame, aber später überschreiben Sie es direkt mit Daten aus einer CSV-Datei.
        end_date = datetime.today()
        start_date = end_date - timedelta(days=2 * 365)
        
        for stock_option in stock_options:
            # Hier scheinen Sie die Daten aus einer CSV-Datei zu lesen, anstatt sie von einer API wie yfinance herunterzuladen.
            data = pd.read_csv('Dataset/stock_data.csv')
            if data.empty:
                st.write('NO Data')
    st.dataframe(data)  # Sie zeigen das DataFrame innerhalb der Schleife an, was dazu führt, dass es mehrmals angezeigt wird.
        
        

        






### mit einer csv es versuchen da versuche begrenzt sind 




