import streamlit as st
from streamlit_lottie import st_lottie
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import time
import requests
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go


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

stock_options = st.text_input("Enter your Stocks (comma-separated)")
stock_options = [stock.strip() for stock in stock_options.split(',')]  # Teilen Sie die Eingabe am Komma und entfernen Sie Leerzeichen


if len(stock_options) == 0:
    st.markdown("**you have nothing entered**")        
    no_data_found = load_lottieurl('https://lottie.host/70333dae-5d9d-4887-ac38-25dcbfe23e80/3TCrl817lO.json')   
    st_lottie( no_data_found,
                quality='high',
                width=650,
                height=400)
else :
    
    start_date, end_date = st.columns((1, 2))
    
    with start_date:
        start_date_input = st.date_input("Start")
    with end_date:
        end_date_input = st.date_input("Last day")

button_for_the_download = st.button('Download Data for the Stocks')
if button_for_the_download:
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        close_df = pd.DataFrame()
        for stock_option in stock_options:
            data = yf.download(stock_option, start=start_date_input, end=end_date_input)
            if 'Close' in data.columns:
                close_df[stock_option] = data['Close']
        if not close_df.empty:
            close_df.reset_index(inplace=True)
            close_df['Date'] = pd.to_datetime(close_df['Date']).dt.date
            st.dataframe(close_df, hide_index=True)
            
            # Line Chart
            st.markdown('## Line Chart')
            line_chart = px.line(close_df, x='Date', y=stock_options, title='Stock Prices Over Time')
            st.plotly_chart(line_chart)
            
            # P/E Ratio
            st.markdown('## P/E Ratio')
            pe_ratio_value = None
            if len(stock_options) > 0:
                # Get P/E ratio for the first stock option
                try:
                    stock_data = yf.Ticker(stock_options[0])
                    pe_ratio_value = stock_data.info['trailingPE']
                except Exception as e:
                    st.warning(f"Error retrieving P/E ratio for {stock_options[0]}: {e}")
            
            if pe_ratio_value is not None:
                st.metric(label="P/E Ratio", value=pe_ratio_value)
            else:
                st.warning("Unable to retrieve P/E ratio.")
        else:
            st.warning("No data available.")
         

 
            
            












