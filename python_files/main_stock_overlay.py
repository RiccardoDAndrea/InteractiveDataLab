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
import requests
from bs4 import BeautifulSoup

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


def get_quote_table(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}"
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.find_all('table')
        
        if len(tables) >= 2:
            data = tables[0].find_all('tr') + tables[1].find_all('tr')
            quote_data = {row.find('td', class_='C($primaryColor)').text: row.find('td', class_='Ta(end)').text for row in data}
            return quote_data
        else:
            print("Error: Unable to find tables on the Yahoo Finance page.")
    else:
        print(f"Error: Unable to fetch data from {url}. Status code: {response.status_code}")







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
                       
            # P/E Ratio and other metrics
            st.markdown('## Metrics')
            st.columns()
            for stock_option in stock_options:
                stock_info = get_quote_table(stock_option)
                if stock_info:
                    PE_Ratio = stock_info.get('PE Ratio (TTM)', 'N/A')
                    st.metric(label=f"P/E Ratio ({stock_option})", value=PE_Ratio)
                    dividends_data = yf.Ticker(stock_option).dividends
                    if not dividends_data.empty:
                        last_dividend = str(dividends_data.iloc[-1])
                        st.metric(label=f"Last Dividend ({stock_option})", value=last_dividend)
                    else:
                        st.warning(f"No dividend data available for {stock_option}")
                else:
                    st.warning(f"Unable to retrieve data for {stock_option}")

        else:
            st.warning("No data available.")

         

 
            
            












