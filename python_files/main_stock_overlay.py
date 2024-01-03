import streamlit as st
from streamlit_lottie import st_lottie
import yfinance as yf
import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import newspaper
import nltk
nltk.download('punkt')



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
        soup = BeautifulSoup(response.text, 
                             'html.parser')
        
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
#no_data_avaible_female = load_lottieurl('https://lottie.host/70333dae-5d9d-4887-ac38-25dcbfe23e80/3TCrl817lO.json')

info_text_stocks = st.expander('Information on how to find you Stocks ?')
with info_text_stocks:

        st.info("""If you want to find your stock, go to https://de.finance.yahoo.com/ and enter only the ticker symbol of the 
                stock on the Streamlit page. For example, Apple would be 'APPL'""", icon="ℹ️")

stock_options = st.text_input("Enter your Stocks (comma-separated)",
                              value='AAPL, TSLA')
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
        my_bar = st.progress(0, 
                             text=progress_text)
        
        close_df = pd.DataFrame()

        for stock_option in stock_options:
            data = yf.download(stock_option, 
                               start=start_date_input, 
                               end=end_date_input)
            
            if 'Close' in data.columns:
                close_df[stock_option] = data['Close']
        if not close_df.empty:
            close_df.reset_index(inplace=True)
            close_df['Date'] = pd.to_datetime(close_df['Date']).dt.date

            st.dataframe(close_df, hide_index=True, 
                         use_container_width=True)
            
            # Line Chart
            st.markdown('## Line Chart')
            line_chart = px.line(close_df, 
                                 x='Date', 
                                 y=stock_options, 
                                 title='Stock Prices Over Time')
            
            st.plotly_chart(line_chart)
                       
            # P/E Ratio and other metrics
            st.markdown('## Metrics')
            PE_ratio_col, dividends_col = st.columns(spec=(2,1))

            for stock_option in stock_options:
                stock_info = get_quote_table(stock_option)

                if stock_info:
                    PE_Ratio = stock_info.get('PE Ratio (TTM)', 'N/A')
                    with PE_ratio_col:
                        st.metric(label=f"P/E Ratio ({stock_option})", 
                                  value=PE_Ratio)
                    dividends_data = yf.Ticker(stock_option).dividends

                    with dividends_col:
                        if not dividends_data.empty:
                            last_dividend = str(dividends_data.iloc[-1])
                            st.metric(label=f"Last Dividend ({stock_option}) in EUR", 
                                      value=last_dividend)
                        else:
                            st.warning(f"No dividend data available for {stock_option}")
                else:
                    st.warning(f"Unable to retrieve data for {stock_option}")

        else:
            st.warning("No data available.")
newspaper_expander = st.expander(label="News about your Stocks")

with newspaper_expander:
    stock_options = st.text_input("Enter your Stocks (comma-separated)", value='AAPL, TSLA', key="input_news")
    stocks = [stock.strip() for stock in stock_options.split(',')]

    for stock_option in stocks:
        st.header(f"News for {stock_option}")
        url = f'https://finance.yahoo.com/quote/{stock_option}/'  # Ändern Sie dies entsprechend Ihrer Website oder Newsquelle
        article = newspaper.Article(url)
       
        try:
            article.download()
            article.parse()
            authors = article.authors
            article_meta_data = article.meta_data
            article_published_date = article_meta_data.get('article:published_time', 'N/A')
            st.write("Authors:", ', '.join(authors))
            st.write("Published Date:", article_published_date)
            article.nlp()

            tab1, tab2 = st.tabs(['Full Text', 'Summary'])
            with tab1:
                st.write(article.authors)
                st.write(article_published_date)
                st.write(article.text)

            with tab2:
                st.write(article.authors)
                st.write(article.summary)

        except Exception as e:
            st.error(f"Error processing news for {stock_option}: {e}")



         

 
            
            












