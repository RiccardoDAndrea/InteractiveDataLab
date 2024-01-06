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
        Overview,Company_Information = st.tabs(['Overview', 
                                                'Company Information'])    
        # L I N E _ C H A R T
        with Overview:
            charts_vis = st.expander(label="Viszualisation")
            with charts_vis:
                st.markdown('## Line Chart')
                stocks_to_display = st.multiselect('Which Stocks should be displayed in the Chart ?',
                            options= stock_options)
                line_chart = px.line(close_df, 
                                        x='Date', 
                                        y=stocks_to_display, 
                                        title='Stock Prices Over Time')
                
                st.plotly_chart(line_chart, use_container_width=True)



            # M E T R I C S
            metrics_expander = st.expander(label="Metrics")
            with metrics_expander:
                st.markdown('## Metrics')
                metrics_filter = st.multiselect(label="Which Metrics do you want to display ?",
                                                options=['PE Ratio (TTM)', 
                                                         'Dividends',
                                                         'PE Ratio',
                                                         'P/B ratio',
                                                         'Debt-to-Equity Ratio',
                                                         'Free Cash Flow',
                                                         'PEG Ratio',
                                                         'metric_8'
                                                        ])

                PE_ratio_ttm_col, dividends_col = st.columns(spec=(2, 1))
                PE_Ratio_col, PB_ratio_col = st.columns(spec=(2, 1))
                debt_equity_ratio_col, Free_cash_flow_col = st.columns(spec=(2, 1))
                PEG_ratio_col, metric_8_col = st.columns(spec=(2, 1))

                for stock_option in stock_options:
                    stock_info = get_quote_table(stock_option)

                    if stock_info:

                        if 'PE Ratio (TTM)' in metrics_filter:
                            PE_Ratio = stock_info.get('PE Ratio (TTM)', 'N/A')
                            with PE_ratio_ttm_col:
                                st.metric(label=f"P/E Ratio ({stock_option})",
                                          value=PE_Ratio)
                                
                        if 'Dividends' in metrics_filter:
                            dividends_data = yf.Ticker(stock_option).dividends
                            with dividends_col:
                                if not dividends_data.empty:
                                    last_dividend = str(dividends_data.iloc[-1])
                                    st.metric(label=f"Last Dividend ({stock_option}) in EUR",
                                            value=last_dividend)
                                else:
                                    st.info(f"No dividend data available for {stock_option}")
                        
                        
                        if 'PE Ratio' in metrics_filter:
                            stock_info = yf.Ticker(stock_option).info
                            pe_ratio = stock_info.get('trailingPE', 'N/A')
                            with PE_Ratio_col:
                                st.metric(label=f"P/E Ratio ({stock_option})", value=pe_ratio)
                        else:
                            st.warning(f"No P/E Ratio data available for {stock_option}")

                        if 'P/B ratio' in metrics_filter:
                            stock_info = yf.Ticker(stock_option).info
                            pb_ratio = stock_info.get('priceToBook', 'N/A')
                            with PB_ratio_col:
                                st.metric(label=f"P/B Ratio ({stock_option})", value=pb_ratio)
                        else:
                            st.warning(f"No P/B Ratio data available for {stock_option}")

                        if 'Debt-to-Equity Ratio' in metrics_filter:
                            stock_info = yf.Ticker(stock_option).info
                            debt_equity_ratio = stock_info.get('debtToEquity', 'N/A')
                            with debt_equity_ratio_col:
                                st.metric(label=f"Debt-to-Equity Ratio ({stock_option})", value=debt_equity_ratio)
                        else:
                            st.warning(f"No Debt-to-Equity data available for {stock_option}")
                        
                        if 'Free Cash Flow' in metrics_filter:
                            Free_cash_flow = yf.Ticker(stock_option)
                            Free_cash_flow_df = Free_cash_flow.cash_flow.loc['Free Cash Flow']

                            # Extrahiere nur das Datum
                            Free_cash_flow_df.index = Free_cash_flow_df.index.date

                            with Free_cash_flow_col:
                                st.dataframe(data=Free_cash_flow_df, use_container_width=False)                       
                        else:
                            st.warning(f"No Free Cash Flow data available for {stock_option}")
                        
                        if 'PEG Ratio' in metrics_filter:
                            with PEG_ratio_col:
                                st.write('Test')
                        else:
                            st.warning(f"No PEG Ratio data available for {stock_option}")
                        if 'metric_8' in metrics_filter:
                            
                            aapl_info = yf.Ticker(stock_option).get_info()
                            aapl_info['longBusinessSummary']
                            with metric_8_col:
                                st.write(aapl_info['longBusinessSummary'])
                        
                        st.divider()




        with Company_Information:
            company_information_expander = st.expander(label='Company Information')
            Company_vizualisation = st.expander(label="Vizusalisation of the Company Key Numbers")
            with company_information_expander:
                Company_info_to_display = st.multiselect('Which Financial information should be displayed?', options=["EBITDA", 
                                                                                                                    "Revenue", 
                                                                                                                    "Short Ratio",
                                                                                                                    "Operating Income"])
                EBITDA_col, Revenue_col = st.columns(2)
                Short_ratio_col, Operating_income_col = st.columns(2)

                for stock_option in stock_options:
                    Company_stock = yf.Ticker(stock_option)
                    financials = Company_stock.get_financials()

                    with EBITDA_col:
                        if 'EBITDA' in Company_info_to_display:
                            ebitda_data = financials.loc['EBITDA', :]
                            st.subheader(f"{stock_option} - EBITDA:")
                            st.write(ebitda_data)

                    with Revenue_col:
                        if 'Revenue' in Company_info_to_display:
                            revenue = financials.loc['CostOfRevenue':'TotalRevenue']
                            revenue = revenue / 1000000000
                            revenue = revenue.T
                            st.subheader(f"{stock_option} - Revenue:")
                            st.write(revenue)

                    with Short_ratio_col:
                        if 'Short Ratio' in Company_info_to_display:
                            short_ratio = Company_stock.info.get('shortRatio', 'N/A')
                            st.subheader(f"{stock_option} - Short Ratio:")
                            st.metric(label='Short Ratio', value=str(short_ratio))

                    with Operating_income_col:
                        if 'Operating Income' in Company_info_to_display:
                            operating_income = financials.loc['OperatingIncome']
                            normalized_operating_income = operating_income / 1_000_000_000
                            transposed_operating_income = normalized_operating_income.T

                            st.subheader(f"{stock_option} - Operating Income:")
                            st.write(transposed_operating_income)
                            
            with Company_vizualisation :
                Company_info_to_display_vis = st.multiselect(label='Which Vizualisation do you want?', options=["EBITDA", 
                                                                                                                "Revenue", 
                                                                                                                "Operating Income"])
                if 'EBITDA' in Company_info_to_display_vis:
                    ebitda_data = financials.loc['EBITDA', :]
                    ebitda_df = pd.DataFrame(ebitda_data).reset_index()  # Resetting index to convert the index to a column
                    ebitda_df = ebitda_df.rename(columns={'index': 'Date'})
                    bar_chart_EBITDA = px.bar(ebitda_df,
                        title='EBITDA Over Time',
                        x='Date',
                        y='EBITDA',
                        text_auto=True)
                    st.plotly_chart(bar_chart_EBITDA, use_container_width=True)

                if 'Operating Income' in Company_info_to_display_vis:
                # Display the chart using Streamlit
                    revenue = financials.loc['OperatingIncome']
                    revenue = revenue/1000000000
                    revenue_df = pd.DataFrame(revenue).reset_index()  # Resetting index to convert the index to a column
                    revenue_df = revenue_df.rename(columns={'index': 'Date'})
                    bar_chart_OperatingIncome = px.bar(revenue_df,
                        title='Operating Income Over Time',
                        x='Date',
                        y='OperatingIncome',
                        text_auto=True)
                    st.plotly_chart(bar_chart_OperatingIncome, use_container_width=True)
                
                
                    
        

st.divider()
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
            #st.write("Authors:", ', '.join(authors))
            #st.write("Published Date:", article_published_date)
            article.nlp()

            tab1, tab2 = st.tabs(['Full Text', 'Summary'])
            with tab1:
                #st.write(article.authors)
                #st.write(article_published_date)
                st.write(article.text)

            with tab2:
                #st.write(article.authors)
                st.write(article.summary)

        except Exception as e:
            st.error(f"Error processing news for {stock_option}: {e}")
    else:
        st.warning("No data available.")



         

 
            
            












