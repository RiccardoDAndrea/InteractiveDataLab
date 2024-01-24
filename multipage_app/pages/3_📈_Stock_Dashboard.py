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


### L O T T I E _ A N I M A T I O N _ S T A R T
no_options_choosen = load_lottieurl('https://lottie.host/afb47212-38e1-4ec5-975d-50eddac6ec7f/oiOK9YPj3b.json')
no_metric_choosen = load_lottieurl('https://lottie.host/c74ab8f3-eeff-45a6-86e1-122efa01fe85/MJAkJKqTYl.json')
no_chart_choosen = load_lottieurl('https://lottie.host/68f31367-9664-481e-b15b-b4abcd8f2366/z9KBlfIHRC.json')
no_company_information_choosen = load_lottieurl('https://lottie.host/6cb318a1-c1ea-4c58-afe0-c2dc7d2d6c85/yUCs9rpnwY.json')
### L O T T I E _ A N I M A T I O N _ E N D 


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


info_text_stocks = st.expander('Information on how to find you Stocks ?')
with info_text_stocks:

        st.info("""If you want to find your stock, go to https://de.finance.yahoo.com/ and enter only the ticker symbol of the 
                stock on the Streamlit page. For example, Apple would be 'AAPL'""", icon="ℹ️")

stock_options = st.text_input("Enter your Stocks (comma-separated)", value='AAPL, TSLA')
stock_options = [stock.strip() for stock in stock_options.split(',')]  # Teilen Sie die Eingabe am Komma und entfernen Sie Leerzeichen    


start_date, end_date = st.columns(2)
with start_date:
    start_date_input = st.date_input("Start")
with end_date:
    end_date_input = st.date_input("Last day")



close_df = pd.DataFrame()

for stock_option in stock_options:
    try:
        data = yf.download(stock_option, start=start_date_input, end=end_date_input)
        
        if 'Close' in data.columns:
            close_df[stock_option] = data['Close']
            
    except Exception as e:
        st.warning('Enter your Stocks')
        st_lottie(no_options_choosen, 
                width=700, 
                height=500, 
                loop=True, 
                quality='medium')

# Check if close_df ]

if not close_df.empty:
    close_df.reset_index(inplace=True)
    close_df['Date'] = pd.to_datetime(close_df['Date']).dt.date

    st.dataframe(close_df, 
                hide_index=True, 
                use_container_width=True)
    # L I N E _ C H A R T
        
    # M E T R I C S 
    metrics_expander = st.expander(label="Metrics")
    with metrics_expander:
        st.markdown('## Metrics')
        metrics_filter = st.multiselect(label="Which Metrics do you want to display ?",
                                        options=[   'Business Summary',
                                                    'Trailing PE', 
                                                    'Dividends',
                                                    'PE Ratio',
                                                    'P/B ratio',
                                                    'Debt-to-Equity Ratio',
                                                    'Free Cash Flow',
                                                    'PEG Ratio',
                                                    'EBITDA', 
                                                    'Revenue', 
                                                    'Short Ratio'
                                                ])
        
        if len(metrics_filter) == 0:
            st.info('Choose you metrics', icon="ℹ️")
            st_lottie(no_metric_choosen, 
            width=1650,
            height=400, 
            loop=True, 
            quality='medium')
        
    


    # M E T R I C S _ F I L T E R _ S T A R T
        if metrics_filter:
            
            for stock_option in stock_options:    
                Company_stock = yf.Ticker(stock_option)
                financials = Company_stock.get_financials()

                Trailing_PE_col, Dividends_col = st.columns(2)
                PE_Ratio_col,P_B_ratio_col = st.columns(2)
                Debt_to_Equity_Ratio_col, Free_cash_flow_col = st.columns(2)
                EBITDA_col, Revenue_col = st.columns(2)
                short_ratio_col, operating_income_col = st.columns(2)
                EBITDA_col, Operating_Income_col = st.columns(2)


                if 'Business Summary' in metrics_filter:
                    stock_info = yf.Ticker(stock_option)
                    BusinessSummary = stock_info.get_info()
                
                    # Extrahiere den Text des langen Geschäftszusammenfassungsfelds
                    long_summary = BusinessSummary.get('longBusinessSummary', '')
                    
                    # Zeige den Text mit Markdown-Formatierung an
                    st.markdown(f"**Business Summary for :orange[**{stock_option}**]:**")
                    st.markdown(long_summary)


                with Trailing_PE_col:
                    if 'Trailing PE' in metrics_filter:
                            stock_info = yf.Ticker(stock_option).info
                            PE_Ratio_ttm = stock_info.get('trailingPE', 'N/A')
                            if PE_Ratio_ttm != 'N/A':
                                    st.metric(label=f"trailing PE (:orange[***{stock_option}***])",
                                                value=PE_Ratio_ttm)
                            else:
                                st.info(f'No data available for trailing PE of **{stock_option}**')
                

                with Dividends_col:
                    if 'Dividends' in metrics_filter:
                        dividends_data = yf.Ticker(stock_option).dividends
                        if not dividends_data.empty:
                                last_dividend = dividends_data.iloc[-1]
                                last_dividend_str = f"{last_dividend:.2f} EUR"  # Format the dividend value
                                st.metric(label=f"Last Dividend (:orange[***{stock_option}***] in EUR)", value=last_dividend_str)
                        else:
                            st.info(f'No dividend data available or (:orange[***{stock_option}***]) does not pay dividends.')    


                with PE_Ratio_col:   
                    if 'PE Ratio' in metrics_filter:
                        stock_info = yf.Ticker(stock_option).info
                        pe_ratio = stock_info.get('trailingPE', 'N/A')
                        st.metric(label=f"P/E Ratio (:orange[***{stock_option}***])", value=pe_ratio)
                

                with P_B_ratio_col:
                    if 'P/B ratio' in metrics_filter:
                        stock_info = yf.Ticker(stock_option).info
                        pb_ratio = stock_info.get('priceToBook', 'N/A')
                        st.metric(label=f"P/B Ratio (:orange[***{stock_option}***])", value=pb_ratio)
                

                with Debt_to_Equity_Ratio_col:
                    if 'Debt-to-Equity Ratio' in metrics_filter:
                        stock_info = yf.Ticker(stock_option).info
                        debt_equity_ratio = stock_info.get('debtToEquity', 'N/A')
                        
                        if debt_equity_ratio != 'N/A':
                            st.metric(label=f"Debt-to-Equity Ratio (:orange[***{stock_option}***])", value=debt_equity_ratio)
                        else:
                            st.write('No data retrieved')
                    

               
                if 'Free Cash Flow' in metrics_filter:
                    Free_cash_flow = yf.Ticker(stock_option)
                    Free_cash_flow_df = Free_cash_flow.cash_flow.loc['Free Cash Flow']
                    Free_cash_flow_df.index = Free_cash_flow_df.index.date
                    st.subheader(f":orange[**{stock_option}**] - Free Cash Flow Over time:")

                    bar_chart_free_cash_flow = px.bar(
                        x=Free_cash_flow_df.index,  # Use the date index as x-axis
                        y=Free_cash_flow_df.values,  # Use the Free Cash Flow values as y-axis
                        labels={'x': 'Date', 'y': 'Free Cash Flow'},
                        text_auto=True
                    )
                    bar_chart_free_cash_flow.update_layout(
                        xaxis=dict(showgrid=True),
                        yaxis=dict(showgrid=True)
)

                    st.plotly_chart(bar_chart_free_cash_flow, 
                                    use_container_width=True)



                if 'EBITDA' in metrics_filter:
                    Company_stock = yf.Ticker(stock_option)
                    EBITDA_df = Company_stock.financials.loc['EBITDA']
                    EBITDA_df = pd.DataFrame(EBITDA_df)
                    EBITDA_df.index.names = ['Date']
                    EBITDA_df = EBITDA_df.reset_index()
                    st.subheader(f":orange[**{stock_option}**] - EBITDA Over Time:")
                    bar_chart_free_cash_flow = px.bar(EBITDA_df, 
                                                      x='Date', 
                                                      y='EBITDA', 
                                                      text_auto=True)
                    st.plotly_chart(bar_chart_free_cash_flow, use_container_width=True)

            

                
                if 'Revenue' in metrics_filter:
                    financials = Company_stock.get_financials()
                    revenue = financials.loc['OperatingIncome':'OperatingExpense'].T
                    revenue.index.names = ['Date']
                    revenue = revenue.reset_index()
                    revenue['OperatingIncome'] = revenue['OperatingIncome'].astype(float)
                    revenue['OperatingExpense'] = revenue['OperatingExpense'].astype(float)

                    # Automatische Skalierung
                    scaling_factor = max(revenue[['OperatingIncome', 'OperatingExpense']].max()) / 10000000
                    revenue[['OperatingIncome', 'OperatingExpense']] /= scaling_factor
                    st.subheader(f":orange[**{stock_option}**] - Operating Income and Expense Over Time:")
                    # Create a bar chart using plotly.express
                    bar_chart_revenue = px.bar(revenue, x='Date', y=['OperatingIncome', 'OperatingExpense'], barmode='group',
                                labels={'value': f'Amount (scaled by {scaling_factor:.0f})', 'variable': 'Category'},
                                    text_auto=True)
                    bar_chart_revenue.update_layout(
                        xaxis=dict(showgrid=True),
                        yaxis=dict(showgrid=True))

                    st.plotly_chart(bar_chart_revenue, 
                                    use_container_width=True)

            
                with short_ratio_col:
                    if 'Short Ratio' in metrics_filter:
                        short_ratio = Company_stock.info.get('shortRatio', 'N/A')
                        st.subheader(f":orange[**{stock_option}**] - Short Ratio:")
                        st.metric(label='Short Ratio', value=str(short_ratio))


               
                
            

                st.divider()
            # M E T R I C S _ F I L T E R _ E N D
        

    # V I S Z U A L I S A T I O N _ S T A R T
charts_vis = st.expander(label="Chart Visualization")

with charts_vis:
    which_company, which_metric = st.columns(2)

    with which_company:
        selected_companies = st.multiselect(label='Which company to display', options=stock_options)

    with which_metric:
        options_for_metric_vis = [
            'Stock Price', 'Trailing PE', 'Dividends', 'PE Ratio', 'P/B ratio',
            'Debt-to-Equity Ratio', 'Free Cash Flow', 'PEG Ratio', 
            'EBITDA', 'Revenue', 'Short Ratio', 'Operating Income'
        ]
        selected_metrics = st.multiselect(label='Which metric to display', options=options_for_metric_vis)

    if 'Stock Price' in selected_metrics:
        st.markdown('## Line Chart')

        line_chart = px.line(close_df,
                                x='Date',
                                y=selected_companies,
                                title=f'Stock Prices Over Time <span style="color:orange">{", ".join(selected_companies)}</span>')

        st.plotly_chart(line_chart, use_container_width=True)
    if 'Free Cash Flow' in selected_metrics:
        for company in selected_companies:
            company_free_cash_flow = Free_cash_flow_df[company]

            bar_chart_free_cash_flow = px.bar(
                x=company_free_cash_flow.index,  # Use the date index as x-axis
                y=company_free_cash_flow.values,  # Use the Free Cash Flow values as y-axis
                labels={'x': 'Date', 'y': 'Free Cash Flow'},
                title=f'Free Cash Flow Over Time for {company} <span style="color:orange">{company}</span>',
                text_auto=True
            )

            st.plotly_chart(bar_chart_free_cash_flow, use_container_width=True)



    # V I S Z U A L I S A T I O N _ E N D      
        



# C O M P A N Y _ I N F O R M A T I O N _ M E T R I C S _ S T A R T
company_information_expander = st.expander(label='Company Information')
Company_vizualisation = st.expander(label="Vizusalisation of the Company Key Numbers")

        
# C O M P A N Y _ I N F O R M A T I O N _ M E T R I C S _ E N D        

st.divider()
newspaper_expander = st.expander(label="News about your Stocks")

# G E T _ N E W S _ F O R _ C O M P A N Y _ S T A R T
with newspaper_expander:


    for stock_option in stock_options:
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

    
# G E T _ N E W S _ F O R _ C O M P A N Y _ E N D



        


        
        












