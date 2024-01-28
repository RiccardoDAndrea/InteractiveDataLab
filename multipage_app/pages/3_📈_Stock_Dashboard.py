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


### START OF THE WEBPAGE ### 

st.title('Stock Dashboard') 


info_text_stocks = st.expander('Information on how to find you Stocks ?')
with info_text_stocks:

        st.info("""If you want to find your stock, go to https://de.finance.yahoo.com/ and enter only the ticker symbol of the 
                stock on the Streamlit page. For example, Apple would be 'AAPL'""", icon="ℹ️")

stock_options = st.text_input("Enter your Stocks (comma-separated)", value = 'AAPL')
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

        if not data.empty:
            if 'Close' in data.columns:
                close_df[stock_option] = data['Close']
                close_df.reset_index(drop=True)
                # Überprüfe auf NaN-Werte in der 'Close'-Spalte
                if data['Close'].isnull().any():
                    st.warning(f'Warnung: {stock_option} contains NaN values in the "Close" column.')
            else:
                st.warning(f'No "Close" column found for {stock_option}')
        else:
            st.warning(f'No data for {stock_option}')
    except Exception as e:
        st.warning(f'Error when retrieving data for {stock_option}: {str(e)}')
        st_lottie(no_options_choosen, 
                width=700, 
                height=500, 
                loop=True, 
                quality='medium')



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
        show_explanation = st.toggle('Show Metric Explanations')

        metrics_filter = st.multiselect(label="Which Metrics do you want to display ?",
                                        options=[   'Business Summary',
                                                    'Stock Performance',
                                                    'Trailing PE', 
                                                    'Dividends',
                                                    'PE Ratio',
                                                    'P/B ratio',
                                                    'Debt-to-Equity Ratio',
                                                    'Free Cash Flow',
                                                    'PEG Ratio',
                                                    'EBITDA', 
                                                    'Revenue', 
                                                    'Short Ratio'],
                                                    key=1)
        
        # comes when the user dont choose any metrics
        if len(metrics_filter) == 0:
            st.info('Choose you metrics', icon="ℹ️")
            st_lottie(no_metric_choosen, 
                        width=1650,
                        height=400, 
                        loop=True, 
                        quality='medium')
        
##############################################################
### W I T H O U T _ E X P L A N A T I O N _ I S _ O N _ S T A R T;
##############################################################    
        
        if show_explanation == False:

    # M E T R I C S _ F I L T E R _ S T A R T
            if metrics_filter and show_explanation == False:

                
                for stock_option in stock_options:
                    # get data from the API generell Information    
                    Company_stock = yf.Ticker(stock_option)

                    # get data from the API financial information
                    financials = Company_stock.get_financials()
                    
                    ### M E T R I C S _ S T A R T 
                    if 'Business Summary' in metrics_filter:
                        stock_info = yf.Ticker(stock_option)
                        BusinessSummary = stock_info.get_info()
                        long_summary = BusinessSummary.get('longBusinessSummary', '')
                        st.subheader(f"**Business Summary for :orange[**{stock_option}**]:**")
                        st.markdown(long_summary)

                    if 'Stock Performance' in metrics_filter:
                        st.markdown('## Line Chart')
                    
                        # Allow user to select companies to show
                        
                        selected_companies_2 = st.multiselect('Which companies to show?', 
                                                options=stock_options, 
                                                default=stock_options,
                                                key=f'unique_key_for_multiselect_2_{stock_option}')
            

                        
                        # Filter the dataframe based on selected companies
                        close_df_selected = close_df[selected_companies_2]
                        
                        # Create a line chart using plotly.express
                        line_chart = px.line(close_df_selected, 
                                            x=close_df_selected.index, 
                                            y=selected_companies_2, 
                                            title=f'Stock Prices Over Time - {", ".join(selected_companies_2)}')
                        
                        # Update layout for better visualization
                        line_chart.update_layout(
                            xaxis_title='Date',
                            yaxis_title='Stock Price',
                            legend_title='Companies',
                            title=dict(text=f'Stock Prices Over Time - {", ".join(selected_companies_2)}', x=0.5),
                        )
                        
                        st.plotly_chart(line_chart, use_container_width=True, key='line_chart_no_explanation')       

                    Trailing_PE_col, Dividends_col = st.columns(2)

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

                    PE_Ratio_col,P_B_ratio_col = st.columns(2)

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
                    
                    Debt_to_Equity_Ratio_col, short_ratio_col = st.columns(2)

                    with Debt_to_Equity_Ratio_col:
                        if 'Debt-to-Equity Ratio' in metrics_filter:
                            stock_info = yf.Ticker(stock_option).info
                            debt_equity_ratio = stock_info.get('debtToEquity', 'N/A')
                            
                            if debt_equity_ratio != 'N/A':
                                st.metric(label=f"Debt-to-Equity Ratio (:orange[***{stock_option}***])", value=debt_equity_ratio)
                            else:
                                st.write('No data retrieved')
                        
                    with short_ratio_col:
                        if 'Short Ratio' in metrics_filter:
                            short_ratio = Company_stock.info.get('shortRatio', 'N/A')
                            #st.markdown(f":orange[**{stock_option}**] - Short Ratio:")
                            st.metric(label='Short Ratio', value=str(short_ratio))

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

                    

                    
                    st.divider()   

##############################################################
### W I T H O U T _ E X P L A N A T I O N _ I S _ O N _ E N D;
##############################################################
                    



##################################################
### E X P L A N A T I O N _ I S _ O N _ S T A R T;
##################################################     
                    

        if show_explanation == True:            
            if metrics_filter:
            
                for stock_option in stock_options:
                    # get data from the API generell Information    
                    Company_stock = yf.Ticker(stock_option[0])

                    # get data from the API financial information
                    financials = Company_stock.get_financials()
                    
                    ### M E T R I C S _ S T A R T 
                if 'Business Summary' in metrics_filter:
                    stock_info = yf.Ticker(stock_option)
                    BusinessSummary = stock_info.get_info()
                    long_summary = BusinessSummary.get('longBusinessSummary', '')
                    st.subheader(f"**Business Summary for :orange[**{stock_option}**]:**")
                    st.markdown(long_summary)

                if 'Stock Performance' in metrics_filter:
                    st.markdown('## Line Chart')
                
                    # Allow user to select companies to show
                    
                    selected_companies_1 = st.multiselect('Which companies to show?', options=stock_options, default=stock_options, key=3)

                    
                    # Filter the dataframe based on selected companies
                    close_df_selected = close_df[selected_companies_1]
                    
                    # Create a line chart using plotly.express
                    line_chart = px.line(close_df_selected, 
                                        x=close_df_selected.index, 
                                        y=selected_companies_1, 
                                        title=f'Stock Prices Over Time - {", ".join(selected_companies_1)}')
                    
                    # Update layout for better visualization
                    line_chart.update_layout(
                        xaxis_title='Date',
                        yaxis_title='Stock Price',
                        legend_title='Companies',
                        title=dict(text=f'Stock Prices Over Time - {", ".join(selected_companies_1)}', x=0.5),
                    )
                    
                    st.plotly_chart(line_chart, use_container_width=True)       

                Trailing_PE_col, expl_Trailing_PE = st.columns(2)
                with Trailing_PE_col:
                    if 'Trailing PE' in metrics_filter:
                            stock_info = yf.Ticker(stock_option).info
                            PE_Ratio_ttm = stock_info.get('trailingPE', 'N/A')
                            if PE_Ratio_ttm != 'N/A':
                                    st.metric(label=f"trailing PE (:orange[***{stock_option}***])",
                                                value=PE_Ratio_ttm)
                            else:
                                st.info(f'No data available for trailing PE of **{stock_option}**')
                
                with expl_Trailing_PE:
                    if 'Trailing PE' in metrics_filter:
                        st.info("""
                                Trailing P/E is a metric analysts use to assess stock value by comparing the current market price to earnings over the last four quarters.

                                - **Apples-to-Apples Evaluation:** Enables comparison of different stocks.
                                - **Economic Moats:** Some justify a higher P/E due to economic moats.
                                - **Dynamic Stock Prices:** Adjusts for changing stock prices.
                                - **Example:** If a stock is $50 with trailing EPS of $2, the trailing P/E is 25x.

                                **Example Usage:**
                                ```python
                                stock_price = 50
                                trailing_eps = 2
                                trailing_pe_ratio = stock_price / trailing_eps
                                print(f"Trailing P/E Ratio: {trailing_pe_ratio:.2f}x
                                """)

                dividends_col, expl_dividends_col = st.columns(2)
                if 'Dividends' in metrics_filter:
                    with dividends_col:
                        dividends_data = yf.Ticker(stock_option).dividends
                        if not dividends_data.empty:
                                last_dividend = dividends_data.iloc[-1]
                                last_dividend_str = f"{last_dividend:.2f} EUR"  # Format the dividend value
                                st.metric(label=f"Last Dividend (:orange[***{stock_option}***] in EUR)", 
                                          value=last_dividend_str)
                        else:
                            st.info(f'No dividend data available or (:orange[***{stock_option}***]) does not pay dividends.')   
                    
                    with expl_dividends_col:
                        st.info("""
                                A dividend is a payment made to shareholders for their investment 
                                in the equity of a company and usually comes from the **net profits of the company**
                                """)

                PE_Ratio_col, expl_PE_Ratio_col = st.columns(2)
                with PE_Ratio_col:
                    if 'PE Ratio' in metrics_filter:
                        stock_info = yf.Ticker(stock_option).info
                        pe_ratio = stock_info.get('trailingPE', 'N/A')
                        st.metric(label=f"P/E Ratio (:orange[***{stock_option}***])", value=pe_ratio)
            
                with expl_PE_Ratio_col:
                    if 'PE Ratio' in metrics_filter:
                        st.info("""
                                **Understanding P/E Ratio:**

                                The Price-to-Earnings (P/E) ratio is a key financial metric calculated by
                                dividing the market price of a stock by its annual earnings per share. 
                                While the stock price indicates what investors are willing to pay for 
                                ownership, the P/E ratio helps assess whether the price accurately 
                                reflects the company's earning potential and its long-term value.

                                For instance, if a company's stock is trading at $100 per share and 
                                generates $4 per share in annual earnings, the P/E ratio would be 25 (100 / 4).
                                 In simple terms, this means it would take 25 years of accumulated earnings 
                                to match the initial investment cost.

                                *Note: A lower P/E ratio may suggest that the stock is undervalued, 
                                while a higher P/E ratio may indicate a potentially overvalued stock.*
                                """)
                

                P_B_ratio_col, expl_P_B_ratio_col_col = st.columns(2)
                with P_B_ratio_col:
                    if 'P/B ratio' in metrics_filter:
                        stock_info = yf.Ticker(stock_option).info
                        pb_ratio = stock_info.get('priceToBook', 'N/A')
                        st.metric(label=f"P/B Ratio (:orange[***{stock_option}***])", value=pb_ratio)
                
                with expl_P_B_ratio_col_col:
                    if 'P/B ratio' in metrics_filter:
                        st.info("""
                                **Price-to-Book Ratio (P/B):**

                                P/B ratio compares a stock's market price to its book value, indicating market valuation of the company’s net worth. 
                                - High-growth companies often have P/B ratios > 1.0, while distressed companies may have ratios < 1.0.
                                - Important for assessing if a stock's price aligns with its balance sheet.

                                *Good P/B Ratio:*
                                - Varies by industry and market conditions.

                                *Bottom Line:*
                                - P/B < 1.0 suggests potential underpricing; > 1.0 may indicate overvaluation.
                                - Valued by investors seeking undervalued stocks.

                                *Note: Use P/B ratio in conjunction with other metrics for comprehensive analysis.*
                                """)

                debt_to_equity_ratio_col, expl_debt_to_equity_ratio_col = st.columns(2)
                with debt_to_equity_ratio_col:
                    if 'Debt-to-Equity Ratio' in metrics_filter:
                        stock_info = yf.Ticker(stock_option).info
                        debt_equity_ratio = stock_info.get('debtToEquity', 'N/A')
                        
                        if debt_equity_ratio != 'N/A':
                            st.metric(label=f"Debt-to-Equity Ratio (:orange[***{stock_option}***])",
                                      value=debt_equity_ratio)
                        else:
                            st.write('No data retrieved')

                with expl_debt_to_equity_ratio_col:
                    if 'Debt-to-Equity Ratio' in metrics_filter:
                        if expl_debt_to_equity_ratio_col: 
                            st.info("""
                                    **Debt-to-Equity Ratio:**

                                    - Compares a company's debt to its equity.
                                    - Indicates the ratio of external debt to shareholder equity.
                                    - High ratios may signal higher financial risk.

                                    *Importance:*
                                    - Helps assess a company's financial health.
                                    - High ratios may lead to increased interest payments and financial instability.

                                    *Optimal Ratio:*
                                    - Varies by industry and risk tolerance.

                                    *Bottom Line:*
                                    - Lower ratios are often preferred, but optimal levels vary.
                                    - Consider alongside other financial metrics for comprehensive analysis.

                                    *Note: Use the Debt-to-Equity Ratio as part of a broader evaluation of a company's financial position.*
                                    """) 
            

                free_cash_flow_col, expl_free_cash_flow_col = st.columns(2)
                with free_cash_flow_col:
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
                            yaxis=dict(showgrid=True))

                        st.plotly_chart(bar_chart_free_cash_flow, 
                                        use_container_width=True)
                with expl_free_cash_flow_col:
                    if 'Free Cash Flow' in metrics_filter:
                        st.info("""
                                **Free Cash Flow Over Time:**

                                *Definition:*
                                - Free Cash Flow (FCF) represents the cash generated by a company's operations that is available for distribution to investors, expansion, or debt reduction.

                                *Displaying Over Time:*
                                - Plotting FCF over time allows users to visualize the trend in a company's ability to generate cash.
                                - Positive trends may signal financial health, while negative trends could indicate potential challenges.

                                *Interpretation:*
                                - Increasing FCF over time is generally positive, indicating improving cash generation.
                                - Declines or fluctuations may require further investigation into operational efficiency and financial management.

                                *Investor Insight:*
                                - Investors often use FCF trends for insights into a company's financial strength and sustainability.

                                *Note: Use FCF in conjunction with other financial metrics for a comprehensive analysis.*
                                """)

                EBITDA_col, expl_EBITDA_col = st.columns(2)
                with EBITDA_col:
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

                with expl_EBITDA_col:
                    if 'EBITDA' in metrics_filter:
                        st.info("""
                                **EBITDA Over Time:**

                                *Definition:*
                                - EBITDA is a measure of a company's operating performance, representing its earnings before deducting interest, taxes, depreciation, and amortization.

                                *Displaying Over Time:*
                                - Plotting EBITDA over time helps visualize the company's operational efficiency and profitability trends.
                                - It provides insights into how well a company generates earnings from its core operations.

                                *Interpretation:*
                                - Rising EBITDA over time is generally positive, indicating improved operational profitability.
                                - Declines may prompt investigation into factors affecting operational performance.

                                *Investor Insight:*
                                - Investors use EBITDA trends to assess the underlying operational strength of a business.

                                *Note: EBITDA should be considered alongside other financial metrics for a comprehensive analysis.*
                                """)
 
            

                revenue_col, expl_revenue_col = st.columns(2)
                with revenue_col:
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
                with expl_revenue_col:
                    if 'Revenue' in metrics_filter:

                        st.info(f"""
                                    The displayed chart illustrates the trend of Operating Income and Operating Expenses over time for the company {stock_option}.

                                    - **Operating Income:**
                                    Operating Income represents the profit generated by a company from its core business operations. It is calculated by subtracting Operating Expenses from Gross Profit.

                                    - **Operating Expense:**
                                    Operating Expenses are the costs directly associated with the day-to-day business activities, such as wages, rents, and material costs.

                                    The X-axis represents time, while the Y-axis shows the amounts for Operating Income and Operating Expenses. The bars in the chart provide a visual insight into the relative magnitudes of these financial metrics.

                                    Please note that the chart is dynamic and can be updated by selecting different metrics in the sidebar.
                                """)

                short_ratio_col, expl_short_ratio_col = st.columns(2)

                with short_ratio_col:
                    if 'Short Ratio' in metrics_filter:
                        short_ratio = Company_stock.info.get('shortRatio', 'N/A')
                        st.subheader(f":orange[**{stock_option}**] - Short Ratio:")
                        st.metric(label='Short Ratio', value=str(short_ratio))

                with expl_short_ratio_col:
                    if 'Short Ratio' in metrics_filter:
                        st.info("""The short ratio is a financial indicator that shows the 
                                   ratio of sold short positions to the average daily trading volume. It is calculated by 
                                 Short Ratio . A high short ratio indicates pessimistic market sentiment, while a low ratio 
                                may indicate little interest in short selling. Investors should consider the ratio 
                                in the context of other factors, as it is not sufficient on its own to make investment decisions.""")
                        st.latex(r'Short Ratio = \frac{Total Short Interest}{Average Daily Trading Volume}')



                st.divider()

            # M E T R I C S _ F I L T E R _ E N D
                

##################################################
### E X P L A N A T I O N _ I S _ O N _ E N D;
##################################################           



    # V I S Z U A L I S A T I O N _ S T A R T
charts_vis = st.expander(label="Chart Visualization")




    # V I S Z U A L I S A T I O N _ E N D      
        



        
# C O M P A N Y _ I N F O R M A T I O N _ M E T R I C S _ E N D        





# G E T _ N E W S _ F O R _ C O M P A N Y _ S T A R T
if stock_option:
    newspaper_expander = st.expander(label="News about your Stocks")
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

if stock_option:
    stock_report_mail = st.expander(label="Personalize your weekly Stock Report")
    with stock_report_mail:
        st.info('currently under development')

        


        
        












