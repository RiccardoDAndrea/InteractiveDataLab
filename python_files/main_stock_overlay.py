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
stock_options = st.multiselect(
    'What are your Stocks',
    options = ['AAPL', 'BYDDF', 'EONGY', 'LNVGF', 'NIO', 'PLUN.F', 'TSLA', 'TKA.DE', 'XIACF'])

expander_df_stocks = st.expander("Do you want to see the stocks Data ?")

with expander_df_stocks:
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
        
expander_charts = st.expander("Dashbaord")
options_of_charts = st.multiselect(
                    'What Graphs do you want?', ('Barchart', 
                                                'Linechart', 
                                                'Scatterchart', 
                                                'Histogramm',
                                                'Boxplot'))
for chart_type in options_of_charts:

    if chart_type == 'Histogramm':
        st.write('You can freely choose your :blue[Histogramm]')
        col1_col ,col2_bins = st.columns(2)
        with col1_col:
            x_axis_val_hist = st.selectbox('Select X-Axis Value', options=close_df.columns,
                                        key='x_axis_hist_multiselect')
        with col2_bins:
            bin_size = st.slider('Bin Size', min_value=1, max_value=30, step=1, value=1, format='%d')
        color = st.color_picker('Pick A Color')
        hist_plot_1 = px.histogram(close_df, 
                                    x=x_axis_val_hist, 
                                    nbins=bin_size,
                                    color_discrete_sequence=[color])
        st.plotly_chart(hist_plot_1)
        # Erstellen des Histogramms mit Plotly
        fig = go.Figure(data=hist_plot_1)
        st.divider()

    elif chart_type == 'Linechart':
        st.markdown('You can freely choose your :blue[Linechart] :chart_with_upwards_trend:')
        col3, col4 = st.columns(2)
        
        # Hier können Sie eine Dropdown-Box hinzufügen, um die ausgewählte Aktienoption festzulegen
        
        # Überprüfen Sie, ob die ausgewählte Aktienoption im DataFrame vorhanden ist
        if stock_option:
            combined_data = pd.DataFrame({'Date': close_df['Date']})  # DataFrame für kombinierte Daten erstellen
            if stock_option in close_df.columns:
                combined_data[stock_option] = close_df[stock_option]
            else:
                st.warning(f"Selected stock option '{stock_option}' not found in the DataFrame.")

            # Plotting the combined data
            st.line_chart(data=combined_data.set_index('Date'), use_container_width=True)
            st.divider()
        else:
            st.warning("Please select at least one stock option.")












