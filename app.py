import streamlit as st 
import pandas as pd 
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
import numpy as np
from streamlit_option_menu import option_menu
import plotly.graph_objs as go
import plotly.io as pio
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import statsmodels.api as sm
# streamlit run app1.py --server.maxMessageSize=1028
# source streamlit_machine_learning/bin/activate

################################################################################################################
############### Hier können die lottie files eingesetzt werden #################################################
################################################################################################################
def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

working_men = load_lottieurl('https://assets4.lottiefiles.com/packages/lf20_8ywcoffy.json')
no_date_col = load_lottieurl('https://assets8.lottiefiles.com/packages/lf20_0pgmwzt3.json')
removed_date_column = load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_ovo7L6.json')
no_data_avaible = load_lottieurl('https://assets7.lottiefiles.com/packages/lf20_rjn0esjh.json')
machine_learning_explanation = load_lottieurl('https://assets7.lottiefiles.com/packages/lf20_LmW6VioIWc.json')
deep_learning_explanation = load_lottieurl('https://assets6.lottiefiles.com/packages/lf20_sibnlnc9.json')
objection_detection_explanation = load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_th55Gb.json')
question_with_NaN_values = load_lottieurl('https://assets7.lottiefiles.com/packages/lf20_lKvkGl.json')
no_X_variable_lottie = load_lottieurl('https://assets10.lottiefiles.com/packages/lf20_ydo1amjm.json')
value_is_zero_in_train_size = load_lottieurl('https://assets7.lottiefiles.com/packages/lf20_usmfx6bp.json')
wrong_data_type_ML = load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_2frpohrv.json')
################################################################################################################
################################################################################################################


# Welcome page

st.title('Project Prometheus') 

# Navigation was an der Seite steht


options = st.sidebar.radio(
    'Select an option',
    ('Homepage',
    'Machine Learning',
    "Object detection",
    'Contact'))

def dataframe():
    uploaded_file = st.sidebar.file_uploader('Upload here your file', key='def')
    if uploaded_file is not None:
        if st.session_state.separator:
            df = pd.read_csv(uploaded_file, sep=st.session_state.separator)
        else:
            df = pd.read_csv(uploaded_file)
        return df

if options == 'Machine Learning':
    st.session_state.separator = st.sidebar.selectbox('How would you like to separate your values?', (",", ";", ".", ":"))

uploaded_file = dataframe()

if options == 'Homepage':

    st.write('# :blue[Welcome]')
    st.write("""Welcome to Project Prometheus. My name is Riccardo D'Andrea and in this website I will guide you through some machine learning processes and explain things as best I can so that we :blue[all understand why machine learning is so great.]
                You will find in the navigation bar on the left side of your screen different navigation points where I will explain metrics, functions as understandable as possible.
                So I suggest you just upload a .csv or a .txt file and let it start.""")
    st.divider()
    st_lottie( working_men,
                quality='low',
                width=650,
                height=400)
    st.divider()
    
    
    
# Erklärungen für Projekte die es auf der Internet Seite gibt
    selected = option_menu("Main Menu", ["Machine Learning",
                                         'Object detection'], 
                                icons = ['house', 'gear'], menu_icon = "cast",
                                orientation = 'horizontal', 
                                default_index = 0)
    if 'Machine Learning' in selected:
        st.markdown(f"<div style='text-align:center;'><h1>Machine Learning</h1></div>",
                    unsafe_allow_html=True)
        st_lottie(machine_learning_explanation, width= 700, 
                                                height=200, 
                                                quality='high')
        
        st.write("""Imagine you're on Netflix and looking for a good movie. But don't worry, 
                    you don't have to spend hours scrolling through endless lists of movies - 
                    thanks to machine learning, Netflix will recommend exactly what you want 
                    to see! And if you get hungry during the movie, Amazon can even suggest 
                    what kind of pizza to order based on your preferences and ordering history. 
                    But be careful,if you identify yourself with a selfie,the facial recognition 
                    program might think you're a robot and lock you out - but hey,we're working 
                    on it!""")
    else:
        st.markdown(f"<div style='text-align:center;'><h1>Objection Detection</h1></div>",
                    unsafe_allow_html=True)
        st_lottie(objection_detection_explanation, 
                                                width= 700, 
                                                height=200, 
                                                quality='high')
        
        st.write("""Object recognition is like a robot that scans its environment and identifies 
        any object that is in its way. It's like a waiter who, every time he serves a new dish, 
        immediately recognises what's on it and whether it contains nuts or gluten so he can warn 
        the allergy sufferers among the guests. Whether it's cars, buildings or faces - thanks to 
        object recognition, we can identify and track everything. But be careful! If you send the 
        object recognition programme to a party, it might try to detect each pair of shoes as a 
        separate object - and that probably wouldn't get it very far!""")

####################################################################################################
############# Bereich Machine Learning  ############################################################
####################################################################################################

elif options == 'Machine Learning':
# Hier werden die Auswahl möglichkeiten gesetzt
    overview,change_data_type, handling_missing_values, Groupby_tab, remove_columns_tab, Visualization, machine_learning = st.tabs([  'Overview',
                                                                                                            'Change the data type',
                                                                                                            'Handling missing Values',
                                                                                                            'Groupings',
                                                                                                            'Remove columns',
                                                                                                            "Viszulisation",
                                                                                                            'Machine Learning'])
# In this Section we are in Change Data types

    if uploaded_file is not None:

        with overview:
                st.write('In the follwoing Tab you can get a :blue[Overview of your data.] It will only show the first :keycap_ten: rows')
                st.dataframe(uploaded_file.head(10),use_container_width=True)
                expander_head = st.expander(':orange[More details about the df.head() method and why its important]')
                expander_head.markdown("""When you're working with big data, you don't always want to repeat the whole 
                                        process of reloading the data. Especially when an API is connected, this can be time intensive. 
                                        The Head function df.head() of Pandas gives you the first 5 lines.""")
                st.divider()

                st.write('Here you can get an overview of your data')
                st.dataframe(uploaded_file.describe(),use_container_width=True)
                expander_describe = st.expander(':orange[More details about the describe method and why its important]')
                expander_describe.write(""" Now let's look at the df.describe function and see if this of all functions is so powerfull.
                                            The describe function not only shows us the rows and columns but also gives us the median. 
                                            So if we build a machine learning model and we get a prediction value that is significantly higher than the max value and the median,
                                            we know that our model is not good at calculating prediction values. """)
                st.divider()

                st.write('Here are the NaN values in your data')
                st.dataframe(uploaded_file.isna().sum(),use_container_width=True)
                expander_NaN = st.expander(':orange[What are NaN values and why should you pay attention to them]')
                expander_NaN.write(""" NaN stands for "Not a Number" and refers to missing or invalid values in a data set. NaN values can come in many forms, including missing values, incorrect entries, or other invalid values.
                                       As a Data Scientist or Analyst, it is important to pay attention to NaN values as they can skew or distort the data set. For example, when we perform a statistical analysis or train a 
                                       machine learning model, missing values can lead to incorrect conclusions or worse predictions.""")
                st.divider()


        with change_data_type:
            st.subheader('Your Dataframe datatype')
            st.dataframe(uploaded_file.dtypes,use_container_width=True)
            st.subheader("Change your Datatypes:")
            selected_columns = st.multiselect("Choose your columns", uploaded_file.columns)
            selected_dtype = st.selectbox("Choose one Datatype", [  "int64", 
                                                                    "float64",
                                                                    "string",
                                                                    "datetime64[ns]"])
            
            expander_datatypes = st.expander(':blue[What are Datatypes and why are they so importen :question:]')
            expander_datatypes.write("""Explination :white_check_mark: """)

        # Ändere den Datentyp und zeige den aktualisierten Datensatz an
            if st.button("Save chanages"):
                if selected_dtype == "datetime64[ns]":
                    for col in selected_columns:
                        uploaded_file[col] = pd.to_datetime(uploaded_file[col], format="%Y")
                for col in selected_columns:
                    uploaded_file[col] = uploaded_file[col].astype(selected_dtype)
                st.write("Updated DataFrame:")
                st.dataframe(uploaded_file.dtypes,use_container_width=True)
                st.write(uploaded_file.head(),use_container_width=True)
            new_data_types = uploaded_file

        with handling_missing_values:
            st.write('How to proceed with NaN values')
            st.dataframe(uploaded_file.isna().sum(), use_container_width=True)
            checkbox_nan_values = st.checkbox("Do you want to replace the NaN values to proceed?", key="disabled")
            if checkbox_nan_values:
                numeric_columns = uploaded_file.select_dtypes(include=[np.number]).columns.tolist()
                missing_values = st.selectbox(
                    "How do you want to replace the NaN values in the numeric columns?",
                    key="visibility",
                    options=["with Median", 
                            "with Mean", 
                            "with Minimum value", 
                            "with Maximum value", 
                            "with Zero"])

                if 'with Median' in missing_values:
                    Nan_values_replaced_file = uploaded_file[numeric_columns].median()
                    uploaded_file[numeric_columns] = uploaded_file[numeric_columns].fillna(Nan_values_replaced_file)
                    st.write('##### You have succesfully change the NaN values :blue[with the Median]')
                    st.dataframe(uploaded_file.isna().sum(), use_container_width=True)
                    st.divider()

                elif 'with Mean' in missing_values:
                    Nan_values_replaced_file = uploaded_file[numeric_columns].mean()
                    repalced_uploaded_file = uploaded_file.copy()
                    repalced_uploaded_file[numeric_columns] = repalced_uploaded_file[numeric_columns].fillna(Nan_values_replaced_file)
                    st.markdown(' ##### You have succesfully change the NaN values :blue[ with the Mean]')
                    st.dataframe(uploaded_file.isna().sum(), use_container_width=True)
                    st.divider()
                elif 'with Minimum value' in missing_values:
                    Nan_values_replaced_file = uploaded_file[numeric_columns].min()
                    repalced_uploaded_file = uploaded_file.copy()
                    repalced_uploaded_file[numeric_columns] = repalced_uploaded_file[numeric_columns].fillna(Nan_values_replaced_file)
                    st.write('##### You have succesfully change the NaN values :blue[with the minimum values]')
                    st.dataframe(uploaded_file.isna().sum(), use_container_width=True)
                    st.divider()
                elif 'with Maximum value' in missing_values:
                    Nan_values_replaced_file = uploaded_file[numeric_columns].max()
                    repalced_uploaded_file = uploaded_file.copy()
                    repalced_uploaded_file[numeric_columns] = repalced_uploaded_file[numeric_columns].fillna(Nan_values_replaced_file)
                    st.write('##### You habe succesfully change the NaN values :blue[with the maximums values]')
                    st.dataframe(uploaded_file.isna().sum(), use_container_width=True)
                    st.divider()
                elif 'with Zero' in missing_values:
                    replaced_uploaded_file = uploaded_file.copy()
                    numeric_columns = replaced_uploaded_file.select_dtypes(include=[np.number]).columns.tolist()
                    replaced_uploaded_file[numeric_columns] = replaced_uploaded_file[numeric_columns].fillna(0)
                    st.write('##### You have successfully changed :blue[the NaN values to 0.]')
                    st.dataframe(uploaded_file.isna().sum(), use_container_width=True)
                    st.divider()

  
                   
                    
                    


                        

        
 ###################################################################################################################
 ####################### wenn date drinne ist ######################################################################
 ###################################################################################################################               
        
        if 'date' in new_data_types:

            with Groupby_tab:
                st.title('Interactive Groupbys with date')
                if 'date' not in uploaded_file.columns:
                    st.write('You have no date column')
                    st_lottie(no_date_col, 
                              quality='low', 
                              loop=False)

                elif 'date' not in uploaded_file.columns:
                    st.write('You have removed the date column')
                    st_lottie(removed_date_column, 
                              quality='low', 
                              loop=False)

                elif uploaded_file is not None:
                    uploaded_file['date'] = pd.to_datetime(uploaded_file['date'])
                    uploaded_file.set_index('date', inplace=True)
                    options_groupby = st.selectbox('Select Groupby Method', options=('Day', 'Month', 'Quarter', 'Yearly'))
                    
                    # Aggregationen der date Spalte
                    if options_groupby == 'Day':
                        uploaded_file = uploaded_file.groupby(pd.Grouper(freq='D')).sum().reset_index()
                        st.dataframe(uploaded_file)

                    elif options_groupby == 'Month':
                        uploaded_file = uploaded_file.groupby(pd.Grouper(freq='M')).sum().reset_index()
                        st.dataframe(uploaded_file)

                    elif options_groupby == 'Quarter':
                        uploaded_file = uploaded_file.groupby(pd.Grouper(freq='Q')).sum().reset_index()
                        st.dataframe(uploaded_file)

                    elif options_groupby == 'Yearly':
                        uploaded_file = uploaded_file.groupby(pd.Grouper(freq='Y')).sum().reset_index()
                        st.dataframe(uploaded_file)


            
            with remove_columns_tab:
                # Dropdown-Box mit Spaltennamen erstellen
                columns_to_drop = st.multiselect("Select columns to drop", uploaded_file.columns)            
                if 'date' in columns_to_drop:
                    columns_to_drop.remove('date')
                uploaded_file = uploaded_file.drop(columns=columns_to_drop)

                st.dataframe(uploaded_file,use_container_width=True)
               



            with Visualization:

                if uploaded_file is not None:
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
                                x_axis_val_hist = st.selectbox('Select X-Axis Value', options=uploaded_file.columns,
                                                            key='x_axis_hist_multiselect')
                            with col2_bins:
                                bin_size = st.slider('Bin Size', min_value=1, max_value=100, step=1, value=1, format='%d')
                            color = st.color_picker('Pick A Color')
                            hist_plot_1 = px.histogram(uploaded_file, 
                                                       x=x_axis_val_hist, 
                                                       nbins=bin_size,
                                                       color_discrete_sequence=[color])
                            st.plotly_chart(hist_plot_1)
                            # Erstellen des Histogramms mit Plotly
                            fig = go.Figure(data=hist_plot_1)
                            # Umwandeln des Histogramm-Graphen in eine Bilddatei
                            img_bytes = pio.to_image(fig, format='png', width=1000, height=600, scale=2)
                            # Herunterladen der Bilddatei als Button
                            with open('histo.png', 'wb') as f:
                                f.write(img_bytes)
                            with open('histo.png', 'rb') as f:
                                image_bytes = f.read()
                                st.download_button(label='Download Histogramm', data=image_bytes, file_name='histo.png')
                            st.divider()


                        elif chart_type == 'Scatterchart':
                            st.write('You can freely choose your :blue[Scatter plot]')
                            x_axis_val_col_, y_axis_val_col_ = st.columns(2)
                            with x_axis_val_col_:
                                x_axis_val = st.selectbox('Select X-Axis Value', options=uploaded_file.columns, key='x_axis_selectbox')
                            with y_axis_val_col_:
                                y_axis_val = st.selectbox('Select Y-Axis Value', options=uploaded_file.columns, key='y_axis_selectbox')
                            scatter_plot_1 = px.scatter(uploaded_file, x=x_axis_val,y=y_axis_val)

                            st.plotly_chart(scatter_plot_1,use_container_width=True)
                            # Erstellen des Histogramms mit Plotly
                            fig_scatter = go.Figure(data=scatter_plot_1)
                            # Umwandeln des Histogramm-Graphen in eine Bilddatei
                            plt.tight_layout()
                            img_bytes_scatter = pio.to_image(fig_scatter, format='png', width=1000, height=600, scale=2)
                            # Herunterladen der Bilddatei als Button
                            with open('Scatter.png', 'wb') as f:
                                f.write(img_bytes_scatter)
                            with open('Scatter.png', 'rb') as f:
                                image_bytes_scatter = f.read()
                                st.download_button(label='Download Scatter', data=image_bytes_scatter, file_name='Scatter.png')
                            st.divider()

                        elif chart_type == 'Linechart':
                            col1, col2 = st.columns(2)
                            with col1:
                                start_date = st.date_input('Start date:', value=uploaded_file["date"].min())
                                start_date = pd.to_datetime(start_date)

                            with col2:
                                end_date = st.date_input('End date:', value=uploaded_file["date"].max())
                                end_date = pd.to_datetime(end_date)

                            col3, col4 = st.columns(2)
                            st.markdown('You can freely choose your :blue[Linechart] :chart_with_upwards_trend:')
                            with col3:
                                x_axis_val_line = st.selectbox('Select :orange[X-Axis Value]', options=uploaded_file.columns, key='x_axis_line_multiselect')
                            with col4:
                                y_axis_vals_line = st.multiselect('Select :blue[Y-Axis Values]', options=uploaded_file.columns, key='y_axis_line_multiselect')

                            # Filtern des Dataframes auf die ausgewählte Zeitrange
                            df_filtered = uploaded_file.loc[(uploaded_file["date"] >= start_date) & 
                                                                            (uploaded_file["date"] <= end_date)]

                            # Erstellen des Lineplots mit der eingeschränkten Zeitachse
                            line_plot = px.line(df_filtered, x=x_axis_val_line, y=y_axis_vals_line)
                            st.plotly_chart(line_plot)

                            fig_line = line_plot.full_figure_for_development(warn=False)
                            # Umwandeln des Lineplots in eine Bilddatei
                            img_bytes_line = fig_line.to_image(format='png', width=1000, height=600, scale=2)
                            # Herunterladen der Bilddatei als Button
                            st.download_button(label='Download Lineplot', data=img_bytes_line, file_name='Lineplot.png')
                            st.divider()



                        elif chart_type == 'Barchart':
                            st.write('You can freely choose your :blue[Barplot]')
                            bar_X_col,bar_Y_col = st.columns(2)
                            with bar_X_col:
                                x_axis_val_bar = st.selectbox('Select X-Axis Value', options=uploaded_file.columns,
                                                        key='x_axis_bar_multiselect')
                            with bar_Y_col:
                                y_axis_val_bar = st.selectbox('Select Y-Axis Value', options=uploaded_file.columns,
                                                        key='Y_axis_bar_multiselect')
                            bar_plot_1 = px.bar(uploaded_file, x=x_axis_val_bar, y=y_axis_val_bar)
                            st.plotly_chart(bar_plot_1)

                            fig_bar = go.Figure(data=bar_plot_1)
                            
                            # Umwandeln des Histogramm-Graphen in eine Bilddatei
                            img_bytes_bar = pio.to_image(fig_bar, format='png', width=1000, height=600, scale=2)
                            
                            # Herunterladen der Bilddatei als Button
                            with open('Barplot.png', 'wb') as f:
                                f.write(img_bytes_bar)
                            with open('Barplot.png', 'rb') as f:
                                img_bytes_line = f.read()
                                st.download_button(label='Download Barplot', data=img_bytes_bar, file_name='Barplot.png')
                            st.divider()

                        elif chart_type == 'Boxplot':
                            st.write('You can freely choose your :blue[Boxplot] plot')
                            y_axis_val_bar = st.selectbox('Select Y-Axis Value', options=uploaded_file.columns,
                                                        key='Y_axis_box_multiselect')
                            box_plot_1 = px.box(uploaded_file,y=y_axis_val_bar)
                            st.plotly_chart(box_plot_1)
                            fig_boxplot = go.Figure(data=box_plot_1)
                            # Umwandeln des Histogramm-Graphen in eine Bilddatei
                            img_bytes_boxplot = pio.to_image(fig_boxplot, format='png', width=1000, height=600, scale=2)
                            # Herunterladen der Bilddatei als Button
                            with open('Boxplot.png', 'wb') as f:
                                f.write(img_bytes_boxplot)
                            with open('Boxplot.png', 'rb') as f:
                                img_bytes_line = f.read()
                                st.download_button(label='Download Boxplot', data=img_bytes_boxplot, file_name='Barplot.png')
                            st.divider()

                    else:
                        st.write('You can choose what you want to visualization')
                else:
                    st.write('Please upload a file to continue')

            with machine_learning:
                
                #Start  von Machine Learning 
                st.dataframe(uploaded_file)

                # Correlation Matrix erster Expander
                Correlation_Matrix = st.expander('Correlation Matrix')
                
                with Correlation_Matrix:
                
                    st.info("""Correlation matrices are important in machine learning because they help 
                                us understand how different variables are related to each other. By identifying 
                                strong correlations, we can select the most useful variables for predicting a 
                                target, and avoid problems with multicollinearity.""", icon="ℹ️")
                    
                    # Korrelationsmatrix
                    corr_matrix = uploaded_file.select_dtypes(include=[np.number]).corr()


                    # Erstellung der Heatmap mit Plotly
                    fig_correlation = px.imshow(corr_matrix.values, color_continuous_scale='purples', zmin=-1, zmax=1,
                                    x=corr_matrix.columns, y=corr_matrix.index,
                                    labels=dict(x="Columns", y="Columns", color="Correlation"))
                    
                    # Anpassung der Plot-Parameter
                    fig_correlation.update_layout(title='Correlation Matrix')
                    fig_correlation.update_layout(
                        title='Correlation Matrix',
                        font=dict(
                            color='grey'
                        )
                    )

                    fig_correlation.update_traces(showscale=False, colorbar_thickness=25)
                    
                    # Hinzufügen der numerischen Werte als Text
                    annotations = []
                    for i, row in enumerate(corr_matrix.values):
                        for j, val in enumerate(row):
                            annotations.append(dict(x=j, y=i, text=str(round(val, 2)), showarrow=False, font=dict(size=16)))
                    fig_correlation.update_layout(annotations=annotations)
                    
                    # Anzeigen der Plot
                    st.plotly_chart(fig_correlation, use_container_width= True)
                    fig_correlationplot = go.Figure(data=fig_correlation)

                    # Umwandeln des Histogramm-Graphen in eine Bilddatei
                    img_bytes_correlationplot = pio.to_image(fig_correlationplot, format='png', width=1000, height=600, scale=2)
                    
                    # Herunterladen der Bilddatei als Button

                    with open('Correlation Matrix.png', 'wb') as f:
                        f.write(img_bytes_correlationplot)
                    with open('Correlation Matrix.png', 'rb') as f:
                        img_bytes_line = f.read()
                        st.download_button(label='Download Correlation Matrix', data=img_bytes_correlationplot, file_name='Barplot.png')
                
                #Ende der Correlations Matrix
              

                MachineLearning_sklearn = st.expander('Machine learning evaluation')
                with MachineLearning_sklearn:
                    # Hier kann der Nutzer dynamisch die unabhängigen und abhängigen Variablen auswählen
                    Target_variable_col, X_variables_col = st.columns(2)
                    Target_variable = Target_variable_col.selectbox('Which is your Target Variable (Y)', options=uploaded_file.columns, key='LR Sklearn Target Variable')
                    X_variables = X_variables_col.multiselect('Which is your Variables (X)', options=uploaded_file.columns, key='LR Sklearn X Variables')
                    
                    if uploaded_file[Target_variable].dtype == object:
                        st.warning('Ups, wrong data type for Target variable!')
                        st_lottie(wrong_data_type_ML, width=700, height=300, quality='low', loop=False)
                        st.dataframe(uploaded_file.dtypes, use_container_width=True)
                        st.stop()
                        
                    if any(uploaded_file[x].dtype == object for x in X_variables):
                        st.warning('Ups, wrong data type for X variables!')
                        st_lottie(wrong_data_type_ML, width=700, height=300, quality='low', loop=False)
                        st.dataframe(uploaded_file.dtypes, use_container_width=True)
                        st.stop()
    
                    elif len(X_variables) == 0 :
                        st_lottie(no_X_variable_lottie)
                        st.warning('X Variable is empty!')
                        st.stop()

                    # Mit dem Slider kann der Nutzer selber aussuchen wie viel Prozentual er als Test und Train definiert
                    # Default Parameters so das keine Fehler meldung ensteht
                    
                    total_size = 100
                    train_size = 60
                    test_size = 40

                    train_size_col, test_size_col = st.columns(2)

                    with train_size_col:
                        train_size = st.slider('Train Size', min_value=0, max_value=total_size, value=train_size, key= 'Sklearn train size')
                        test_size = total_size - train_size

                    with test_size_col:
                        test_size = st.slider('Test Size', min_value=0, max_value=total_size, value=test_size, key= 'Sklearn test size')
                        train_size = total_size - test_size

                    # Relevant damit das Skript weiter läuft und nicht immer in Fehlermeldungen läuft
                    if train_size <= 0:
                        st_lottie(value_is_zero_in_train_size)
                        st.warning('Train size should be greater than zero.')
                        st.stop()
                    elif test_size <= 0:
                        st.warning('Test size should be greater than zero.')
                        st.stop()
                    # elif train_size + test_size > len(df_dtype_and_groupby_and_dropped):
                    #     st.warning('Train size and Test size exceed the number of samples in the dataset.')
                    #     st.stop()
                    elif train_size == len(uploaded_file):
                        st.warning('Train size cannot be equal to the number of samples in the dataset.')
                        st.stop()
                    
                    # Unanhängige Varible sowie Abhängige 
                    X = uploaded_file[X_variables]
                    y = uploaded_file[Target_variable]
                    
                    # Aufteilung der Train und Test Datensätze
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, train_size=train_size/100, random_state=42)

                    # Initalisierung des Models
                    lm = LinearRegression(fit_intercept=True)
                    lm.fit(X_train, y_train)
                    
                    # Evaluierung
                    # R2 oder auch Score

                    R2_sklearn_train = lm.score(X_train, y_train)
                    R2_sklearn_test = lm.score(X_test, y_test)
                   
                    R_2_training_col,R_2_test_col = st.columns(2)
                    with R_2_training_col:
                        st.metric(label = 'R2 of the Training Data', value = R2_sklearn_train.round(3))
                    with R_2_test_col:
                        st.metric(label = 'R2 of the Test Data', value = R2_sklearn_test.round(3))

                    st.divider()
                    
                    # Root Mean Square error
                    y_pred_train = lm.predict(X_train)
                    y_pred_test = lm.predict(X_test)

                    RMSE_train = sqrt(mean_squared_error(y_train, y_pred_train))
                    RMSE_test = sqrt(mean_squared_error(y_test, y_pred_test))

                    RMSE_train_col,RMSE_test_col = st.columns(2)
                    with RMSE_train_col:
                        st.metric(label = 'RMSE of the Training Data', value = round(RMSE_train,3))
                    with RMSE_test_col:
                        st.metric(label = 'RMSE of the Test Data', value = round(RMSE_test,3))

                    st.divider()
                    
                    # R2-Wert auf den Trainingsdaten berechnen
                    # r2_train = r2_score(y_train, y_pred_train)
                    # st.write('R2 of the Training Data (alternative calculation)', r2_train)
                    
                    # st.divider()
                    
                    # The Mean Absolute Error (MAE)
                    MAE_train = mean_absolute_error(y_train, y_pred_train)
                    MAE_test = mean_absolute_error(y_test, lm.predict(X_test))


                    MAE_train_col,MAE_test_col = st.columns(2)
                    with RMSE_train_col:
                        st.metric(label = 'MAE of the Training Data', value = round(MAE_train,3))
                    with RMSE_test_col:
                        st.metric(label = 'MAE of the Test Data', value = round(MAE_test,3))

                    # Coefficient
                    coefficiensts = lm.coef_
                    coefficients = pd.DataFrame(lm.coef_.reshape(-1, 1), columns=['Coefficient'], index=X.columns)
                    st.table(coefficients)
                
                    # Intercept
                    
                    st.metric(label='Intercept', value=lm.intercept_)

                    results = pd.concat([y_test.reset_index(drop=True), pd.Series(y_pred_test)], axis=1)
                    results.columns = ['y_test', 'y_pred_test']
                    results['difference'] = results['y_test'] - results['y_pred_test']
                    st.dataframe(results,use_container_width= True)

                plt_Scatter = st.expander('Visualization of Scatter Plot')
                with plt_Scatter:
                    fig = px.scatter(results, x='y_test', y='y_pred_test', trendline="ols")
                    fig.update_layout(
                        title="Scatter Plot: Actual vs. Predicted",
                        xaxis_title="Actual values",
                        yaxis_title="Predicted values",
                        font=dict(size=12)
                    )
                    st.plotly_chart(fig)

                    residuals_plot = px.scatter(results, x='y_pred_test', y='difference')
                    residuals_plot.update_layout(
                        title="Residual Plot",
                        xaxis_title="Predicted values",
                        yaxis_title="Residuals",
                        font=dict(size=12)
                    )
                    st.plotly_chart(residuals_plot, use_container_width=True)

                try_machine_learning = st.expander('Try Your Machine Learning')

                with try_machine_learning:
                    user_input_min = None
                    user_input_max = None
                    if len(X.columns) == 1:
                        x0 = st.number_input(X_train.columns[0], min_value=user_input_min, max_value=user_input_max,step=1,key='x1')
                        y_pred_ = lm.intercept_ + coefficiensts[0] * x0

                    elif len(X.columns) == 2:

                        regression_options_2_1,regression_options_2_2 = st.columns(2)

                        x0 = regression_options_2_1.number_input(X_train.columns[0], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_2_1')
                        x1 = regression_options_2_2.number_input(X_train.columns[1], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_2_2')
                        y_pred_ = lm.intercept_ + coefficiensts[0] * x0 + coefficiensts[1] * x1

                    elif len(X.columns) == 3:

                        regression_options_3_1,regression_options_3_2,regression_options_3_3 = st.columns(3)
                        
                        x0 = regression_options_3_1.number_input(X_train.columns[0], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_3_1')
                        x1 = regression_options_3_2.number_input(X_train.columns[1], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_3_2')
                        x2 = regression_options_3_3.number_input(X_train.columns[2], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_3_3')
                        y_pred_ = lm.intercept_ + coefficiensts[0] * x0 + coefficiensts[1] * x1 + coefficiensts[2] * x2

                    elif len(X.columns) == 4:

                        regression_options_4_1,regression_options_4_2,regression_options_4_3,regression_options_4_4 = st.columns(4)
                        
                        x0 = regression_options_4_1.number_input(X_train.columns[0], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_4_1')
                        x1 = regression_options_4_2.number_input(X_train.columns[1], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_4_2')
                        x2 = regression_options_4_3.number_input(X_train.columns[2], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_4_3')
                        x3 = regression_options_4_4.number_input(X_train.columns[3], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_4_4')
                        y_pred_ = lm.intercept_ + coefficiensts[0] * x0 + coefficiensts[1] * x1 + coefficiensts[2] * x2 + coefficiensts[3] * x3

                    
                    elif len(X.columns) == 5:
                        regression_options_5_1,regression_options_5_2,regression_options_5_3 = st.columns(3)
                        
                        x0 = regression_options_5_1.number_input(X_train.columns[0], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_5_1')
                        x1 = regression_options_5_2.number_input(X_train.columns[1], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_5_2')
                        x2 = regression_options_5_3.number_input(X_train.columns[2], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_5_3')

                        regression_options_5_4,regression_options_5_5 = st.columns(2)
                       
                        x3 = regression_options_5_4.number_input(X_train.columns[3], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_5_4')
                        x4 = regression_options_5_5.number_input(X_train.columns[4], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_5_5')
                        y_pred_ = lm.intercept_ + coefficiensts[0] * x0 + coefficiensts[1] * x1 + coefficiensts[2] * x2 + coefficiensts[3] * x3 + coefficiensts[4] * x4

                    elif len(X.columns) == 6:
                        
                        regression_options_6_1,regression_options_6_2,regression_options_6_3 = st.columns(3)
                        
                        x0 = regression_options_6_1.number_input(X_train.columns[0], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_6_1')
                        x1 = regression_options_6_2.number_input(X_train.columns[1], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_6_2')
                        x2 = regression_options_6_3.number_input(X_train.columns[2], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_6_3')
                        
                        regression_options_6_4,regression_options_6_5,regression_options_6_6 = st.columns(3)
                        
                        x3 = regression_options_6_4.number_input(X_train.columns[3], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_6_4')
                        x4 = regression_options_6_5.number_input(X_train.columns[4], min_value=user_input_min, max_value=user_input_max,step=1,key='regregression_options_6_5')
                        x5 = regression_options_6_6.number_input(X_train.columns[5], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_6_6')
                        y_pred_ = lm.intercept_ + coefficiensts[0] * x0 + coefficiensts[1] * x1 + coefficiensts[2] * x2 + coefficiensts[3] * x3 + coefficiensts[4] * x4 + coefficiensts[5] * x5
                    
                    elif len(X.columns) == 7:
                        
                        regression_options_7_1,regression_options_7_2,regression_options_7_3 = st.columns(3)
                        
                        x0 = regression_options_7_1.number_input(X_train.columns[0], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_7_1')
                        x1 = regression_options_7_2.number_input(X_train.columns[1], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_7_2')
                        x2 = regression_options_7_3.number_input(X_train.columns[2], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_7_3')
                        
                        regression_options_7_4,regression_options_7_5,regression_options_7_6,regression_options_7_7 = st.columns(4)
                        
                        x3 = regression_options_7_4.number_input(X_train.columns[3], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_7_4')
                        x4 = regression_options_7_5.number_input(X_train.columns[4], min_value=user_input_min, max_value=user_input_max,step=1,key='regregression_options_7_5')
                        x5 = regression_options_7_6.number_input(X_train.columns[5], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_7_6')
                        x6 = regression_options_7_7.number_input(X_train.columns[6], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_7_7')
                        y_pred_ = lm.intercept_ + coefficiensts[0] * x0 + coefficiensts[1] * x1 + coefficiensts[2] * x2 + coefficiensts[3] * x3 + coefficiensts[4] * x4 + coefficiensts[5] * x5  + coefficiensts[6] * x6
                    
                    else:
                        st.warning("""I am still working on increasing the number of columns for the regression. 
                                      Please choose up to 7 columns to avoid this error message""",icon="ℹ️")
                        
                        st.stop()
                    st.markdown(f'Your predicted value for :blue[{Target_variable}] is: **{y_pred_.round(3)}**')

###########################################################################################################################
######################################## Wenn kein date drinne ist #######################################################
##########################################################################################################################
        else:
            with Groupby_tab:
                st.title('Interactive Groupbys with date')

                if 'date' not in uploaded_file.columns:
                    st.write('You have no date column')
                    st_lottie(no_date_col, 
                              quality='low', 
                              loop=False)

            with remove_columns_tab:
                # Dropdown-Box mit Spaltennamen erstellen
                columns_to_drop = st.multiselect("Select columns to drop", uploaded_file.columns)

                # Ausgewählte Spalten aus dem DataFrame entfernen
                uploaded_file = uploaded_file.drop(columns=columns_to_drop)

                only_numeric_columns = st.button(label=('Only numeric values'))

                if only_numeric_columns:
                    # Nur numerische Spalten beibehalten und 'date' falls vorhanden
                    numeric_cols = uploaded_file.select_dtypes(include=[np.number]).columns.tolist()

                    if 'date' in uploaded_file.columns:
                        numeric_cols.append('date')
                    uploaded_file = uploaded_file[numeric_cols]
                    st.dataframe(uploaded_file.head(20), use_container_width=True)
                else:
                    st.dataframe(uploaded_file.head(20), use_container_width=True)

                reset_selection = st.button('Reset Selection')

                if reset_selection:
                    uploaded_file = uploaded_file.copy()
                    st.dataframe(uploaded_file)
                    st.success('Selection reset.')


            with Visualization:

                if uploaded_file is not None:
        
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
                                x_axis_val_hist = st.selectbox('Select X-Axis Value', options=uploaded_file.columns,
                                                            key='x_axis_hist_multiselect')
                            with col2_bins:
                                bin_size = st.slider('Bin Size', min_value=1, max_value=30, step=1, value=1, format='%d')
                            color = st.color_picker('Pick A Color')
                            hist_plot_1 = px.histogram(uploaded_file, 
                                                       x=x_axis_val_hist, 
                                                       nbins=bin_size,
                                                       color_discrete_sequence=[color])
                            st.plotly_chart(hist_plot_1)
                            # Erstellen des Histogramms mit Plotly
                            # fig = go.Figure(data=hist_plot_1)
                            # # Umwandeln des Histogramm-Graphen in eine Bilddatei
                            # img_bytes = pio.to_image(fig, format='png', width=1000, height=600, scale=2)
                            # # Herunterladen der Bilddatei als Button
                            # with open('histo.png', 'wb') as f:
                            #     f.write(img_bytes)
                            # with open('histo.png', 'rb') as f:
                            #     image_bytes = f.read()
                            #     st.download_button(label='Download Histogramm', data=image_bytes, file_name='histo.png')
                            st.divider()

                        elif chart_type == 'Scatterchart':
                            st.write('You can freely choose your :blue[Scatter plot]')
                            x_axis_val_col_, y_axis_val_col_ = st.columns(2)
                            with x_axis_val_col_:
                                x_axis_val = st.selectbox('Select X-Axis Value', options=uploaded_file.columns, key='x_axis_selectbox')
                            with y_axis_val_col_:
                                y_axis_val = st.selectbox('Select Y-Axis Value', options=uploaded_file.columns, key='y_axis_selectbox')
                            scatter_plot_1 = px.scatter(uploaded_file, x=x_axis_val,y=y_axis_val)

                            st.plotly_chart(scatter_plot_1,use_container_width=True)
                            # # Erstellen des Histogramms mit Plotly
                            # fig_scatter = go.Figure(data=scatter_plot_1)
                            # # Umwandeln des Histogramm-Graphen in eine Bilddatei
                            # plt.tight_layout()
                            # img_bytes_scatter = pio.to_image(fig_scatter, format='png', width=1000, height=600, scale=2)
                            # # Herunterladen der Bilddatei als Button
                            # with open('Scatter.png', 'wb') as f:
                            #     f.write(img_bytes_scatter)
                            # with open('Scatter.png', 'rb') as f:
                            #     image_bytes_scatter = f.read()
                            #     st.download_button(label='Download Scatter', data=image_bytes_scatter, file_name='Scatter.png')
                            st.divider()

                        elif chart_type == 'Linechart':
                            st.markdown('You can freely choose your :blue[Linechart] :chart_with_upwards_trend:')
                            Line_date_not_col1, Line_date_not_col2= st.columns(2)
                            with Line_date_not_col1:
                                start_date = st.date_input('Start date')
                            with Line_date_not_col2:
                                end_date = st.date_input('End date')
                            
                            col3,col4 = st.columns(2)
                            
                            with col3:
                                x_axis_val_line = st.selectbox('Select X-Axis Value', options=uploaded_file.columns,
                                                            key='x_axis_line_multiselect')
                            with col4:
                                y_axis_vals_line = st.multiselect('Select :blue[Y-Axis Values]', options=uploaded_file.columns,
                                                                key='y_axis_line_multiselect')

                            line_plot_1 = px.line(uploaded_file, x=x_axis_val_line, y=y_axis_vals_line)
                            st.plotly_chart(line_plot_1)
                            # fig_line = go.Figure(data=line_plot_1)
                            # # Umwandeln des Histogramm-Graphen in eine Bilddatei
                            # img_bytes_line = pio.to_image(fig_line, format='png', width=1000, height=600, scale=2)
                            # # Herunterladen der Bilddatei als Button
                            # with open('Lineplot.png', 'wb') as f:
                            #     f.write(img_bytes_line)
                            # with open('Lineplot.png', 'rb') as f:
                            #     img_bytes_line = f.read()
                            #     st.download_button(label='Download Lineplot', data=img_bytes_line, file_name='histo.png')
                            st.divider()

                        elif chart_type == 'Barchart':
                            st.write('You can freely choose your :blue[Barplot]')
                            bar_X_col,bar_Y_col = st.columns(2)
                            with bar_X_col:
                                x_axis_val_bar = st.selectbox('Select X-Axis Value', options=uploaded_file.columns,
                                                        key='x_axis_bar_multiselect')
                            with bar_Y_col:
                                y_axis_val_bar = st.selectbox('Select Y-Axis Value', options=uploaded_file.columns,
                                                        key='Y_axis_bar_multiselect')
                            bar_plot_1 = px.bar(uploaded_file, x=x_axis_val_bar, y=y_axis_val_bar)
                            st.plotly_chart(bar_plot_1)

                            # fig_bar = go.Figure(data=bar_plot_1)
                            
                            # # Umwandeln des Histogramm-Graphen in eine Bilddatei
                            # img_bytes_bar = pio.to_image(fig_bar, format='png', width=1000, height=600, scale=2)
                            
                            # # Herunterladen der Bilddatei als Button
                            # with open('Barplot.png', 'wb') as f:
                            #     f.write(img_bytes_bar)
                            # with open('Barplot.png', 'rb') as f:
                            #     img_bytes_line = f.read()
                            #     st.download_button(label='Download Barplot', data=img_bytes_bar, file_name='Barplot.png')
                            st.divider()

                        elif chart_type == 'Boxplot':
                            st.write('You can freely choose your :blue[Boxplot]')
                            y_axis_val_bar = st.selectbox('Select Y-Axis Value', options=uploaded_file.columns,
                                                        key='Y_axis_box_multiselect')
                            box_plot_1 = px.box(uploaded_file,y=y_axis_val_bar)
                            st.plotly_chart(box_plot_1)


                            # fig_boxplot = go.Figure(data=box_plot_1)
                            # # Umwandeln des Histogramm-Graphen in eine Bilddatei
                            # img_bytes_boxplot = pio.to_image(fig_boxplot, format='png', width=1000, height=600, scale=2)
                            # # Herunterladen der Bilddatei als Button
                            # with open('Boxplot.png', 'wb') as f:
                            #     f.write(img_bytes_boxplot)
                            # with open('Boxplot.png', 'rb') as f:
                            #     img_bytes_line = f.read()
                            #     st.download_button(label='Download Boxplot', data=img_bytes_boxplot, file_name='Barplot.png')
                            st.divider()
                           
                else:
                    st.write('Please upload a file to continue')

            with machine_learning:
                
                #Start  von Machine Learning 
                st.dataframe(uploaded_file,use_container_width= True)
                # Correlation Matrix erster Expander
                Correlation_Matrix = st.expander('Correlation Matrix')
                
                with Correlation_Matrix:
                
                    st.info("""Correlation matrices are important in machine learning because they help 
                                us understand how different variables are related to each other. By identifying 
                                strong correlations, we can select the most useful variables for predicting a 
                                target, and avoid problems with multicollinearity.""", icon="ℹ️")
                    
                    # Korrelationsmatrix
                    corr_matrix = uploaded_file.select_dtypes(include=[np.number]).corr()


                    # Erstellung der Heatmap mit Plotly
                    fig_correlation = px.imshow(corr_matrix.values, color_continuous_scale='purples', zmin=-1, zmax=1,
                                    x=corr_matrix.columns, y=corr_matrix.index,
                                    labels=dict(x="Columns", y="Columns", color="Correlation"))
                    
                    # Anpassung der Plot-Parameter
                    fig_correlation.update_layout(title='Correlation Matrix')
                    fig_correlation.update_layout(
                        title='Correlation Matrix',
                        font=dict(
                            color='grey'
                        )
                    )

                    fig_correlation.update_traces(showscale=False, colorbar_thickness=25)
                    
                    # Hinzufügen der numerischen Werte als Text
                    annotations = []
                    for i, row in enumerate(corr_matrix.values):
                        for j, val in enumerate(row):
                            annotations.append(dict(x=j, y=i, text=str(round(val, 2)), showarrow=False, font=dict(size=16)))
                    fig_correlation.update_layout(annotations=annotations)
                    
                    # Anzeigen der Plot
                    st.plotly_chart(fig_correlation, use_container_width= True)
                    fig_correlationplot = go.Figure(data=fig_correlation)
                    fig_correlationplot
                    # Umwandeln des Histogramm-Graphen in eine Bilddatei
                    #img_bytes_correlationplot = pio.to_image(fig_correlationplot, format='png', width=1000, height=600, scale=2)
                    
                    # Herunterladen der Bilddatei als Button

                    # with open('Correlation Matrix.png', 'wb') as f:
                    #     f.write(img_bytes_correlationplot)
                    # with open('Correlation Matrix.png', 'rb') as f:
                    #     img_bytes_line = f.read()
                    #     st.download_button(label='Download Correlation Matrix', data=img_bytes_correlationplot, file_name='Barplot.png')
                
                #Ende der Correlations Matrix
              
                MachineLearning_sklearn = st.expander('Machine learning evaluation')
                with MachineLearning_sklearn:
                    # Hier kann der Nutzer dynamisch die unabhängigen und abhängigen Variablen auswählen
                    Target_variable_col, X_variables_col = st.columns(2)
                    Target_variable = Target_variable_col.selectbox('Which is your Target Variable (Y)', options=uploaded_file.columns, key='LR Sklearn Target Variable')
                    X_variables = X_variables_col.multiselect('Which is your Variables (X)', options=uploaded_file.columns, key='LR Sklearn X Variables')
                    if any(uploaded_file[x].dtype == object for x in Target_variable):
                        st.warning('Ups, wrong data type for X variables!')
                        st_lottie(wrong_data_type_ML, width=700, height=300, quality='low', loop=False)
                        st.dataframe(uploaded_file.dtypes,use_container_width=True)
                        st.stop()
                    if any(uploaded_file[x].dtype == object for x in X_variables):
                        st.warning('Ups, wrong data type for X variables!')
                        st_lottie(wrong_data_type_ML, width=700, height=300, quality='low', loop=False)
                        st.dataframe(uploaded_file.dtypes,use_container_width=True)
                        st.stop()
                    
                    elif len(X_variables) == 0 :
                        st_lottie(no_X_variable_lottie)
                        st.warning('X Variable is empty!')
                        st.stop()

                    # Mit dem Slider kann der Nutzer selber aussuchen wie viel Prozentual er als Test und Train definiert
                    # Default Parameters so das keine Fehler meldung ensteht
                    
                    total_size = 100
                    train_size = 60
                    test_size = 40

                    train_size_col, test_size_col = st.columns(2)

                    with train_size_col:
                        train_size = st.slider('Train Size', min_value=0, max_value=total_size, value=train_size, key= 'Sklearn train size')
                        test_size = total_size - train_size

                    with test_size_col:
                        test_size = st.slider('Test Size', min_value=0, max_value=total_size, value=test_size, key= 'Sklearn test size')
                        train_size = total_size - test_size

                    # Relevant damit das Skript weiter läuft und nicht immer in Fehlermeldungen läuft
                    if train_size <= 0:
                        st_lottie(value_is_zero_in_train_size)
                        st.warning('Train size should be greater than zero.')
                        st.stop()
                    elif test_size <= 0:
                        st.warning('Test size should be greater than zero.')
                        st.stop()
                    # elif train_size + test_size > len(df_dtype_and_groupby_and_dropped):
                    #     st.warning('Train size and Test size exceed the number of samples in the dataset.')
                    #     st.stop()
                    elif train_size == len(uploaded_file):
                        st.warning('Train size cannot be equal to the number of samples in the dataset.')
                        st.stop()
                    
                    # Unanhängige Varible sowie Abhängige 
                    X = uploaded_file[X_variables]
                    y = uploaded_file[Target_variable]
                    
                    # Aufteilung der Train und Test Datensätze
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, train_size=train_size/100, random_state=42)

                    # Initalisierung des Models
                    lm = LinearRegression(fit_intercept=True)
                    lm.fit(X_train, y_train)
                    
                    # Evaluierung
                    # R2 oder auch Score

                    R2_sklearn_train = lm.score(X_train, y_train)
                    R2_sklearn_test = lm.score(X_test, y_test)
                   
                    R_2_training_col,R_2_test_col = st.columns(2)
                    with R_2_training_col:
                        st.metric(label = 'R2 of the Training Data', value = R2_sklearn_train.round(3))
                    with R_2_test_col:
                        st.metric(label = 'R2 of the Test Data', value = R2_sklearn_test.round(3))

                    st.divider()
                    
                    # Root Mean Square error
                    y_pred_train = lm.predict(X_train)
                    y_pred_test = lm.predict(X_test)

                    RMSE_train = sqrt(mean_squared_error(y_train, y_pred_train))
                    RMSE_test = sqrt(mean_squared_error(y_test, y_pred_test))

                    RMSE_train_col,RMSE_test_col = st.columns(2)
                    with RMSE_train_col:
                        st.metric(label = 'RMSE of the Training Data', value = round(RMSE_train,3))
                    with RMSE_test_col:
                        st.metric(label = 'RMSE of the Test Data', value = round(RMSE_test,3))

                    st.divider()
                    
                    # R2-Wert auf den Trainingsdaten berechnen
                    # r2_train = r2_score(y_train, y_pred_train)
                    # st.write('R2 of the Training Data (alternative calculation)', r2_train)
                    
                    # st.divider()
                    
                    # The Mean Absolute Error (MAE)
                    MAE_train = mean_absolute_error(y_train, y_pred_train)
                    MAE_test = mean_absolute_error(y_test, lm.predict(X_test))


                    MAE_train_col,MAE_test_col = st.columns(2)
                    with RMSE_train_col:
                        st.metric(label = 'MAE of the Training Data', value = round(MAE_train,3))
                    with RMSE_test_col:
                        st.metric(label = 'MAE of the Test Data', value = round(MAE_test,3))

                    # Coefficient
                    coefficiensts = lm.coef_
                    coefficients = pd.DataFrame(lm.coef_.reshape(-1, 1), columns=['Coefficient'], index=X.columns)
                    st.table(coefficients)
                
                    # Intercept
                    
                    st.metric(label='Intercept', value=lm.intercept_)

                    results = pd.concat([y_test.reset_index(drop=True), pd.Series(y_pred_test)], axis=1)
                    results.columns = ['y_test', 'y_pred_test']
                    results['difference'] = results['y_test'] - results['y_pred_test']
                    st.dataframe(results,use_container_width= True)

                plt_Scatter = st.expander('Visualization of Scatter Plot')
                with plt_Scatter:
                    fig = px.scatter(results, x='y_test', y='y_pred_test', trendline="ols")
                    fig.update_layout(
                        title="Scatter Plot: Actual vs. Predicted",
                        xaxis_title="Actual values",
                        yaxis_title="Predicted values",
                        font=dict(size=12)
                    )
                    st.plotly_chart(fig)

                    residuals_plot = px.scatter(results, x='y_pred_test', y='difference')
                    residuals_plot.update_layout(
                        title="Residual Plot",
                        xaxis_title="Predicted values",
                        yaxis_title="Residuals",
                        font=dict(size=12)
                    )
                    st.plotly_chart(residuals_plot, use_container_width=True)

                try_machine_learning = st.expander('Try Your Machine Learning')

                with try_machine_learning:
                    user_input_min = None
                    user_input_max = None
                    if len(X.columns) == 1:
                        x0 = st.number_input(X_train.columns[0], min_value=user_input_min, max_value=user_input_max,step=1,key='x1')
                        y_pred_ = lm.intercept_ + coefficiensts[0] * x0

                    elif len(X.columns) == 2:

                        regression_options_2_1,regression_options_2_2 = st.columns(2)

                        x0 = regression_options_2_1.number_input(X_train.columns[0], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_2_1')
                        x1 = regression_options_2_2.number_input(X_train.columns[1], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_2_2')
                        y_pred_ = lm.intercept_ + coefficiensts[0] * x0 + coefficiensts[1] * x1

                    elif len(X.columns) == 3:

                        regression_options_3_1,regression_options_3_2,regression_options_3_3 = st.columns(3)
                        
                        x0 = regression_options_3_1.number_input(X_train.columns[0], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_3_1')
                        x1 = regression_options_3_2.number_input(X_train.columns[1], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_3_2')
                        x2 = regression_options_3_3.number_input(X_train.columns[2], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_3_3')
                        y_pred_ = lm.intercept_ + coefficiensts[0] * x0 + coefficiensts[1] * x1 + coefficiensts[2] * x2

                    elif len(X.columns) == 4:

                        regression_options_4_1,regression_options_4_2,regression_options_4_3,regression_options_4_4 = st.columns(4)
                        
                        x0 = regression_options_4_1.number_input(X_train.columns[0], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_4_1')
                        x1 = regression_options_4_2.number_input(X_train.columns[1], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_4_2')
                        x2 = regression_options_4_3.number_input(X_train.columns[2], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_4_3')
                        x3 = regression_options_4_4.number_input(X_train.columns[3], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_4_4')
                        y_pred_ = lm.intercept_ + coefficiensts[0] * x0 + coefficiensts[1] * x1 + coefficiensts[2] * x2 + coefficiensts[3] * x3

                    
                    elif len(X.columns) == 5:
                        regression_options_5_1,regression_options_5_2,regression_options_5_3 = st.columns(3)
                        
                        x0 = regression_options_5_1.number_input(X_train.columns[0], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_5_1')
                        x1 = regression_options_5_2.number_input(X_train.columns[1], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_5_2')
                        x2 = regression_options_5_3.number_input(X_train.columns[2], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_5_3')

                        regression_options_5_4,regression_options_5_5 = st.columns(2)
                       
                        x3 = regression_options_5_4.number_input(X_train.columns[3], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_5_4')
                        x4 = regression_options_5_5.number_input(X_train.columns[4], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_5_5')
                        y_pred_ = lm.intercept_ + coefficiensts[0] * x0 + coefficiensts[1] * x1 + coefficiensts[2] * x2 + coefficiensts[3] * x3 + coefficiensts[4] * x4

                    elif len(X.columns) == 6:
                        
                        regression_options_6_1,regression_options_6_2,regression_options_6_3 = st.columns(3)
                        
                        x0 = regression_options_6_1.number_input(X_train.columns[0], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_6_1')
                        x1 = regression_options_6_2.number_input(X_train.columns[1], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_6_2')
                        x2 = regression_options_6_3.number_input(X_train.columns[2], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_6_3')
                        
                        regression_options_6_4,regression_options_6_5,regression_options_6_6 = st.columns(3)
                        
                        x3 = regression_options_6_4.number_input(X_train.columns[3], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_6_4')
                        x4 = regression_options_6_5.number_input(X_train.columns[4], min_value=user_input_min, max_value=user_input_max,step=1,key='regregression_options_6_5')
                        x5 = regression_options_6_6.number_input(X_train.columns[5], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_6_6')
                        y_pred_ = lm.intercept_ + coefficiensts[0] * x0 + coefficiensts[1] * x1 + coefficiensts[2] * x2 + coefficiensts[3] * x3 + coefficiensts[4] * x4 + coefficiensts[5] * x5
                    
                    elif len(X.columns) == 7:
                        
                        regression_options_7_1,regression_options_7_2,regression_options_7_3 = st.columns(3)
                        
                        x0 = regression_options_7_1.number_input(X_train.columns[0], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_7_1')
                        x1 = regression_options_7_2.number_input(X_train.columns[1], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_7_2')
                        x2 = regression_options_7_3.number_input(X_train.columns[2], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_7_3')
                        
                        regression_options_7_4,regression_options_7_5,regression_options_7_6,regression_options_7_7 = st.columns(4)
                        
                        x3 = regression_options_7_4.number_input(X_train.columns[3], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_7_4')
                        x4 = regression_options_7_5.number_input(X_train.columns[4], min_value=user_input_min, max_value=user_input_max,step=1,key='regregression_options_7_5')
                        x5 = regression_options_7_6.number_input(X_train.columns[5], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_7_6')
                        x6 = regression_options_7_7.number_input(X_train.columns[6], min_value=user_input_min, max_value=user_input_max,step=1,key='regression_options_7_7')
                        y_pred_ = lm.intercept_ + coefficiensts[0] * x0 + coefficiensts[1] * x1 + coefficiensts[2] * x2 + coefficiensts[3] * x3 + coefficiensts[4] * x4 + coefficiensts[5] * x5  + coefficiensts[6] * x6
                    
                    else:
                        st.warning("""I am still working on increasing the number of columns for the regression. 
                                      Please choose up to 7 columns to avoid this error message""",icon="ℹ️")
                        
                        st.stop()
                    st.markdown(f'Your predicted value for :blue[{Target_variable}] is: **{y_pred_.round(3)}**')

                    







                
                        

                

    else:
        st.write('#### You :blue[**_have not uploaded any data_**] if you want to start just upload a dataset or use one of the example datasets')
        st_lottie(no_data_avaible)



































##########################################################################################
################## Bereich Object detection ##############################################
##########################################################################################

elif options == 'Object detection':
    st.title('Falken Auge')
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # To read image file buffer as bytes:
        bytes_data = img_file_buffer.getvalue()
        # Check the type of bytes_data:
        # Should output: <class 'bytes'>
        st.write(type(bytes_data))







elif options == 'Contact':
    st.write('Still nothing')
    
    
