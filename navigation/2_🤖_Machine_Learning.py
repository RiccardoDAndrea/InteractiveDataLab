import streamlit as st 
import pandas as pd 
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import os
import requests
import statsmodels as sm

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

no_date_col = load_lottieurl('https://assets8.lottiefiles.com/packages/lf20_0pgmwzt3.json')
removed_date_column = load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_ovo7L6.json')
no_data_avaible = load_lottieurl('https://assets7.lottiefiles.com/packages/lf20_rjn0esjh.json')
question_with_NaN_values = load_lottieurl('https://assets7.lottiefiles.com/packages/lf20_lKvkGl.json')
no_X_variable_lottie = load_lottieurl('https://assets10.lottiefiles.com/packages/lf20_ydo1amjm.json')
value_is_zero_in_train_size = load_lottieurl('https://assets7.lottiefiles.com/packages/lf20_usmfx6bp.json')
wrong_data_type_ML = load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_2frpohrv.json')
rocket_for_cv = load_lottieurl('https://assets4.lottiefiles.com/packages/lf20_atskiwym.json')

################################################################################################################
################################################################################################################


# Let the user choose the separator
with st.sidebar.expander('Upload settings'):

    separator, thousands = st.columns(2)
    
    with separator:
        selected_separator = st.selectbox('Value separator:', (";", ",", ".", ":"))
    
    with thousands:
        selected_thousands = st.selectbox('Thousands separator:', (",", "."), key='thousands')
    
    decimal, unicode = st.columns(2)
    
    with decimal:
        selected_decimal = st.selectbox('Decimal separator:', (".", ","), key='decimal')
    
    with unicode:
        selected_unicode = st.selectbox('File encoding:', ('utf-8', 'utf-16', 'utf-32', 'iso-8859-1', 'cp1252'))


# Ausgabe des Datensatzes
datasets = ['Car dataset', 'Wage dataset', 'Own dataset']  # Liste der verfügbaren Datensätze
selected_datasets = st.sidebar.selectbox('Choose your Dataset:', options=datasets)

if 'Car dataset' in selected_datasets:
    uploaded_file = pd.read_csv('https://raw.githubusercontent.com/RiccardoDAndrea/InteractiveDataLab/main/Dataset/Car.csv',
                                sep=selected_separator, thousands=selected_thousands, decimal=selected_decimal, encoding=selected_unicode)
    
    
elif 'Wage dataset' in selected_datasets:
     uploaded_file = pd.read_csv('https://raw.githubusercontent.com/RiccardoDAndrea/Streamlit-Regression-App/main/Dataset/wage.csv',
                                 sep=selected_separator, thousands=selected_thousands, decimal=selected_decimal, encoding=selected_unicode)


elif 'Own dataset' in selected_datasets:
    uploaded_file = st.file_uploader("Upload your own dataset", type=['csv', 'txt'], key='dataframe')
    if uploaded_file is not None:
        uploaded_file = pd.read_csv(uploaded_file, sep=selected_separator, thousands=selected_thousands, decimal=selected_decimal, encoding=selected_unicode)



####################################################################################################
#############  M A C H I N E - L E A R N I N G ####################################################
####################################################################################################


# Possibilitys for the use 
overview,change_data_type, handling_missing_values, remove_columns_tab, Visualization, machine_learning = st.tabs(['Overview',
                                                                                                                    'Change the data type',
                                                                                                                    'Handling missing Values',
                                                                                                                    'Remove columns',
                                                                                                                    "Visualisation",
                                                                                                                    'Machine Learning'])

# In this Section we are in Change Data types
if uploaded_file is not None:

    with overview:
            st.write('In the follwoing Tab you can get a :blue[Overview of your data.] It will only show the first :keycap_ten: rows')
            st.dataframe(uploaded_file.head(10),use_container_width=True)
            expander_head = st.expander(':grey[ :information_source: More details about the df.head() method and why it\'s important]')
            expander_head.info(f"""
                The df.head() method in Pandas is used to quickly get an overview of your 
                data. It returns the first 5 rows of your DataFrame, showing the column 
                names, data types, and sample data. This method is particularly useful 
                when dealing with large datasets, as it allows you to quickly understand 
                the structure of your data without having to load the entire dataset.

                By using df.head(), you can make rapid decisions about data cleaning, 
                transformation, or analysis. It saves time and ensures efficient 
                utilization of computational resources. Instead of reloading the 
                entire dataset, you can get a quick glimpse of the data and use that 
                               information to perform necessary operations.""")

            st.divider()

            st.write('Here you can get an overview of your data')
            st.dataframe(uploaded_file.describe(),use_container_width=True)
            expander_describe = st.expander(':grey[:information_source: More details about the describe method and its importance]')
            expander_describe.info("""
                Let's delve into the power of the 'df.describe()' function. This function offers more than just a summary of rows and columns; 
                it provides essential statistical insights. 

                The describe function not only reveals the basic statistics like mean and standard deviation but also includes the median. 
                This is particularly valuable when building machine learning models. For instance, if a prediction value from your model 
                significantly deviates from the maximum value and the median provided by describe, it indicates potential issues with your model's predictions.

                In essence, df.describe() empowers you to quickly grasp the distribution and central tendencies of your data, 
                enabling you to make informed decisions about the performance and reliability of your models.
            """)


            st.divider()

            st.write('Here are the NaN values in your data')
            st.dataframe(uploaded_file.isna().sum(), use_container_width=True)
            expander_NaN = st.expander(':grey[:information_source: What are NaN values and why should you pay attention to them ?]')
            expander_NaN.info("""
                                NaN stands for "Not a Number" and denotes missing or invalid values within a dataset. These values can manifest as missing entries, incorrect data, or other forms of invalid information.

                                For Data Scientists or Analysts, it's crucial to be vigilant about NaN values because they have the potential to distort the dataset. 
                                When conducting statistical analyses or training machine learning models, the presence of missing values can lead to erroneous conclusions or, in the case of models, inaccurate predictions.

                                Addressing NaN values through appropriate handling methods is essential for maintaining the integrity and accuracy of your analyses and models.
                            """)
            st.divider()
    with change_data_type:
        # Display original data types
        st.subheader('Your DataFrame data types:')
        st.dataframe(uploaded_file.dtypes, use_container_width=True)

        # Section for changing data types
        st.subheader("Change your Data Types:")

        # Split into two columns for selecting columns and data types
        change_data_type_col_1, change_data_type_col_2 = st.columns(2)

        with change_data_type_col_1:
            selected_columns_1 = st.multiselect("Choose your columns", uploaded_file.columns, key='change_data_type_1')
            selected_dtype_1 = st.selectbox("Choose a data type", ['None', 'int64', 'float64', 'string', 'datetime64[ns]'], 
                                            key='selectbox_1')

        with change_data_type_col_2:
            selected_columns_2 = st.multiselect("Choose your columns", uploaded_file.columns, key='change_data_type_2')
            selected_dtype_2 = st.selectbox("Choose a data type", ['None', 'int64', 'float64', 'string', 'datetime64[ns]'], 
                                            key='selectbox_2')

    # Function to change data types
        def change_data_types(uploaded_file, columns, dtype):
            if columns and dtype != 'None':  # Ensure columns are selected and dtype is not 'None'
                try:
                    if dtype == "int64":
                        uploaded_file[columns] = uploaded_file[columns].apply(pd.to_numeric, errors='coerce').round(0).astype('Int64')
                    elif dtype == "float64":
                        uploaded_file[columns] = uploaded_file[columns].apply(pd.to_numeric, errors='coerce').astype('float64')
                    elif dtype == "string":
                        uploaded_file[columns] = uploaded_file[columns].astype('string')
                    elif dtype == "datetime64[ns]":
                        uploaded_file[columns] = pd.to_datetime(uploaded_file[columns], errors='coerce')
                except Exception as e:
                    st.error(f"Error converting columns {columns} to {dtype}: {e}")

            return uploaded_file

        # Apply data type changes for both sets of selections
        uploaded_file = change_data_types(uploaded_file, selected_columns_1, selected_dtype_1)
        uploaded_file = change_data_types(uploaded_file, selected_columns_2, selected_dtype_2)

        # Display the modified DataFrame data types
        st.subheader('Modified DataFrame data types:')
        st.dataframe(uploaded_file.dtypes, use_container_width=True)

        # Display the modified DataFrame
        st.subheader('Modified DataFrame:')
        st.dataframe(uploaded_file, use_container_width=True)

   

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
                uploaded_file_median = uploaded_file[numeric_columns].median()
                uploaded_file[numeric_columns] = uploaded_file[numeric_columns].fillna(uploaded_file_median)
                st.write('##### You have succesfully change the NaN values :blue[with the Median]')
                st.dataframe(uploaded_file.isna().sum(), use_container_width=True)
                st.divider()
                
            elif 'with Mean' in missing_values:
                uploaded_file_mean = uploaded_file[numeric_columns].mean()
                uploaded_file[numeric_columns] = uploaded_file[numeric_columns].fillna(uploaded_file_mean)
                st.markdown(' ##### You have succesfully change the NaN values :blue[ with the Mean]')
                st.dataframe(uploaded_file.isna().sum(), use_container_width=True)
                st.divider()

            elif 'with Minimum value' in missing_values:
                uploaded_file_min = uploaded_file[numeric_columns].min()
                uploaded_file[numeric_columns] = uploaded_file[numeric_columns].fillna(uploaded_file_min)
                st.write('##### You have succesfully change the NaN values :blue[with the minimum values]')
                st.dataframe(uploaded_file.isna().sum(), use_container_width=True)
                st.divider()
                
            elif 'with Maximum value' in missing_values:
                uploaded_file_max = uploaded_file[numeric_columns].max()
                uploaded_file[numeric_columns] = uploaded_file[numeric_columns].fillna(uploaded_file_max)
                st.write('##### You have succesfully change the NaN values :blue[with the maximums values]')
                st.dataframe(uploaded_file.isna().sum(), use_container_width=True)
                st.divider()
                
            elif 'with Zero' in missing_values:
                numeric_columns = uploaded_file.select_dtypes(include=[np.number]).columns.tolist()
                uploaded_file[numeric_columns] = uploaded_file[numeric_columns].fillna(0)
                st.write('##### You have successfully changed :blue[the NaN values to 0.]')
                st.dataframe(uploaded_file.isna().sum(), use_container_width=True)
                st.divider()


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
                        fig = go.Figure(data=hist_plot_1)
                        # Umwandeln des Histogramm-Graphen in eine Bilddatei
                        img_bytes = pio.to_image(fig, format='png', width=1000, height=600, scale=2)
                        
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
                        
                        st.divider()

                    elif chart_type == 'Linechart':
                        st.markdown('You can freely choose your :blue[Linechart] :chart_with_upwards_trend:')

                        col3,col4 = st.columns(2)
                        
                        with col3:
                            x_axis_val_line = st.selectbox('Select X-Axis Value', options=uploaded_file.columns,
                                                        key='x_axis_line_multiselect')
                        with col4:
                            y_axis_vals_line = st.multiselect('Select :blue[Y-Axis Values]', options=uploaded_file.columns,
                                                            key='y_axis_line_multiselect')

                        line_plot_1 = px.line(uploaded_file, x=x_axis_val_line, y=y_axis_vals_line)
                        st.plotly_chart(line_plot_1)
                        fig_line = go.Figure(data=line_plot_1)
                        # Umwandeln des Histogramm-Graphen in eine Bilddatei
                        img_bytes_line = pio.to_image(fig_line, format='png', width=1000, height=600, scale=2)
                        
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
                        
                        st.divider()

                    elif chart_type == 'Boxplot':
                        st.write('You can freely choose your :blue[Boxplot]')
                        y_axis_val_bar = st.selectbox('Select Y-Axis Value', options=uploaded_file.columns,
                                                    key='Y_axis_box_multiselect')
                        box_plot_1 = px.box(uploaded_file,y=y_axis_val_bar)
                        st.plotly_chart(box_plot_1)


                        fig_boxplot = go.Figure(data=box_plot_1)
                        # Umwandeln des Histogramm-Graphen in eine Bilddatei
                        img_bytes_boxplot = pio.to_image(fig_boxplot, format='png', width=1000, height=600, scale=2)
                        # Herunterladen der Bilddatei als Button
                        
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
                corr_matrix = uploaded_file.select_dtypes(include=['float64', 
                                                                    'int64']).corr()


                # Erstellung der Heatmap mit Plotly
                fig_correlation = px.imshow(corr_matrix.values, 
                                            color_continuous_scale = 'purples', 
                                            zmin = -1, 
                                            zmax = 1,
                                            x = corr_matrix.columns, 
                                            y = corr_matrix.index,
                                            labels = dict( x = "Columns", 
                                                            y = "Columns", 
                                                            color = "Correlation"))
                
                # Anpassung der Plot-Parameter
                fig_correlation.update_layout(
                                            title='Correlation Matrix',
                                            font=dict(
                                                color='grey'
                    )
                )

                fig_correlation.update_traces( showscale = False, 
                                                colorbar_thickness = 25)
                # Hinzufügen der numerischen Werte als Text
                annotations = []
                for i, row in enumerate(corr_matrix.values):
                    for j, val in enumerate(row):
                        annotations.append(dict(x=j, y=i, text=str(round(val, 2)), showarrow=False, font=dict(size=16)))
                fig_correlation.update_layout(annotations=annotations)

                # Anzeigen der Plot
                st.plotly_chart(fig_correlation, use_container_width= True)
                fig_correlationplot = go.Figure(data=fig_correlation)
                
            #Ende der Correlations Matrix
            
            MachineLearning_sklearn = st.expander('Machine learning evaluation')
            with MachineLearning_sklearn:
                # Hier kann der Nutzer dynamisch die unabhängigen und abhängigen Variablen auswählen
                Target_variable_col, X_variables_col = st.columns(2)
                Target_variable = Target_variable_col.selectbox('Which is your Target Variable (Y)', options=uploaded_file.columns, key='LR Sklearn Target Variable')
                X_variables = X_variables_col.multiselect('Which are your Variables (X)', options=uploaded_file.columns, key='LR Sklearn X Variables')

                # Überprüfung des Datentyps der ausgewählten Variablen
                if uploaded_file[Target_variable].dtype == str or uploaded_file[Target_variable].dtype == str :
                    st.warning('Ups, wrong data type for Target variable!')
                    st_lottie(wrong_data_type_ML, width=700, height=300, quality='low', loop=False)
                    st.dataframe(uploaded_file.dtypes, use_container_width=True)
                    st.stop()

                if any(uploaded_file[x].dtype == object for x in X_variables):
                    st.warning('Ups, wrong data type for X variables!')
                    st_lottie(wrong_data_type_ML, width=700, height=300, quality='low', loop=False)
                    st.dataframe(uploaded_file.dtypes, use_container_width=True)
                    st.stop()

                
                if len(X_variables) == 0 :
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
                X = uploaded_file[X_variables].round(2)
                y = uploaded_file[Target_variable].round(2)
                
                # Aufteilung der Train und Test Datensätze
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, train_size=train_size/100, random_state=42)

                # Initalisierung des Models
                lm = LinearRegression(fit_intercept=True)
                lm.fit(X_train, y_train)
                
                # Evaluierung
                # R2 oder auch Score
                # Berechnung der R2-Scores

                try:
                    R2_sklearn_train = lm.score(X_train.round(2), y_train.round(2))
                    
                except Exception as e:
                    st.error(f"""Error occurred during R2 score calculation for training data: {str(e)}.
                                    Please check the data type of your target variable in the 'Change Data Types' section and make sure it is compatible for regression analysis.""")
                    st.stop()



                R2_sklearn_test = lm.score(X_test.round(2), y_test.round(2))
                
                R_2_training_col,R_2_test_col = st.columns(2)
                with R_2_training_col:
                    st.metric(label = 'R2 of the Training Data', value = round(R2_sklearn_train,2))
                with R_2_test_col:
                    st.metric(label = 'R2 of the Test Data', value = round(R2_sklearn_test,2))

                st.divider()
                
                # Root Mean Square error
                y_pred_train = lm.predict(X_train)
                y_pred_test = lm.predict(X_test)

                RMSE_train = sqrt(mean_squared_error(y_train, y_pred_train))
                RMSE_test = sqrt(mean_squared_error(y_test, y_pred_test))

                RMSE_train_col,RMSE_test_col = st.columns(2)
                with RMSE_train_col:
                    st.metric(label = 'RMSE of the Training Data', value = round(RMSE_train,2))
                with RMSE_test_col:
                    st.metric(label = 'RMSE of the Test Data', value = round(RMSE_test,2))

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
                    st.metric(label = 'MAE of the Training Data', value = round(MAE_train,2))
                with RMSE_test_col:
                    st.metric(label = 'MAE of the Test Data', value = round(MAE_test,2))

                # Coefficient
                coefficiensts = lm.coef_
                coefficients = pd.DataFrame(lm.coef_.reshape(-1, 1), columns=['Coefficient'], index=X.columns)
                st.table(coefficients)
            
                # Intercept
                
                st.metric(label='Intercept', value= round(lm.intercept_,3))

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