""" 
The following script was created to give users the opportunity to learn how to handle a simple machine learning concept.

1. the user starts on the home page
--> Afterwards the User has the possibility to choose between the section 

2. Machine learning
Here the user has the possibility to choose between two predefined datasets or to use his own dataset. As well as to choose the column separator freely.
Then follows the Machine Learning process which is structured as follows.

    - Overview',
        -> df.head(), df.describe(), df.isna().sum() quick overview for the data

    - 'Change the data type',
        -> The user can change the data type if the data is not imported as he/she imagined it.

    - 'Handling missing values',
        -> Each Dataset can have NaN values so we can SOLVE this problem. Here the user can replace the NaN with different possibilities. 
           Depending on the dataset, he can make the best choice of how he wants to replace the NaN values.

    - 'Remove columns',
        -> For better clarity, it is possible to remove entire columns. 

    - 'Visualisation',
        -> In the visualisation area, many different charts can be developed from 
            1. Barchart
            2. Linechart
            3. Scatterchart
            4. Boxplot
            5. Histogram

    - Machine Learning
        ->  In the final area of machine learning, a correlation matrix is created that the user can look at in terms of multicollinearity. But also to determine / evaluate the correlation between the variables. 
            After that, a "target variable" and the "X variables" can be freely selected.
            After successful entry of the X and Y variables, the size of the train and test data set can be determined with a link.
            The most important metrics of a regression model are briefly described below. From Intercept, Slope, RMSE, MAE. 
            In the final area of machine learning, a correlation matrix is created that the user can look at in terms of multicollinearity. 
            But also to determine / evaluate the correlation between the variables. 
            After that, a "target variable" and the "X variables" can be freely selected.
            After successful entry of the X and Y variables, the size of the train and test data set can be determined with a link.
            The most important metrics of a regression model are briefly described below. From Intercept, Slope, RMSE, MAE.

            Two dynamised scatter charts follow 
            1. scatterchart are the Acutal versus the Predicted values
            2. scatterchart are the residuals

            In the last section, after importing the data, Data manipulation, Data visualization. 
            The user can test his regression through freely selectable input options and the result as Output.

3. Objection detection (under construction)
    - under contruction...

4. Contact
    - under contruction... 

To all of you who have taken the time to visit my site and are now reading the script. Thank you so much for taking the time - I really appreciate it. 

If YOU see any improvements, potential errors or optimisation potential please don't hesitate to let me know.
-- Thank you in advance

# https://riccardo-dandrea.streamlit.app/
"""

import streamlit as st 
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu
import requests



################################################################################################################
############### Hier werden die Lottie files eingesetzt  #######################################################
################################################################################################################

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

working_men = load_lottieurl('https://assets1.lottiefiles.com/packages/lf20_w6dptksf.json')
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
rocket_for_cv = load_lottieurl('https://assets4.lottiefiles.com/packages/lf20_atskiwym.json')
################################################################################################################
################################################################################################################


####################  H O M E P A G E   ########################################################################    
st.set_page_config(page_title="Multipage App",
                   page_icon="🧊")






st.write('# :blue[Welcome]')

st.write("""to my Regression App. My name is **Riccardo D'Andrea** and in this website I will guide you through some machine learning 
            processes and explain things as best I 
            can so that we :blue[**all understand why machine learning is so great.**]
            You will find in the navigation bar on the left side of your screen different 
            navigation points where I will explain metrics, functions as understandable as possible.
            So I suggest you just upload a .csv or a .txt file and let it start.""")
st.warning("Hold your horses! This page is still under construction. Don't be surprised if you encounter a wild error or two. But fear not! I'm on the case, working my coding magic to make it all better!")


st_lottie( working_men,
            quality='high',
            width=650,
            height=400)

st.divider()

# Give the user a sort overview what the different section in the homepage
explination_homepage = option_menu("Main Menu", 
                                    ["Machine Learning",
                                    'Object detection'], 

                            icons = ['house', 'gear'], 

                            menu_icon = "cast",

                            orientation = 'horizontal', 

                            default_index = 0)
    
if 'Machine Learning' in explination_homepage:
    # use of ccs because than we can center the tile otherwise it would be left orientited on the homepage
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
    
#### Explination of what is Objection Detection
if 'Object detection' in explination_homepage:

    # use of ccs because than we can center the tile otherwise it would be left orientited on the homepage
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

