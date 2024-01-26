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
## streamlit run 1_🏡_Home.py


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
machine_learning_explanation = load_lottieurl('https://assets7.lottiefiles.com/packages/lf20_LmW6VioIWc.json')
deep_learning_explanation = load_lottieurl('https://assets6.lottiefiles.com/packages/lf20_sibnlnc9.json')
objection_detection_explanation = load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_th55Gb.json')
stock_dashboard_explanation = load_lottieurl('https://lottie.host/562bdf3a-49ea-4c06-941b-706dced0741e/wV9uEdTUyV.json')
################################################################################################################
################################################################################################################


####################  H O M E P A G E   ########################################################################    


st.set_page_config(
    page_title="Portfolio Projects",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.write('# :blue[Welcome]')

st.write("""Welcome to my portfolio homepage! 🚀 Here, I, :blue[Riccardo D'Andrea], 
            an avid :blue[data enthusiast], proudly showcase my projects. Join me on 
            an exciting journey into the realm of :blue[machine learning]. 🤖 On this 
            website, you'll explore not only a :blue[stock market dashboard] and an 
            :blue[object detection system] but also feel the contagious passion of a 
            data enthusiast. 📊 As a dedicated follower of data science, my 
            mission is to transform these concepts into tangible projects 
            through thorough research and hands-on practice. 🛠️ Come along and 
            witness how I turn my passion for data into captivating projects. 
            🌐 Experience firsthand how data has the power to change the world, 
            and let the possibilities inspire you! 💡""")

st.warning("""Hold your horses! This page is still under construction. 
              Don't be surprised 
              if you encounter a wild error or two. But fear not! I'm on the case, 
              working my coding magic to make it all better!""")


st_lottie( working_men,
            quality='high',
            width=1500,
            height=400,
                )

st.divider()

# Give the user a sort overview what the different section in the homepage
explination_homepage = option_menu("Main Menu", 
                                    ["Machine Learning",
                                    'Object detection',
                                    "Stock Dashboard"], 

                            icons = ['bar-chart-line-fill', 
                                     'eye-fill',
                                     'building-up'], 

                            menu_icon = "cast",

                            orientation = 'horizontal', 

                            default_index = 0)
    
if 'Machine Learning' in explination_homepage:
    # use of ccs because than we can center the tile otherwise it would be left orientited on the homepage
    st.markdown(f"<div style='text-align:center;'><h1>Machine Learning</h1></div>",
                unsafe_allow_html=True)
    
    st_lottie(machine_learning_explanation, 
              width=1500,
              height=400,
              quality='high')
    
    st.write("""Picture this: You're on Netflix, craving a good movie night. 
                No need to spend hours scrolling through endless lists, thanks 
                to machine learning! Netflix now tailors recommendations to 
                exactly what you want to see. 🎬 And if hunger strikes mid-movie, 
                Amazon's got your back, suggesting the perfect pizza based on your 
                preferences and order history. 🍕 But here's a fun twist: if you 
                try to identify yourself with a selfie, watch out! The facial 
                recognition program might mistakenly think you're a robot and 
                lock you out. 😄 Don't worry, though – we're working on perfecting 
                that glitch! 🤖✨""")
    
#### Explination of what is Objection Detection
if 'Object detection' in explination_homepage:

    # use of ccs because than we can center the tile otherwise it would be left orientited on the homepage
    st.markdown(f"<div style='text-align:center;'><h1>Objection Detection</h1></div>",
                unsafe_allow_html=True)
    
    st_lottie(objection_detection_explanation, 
                width=1500,
                height=400, 
                quality='high')
    
    st.write("""Imagine object recognition as a robot navigating its surroundings, 
                swiftly identifying any object in its path. 🤖 It's akin to 
                having a waiter who, with each new dish served, instantly 
                recognizes its contents, checking for nuts or gluten to alert 
                guests with allergies. 🍽️ Whether it's cars, buildings, or faces, 
                object recognition allows us to identify and track everything in 
                our environment.

But here's a humorous twist: if you send the object recognition program to a party, it might hilariously attempt to label each pair of shoes as a separate object. 👠👞 That might not be the most practical application, but it sure adds a touch of whimsy to the capabilities of object recognition! 😄🌐""")

if 'Stock Dashboard' in explination_homepage:

    st.markdown(f"<div style='text-align:center;'><h1>Stock Dashboard</h1></div>",
                unsafe_allow_html=True)
    
    st_lottie(stock_dashboard_explanation, 
                width=1500,
                height=400,
                quality='high')

    st.write("""Introducing your stock dashboard, your financial GPS navigating the
                twists and turns of the market! 📈 Imagine it as your money-savvy 
                sidekick – let's call it 'StockSavvy.' This savvy companion keeps a 
                vigilant eye on your investments and discusses numbers with a touch 
                of wit.
                StockSavvy is not just a messenger of financial updates; it's your 
                financial wingman, ready for action as you navigate the peaks and 
                valleys of your stock adventures. 🚀 Picture it as a comedian in a 
                suit, cracking jokes when the market roller coaster takes unexpected 
                turns.
                But StockSavvy offers more than just laughs; it's a memory maestro. 
                It reminds you to take a coffee break, because a caffeinated investor is a happy investor. ☕📉 Celebrating gains with virtual confetti and consoling you through losses with a digital pat on the back, saying, 'Don't worry, we'll bounce back!'
                With a stock dashboard, your investment journey becomes a comedy show, 
                and StockSavvy is your financial stand-up, making the financial world a 
                bit more entertaining, one trade at a time. 🎤💸""")


