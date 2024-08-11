## streamlit run 1_Home.py
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





