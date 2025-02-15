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
rnn_explanation = load_lottieurl("https://lottie.host/1fc8d735-8a9d-4d28-aab8-f707bef6e590/N42d1J8UoU.json")
################################################################################################################
################################################################################################################


####################  H O M E P A G E   ########################################################################    


st.write("# :green[Hey, I'm Riccardo D'Andrea - welcome to my site!]")

st.markdown("###### I'm glad you stopped by! This is all about my passion for **data science** and data engineering. I love tinkering in the infinite vastness of code and data — whether it's building clever data pipelines or bringing machine learning models to life.")

st.markdown("###### On this page, you can take a look behind the scenes of my projects and see how I strive to understand and make sense of data. From optimising complex queries in SQL to modelling with Python, I'm constantly learning and eager to develop better and better solutions.")



st.divider()
motvation_col, motivation_pic_col = st.columns(2)
with motvation_col:
    st.markdown("""
### :green[🚀 Turning Theory into Action]
My projects are driven by a clear mission: to bridge the gap between theoretical concepts and real-world applications, making them more accessible and impactful for users. By keeping my code open source, I encourage everyone to explore, test, and, ideally, contribute to enhancing it further.

### 🌍 :green[Embracing Challenges as a European and a German]
As a proud European—and even more so as a German—I am committed to confronting challenges directly, be it pandemics, wars, political upheavals, or economic crises. In times of adversity, I believe that retreating is not an option; instead, facing problems with determination is essential. Maintaining a clear mind and readiness to solve even the toughest issues is crucial. This approach requires dedication, discipline, and an unwavering belief in the European project and the German ideal. We have the capability in Europe to shape a sustainable future that empowers the next generations to build and progress further.
""")
st.markdown("""
<div style="display: inline-flex; align-items: center;">
    <span>Down below you will find an explanation about my projects</span>
    <span style="font-size: 24px; margin-left: 8px;">&#8964;</span> <!-- Unicode for a downward arrow -->
</div>
""", unsafe_allow_html=True)



with motivation_pic_col:
    st_lottie(working_men, width=1000, height=400)
st.divider()

# Give the user a sort overview what the different section in the homepage
explination_homepage = option_menu("Main Menu", 
                                    ["Machine Learning",
                                    "Object detection",
                                    "Stock Dashboard",
                                    "Recurrent Neural Network"], 

                            icons = ['bar-chart-line-fill', 
                                     'eye-fill',
                                     'building-up',
                                     "bi bi-cpu"], 

                            menu_icon = "cast",

                            orientation = 'horizontal', 

                            default_index = 0)
    
if 'Machine Learning' in explination_homepage:
    # use of ccs because than we can center the tile otherwise it would be left orientited on the homepage
    st.markdown(f"<div style='text-align:center;'><h1>Machine Learning</h1></div>",
                unsafe_allow_html=True)
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
    st_lottie(machine_learning_explanation, 
              width=2000,
              height=400,
              quality='high' )
    
    
    
#### Explination of what is Objection Detection
if 'Object detection' in explination_homepage:

    # use of ccs because than we can center the tile otherwise it would be left orientited on the homepage
    st.markdown(f"<div style='text-align:center;'><h1>Objection Detection</h1></div>",
                unsafe_allow_html=True)
    st.write("""Imagine object recognition as a robot navigating its surroundings, 
                swiftly identifying any object in its path. 🤖 It's akin to 
                having a waiter who, with each new dish served, instantly 
                recognizes its contents, checking for nuts or gluten to alert 
                guests with allergies. 🍽️ Whether it's cars, buildings, or faces, 
                object recognition allows us to identify and track everything in 
                our environment. But here's a humorous twist: if you send the object recognition 
                program to a party, it might hilariously attempt to label each pair of shoes as a separate object. 👠👞 
                That might not be the most practical application, but it sure adds a touch of whimsy to the capabilities of object recognition! 😄🌐""")
    st_lottie(objection_detection_explanation, 
                width=2000,
                height=400,
                quality='high')
    
    

if 'Stock Dashboard' in explination_homepage:

    st.markdown(f"<div style='text-align:center;'><h1>Stock Dashboard</h1></div>",
                unsafe_allow_html=True)
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
                It reminds you to take a coffee break, because a caffeinated investor is a happy investor. ☕📉 
                Celebrating gains with virtual confetti and consoling you through losses with a digital pat on the back, saying, 'Don't worry, we'll bounce back!'
                With a stock dashboard, your investment journey becomes a comedy show, 
                and StockSavvy is your financial stand-up, making the financial world a 
                bit more entertaining, one trade at a time. 🎤💸""")
    
    st_lottie(stock_dashboard_explanation, 
                width=2000,
                height=400,
                quality='high')

    


if 'Recurrent Neural Network' in explination_homepage:

    st.markdown(f"<div style='text-align:center;'><h1>Recurrent Neural Network</h1></div>",
                unsafe_allow_html=True)
    st.markdown("""Introducing your Recurrent Neural Network (RNN), the model that doesn’t just look at the present but also remembers the past to help predict the future! 🔮 Let’s call it "NeuroTime". NeuroTime is like a time traveler that can glance back at what’s already happened, learning from it and using that knowledge to make better decisions going forward.
                NeuroTime shines when it comes to sequences. 📊 Just like you remember the lyrics of your favorite song after hearing it a few times, RNNs remember the important patterns in data over time — whether it's stock prices, weather data, or even a sentence in a story. As trends shift and the market moves, NeuroTime keeps track of everything it’s seen, constantly adapting and making predictions based on what’s happened before. 🚀
                It’s not just a one-hit wonder, though. Sometimes, memories fade (let’s face it, no one has perfect recall), but that’s where Long Short-Term Memory (LSTM) comes in — an upgraded version of RNNs that helps keep important info from slipping away. Think of it as the model’s personal assistant, making sure crucial details are never forgotten. 🧳
                Imagine NeuroTime as your reliable guide through time, turning messy data into a smooth, predictive path. Whether you're using it to translate languages, predict the stock market, or forecast the weather, NeuroTime brings the magic of memory to every sequence. 🌟
                With NeuroTime, you're not just making predictions — you’re learning from the past to shape a smarter future, one sequence at a time! 🌤️💸""")
    
    st_lottie(rnn_explanation, 
                width=2000,
                height=400,
                quality='high')



