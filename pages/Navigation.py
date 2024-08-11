## streamlit run 1_Home.py
import streamlit as st 
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu
import requests

Home = st.Page(
    "1_Home.py",
    title="Home",
    icon=":material/home:",
    )

Linear_Regression = st.Page(
    "2_🤖_Machine_Learning.py",
    title="Linear Regression",
    icon=":material/bar_chart:",
)

Stock_Dashboard = st.Page(
    "3_📈_Stock_Dashboard.py",
    title="Stocks Dashboard",
    icon=":material/smart_toy:",
)

Explore_the_power_of_rnns = st.Page(
    "4_🤖_Recurrent_Neural_Network.py",
    title="Recurent Neural Network",
    icon=":material/smart_toy:",
)

Contact = st.Page(
    "5_🤵‍♂️_Contact.py",
    title="Contact",
    icon=":material/smart_toy:",
)

pg = st.navigation(
    {
            
        "Projects": [Linear_Regression, Stock_Dashboard,Explore_the_power_of_rnns],
        "Info": [Contact],
    }
)


pg = st.navigation(
    {
        "Home": [Home],
        "Projects": [Linear_Regression, Stock_Dashboard,Explore_the_power_of_rnns],
        "Info": [Contact],
    }
)


pg.run()