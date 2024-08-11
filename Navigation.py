## streamlit run 1_Home.py
import streamlit as st 


Home = st.Page(
    "pages/1_Home.py",
    title="Home",
    icon=":material/home:",
    default=True)

Linear_Regression = st.Page(
    "pages/2_🤖_Machine_Learning.py",
    title="Linear Regression",
    icon=":material/bar_chart:",
)

Stock_Dashboard = st.Page(
    "pages/3_📈_Stock_Dashboard.py",
    title="Stocks Dashboard",
    icon=":material/smart_toy:",
)

Explore_the_power_of_rnns = st.Page(
    "pages/4_🤖_Recurrent_Neural_Network.py",
    title="Recurent Neural Network",
    icon=":material/smart_toy:",
)

face_detection = st.Page(
    "pages/face_detection.py",
    title="Face Recognition",
    icon=":material/smart_toy:",
)

Contact = st.Page(
    "pages/5_🤵‍♂️_Contact.py",
    title="Contact",
    icon=":material/smart_toy:",
)


pg = st.navigation(
    {
        "Home": [Home],
        "Projects": [Linear_Regression, Stock_Dashboard,Explore_the_power_of_rnns, face_detection],
        "Info": [Contact],
    }
)


pg.run()