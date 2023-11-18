import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import requests
import pandas as pd

contact_possibilities = option_menu("Contact", 
                                       ["Contact",
                                        'Vita'], 

                                icons = ['bi-send-fill', 'person-badge-fill'], 

                                menu_icon = "cast",

                                orientation = 'horizontal', 

                                default_index = 0)

if 'Contact' in contact_possibilities:
       
    st.markdown('''
            <h1 align="center">Hi 👋, I'm Riccardo D'Andrea</h1>
            <h3 align="center">A passionate Data Scientist from Germany</h3>

            <p>
                I'm currently working on a Streamlit Regression App, where I explore the fascinating world of machine learning and data analysis. 📊🔬
            </p>

            <p>
                Learning about object detection and classification has been my recent focus, and I'm excited to dive deeper into this field. 🕵️‍♂️🔍
            </p>

            <p>
                If you have any questions or just want to chat about anything related to data science, feel free to reach out to me. I'm always open for discussions and collaborations! 💬📫
            </p>

            <h3 align="left">Connect with me:</h3>
            <p align="left">
                <a href="https://www.linkedin.com/in/riccardo-dandrea-670426234/" target="_blank">LinkedIn</a> |
                <a href="https://stackoverflow.com/users/19773284" target="_blank">Stack Overflow</a>
            </p>

            <h3 align="left">Languages and Tools:</h3>
            <p align="left">
                <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="Git" width="40" height="40"/>
                <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original-wordmark.svg" alt="MySQL" width="40" height="40"/>
                <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="OpenCV" width="40" height="40"/>
                <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="Pandas" width="40" height="40"/>
                <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python" width="40" height="40"/>
                <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="PyTorch" width="40" height="40"/>
                <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="Scikit-Learn" width="40" height="40"/>
                <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="Seaborn" width="40" height="40"/>
                <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="TensorFlow" width="40" height="40"/>
            </p>
        ''', unsafe_allow_html=True)
        
if 'Vita' in contact_possibilities:
       
        langugage_version = st.radio('Which langauge do you want the CV', ['German version', 
                                                                           'English version'])
        
        ##### G E R M A N _ V E R S I O N ####

        if 'German version' in langugage_version:
            st.markdown(f"<div style='text-align:center;'><h1>Lebenslauf</h1></div>",
                        unsafe_allow_html=True)
            st.divider()

            url = "https://media.licdn.com/dms/image/D4D03AQG-a7zlIBBcYw/profile-displayphoto-shrink_800_800/0/1699013217528?e=1704326400&v=beta&t=NAWAas_NHznblsOZpugDM3bDGE3FR7VssU4CRbBWbUs"

            image = Image.open(requests.get(url, stream = True).raw)
            resized_image = image.resize((200, 200))  # Ändern Sie die Größe nach Bedarf
  # Ändern Sie die Größe nach Bedarf

            CV_title, CV_image = st.columns([2, 1])  # Aufteilung in zwei Spalten

            with CV_title:
                #st_lottie(rocket_for_cv, width=400, height=300, quality='high', loop=True)
                st.markdown(f"<div style='text-align:center;'><h5>Fertigkeiten</h5></div>",
                unsafe_allow_html=True)
                st.markdown("""<p align="left">
                <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="Git" width="40" height="40"/>
                <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original-wordmark.svg" alt="MySQL" width="40" height="40"/>
                <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="OpenCV" width="40" height="40"/>
                <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="Pandas" width="40" height="40"/>
                <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python" width="40" height="40"/>
            </p>""", unsafe_allow_html=True)
                st.markdown("""<p align="left">
                <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="PyTorch" width="40" height="40"/>
                <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="Scikit-Learn" width="40" height="40"/>
                <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="Seaborn" width="40" height="40"/>
                <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="TensorFlow" width="40" height="40"/>
                <img src="https://www.r-project.org/logo/Rlogo.svg" alt="R" width="40" height="40"/>
            </p>""", unsafe_allow_html=True)

                # Leere DataFrame mit Spalten erstellen
                data = {
                "Hard Skill": ["Python 🐍", "Machine Learning 🤖", "Datenvisualisierung 📊", "Statistik 📈"],
                "Soft Skill": ["Problem lösen 💡", "Kommunikation 🗣️", "Kritisches Denken 🤔", "Teamarbeit 🤝"],
            }
            df = pd.DataFrame(data)

            # Spaltenkonfigurationen festlegen
            column_config = {
                "Hard Skill": {},
                "Soft Skill": {}
            }

            # DataFrame anzeigen
            st.dataframe(df, column_config=column_config, hide_index=True, use_container_width=True)
            with CV_image:
                # Hier können Sie den restlichen Inhalt der Spalte platzieren
                st.image(resized_image, use_column_width=True)
            st.divider()

            current_activity_date, current_activity = st.columns((1,2))

            with current_activity_date:
                st.markdown("""09/2020 – jetzt""")
                

            with current_activity:
                st.markdown("""**Bachelor of Arts in Angewandter Volkswirtschaft**
                                im 5. Semester Hochschule Osnabrück, Osnabrück""")
            intership_date, intership_activity = st.columns((1,2))

            with intership_date:
                st.markdown("""> 03/2023 - 06/2023""")
            with intership_activity:
                st.markdown("> **Praktikum bei der mso digital GmbH & Co. KG**\n\n > - Abteilung **Data & Process Analytics**")

                
            st.divider()
            handelsfachwirt_date, handelsfachwirt_activity = st.columns((1,2))

            with handelsfachwirt_date:
                st.markdown("02/2019 – 02/2021")

            with handelsfachwirt_activity:

                st.markdown("""
                            **Handelsfachwirt** 
                            \n
                            Bachelor of Professional in Trade and Commerce Deichmann SE, Osnabrück Nahne
                            - Stellvertretende Leitung
                            - Führung und Motivation der Azubis
                            - Planung und Analyse der Filialkennzahlen
                            """)
                
            st.divider()

            Deichmann_KIE_date, Deichmann_KIE_activity = st.columns((1,2))

            with Deichmann_KIE_date:
                st.markdown("08/2017 – 01/2019")
                
            with Deichmann_KIE_activity:
                st.markdown("""
                            **Kaufmann im Einzelhandel**
                            \n
                            Deichmann SE, Osnabrück
                            """)
            st.divider()

            Fachabitur_date, Fachabitur_activity = st.columns((1,2))
            with Fachabitur_date:
                st.markdown("08/2015 – 07/2017")   

            with Fachabitur_activity:
                st.markdown(""" **Fachabitur in Wirtschaft und Verwaltung**""")
                st.markdown("""BBS am Pottgraben, Osnabrück""")
                
            st.divider()
            Realabschluss_date, Realabschluss_activity = st.columns((1,2))
            with Realabschluss_date:
                st.markdown("08/2015 – 07/2017")

            with Realabschluss_activity:
                st.markdown("""**Sekundarabschluss I**""") 
                st.markdown("""Gesamtschule Schinkel, Osnabrück""")

 ###########################################################################################################################################################################
 ########### E N G L I S H _ V E R S I O N #######################################################################################################################################
 ###########################################################################################################################################################################
        if 'English version' in langugage_version:
            st.markdown(f"<div style='text-align:center;'><h1>Curriculum Vitae</h1></div>",
                        unsafe_allow_html=True)
            st.divider()

            url = "https://media.licdn.com/dms/image/D4D03AQG-a7zlIBBcYw/profile-displayphoto-shrink_800_800/0/1699013217528?e=1704326400&v=beta&t=NAWAas_NHznblsOZpugDM3bDGE3FR7VssU4CRbBWbUs"

            image = Image.open(requests.get(url, stream=True).raw)
            resized_image = image.resize((200, 200))  # Ändern Sie die Größe nach Bedarf
  # Ändern Sie die Größe nach Bedarf

            CV_title, CV_image = st.columns([2, 1])  # Aufteilung in zwei Spalten

            with CV_title:
                #st_lottie(rocket_for_cv, width=400, height=300, quality='high', loop=True)
                st.markdown(f"<div style='text-align:center;'><h5>Skills</h5></div>",
                unsafe_allow_html=True)
                st.markdown("""<p align="left">
                <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="Git" width="40" height="40"/>
                <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original-wordmark.svg" alt="MySQL" width="40" height="40"/>
                <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="OpenCV" width="40" height="40"/>
                <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="Pandas" width="40" height="40"/>
                <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python" width="40" height="40"/>
            </p>""", unsafe_allow_html=True)
                st.markdown("""<p align="left">
                <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="PyTorch" width="40" height="40"/>
                <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="Scikit-Learn" width="40" height="40"/>
                <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="Seaborn" width="40" height="40"/>
                <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="TensorFlow" width="40" height="40"/>
                <img src="https://www.r-project.org/logo/Rlogo.svg" alt="R" width="40" height="40"/>
            </p>""", unsafe_allow_html=True)
              

                # Leere DataFrame mit Spalten erstellen
                data = {
                        "Hard Skill": ["Python 🐍", "Machine Learning 🤖", "Data Visualization 📊", "Statistics 📈"],
                        "Soft Skill": ["Problem Solving 💡", "Communication 🗣️", "Critical Thinking 🤔", "Teamwork 🤝"],
                    }
            df = pd.DataFrame(data)

            # Spaltenkonfigurationen festlegen
            column_config = {
                "Hard Skill": {},
                "Soft Skill": {}
            }

            # DataFrame anzeigen
            st.dataframe(df, column_config=column_config, hide_index=True, use_container_width=True)
            with CV_image:
                # Hier können Sie den restlichen Inhalt der Spalte platzieren
                st.image(resized_image, use_column_width=True)
            st.divider()

            current_activity_date, current_activity = st.columns((1,2))

            with current_activity_date:
                st.markdown("""09/2020 – jetzt""")
                

            with current_activity:
                st.markdown("""**Bachelor of Arts in Applied Economics** in the 5th semester Osnabrück University 
                            of Applied Sciences, Osnabrück""")
            intership_date, intership_activity = st.columns((1,2))

            with intership_date:
                st.markdown("""> 03/2023 - 06/2023""")
            with intership_activity:
                st.markdown("> **Internship mso digital GmbH & Co. KG**\n\n > - Department **Data & Process Analytics**")

                
            st.divider()
            handelsfachwirt_date, handelsfachwirt_activity = st.columns((1,2))

            with handelsfachwirt_date:
                st.markdown("02/2019 – 02/2021")

            with handelsfachwirt_activity:

                st.markdown("""
                            **Handelsfachwirt** 
                            \n
                            Bachelor of Professional in Trade and Commerce Deichmann SE, Osnabrück Nahne
                            - Deputy management
                            - Leading and motivating the trainees
                            - Planning and analysis of branch key figures
                            """)
                
            st.divider()

            Deichmann_KIE_date, Deichmann_KIE_activity = st.columns((1,2))

            with Deichmann_KIE_date:
                st.markdown("08/2017 – 01/2019")
                
            with Deichmann_KIE_activity:
                st.markdown("""
                            **Retail sales assistant**
                            \n
                            Deichmann SE, Osnabrück
                            """)
            st.divider()

            Fachabitur_date, Fachabitur_activity = st.columns((1,2))
            with Fachabitur_date:
                st.markdown("08/2015 – 07/2017")   

            with Fachabitur_activity:
                st.markdown(""" **Vocational baccalaureate in economics and administration**""")
                st.markdown("""BBS am Pottgraben, Osnabrück""")
                
            st.divider()
            Realabschluss_date, Realabschluss_activity = st.columns((1,2))
            with Realabschluss_date:
                st.markdown("08/2015 – 07/2017")

            with Realabschluss_activity:
                st.markdown("""**Secondary Certificate I**""") 
                st.markdown("""Gesamtschule Schinkel, Osnabrück""")