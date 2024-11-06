import streamlit as st 
import pandas as pd 
import requests
from streamlit_option_menu import option_menu
import requests
from PIL import Image


contact_possibilities = option_menu("Contact", 
                                ["Contact", "Vita"], 
                                icons=['bi-send-fill', 'person-badge-fill'], 
                                menu_icon="cast",
                                orientation='horizontal', 
                                default_index=0)

if "Contact" in contact_possibilities:

    st.markdown('''
        <h1 align="center">Hi 👋, I'm Riccardo D'Andrea</h1>
        <h3 align="center">A passionate Data Scientist / Engineer from Germany</h3>

        <p>
            I'm currently working on a Streamlit Regression App, where I explore the fascinating world of machine learning and data analysis. 📊🔬
        </p>

        <p>
            Learning about object detection and classification has been my recent focus, and I'm excited to dive deeper into this field. 🕵️‍♂️🔍
        </p>

        <p>
            If you have any questions or just want to chat about anything related to data science or data engineer, feel free to reach out to me. I'm always open for discussions and collaborations! 💬📫
        </p>

        <h3 align="left">Connect with me:</h3>
        <p align="left">
            <a href="https://www.linkedin.com/in/riccardo-dandrea-670426234/" target="_blank">LinkedIn</a> |
            <a href="https://stackoverflow.com/users/19773284" target="_blank">Stack Overflow</a>
        </p>
    ''', unsafe_allow_html=True)

if "Vita" in contact_possibilities:

    language_version = st.radio('Which langauge do you want the CV', ['German version', 
                                                                      'English version'])
        
    ##### G E R M A N _ V E R S I O N ####

    if 'German version' in language_version:
        st.markdown("<div style='text-align:center;'><h1>Fertigkeiten</h1></div>", unsafe_allow_html=True)
        st.divider()

        CV_title, CV_image = st.columns([2, 1])  # Aufteilung in zwei Spalten

        with CV_title:

            

            st.subheader("*Programmier Sprachen*")
            st.markdown("""
            <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python" width="40" height="40"/>
            <img src="https://www.r-project.org/logo/Rlogo.svg" alt="R" width="40" height="40"/>
            <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/cplusplus/cplusplus-original.svg" alt="C++" width="40" height="40"/>
            """, unsafe_allow_html=True)

            st.divider()
            st.subheader("*Libaries*")
            st.markdown("""
                <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original-wordmark.svg" alt="MySQL" width="40" height="40"/>
                <span style="margin-left: 5px;">MySQL</span> |

                <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="OpenCV" width="40" height="40"/>
                <span style="margin-left: 5px;">OpenCV</span> |

                <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="Pandas" width="40" height="40"/>
                <span style="margin-left: 5px;">Pandas</span> |

                <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="PyTorch" width="40" height="40"/>
                <span style="margin-left: 5px;">PyTorch</span> |

                <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="Scikit-Learn" width="40" height="40"/>
                <span style="margin-left: 5px;">Scikit-Learn</span> |

                <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="Seaborn" width="40" height="40"/>
                <span style="margin-left: 5px;">Seaborn</span> |

                <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="TensorFlow" width="40" height="40"/>
                <span style="margin-left: 5px;">TensorFlow</span> |

                <img src="https://upload.wikimedia.org/wikipedia/commons/a/ae/Keras_logo.svg" alt="Keras" width="40" height="40"/>
                <span style="margin-left: 5px;">Keras</span> |

                <!-- Alternative OpenAI Logo -->
                <img src="https://www.svgrepo.com/show/306500/openai.svg" alt="OpenAI" width="40" height="40"/>
                <span style="margin-left: 5px;">OpenAI</span> |

                <!-- Alternative Langchain Logo -->
                <img src="https://www.svgrepo.com/show/312765/parrot.svg" alt="Langchain" width="40" height="40"/>
                <span style="margin-left: 5px;">Langchain</span> |
            
            """, unsafe_allow_html=True)
            st.divider()

            st.subheader("*Tools*")
            
            # Tools mit entsprechenden Icons in Markdown anzeigen
            st.markdown("""
                    <div style="display: flex; align-items: center;">
                        <div style="margin-right: 20px;">
                            <img src="https://www.vectorlogo.zone/logos/docker/docker-icon.svg" alt="Docker" width="40" height="40"/>
                            <span style="margin-left: 5px;">Docker</span>
                        </div>
                        <div style="margin-right: 20px;">
                            <img src="https://git-scm.com/images/logos/downloads/Git-Icon-1788C.png" alt="Git" width="40" height="40"/>
                            <span style="margin-left: 5px;">Git</span>
                        </div>
                        <div>
                            <img src="https://www.vectorlogo.zone/logos/google_bigquery/google_bigquery-icon.svg" alt="Google Big Query" width="40" height="40"/>
                            <span style="margin-left: 5px;">Google Big Query</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            st.divider()
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
            st.image("pic/mso_Bild.jpeg", width=200, use_column_width=True)
        st.divider()

        st.markdown(f"<div style='text-align:center;'><h2>Lebenslauf</h2></div>", unsafe_allow_html=True)
        
        berufs_erfahrung_col, bildung_col = st.columns(2)
        
        with berufs_erfahrung_col:
              
            st.subheader("Berufserfahrung")
            current_activity_date, current_activity = st.columns((1,2))  
            with berufs_erfahrung_col:
                with current_activity_date:
                    st.markdown("""09/2020 – 08/2024""")
                    

                with current_activity:
                    st.markdown("*STATY.AI*")
                    st.markdown("""Ziel: Entwicklung einer Streamlit-App, die es Nutzern ermöglicht, eigene Bilder hochzuladen, 
                                 um ein CNN-Modell mithilfe von ***MobileNetV2*** und ***SENet*** zu kalibrieren und für die Objekterkennung nutzen.""")
                    st.markdown("*Empirisches Projekt*")
                    st.markdown("""Unterstützung der Studierenden bei der Umsetzung der Python Skripte für Daten beschaffung und der Regressionen erstellung und evaluierung""")
                    st.markdown("""*Leitung einer einwöchige Blockwoche an der Hochschule Osnabrück*""")
                    st.markdown("Thema: Python-Crashkurs: Von der Idee zur interaktiven Data Science Web-App") 
                                    
                st.divider()
                
                
                intership_date, intership_activity = st.columns((1,2))

                with intership_date:
                    st.markdown("""> 03/2023 - 06/2023""")
                with intership_activity:
                    st.markdown(" **Praktikum bei der mso digital GmbH & Co. KG**")
                    st.markdown("Abteilung **Data & Process Analytics**")    
                    st.markdown("""- Python-Skript zur Anbindung der Google Ads und Criteo API mit automatischer Speicherung in BigQuery und regelmäßigem Daten-Update, was erhebliche Kosten sparte.""")
                    st.markdown(""" - Erstellung linearer Regressionen für ein Marketing mix modeling Modell""")        
                st.divider()
                

            with bildung_col:
                
                st.subheader("Bildung")
                studium_date_col, studium_desc_col = st.columns((1,2)) 
                with studium_date_col:
                    st.write("2020 - 2024")

                with studium_desc_col:
                    st.markdown("""**Bachelor of Arts in Angewandter Volkswirtschaft**
                                     an der Hochschule Osnabrück""")
                    st.markdown("Bachelorarbeit mit 1,0 bestanden")
                    st.markdown("""
                                <p><u><strong>Thema</strong>:</u> 
                                Exploring the Power of RNNs and LLMs: 
                                Designing a Web App for Enhanced Predictions Using 
                                Recurrent Neural Networks and OpenAI API-powered Retrieval 
                                Augmented Generation</p>""", unsafe_allow_html=True)
                    
                italy_date_col, italy_desc_col = st.columns((1,2)) 
                with italy_date_col:
                    st.markdown("09.2023 - 12.2023")
                with italy_desc_col:
                    st.markdown("""
                                **Auslandssemester Italien, Turin**
                                
                                SAA - School of Management
                                - Internationale Team Erfahrung
                                - Englische Sprachkenntnisse""")
                st.divider()                    
                hfw_date_col, hfw_desc_col = st.columns((1,2))

                with hfw_date_col:
                    st.markdown("02/2019 – 02/2021")

                with hfw_desc_col:

                    st.markdown("""**Handelsfachwirt**""")
                    st.markdown("Deichmann SE, Osnabrück Nahne")
                    st.markdown("""
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
    if 'English version' in language_version:
        st.markdown(f"<div style='text-align:center;'><h1>Curriculum Vitae</h1></div>",
                    unsafe_allow_html=True)
        st.divider()

        # url = "https://media.licdn.com/dms/image/D4D03AQG-a7zlIBBcYw/profile-displayphoto-shrink_200_200/0/1699013217528?e=1710979200&v=beta&t=qMYfiGkQEJt8hB45L0WRUdg0YCGgqp946wrHXmXn4vw"

        # image = Image.open(requests.get(url, stream=True).raw)
        # resized_image = image.resize((200, 200))  # Ändern Sie die Größe nach Bedarf
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
            st.image("pic/mso_Bild.jpeg", width=200, use_column_width=True)
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
