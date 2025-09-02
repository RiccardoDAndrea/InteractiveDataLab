import os
import streamlit as st
from streamlit_lottie import st_lottie
import torch
#from navigation.utils import load_lottieurl, Pipeline_for_text2Image
from utils import load_lottieurl, Pipeline_for_text2Image

st.title("Text :two: Image Generator")

# ---- SessionState ----
if "pipe" not in st.session_state:
    st.session_state.pipe = None
if "show_path_input" not in st.session_state:
    st.session_state.show_path_input = False

# ---- Animation ----
working_men = load_lottieurl(
    "https://lottie.host/a2786f75-598c-457d-83b8-da7d5c45b91f/g06V88qWpk.json"
)

welcome_container_lottie, text_container = st.columns(2)
with welcome_container_lottie:
    st_lottie(working_men, width=400, height=200)

with text_container:
    st.markdown("""
        Willkommen zu einer kurzen Demo, die zeigt, wie Sie ein neues Bild durch Textbeschreibung erstellen können.
        Wir werden die HuggingFace API dafür nutzen, aber Sie können
        auch lokal gespeicherte Modelle verwenden.""")



### Applikationsfluss mit `st.form`

# Button zum Anzeigen des Pfad-Eingabefeldes
if st.session_state.pipe is None:
    if st.button("Load a Model"):
        st.session_state.show_path_input = True

# Einbettung der Eingabe in ein Formular
if st.session_state.show_path_input:
    # Beginnen Sie ein Formular
    with st.form("model_loader_form"):
        st.write("Enter your model directory")
        path_input = st.text_input(label="Model Path")

        # Jede Eingabe innerhalb des Formulars wird erst bei Klick auf den Submit-Button verarbeitet
        submit_button = st.form_submit_button("Load Model")

        if submit_button:
            if path_input:
                st.info(f"Selected model path: {path_input}")
                try:
                    # Annahme: Pipeline_for_text2Image ist in navigation.utils definiert
                    pipe = Pipeline_for_text2Image(path_input)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    st.session_state.pipe = pipe.to(device)
                    st.session_state.show_path_input = False
                    st.success(f"✅ Pipeline geladen auf {device.upper()}")
                    # st.rerun() kann hier vermieden werden, da die Form
                    # automatisch das Skript neu ausführt, sobald sie abgeschickt wird.
                except Exception as e:
                    st.error(f"Konnte das Modell nicht laden: {e}")
            else:
                st.warning("Bitte geben Sie ein Verzeichnis ein.")



### Bild-Generierung

# Nur diesen Abschnitt anzeigen, wenn die Pipeline geladen ist und der Pfad-Input nicht sichtbar ist
if st.session_state.pipe and not st.session_state.show_path_input:
    prompt = st.text_input("Beschreiben Sie Ihr Bild:", key="prompt_input")
    if st.button("Bild generieren"):
        if prompt:
            with st.spinner("Bild wird generiert..."):
                image = st.session_state.pipe(prompt, num_inference_steps=25).images[0]
                st.image(image, caption=prompt)
        else:
            st.warning("Bitte geben Sie einen Prompt ein, um ein Bild zu generieren.")