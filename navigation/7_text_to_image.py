import os
import streamlit as st
from streamlit_lottie import st_lottie
import torch
from navigation.utils import load_lottieurl, Pipeline_for_text2Image
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
        Welcome to a short demo showing how you can generate a new image by describing text. 
        We will use the HuggingFace API for this, but you can 
        also use models saved locally.""")

# ---- Button zum Umschalten ----
if st.button("Change Model Path"):
    st.session_state.show_path_input = not st.session_state.show_path_input

# ---- Modell laden (nur wenn nötig oder explizit angefordert) ----
if st.session_state.pipe is None or st.session_state.show_path_input:
    Path_to_models = st.text_input(
        label="Enter your model directory",
        key="path_input"
    )

    if Path_to_models:
        st.info(f"Selected model path: {Path_to_models}")

        if st.button("Load Model"):
            try:
                pipe = Pipeline_for_text2Image(Path_to_models)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                st.session_state.pipe = pipe.to(device)
                st.success(f"✅ Pipeline loaded on {device.upper()}")
                st.session_state.show_path_input = False  # Pfadeingabe wieder schließen
            except Exception as e:
                st.error(f"Could not load model: {e}")
    else:
        st.info("error")
        st.warning("No directory entered")



# ---- Prompt Input + Image Generation ----
if st.session_state.pipe and not st.session_state.show_path_input:
    prompt = st.text_input("Describe your image prompt:")
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            image = st.session_state.pipe(prompt, num_inference_steps=25).images[0]
            st.image(image, caption=prompt)
