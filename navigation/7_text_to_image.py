import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import os

# Titel
st.title("🎨 Stable Diffusion Image Generator")

# Eingabefeld für optionalen Modellpfad oder Repo-Namen
location_of_image_models = st.text_input(
    "Optional: Specify local model path (with model_index.json) or leave empty to use default repo:",
    value=""
)

# Auswahl Modell
model_type = st.radio(
    "Select Model Type:",
    ("SDXL Base 1.0 (High Quality, ~7GB)",
     "SD 1.5 Pruned / Lite (~2GB)",
     "SDXL Turbo (fast, ~7GB)")
)

# Device check
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: **{device}**")

# Pipeline Loader
@st.cache_resource
def load_pipeline(model_choice, custom_path=None):
    # Repo-Namen als Fallback
    if model_choice == "SDXL Base 1.0 (High Quality, ~7GB)":
        default_repo = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe_cls = StableDiffusionXLPipeline
    elif model_choice == "SDXL Turbo (fast, ~7GB)":
        default_repo = "stabilityai/sdxl-turbo"
        pipe_cls = StableDiffusionXLPipeline
    else:  # SD 1.5
        default_repo = "runwayml/stable-diffusion-v1-5"
        pipe_cls = StableDiffusionPipeline

    model_path = default_repo

    if custom_path:
        # wenn Nutzer einen lokalen Ordner angegeben hat
        if os.path.exists(os.path.join(custom_path, "model_index.json")):
            model_path = custom_path
        elif os.path.isdir(custom_path):
            # prüfen, ob es ein Hub-Cache Ordner ist (models--xxx)
            snapshots_dir = os.path.join(custom_path, "snapshots")
            if os.path.exists(snapshots_dir):
                subfolders = os.listdir(snapshots_dir)
                if subfolders:
                    model_path = os.path.join(snapshots_dir, subfolders[0])
                    st.info(f"Using snapshot: {model_path}")
        else:
            st.warning("⚠️ Path not valid, falling back to default repo.")

    pipe = pipe_cls.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    return pipe

pipe = load_pipeline(model_type, location_of_image_models.strip())

# Prompt Input
prompt = st.text_input("Enter your prompt:")

# Zusätzliche Optionen
steps = st.slider("Inference steps", 5, 50, 20)
guidance = st.slider("Guidance scale", 1.0, 15.0, 7.5)
seed = st.number_input("Random Seed (0 = random)", value=0, step=1)

# Bild generieren
if st.button("Generate Image"):
    if prompt.strip():
        with st.spinner("Generating image... ⏳"):
            generator = torch.manual_seed(seed) if seed != 0 else None
            try:
                image = pipe(
                    prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator
                ).images[0]
                st.image(image, caption="Generated Image")
            except torch.cuda.OutOfMemoryError:
                st.error("⚠️ Out of GPU memory! Try lower resolution or a smaller model.")
                torch.cuda.empty_cache()
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
    else:
        st.warning("Please enter a prompt!")
