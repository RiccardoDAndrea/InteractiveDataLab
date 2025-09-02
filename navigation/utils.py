import os
import requests
from diffusers import AutoPipelineForText2Image, DEISMultistepScheduler
import torch


def load_lottieurl(url: str):
    """Lädt eine Lottie-Animation von einer URL (JSON)."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[WARN] Konnte Lottie-Animation nicht laden: {e}")
    return None


def Pipeline_for_text2Image(Path_to_models: str):
    """
    Lädt eine Text2Image-Pipeline aus einem lokalen Diffusers-Modellordner.
    Erwartet, dass im Ordner eine model_index.json liegt.
    """
    # Falls nur .../snapshots angegeben → neuesten Hash-Ordner nehmen
    if Path_to_models.endswith("snapshots"):
        snapshots = os.listdir(Path_to_models)
        if not snapshots:
            raise FileNotFoundError("No snapshot folders found in given path.")
        snapshots = sorted(
            snapshots,
            key=lambda x: os.path.getmtime(os.path.join(Path_to_models, x)),
            reverse=True
        )
        Path_to_models = os.path.join(Path_to_models, snapshots[0])

    if not os.path.exists(os.path.join(Path_to_models, "model_index.json")):
        raise FileNotFoundError(
            f"❌ {Path_to_models} enthält keine model_index.json. "
            "Bitte ins Diffusers-Format konvertieren!"
        )

    # Pipeline laden (nur lokal, kein Download!)
    pipe = AutoPipelineForText2Image.from_pretrained(
        Path_to_models,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True
    )

    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)

    return pipe

