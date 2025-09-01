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
    Lädt eine Text2Image-Pipeline aus einem lokalen Modellpfad.
    Es werden keine Dateien heruntergeladen (local_files_only=True).
    """
    # Wenn user nur .../snapshots übergibt → automatisch Hash-Ordner wählen
    if Path_to_models.endswith("snapshots"):
        snapshots = os.listdir(Path_to_models)
        # Nimm den ersten Hash-Ordner (meist nur einer vorhanden)
        Path_to_models = os.path.join(Path_to_models, snapshots[0])

    

    # Pipeline laden
    pipe = AutoPipelineForText2Image.from_pretrained(
        Path_to_models,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,  # 🔒 kein Download
    )

    # Scheduler direkt setzen
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)

    return pipe
