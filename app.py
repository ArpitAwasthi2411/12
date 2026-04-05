from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import librosa
import io

app = FastAPI()

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


def compute_stats(data):
    data = np.array(data)

    result = {
        "rows": int(data.shape[0]),
        "columns": [int(data.shape[0])],

        "mean": {"col0": float(np.mean(data))},
        "std": {"col0": float(np.std(data))},
        "variance": {"col0": float(np.var(data))},
        "min": {"col0": float(np.min(data))},
        "max": {"col0": float(np.max(data))},
        "median": {"col0": float(np.median(data))},
        "mode": {"col0": float(np.bincount(data.astype(int)).argmax()) if len(data) > 0 else 0},
        "range": {"col0": float(np.max(data) - np.min(data))},

        "allowed_values": {"col0": list(np.unique(data)[:10])},  # limit size
        "value_range": {"col0": [float(np.min(data)), float(np.max(data))]},

        "correlation": []
    }

    return result


@app.post("/")
async def process_audio(req: AudioRequest):
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(req.audio_base64)

        # Load audio
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

        # Compute stats
        result = compute_stats(audio)

        return result

    except Exception as e:
        return {"error": str(e)}
