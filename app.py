#!/usr/bin/env python3
"""
Korean Audio Dataset API Server
================================
Receives: POST /analyze  { "audio_id": "q0", "audio_base64": "..." }
Returns:  JSON with all required dataset statistics keys

Deploy options:
  1. Local:  python server.py  (runs on port 5050)
  2. Render / Railway / Fly.io: push this folder, set start command to: python server.py
  3. Docker: see Dockerfile in this folder
"""

import base64, io, json, warnings
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, request, jsonify

warnings.filterwarnings("ignore")
app = Flask(__name__)


# ── Core analysis ────────────────────────────────────────────────────────────

def analyze_audio(audio_base64: str) -> dict:
    # Decode → load
    audio_bytes  = base64.b64decode(audio_base64)
    audio_buffer = io.BytesIO(audio_bytes)

    try:
        y, sr = librosa.load(audio_buffer, sr=None, mono=True)
    except Exception:
        audio_buffer.seek(0)
        y, sr = sf.read(audio_buffer)
        y = y.mean(axis=1) if y.ndim > 1 else y
        y = y.astype(np.float32)

    # 20 MFCC coefficients  (shape: 20 × T)
    mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    columns = [f"mfcc_{i}" for i in range(20)]

    def safe_mode(arr: np.ndarray) -> float:
        counts, edges = np.histogram(arr, bins=10)
        k = int(np.argmax(counts))
        return float((edges[k] + edges[k + 1]) / 2)

    mean     = {c: float(np.mean(mfcc[i]))    for i, c in enumerate(columns)}
    std      = {c: float(np.std(mfcc[i]))     for i, c in enumerate(columns)}
    variance = {c: float(np.var(mfcc[i]))     for i, c in enumerate(columns)}
    minimum  = {c: float(np.min(mfcc[i]))     for i, c in enumerate(columns)}
    maximum  = {c: float(np.max(mfcc[i]))     for i, c in enumerate(columns)}
    median   = {c: float(np.median(mfcc[i]))  for i, c in enumerate(columns)}
    mode     = {c: safe_mode(mfcc[i])         for i, c in enumerate(columns)}
    rng      = {c: float(np.ptp(mfcc[i]))     for i, c in enumerate(columns)}

    allowed_values = {
        c: [float(np.percentile(mfcc[i], p)) for p in [0, 25, 50, 75, 100]]
        for i, c in enumerate(columns)
    }
    value_range = {
        c: [float(np.min(mfcc[i])), float(np.max(mfcc[i]))]
        for i, c in enumerate(columns)
    }
    correlation = np.corrcoef(mfcc).tolist()

    return {
        "rows":           int(mfcc.shape[1]),
        "columns":        columns,
        "mean":           mean,
        "std":            std,
        "variance":       variance,
        "min":            minimum,
        "max":            maximum,
        "median":         median,
        "mode":           mode,
        "range":          rng,
        "allowed_values": allowed_values,
        "value_range":    value_range,
        "correlation":    correlation,
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data         = request.get_json(force=True)
        audio_base64 = data.get("audio_base64", "")
        if not audio_base64:
            return jsonify({"error": "audio_base64 is required"}), 400
        return jsonify(analyze_audio(audio_base64))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    print(f"🎧 Audio Analysis API running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
