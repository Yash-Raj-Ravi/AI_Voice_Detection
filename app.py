from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import tempfile
import os
import traceback
import joblib
import tensorflow as tf

from feature_extractor import extract_features_with_stats

# =====================
# CONFIG
# =====================
API_KEY = "YOUR_API_KEY"

SUPPORTED_LANGUAGES = {
    "english", "hindi", "tamil", "telugu", "malayalam"
}

# =====================
# LOAD MODEL & SCALER
# =====================
model = tf.keras.models.load_model("voice_authenticity_model.h5")
scaler = joblib.load("scaler.joblib")

# =====================
# FASTAPI INIT
# =====================
app = FastAPI(title="AI Generated Voice Detection API")

# =====================
# REQUEST SCHEMA
# =====================
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


# =====================
# EXPLANATION LOGIC
# =====================
def generate_explanation(features, prediction, confidence):
    pitch_mean = features[-6]
    pitch_std = features[-5]
    jitter = features[-4]
    shimmer = features[-3]

    reasons = []

    if pitch_std < 15:
        reasons.append("low pitch variation")
    if jitter < 0.02:
        reasons.append("unnaturally stable pitch")
    if shimmer < 0.03:
        reasons.append("low amplitude variation")

    if not reasons:
        reasons.append("natural vocal dynamics")

    base = ", ".join(reasons)

    if prediction == "AI_GENERATED":
        return f"Speech shows {base}, commonly observed in synthetic voices."
    else:
        return f"Speech exhibits {base}, typical of natural human speech."


# =====================
# API ENDPOINT
# =====================
@app.post("/api/voice-detection")
def voice_detection(payload: dict, x_api_key: str = Header(...)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        language = payload.get("language", "").lower()
        audio_format = payload.get("audioFormat")
        audio_b64 = payload.get("audioBase64")

        if language not in SUPPORTED_LANGUAGES:
            return {
                "status": "error",
                "message": "Unsupported language"
            }

        if audio_format != "mp3":
            return {
                "status": "error",
                "message": "Only mp3 audio is supported"
            }

        if not audio_b64:
            return {
                "status": "error",
                "message": "Missing audio data"
            }

        # Decode audio
        audio_bytes = base64.b64decode(audio_b64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Feature extraction
        features = extract_features_with_stats(tmp_path)
        features_scaled = scaler.transform(features.reshape(1, -1))

        prob = float(model.predict(features_scaled)[0][0])

        label = "AI_GENERATED" if prob >= 0.5 else "HUMAN"
        confidence = round(prob if label == "AI_GENERATED" else 1 - prob, 3)

        explanation = generate_explanation(features, label, confidence)

        os.remove(tmp_path)

        return {
            "status": "success",
            "language": language.capitalize(),
            "classification": label,
            "confidenceScore": confidence,
            "explanation": explanation
        }

    except Exception:
        traceback.print_exc()
        return {
            "status": "error",
            "message": "Internal server error during audio processing"
        }
