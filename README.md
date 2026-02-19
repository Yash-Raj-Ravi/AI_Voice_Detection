# AI Generated Voice Detection API

This project provides a FastAPI-based backend service that detects whether a voice sample is human or AI-generated using machine learning and audio feature analysis.

## Overview

The system accepts base64-encoded MP3 audio, extracts statistical voice features, scales them using a pretrained scaler, and performs classification using a TensorFlow / Keras model.

The API returns:

- Voice classification (HUMAN / AI_GENERATED)
- Confidence score
- Human-readable explanation

## Features

- FastAPI REST API
- API key authentication
- Base64 audio input support
- MP3 audio validation
- Audio feature extraction
- Pretrained ML model inference
- Confidence scoring
- Explanation generation based on vocal characteristics

## Tech Stack

- Python
- FastAPI
- TensorFlow / Keras
- Librosa (via feature extractor)
- Scikit-learn
- Joblib

## API Endpoint

**POST** `/api/voice-detection`

## Headers

`x-api-key` : YOUR_API_KEY

## Request Body

```json
{
  "language": "english",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_ENCODED_AUDIO"
}
```
## Response Example

```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.87,
  "explanation": "Speech shows low pitch variation, commonly observed in synthetic voices."
}
```
## Error Response Example
```json
{
  "status": "error",
  "message": "Only mp3 audio is supported"
}
```
## Supported Languages

- English
- Hindi
- Tamil
- Telugu
- Malayalam
