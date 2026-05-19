"""Configuration file for Emotion Chat application."""

import os

# Emotion Models Configuration
EMOTION_MODELS = {
    "ml": {
        "name": "Machine Learning",
        "service": "ml_emotion_service",
        "enabled": True
    },
    "phobert": {
        "name": "PhoBERT Multitask",
        "service": "phobert_multitask_service",
        "enabled": True
    }
}

# Get list of enabled emotion models
ENABLED_EMOTION_MODELS = [
    config["name"] 
    for config in EMOTION_MODELS.values() 
    if config["enabled"]
]

# LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "qwen2.5-1.5b-chat-tamly-markdown-withemotion:latest")
WHISPER_MODEL_ID = os.getenv("WHISPER_MODEL_ID", "usernone1234/whisper-vi-audio")
WHISPER_TEMP_DIR = os.getenv("WHISPER_TEMP_DIR", "")
