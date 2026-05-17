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
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "tamly-model-withoutemotion")