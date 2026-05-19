from pydantic import BaseModel
from typing import Optional

class UserInput(BaseModel):
    message: str
    model: Optional[str] = "qwen2.5-1.5b-chat-tamly-no-emotion:latest"
    emotion_model: Optional[str] = "default"
    conversation_id: Optional[int] = None
    user_id: Optional[str] = "guest_user"

class ChatResponse(BaseModel):
    emotion: str
    response: str


class EmotionPredictRequest(BaseModel):
    message: str
    emotion_model: Optional[str] = "default"