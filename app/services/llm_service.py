import json
import re
import uuid
from typing import List, Dict, Tuple, Optional, AsyncIterator
import numpy as np
import httpx
from fastapi import HTTPException

# Import Agno và Memori
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from app.models.db import mem

from app.config import ENABLED_EMOTION_MODELS, OLLAMA_BASE_URL, DEFAULT_LLM_MODEL

class LLMService:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.tags_url = f"{base_url}/api/tags"
        self.default_model = DEFAULT_LLM_MODEL
        
        # Ollama cung cấp API tương thích với OpenAI ở đuôi /v1
        openai_base_url = f"{base_url.rstrip('/')}/v1"
        
        # Khởi tạo model mặc định qua giao thức OpenAI tương thích
        self.model = OpenAIChat(
            id=self.default_model,
            base_url=openai_base_url,
            api_key="ollama" # Ollama không yêu cầu key thực
        )
        
        # Móc Memori vào model
        mem.llm.register(openai_chat=self.model)

    async def get_available_models(self) -> List[Dict]:
        """Lấy danh sách các model hiện có trong Ollama."""
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(self.tags_url, timeout=5.0)
                res.raise_for_status()
                return res.json().get("models", [])
        except Exception as e:
            print(f"Ollama connection error: {e}")
            return []

    async def get_available_emotion_models(self) -> List[str]:
        return ENABLED_EMOTION_MODELS

    def _strip_markdown_fence(self, text: str) -> str:
        raw = text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                return "\n".join(lines[1:-1]).strip()
        return raw

    def _extract_json_payload(self, text: str) -> Optional[Dict]:
        candidate = self._strip_markdown_fence(text)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        start = candidate.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False

        for i in range(start, len(candidate)):
            ch = candidate[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = candidate[start:i + 1]
                    try:
                        parsed = json.loads(snippet)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        return None
        return None

    def _parse_ai_response(self, text: str) -> Tuple[str, str]:
        raw = text.strip()
        payload = self._extract_json_payload(raw)
        if payload is not None:
            emotion = str(payload.get("Emotion") or payload.get("emotion") or "Bình thường").strip()
            advice = str(payload.get("Response") or payload.get("response") or "").strip()
            if advice:
                return emotion or "Bình thường", advice
            return emotion or "Bình thường", raw

        emotion_match = re.search(r"\"?Emotion\"?\s*:\s*\"?([^\"\n]+)\"?", raw, re.IGNORECASE)
        response_match = re.search(r"\"?Response\"?\s*:\s*(.*)", raw, re.IGNORECASE | re.DOTALL)

        emotion = emotion_match.group(1).strip() if emotion_match else "Bình thường"
        advice = response_match.group(1).strip().strip('"') if response_match else raw

        return emotion, advice

    # ---> QUAN TRỌNG: Thêm tham số user_id và conversation_id
    def _create_agent(self, user_id: str, model_name: Optional[str] = None) -> Agent:
        """Hàm khởi tạo Agent chung cho cả stream và non-stream"""
        # 1. Đặt định danh để Memori biết đang lưu ký ức cho ai
        mem.attribution(entity_id=str(user_id), process_id="emotion_chat")
        
        # 2. Xử lý model nếu người dùng chọn model khác mặc định
        current_model = self.model
        if model_name and model_name != self.default_model:
            openai_base_url = f"{self.base_url.rstrip('/')}/v1"
            current_model = OpenAIChat(id=model_name, base_url=openai_base_url, api_key="ollama")
            mem.llm.register(openai_chat=current_model)

        # 3. Khởi tạo Agent với chỉ thị trả về JSON
        return Agent(
            model=current_model,
            instructions=[
                "Bạn là một AI tư vấn tâm lý thấu cảm, tinh tế và có khả năng suy luận tốt.",
                "Hãy sử dụng những thông tin đã biết về người dùng để đưa ra lời khuyên cá nhân hóa.",
                "BẮT BUỘC: Bạn phải trả lời bằng MỘT chuỗi JSON hợp lệ chứa 2 key: 'Emotion' (cảm xúc hiện tại của người dùng) và 'Response' (lời khuyên của bạn)."
            ]
        )

    async def generate_response(self, message: str, user_id: Optional[str] = None, conversation_id: Optional[str] = None, model_name: Optional[str] = None, **kwargs) -> Dict:
        """Tạo phản hồi có tích hợp Memori (Non-stream)

        Hỗ trợ hai cách gọi:
        - Legacy: generate_response(message, model_name)
        - New: generate_response(message, user_id, conversation_id, model_name)
        """
        # Backward compatibility: if caller passed second positional arg (legacy),
        # it will be mapped to `user_id` here — treat that as `model_name` when
        # `conversation_id` and `model_name` are not provided.
        if conversation_id is None and model_name is None and user_id is not None:
            model_name = user_id
            user_id = "guest_user"
        user_id = user_id or "guest_user"

        agent = self._create_agent(user_id, model_name)
        selected_model = model_name or self.default_model

        try:
            # Gọi LLM (Memori tự động nhúng ký ức vào context)
            response = agent.run(message, session_id=str(conversation_id))
            raw_text = response.content

            emotion, advice = self._parse_ai_response(raw_text)

            return {
                "status": "success",
                "emotion": emotion,
                "advice": advice,
                "model_used": selected_model,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"LLM/Memori Error: {str(e)}"
            }

    async def generate_response_stream(self, message: str, user_id: Optional[str] = None, conversation_id: Optional[str] = None, model_name: Optional[str] = None, **kwargs) -> AsyncIterator[Dict]:
        """Tạo phản hồi stream có tích hợp Memori

        Hỗ trợ legacy call pattern: generate_response_stream(message, model_name)
        and new pattern: generate_response_stream(message, user_id, conversation_id, model_name)
        """
        # Backward compatibility: if caller passed model as second positional arg,
        # it will be mapped to `user_id` here — treat that as `model_name`.
        if conversation_id is None and model_name is None and user_id is not None:
            model_name = user_id
            user_id = "guest_user"
        user_id = user_id or "guest_user"

        agent = self._create_agent(user_id, model_name)
        selected_model = model_name or self.default_model
        aggregated_text = ""

        try:
            # Agno hỗ trợ run_stream để lấy từng chunk
            stream_response = agent.run(message, session_id=str(conversation_id), stream=True)
            
            for chunk in stream_response:
                if chunk.content:
                    aggregated_text += chunk.content
                    yield {
                        "type": "chunk",
                        "content": chunk.content,
                    }

            emotion, advice = self._parse_ai_response(aggregated_text)
            yield {
                "type": "final",
                "emotion": emotion,
                "advice": advice,
                "raw_text": aggregated_text,
                "show_emotion": True,
                "reliability_score": 1.0,
                "model_used": selected_model,
            }
        except Exception as e:
            yield {
                "type": "error",
                "message": f"LLM/Memori Error: {str(e)}",
            }

    async def calculate_cosine_similarity_between_two_labels(self, label_a: str, label_b: str, embedder) -> float:
        vec_a = embedder.encode([label_a])[0]
        vec_b = embedder.encode([label_b])[0]
        
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

# Khởi tạo instance
llm_service = LLMService()