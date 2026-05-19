import os
import tempfile
import json
import uuid
from fastapi import APIRouter, Depends, Request, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from pathlib import Path
from app.services.llm_service import llm_service
from app.services.history_service import history_service
from app.schemas.chat import EmotionPredictRequest, UserInput
from app.services.ml_emotion_service import ml_emotion_service
from app.services.phobert_multitask_service import get_phobert_multitask_service
from app.services.whisper_service import whisper_service
from app.config import EMOTION_MODELS
from sqlalchemy.orm import Session
from app.models.db import get_db

router = APIRouter()
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _resolve_emotion_model_key(selected_name: str | None) -> str:
    if not selected_name:
        return "ml"

    normalized = selected_name.strip().lower()
    if normalized in {"", "default"}:
        return "ml"

    for key, config in EMOTION_MODELS.items():
        if normalized in {key.lower(), str(config.get("name", "")).lower()}:
            return key

    if "phobert" in normalized:
        return "phobert"
    if "machine learning" in normalized or normalized == "ml":
        return "ml"

    return "ml"


def _build_sse_payload(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _predict_emotion_with_fallback(message: str, selected_emotion_model_key: str):
    warning = None
    if selected_emotion_model_key == "phobert":
        try:
            phobert_service = get_phobert_multitask_service()
            return phobert_service.predict(message), warning
        except Exception as exc:
            warning = f"Không tải được PhoBERT ({exc}). Đang fallback về Machine Learning."

    return ml_emotion_service.predict(message), warning

@router.get("/")
async def load_index(request: Request):
    models = await llm_service.get_available_models()
    emotion_models = await llm_service.get_available_emotion_models()
    return templates.TemplateResponse("index.html", {"request": request, "models": models, "emotion_models": emotion_models})

async def _handle_chat(data: UserInput, db: Session):
    llm_response = await llm_service.generate_response(data.message, data.model)

    if llm_response.get("status") != "success":
        return {
            "status": "error",
            "message": llm_response.get("message", "Unknown error")
        }

    selected_emotion_model_key = _resolve_emotion_model_key(data.emotion_model)
    ml_result, warning = _predict_emotion_with_fallback(data.message, selected_emotion_model_key)

    llm_emotion = llm_response.get("emotion", "Bình thường")
    ml_emotion = ml_result.get("emotion", "Không xác định")
    ml_detail_emotion = ml_result.get("detail_emotion", ml_emotion)

    emotion_similarity = None
    try:
        embedder = ml_emotion_service._get_embedder()
        emotion_similarity = await llm_service.calculate_cosine_similarity_between_two_labels(
            llm_emotion,
            ml_detail_emotion,
            embedder,
        )
    except Exception:
        emotion_similarity = None

    history = history_service.save_chat(
        db=db,
        user_msg=data.message,
        ai_res=llm_response.get("advice", ""),
        emotion=llm_emotion,
        ml_detail_emotion=ml_detail_emotion,
        conversation_id=data.conversation_id,
        emotion_model_used=EMOTION_MODELS.get(selected_emotion_model_key, {}).get("name", "Machine Learning"),
    )

    conversation, assistant_message = history

    return {
        "status": "success",
        "data": {
            "emotion": llm_emotion,
            "llm_emotion": llm_emotion,
            "ml_emotion": ml_emotion,
            "ml_detail_emotion": ml_detail_emotion,
            "advice": llm_response.get("advice", ""),
            "show_emotion": llm_response.get("show_emotion", True),
            "reliability_score": llm_response.get("reliability_score", 1.0),
            "emotion_similarity": emotion_similarity,
            "model_used": llm_response.get("model_used"),
            "emotion_model_used": EMOTION_MODELS.get(selected_emotion_model_key, {}).get("name", "Machine Learning"),
            "emotion_model_warning": warning,
            "history_id": conversation.id,
            "message_id": assistant_message.id,
        }
    }


@router.post("/chat")
async def chat_endpoint(data: UserInput, db: Session = Depends(get_db)):
    return await _handle_chat(data, db)


@router.post("/consult-api")
async def consult_api(data: UserInput, db: Session = Depends(get_db)):
    return await _handle_chat(data, db)


@router.post("/consult-api/stream")
async def consult_api_stream(data: UserInput, db: Session = Depends(get_db)):
    selected_emotion_model_key = _resolve_emotion_model_key(data.emotion_model)
    ml_result, warning = _predict_emotion_with_fallback(data.message, selected_emotion_model_key)

    ml_emotion = ml_result.get("emotion", "Không xác định")
    ml_detail_emotion = ml_result.get("detail_emotion", ml_emotion)

    current_user_id = getattr(data, 'user_id', 'hoang_dev_user') 
    session_conv_id = str(data.conversation_id) if data.conversation_id else f"new_chat_{uuid.uuid4().hex[:8]}"
    
    async def event_generator():
        final_event = None
        
        async for event in llm_service.generate_response_stream(
            message = data.message,
            user_id = current_user_id,
            conversation_id = session_conv_id,
            model_name = data.model):
            event_type = event.get("type")

            if event_type == "chunk":
                yield _build_sse_payload({
                    "type": "chunk",
                    "content": event.get("content", ""),
                })
                continue

            if event_type == "error":
                yield _build_sse_payload({
                    "type": "error",
                    "message": event.get("message", "Unknown error"),
                })
                return

            if event_type == "final":
                final_event = event
                break

        if not final_event:
            yield _build_sse_payload({
                "type": "error",
                "message": "Không nhận được phản hồi cuối từ LLM.",
            })
            return

        llm_emotion = final_event.get("emotion", "Bình thường")
        advice = final_event.get("advice", "")

        emotion_similarity = None
        try:
            embedder = ml_emotion_service._get_embedder()
            emotion_similarity = await llm_service.calculate_cosine_similarity_between_two_labels(
                llm_emotion,
                ml_detail_emotion,
                embedder,
            )
        except Exception:
            emotion_similarity = None

        history = history_service.save_chat(
            db=db,
            user_msg=data.message,
            ai_res=advice,
            emotion=llm_emotion,
            ml_detail_emotion=ml_detail_emotion,
            conversation_id=data.conversation_id,
            emotion_model_used=EMOTION_MODELS.get(selected_emotion_model_key, {}).get("name", "Machine Learning"),
        )

        conversation, assistant_message = history

        yield _build_sse_payload({
            "type": "done",
            "data": {
                "emotion": llm_emotion,
                "llm_emotion": llm_emotion,
                "ml_emotion": ml_emotion,
                "ml_detail_emotion": ml_detail_emotion,
                "advice": advice,
                "show_emotion": final_event.get("show_emotion", True),
                "reliability_score": final_event.get("reliability_score", 1.0),
                "emotion_similarity": emotion_similarity,
                "model_used": final_event.get("model_used"),
                "emotion_model_used": EMOTION_MODELS.get(selected_emotion_model_key, {}).get("name", "Machine Learning"),
                "emotion_model_warning": warning,
                "history_id": conversation.id,
                "message_id": assistant_message.id,
            },
        })

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@router.post("/api/emotion/predict")
async def predict_emotion(data: EmotionPredictRequest):
    selected_emotion_model_key = _resolve_emotion_model_key(data.emotion_model)
    ml_result, warning = _predict_emotion_with_fallback(data.message, selected_emotion_model_key)
    if warning:
        ml_result = {**ml_result, "warning": warning}
    return ml_result


@router.post("/api/voice/transcribe")
async def transcribe_voice(file: UploadFile = File(...)):
    content_type = (file.content_type or "").lower()
    if not content_type.startswith("audio/"):
        return {"status": "error", "message": "File upload phải là audio."}

    extension = Path(file.filename or "voice.webm").suffix or ".webm"
    raw_data = await file.read()

    if not raw_data:
        return {"status": "error", "message": "Không nhận được dữ liệu audio."}

    try:
        result = await run_in_threadpool(whisper_service.transcribe_audio, raw_data)
        return result
    except OSError as exc:
        if getattr(exc, "errno", None) == 28:
            raise HTTPException(
                status_code=507,
                detail="Không đủ dung lượng để xử lý audio trong bộ nhớ hoặc tạm.",
            ) from exc
        raise

@router.get("/api/history")
async def list_history(db: Session = Depends(get_db)):
    """Lấy danh sách tất cả các cuộc trò chuyện."""
    conversations = history_service.get_all_history(db)
    return [{"id": h.id, "title": h.title, "time": h.created_at} for h in conversations]

@router.get("/api/history/{chat_id}")
async def get_chat_detail(chat_id: int, db: Session = Depends(get_db)):
    """Lấy chi tiết một cuộc trò chuyện với tất cả các tin nhắn."""
    conversation = history_service.get_chat_detail(db, chat_id)
    if not conversation:
        return {"status": "error", "message": "Không tìm thấy hội thoại"}
    
    # Phát hiện records legacy (lưu nhầm ML coarse vào cột emotion)
    legacy_ml_coarse_labels = {
        "negative_sad", "negative_anger", "negative_fear", 
        "negative_anxiety", "positive", "neutral"
    }

    messages = []
    cosine_similarities = []
    
    for msg in conversation.messages:
        raw_emotion = (msg.emotion or "").strip()
        raw_emotion_key = raw_emotion.lower()
        is_legacy = raw_emotion_key in legacy_ml_coarse_labels and bool(msg.ml_detail_emotion)
        
        # Với legacy: không có LLM emotion hợp lệ, chỉ có ML
        llm_emoji = None if is_legacy else raw_emotion
        
        messages.append({
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "emotion": raw_emotion or None,
            "llm_emotion": llm_emoji,
            "ml_emotion": msg.ml_detail_emotion,
            "ml_detail_emotion": msg.ml_detail_emotion,
            "timestamp": msg.timestamp
        })
        
        # Chỉ tính similarity khi có LLM emotion hợp lệ (không legacy)
        similarity_value = None
        if msg.role == "assistant" and not is_legacy and raw_emotion and msg.ml_detail_emotion:
            try:
                similarity_value = await llm_service.calculate_cosine_similarity_between_two_labels(
                    raw_emotion,
                    msg.ml_detail_emotion,
                    ml_emotion_service._get_embedder(),
                )
            except Exception:
                pass
        
        cosine_similarities.append({
            "message_id": msg.id,
            "emotion_similarity": similarity_value,
        })
    
    return {
        "status": "success",
        "id": conversation.id,
        "title": conversation.title,
        "messages": messages,
        "cosine_similarities": cosine_similarities,
    }

# Delete action
@router.delete("/api/history/{chat_id}")
async def delete_chat(chat_id: int, db: Session = Depends(get_db)):
    """Xóa một cuộc trò chuyện."""
    conversation = history_service.get_chat_detail(db, chat_id)
    if not conversation:
        return {"status": "error", "message": "Không tìm thấy hội thoại"}
    db.delete(conversation)
    db.commit()
    return {"status": "success", "message": "Đã xóa hội thoại"}

@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    
    try:
        while True:
            # 1. Nhận data từ Frontend gửi lên qua Socket
            data_str = await websocket.receive_text()
            payload = json.loads(data_str)
            
            message = payload.get("message", "")
            if not message:
                continue
                
            model = payload.get("model")
            emotion_model = payload.get("emotion_model")
            conversation_id = payload.get("conversation_id")
            
            # Lấy định danh người dùng (Memori)
            current_user_id = payload.get("user_id", "hoang_dev_user")
            session_conv_id = str(conversation_id) if conversation_id else f"ws_chat_{uuid.uuid4().hex[:8]}"

            # 2. Xử lý ML Emotion song song (Fallback)
            selected_emotion_model_key = _resolve_emotion_model_key(emotion_model)
            ml_result, warning = _predict_emotion_with_fallback(message, selected_emotion_model_key)

            ml_emotion = ml_result.get("emotion", "Không xác định")
            ml_detail_emotion = ml_result.get("detail_emotion", ml_emotion)

            # 3. Gọi LLM Stream
            final_event = None
            async for event in llm_service.generate_response_stream(
                message=message,
                user_id=current_user_id,
                conversation_id=session_conv_id,
                model_name=model
            ):
                event_type = event.get("type")

                if event_type == "chunk":
                    await websocket.send_json({
                        "type": "chunk",
                        "content": event.get("content", ""),
                    })
                    continue

                if event_type == "error":
                    await websocket.send_json({
                        "type": "error",
                        "message": event.get("message", "Unknown error"),
                    })
                    break # Dừng stream nếu lỗi

                if event_type == "final":
                    final_event = event
                    break

            # 4. Xử lý sau khi stream xong (Lưu DB & Tính toán thêm)
            if final_event:
                llm_emotion = final_event.get("emotion", "Bình thường")
                advice = final_event.get("advice", "")

                emotion_similarity = None
                try:
                    embedder = ml_emotion_service._get_embedder()
                    emotion_similarity = await llm_service.calculate_cosine_similarity_between_two_labels(
                        llm_emotion,
                        ml_detail_emotion,
                        embedder,
                    )
                except Exception:
                    emotion_similarity = None

                # Lưu Database
                history = history_service.save_chat(
                    db=db,
                    user_msg=message,
                    ai_res=advice,
                    emotion=llm_emotion,
                    ml_detail_emotion=ml_detail_emotion,
                    conversation_id=conversation_id,
                    emotion_model_used=EMOTION_MODELS.get(selected_emotion_model_key, {}).get("name", "Machine Learning"),
                )

                conversation, assistant_message = history

                # Gửi cục data cuối cùng chốt hạ
                await websocket.send_json({
                    "type": "done",
                    "data": {
                        "emotion": llm_emotion,
                        "llm_emotion": llm_emotion,
                        "ml_emotion": ml_emotion,
                        "ml_detail_emotion": ml_detail_emotion,
                        "advice": advice,
                        "show_emotion": final_event.get("show_emotion", True),
                        "reliability_score": final_event.get("reliability_score", 1.0),
                        "emotion_similarity": emotion_similarity,
                        "model_used": final_event.get("model_used"),
                        "emotion_model_used": EMOTION_MODELS.get(selected_emotion_model_key, {}).get("name", "Machine Learning"),
                        "emotion_model_warning": warning,
                        "history_id": conversation.id,
                        "message_id": assistant_message.id,
                    },
                })
            elif not final_event:
                await websocket.send_json({
                    "type": "error",
                    "message": "Không nhận được phản hồi cuối từ LLM.",
                })

    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Server error: {str(e)}"
        })