import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict


class WhisperService:
    def __init__(self, model_dir: str | None = None, language: str = "vi"):
        # Lấy đường dẫn thư mục gốc của dự án (capstone-project-SP26AI20-final)
        root_dir = Path(__file__).resolve().parents[2]

        # Ưu tiên model_dir truyền vào, sau đó env, sau cùng tự dò một số vị trí local phổ biến.
        env_model_dir = os.getenv("WHISPER_LOCAL_MODEL_DIR", "").strip()
        default_candidates = [
            Path(env_model_dir) if env_model_dir else None,
            root_dir / "whisper_vi_final_model",
            root_dir.parent / "whisper_vi_final_model",
            Path("C:\\Users\\User\\Downloads\\whisper_vi_final_model"),
            Path("C:\\Users\\User\\Downloads\\whisper_vi_final_model-20260418T144836Z-3-001\\whisper_vi_final_model"),
        ]
        default_local_dir = next(
            (candidate for candidate in default_candidates if candidate and candidate.exists()),
            Path("C:\\Users\\User\\Downloads\\whisper_vi_final_model"),
        )

        self.model_dir = Path(model_dir) if model_dir else default_local_dir
        self.language = language

        self._pipe = None
        self._load_lock = Lock()

        # Kiểm tra ngay khi khởi động để đảm bảo đường dẫn chuẩn xác
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục chứa weight finetune local tại: {self.model_dir}. "
                f"Vui lòng kiểm tra lại vị trí đặt thư mục."
            )

    def _build_pipeline(self):
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except Exception as exc:
            raise RuntimeError(
                "Thiếu thư viện cho Whisper. Hãy cài torch và transformers."
            ) from exc

        use_cuda = torch.cuda.is_available()
        device = 0 if use_cuda else -1
        dtype = torch.float16 if use_cuda else torch.float32

        print(f"[WhisperService] Đang load weight finetune OFFLINE từ thư mục: {self.model_dir}")

        # Khóa chặt transformers chỉ đọc local_files_only, tuyệt đối không gọi mạng lên HF Hub
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(self.model_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=True,
        )

        processor = AutoProcessor.from_pretrained(
            str(self.model_dir),
            local_files_only=True
        )

        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
            chunk_length_s=30,
            batch_size=8,
            generate_kwargs={
                "task": "transcribe",
                "language": self.language,
            },
        )

    def _get_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        with self._load_lock:
            if self._pipe is None:
                self._pipe = self._build_pipeline()

        return self._pipe

    def transcribe_audio(self, audio_data: bytes) -> Dict[str, Any]:
        try:
            asr = self._get_pipeline()
            result = asr(audio_data)
            text = ""

            if isinstance(result, dict):
                text = str(result.get("text", "")).strip()
            elif isinstance(result, str):
                text = result.strip()

            if not text:
                return {
                    "status": "error",
                    "message": "Whisper không nhận dạng được nội dung giọng nói.",
                }

            return {
                "status": "success",
                "text": text,
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Lỗi Whisper local: {exc}",
            }

    def transcribe_file(self, audio_path: Path) -> Dict[str, Any]:
        try:
            return self.transcribe_audio(audio_path.read_bytes())
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Lỗi Whisper local: {exc}",
            }


whisper_service = WhisperService()