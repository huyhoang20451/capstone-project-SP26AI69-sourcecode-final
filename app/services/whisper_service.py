import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from app.services.hf_hub_service import resolve_model_source


class WhisperService:
	def __init__(self, model_dir: str | None = None, language: str = "vi"):
		root_dir = Path(__file__).resolve().parents[2]
		default_local_dir = Path(os.getenv("WHISPER_MODEL_DIR", str(root_dir / "whisper_vi_final_model")))

		self._local_source = Path(model_dir) if model_dir else default_local_dir
		self._repo_id = os.getenv("WHISPER_MODEL_ID") or None
		self.model_dir: Path | None = None
		self.language = language

		self._pipe = None
		self._load_lock = Lock()

	def _resolve_model_dir(self, path: Path) -> Path:
		if path.exists():
			return path
		raise FileNotFoundError(f"Không tìm thấy thư mục model Whisper tại: {path}")

	def _get_model_dir(self) -> Path:
		if self.model_dir is None:
			self.model_dir = self._resolve_model_dir(
				resolve_model_source(repo_id=self._repo_id, local_path=self._local_source)
			)

		return self.model_dir

	def _build_pipeline(self):
		try:
			import torch
			from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
		except Exception as exc:
			raise RuntimeError(
				"Thiếu thư viện cho Whisper. Hãy cài torch và transformers."
			) from exc

		model_dir = self._get_model_dir()

		use_cuda = torch.cuda.is_available()
		device = 0 if use_cuda else -1
		dtype = torch.float16 if use_cuda else torch.float32

		model = AutoModelForSpeechSeq2Seq.from_pretrained(
			str(model_dir),
			torch_dtype=dtype,
			low_cpu_mem_usage=True,
			use_safetensors=True,
		)

		processor = AutoProcessor.from_pretrained(str(model_dir))

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

	def transcribe_file(self, audio_path: Path) -> Dict[str, Any]:
		try:
			asr = self._get_pipeline()
			result = asr(str(audio_path))
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
				"message": f"Lỗi Whisper: {exc}",
			}


whisper_service = WhisperService()
