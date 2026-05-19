from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


def resolve_model_source(
	*,
	repo_id: str | None = None,
	local_path: str | Path | None = None,
) -> Path:
	if local_path is not None:
		candidate = Path(local_path).expanduser()
		if candidate.exists():
			return candidate

	if repo_id:
		cache_dir = os.getenv("HF_HOME") or os.getenv("HF_HUB_CACHE")
		token = os.getenv("HF_TOKEN")
		try:
			return Path(snapshot_download(repo_id=repo_id, cache_dir=cache_dir, token=token))
		except Exception as exc:
			raise RuntimeError(
				"Không tải được model từ Hugging Face Hub. "
				f"repo_id={repo_id}. Nếu repo gated/private, hãy đặt biến môi trường HF_TOKEN "
				"(token có quyền đọc model) hoặc cấu hình local_path tới model đã tải sẵn."
			) from exc

	if local_path is not None:
		candidate = Path(local_path).expanduser()
		if candidate.exists():
			return candidate

	raise FileNotFoundError("Không tìm thấy model local và cũng chưa cấu hình Hugging Face repo id.")