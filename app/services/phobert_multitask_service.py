from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaModel

from app.services.hf_hub_service import resolve_model_source


FINE_LABELS = [
	"Buồn bã",
	"Chán ghét",
	"Cô đơn",
	"Highly negative",
	"Hối tiếc",
	"Lo âu",
	"Lạc quan",
	"Ngạc nhiên",
	"Other",
	"Sợ hãi",
	"Trung lập",
	"Tức giận",
	"Vui vẻ",
]

FINE_TO_COARSE = {
	"Buồn bã": "Negative_Sad",
	"Chán ghét": "Negative_Sad",
	"Cô đơn": "Negative_Sad",
	"Highly negative": "Negative_Sad",
	"Hối tiếc": "Negative_Sad",
	"Lo âu": "Negative_Sad",
	"Lạc quan": "Positive",
	"Vui vẻ": "Positive",
	"Ngạc nhiên": "Surprise",
	"Tức giận": "Anger",
	"Sợ hãi": "Fear",
	"Trung lập": "Neutral_Other",
	"Other": "Neutral_Other",
}


class PhoBertMultitaskModel(torch.nn.Module):
	def __init__(self, config: AutoConfig, coarse_labels: int, fine_labels: int):
		super().__init__()
		self.encoder = RobertaModel(config, add_pooling_layer=True)
		self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.classifier_coarse = torch.nn.Linear(config.hidden_size, coarse_labels)
		self.classifier_fine = torch.nn.Linear(config.hidden_size, fine_labels)

	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
		outputs = self.encoder(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
		)

		pooled_output = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
		hidden = self.dropout(torch.tanh(self.dense(pooled_output)))
		coarse_logits = self.classifier_coarse(hidden)
		fine_logits = self.classifier_fine(hidden)
		return coarse_logits, fine_logits


class PhoBertMultitaskService:
	def __init__(self, model_path: Optional[str] = None):
		root_dir = Path(__file__).resolve().parents[2]
		default_local_path = Path(
			os.getenv(
				"PHOBERT_MULTITASK_MODEL_PATH",
				str(root_dir / "checkpoint-3824-20260411T075134Z-3-002" / "checkpoint-3824"),
			)
		)
		default_repo_id = os.getenv("PHOBERT_MULTITASK_MODEL_ID") or None
		self.device = torch.device("cpu")
		local_source = Path(model_path) if model_path else default_local_path
		self.model_path = self._resolve_path(
			resolve_model_source(repo_id=default_repo_id, local_path=local_source)
		)
		self.tokenizer = self._load_tokenizer()
		self.config = self._load_config()
		self.model = self._load_model()

	def _resolve_path(self, path: Path) -> Path:
		weight_files = ["model.safetensors", "pytorch_model.bin"]
		if any((path / filename).exists() for filename in weight_files):
			return path

		candidates = []
		for checkpoint in path.parent.glob("checkpoint-*"):
			if any((checkpoint / filename).exists() for filename in weight_files):
				try:
					step = int(checkpoint.name.split("-")[-1])
				except ValueError:
					step = 0
				candidates.append((step, checkpoint))

		if not candidates:
			raise FileNotFoundError(f"Không tìm thấy model weights tại {path}")

		candidates.sort(key=lambda item: item[0], reverse=True)
		return candidates[0][1]

	def _load_tokenizer(self):
		try:
			return AutoTokenizer.from_pretrained(self.model_path)
		except Exception:
			return AutoTokenizer.from_pretrained("vinai/phobert-large")

	def _load_config(self):
		config_path = self.model_path / "config.json"
		if config_path.exists():
			with open(config_path, "r", encoding="utf-8") as file:
				config_dict = json.load(file)

			raw_id2label = config_dict.get("id2label")
			if isinstance(raw_id2label, list):
				config_dict["id2label"] = {index: str(label) for index, label in enumerate(raw_id2label)}
			elif isinstance(raw_id2label, dict):
				config_dict["id2label"] = {int(key): str(value) for key, value in raw_id2label.items()}
			else:
				num_labels = int(config_dict.get("num_labels", 2))
				config_dict["id2label"] = {index: f"LABEL_{index}" for index in range(num_labels)}

			config_dict["label2id"] = {label: index for index, label in config_dict["id2label"].items()}
			config_dict["num_labels"] = len(config_dict["id2label"])
			model_type = config_dict.pop("model_type", "roberta")
			return AutoConfig.for_model(model_type, **config_dict)

		from safetensors.torch import load_file

		weights = load_file(str(self.model_path / "model.safetensors"))
		hidden_size = weights["encoder.embeddings.word_embeddings.weight"].shape[1]
		vocab_size = weights["encoder.embeddings.word_embeddings.weight"].shape[0]
		max_position_embeddings = weights["encoder.embeddings.position_embeddings.weight"].shape[0]
		num_layers = len(
			{
				int(key.split(".")[3])
				for key in weights.keys()
				if key.startswith("encoder.encoder.layer.")
			}
		)

		return AutoConfig.for_model(
			"roberta",
			vocab_size=vocab_size,
			hidden_size=hidden_size,
			num_hidden_layers=num_layers,
			num_attention_heads=hidden_size // 64,
			intermediate_size=hidden_size * 4,
			hidden_act="gelu",
			hidden_dropout_prob=0.1,
			attention_probs_dropout_prob=0.1,
			max_position_embeddings=max_position_embeddings,
			type_vocab_size=1,
			layer_norm_eps=1e-5,
			pad_token_id=1,
			bos_token_id=0,
			eos_token_id=2,
		)

	def _load_model(self):
		from safetensors.torch import load_file

		weights_path = self.model_path / "model.safetensors"
		if not weights_path.exists():
			weights_path = self.model_path / "pytorch_model.bin"

		if weights_path.suffix == ".safetensors":
			weights = load_file(str(weights_path))
		else:
			weights = torch.load(weights_path, map_location="cpu")

		coarse_labels = weights["classifier_coarse.bias"].shape[0]
		fine_labels = weights["classifier_fine.bias"].shape[0]

		model = PhoBertMultitaskModel(self.config, coarse_labels, fine_labels)
		model.load_state_dict(weights, strict=True)
		model.to(self.device)
		model.eval()
		return model

	def predict(self, text: str) -> Dict[str, object]:
		inputs = self.tokenizer(
			text,
			return_tensors="pt",
			truncation=True,
			max_length=256,
			padding=False,
		)
		inputs = {key: value.to(self.device) for key, value in inputs.items()}

		with torch.no_grad():
			coarse_logits, fine_logits = self.model(**inputs)

		coarse_id = int(coarse_logits.argmax(dim=-1).item())
		fine_id = int(fine_logits.argmax(dim=-1).item())
		fine_label = FINE_LABELS[fine_id] if 0 <= fine_id < len(FINE_LABELS) else f"LABEL_{fine_id}"
		coarse_label = FINE_TO_COARSE.get(fine_label, "Unknown")

		return {
			"status": "success",
			"coarse_id": coarse_id,
			"coarse_emotion": coarse_label,
			"fine_id": fine_id,
			"emotion": coarse_label,
			"detail_emotion": fine_label,
			"fine_emotion": fine_label,
			"model_used": "PhoBERT Multitask",
		}


@lru_cache(maxsize=1)
def get_phobert_multitask_service() -> PhoBertMultitaskService:
	return PhoBertMultitaskService()
