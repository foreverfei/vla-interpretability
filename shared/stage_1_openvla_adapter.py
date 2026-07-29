"""Adapter for OpenVLA using HuggingFace Transformers (official way)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def create_adapter(**kwargs: Any) -> "OpenVLARepresentationAdapter":
    return OpenVLARepresentationAdapter(**kwargs)


class OpenVLARepresentationAdapter:
    def __init__(
        self,
        *,
        model_path: str | None = None,
        data_root: str | None = None,
        device: str = "cuda",
        **_: Any,
    ) -> None:
        self.model_path = model_path
        self.data_root = data_root
        self.device = device
        self.model = None
        self.processor = None

    def load(self) -> None:
        if not self.model_path:
            raise ValueError("OpenVLA adapter requires model_path")

        print(f"Loading OpenVLA from {self.model_path}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("Model loaded successfully.")

    def extract_hidden_states(self, row: dict[str, Any]) -> dict[int, np.ndarray]:
        image_path = Path(str(row["image_path"]))
        instruction = str(row.get("instruction") or row.get("task_description") or "")
        image = Image.open(image_path).convert("RGB")

        # 正确的 Prompt 格式
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        # 获取模型设备和数据类型
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype

        # 将输入张量移动到模型设备，但仅将 pixel_values 转换为 model_dtype
        # 其他张量（input_ids, attention_mask）保持原始整数类型
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_dict = {}
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                # bfloat16 -> float32 -> numpy (numpy 不支持 bfloat16)
                hidden_dict[layer_idx] = hidden[0].cpu().float().numpy()
        else:
            raise RuntimeError(
                "Model did not return hidden_states. Ensure model supports output_hidden_states."
            )

        return hidden_dict
