from typing import Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.config import MODEL_ID
from src.utils.runtime import dtype_name, select_compute_dtype


def load_model(mode: str) -> tuple[Any, Any, dict[str, Any]]:
    """Load either the full-precision or 4-bit NF4 model."""
    if mode not in {"full", "quantized"}:
        raise ValueError("mode must be 'full' or 'quantized'")

    compute_dtype = select_compute_dtype()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    common_kwargs = {
        "device_map": {"": 0},
        "dtype": compute_dtype,
        "low_cpu_mem_usage": True,
    }

    if mode == "full":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            **common_kwargs,
        )

        precision = {
            "label": f"Non-quantized {dtype_name(compute_dtype)}",
            "weight_precision": dtype_name(compute_dtype),
            "compute_dtype": dtype_name(compute_dtype),
            "quantization": "None",
        }
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            **common_kwargs,
        )

        precision = {
            "label": f"4-bit NF4 / {dtype_name(compute_dtype)} compute",
            "weight_precision": "4-bit NF4",
            "compute_dtype": dtype_name(compute_dtype),
            "quantization": "bitsandbytes NF4 + double quantization",
        }

    model.eval()
    model.config.use_cache = True

    return model, tokenizer, precision
