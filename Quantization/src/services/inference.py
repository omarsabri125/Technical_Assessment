import time
from typing import Any

import torch

from src.config import MAX_NEW_TOKENS
from src.prompts import SYSTEM_PROMPT
from src.utils.runtime import cuda_sync, get_input_device, gib


def build_inputs(
    tokenizer: Any,
    prompt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Apply the chat template and move tensors to the model device."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return tokenizer(
        rendered,
        return_tensors="pt",
    ).to(device)


@torch.inference_mode()
def generate_once(
    model: Any,
    tokenizer: Any,
    prompt: str,
) -> dict[str, Any]:
    """Generate one answer and measure inference time and VRAM."""
    device = get_input_device(model)
    inputs = build_inputs(tokenizer, prompt, device)
    input_tokens = int(inputs["input_ids"].shape[-1])

    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()

    cuda_sync()
    started = time.perf_counter()

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    cuda_sync()
    elapsed = time.perf_counter() - started

    generated_ids = output_ids[0, input_tokens:]
    generated_tokens = int(generated_ids.numel())

    output_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    peak_total = torch.cuda.max_memory_allocated()
    allocated_after = torch.cuda.memory_allocated()

    del inputs, output_ids, generated_ids

    return {
        "output": output_text,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": round(elapsed, 6),
        "tokens_per_second": (
            round(generated_tokens / elapsed, 4)
            if elapsed
            else None
        ),
        "vram_allocated_before_gib": gib(allocated_before),
        "vram_allocated_after_gib": gib(allocated_after),
        "peak_vram_total_gib": gib(peak_total),
        "peak_vram_increment_gib": gib(
            max(0, peak_total - allocated_before)
        ),
    }
