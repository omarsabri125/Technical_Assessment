import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import set_seed

from src.config import MAX_NEW_TOKENS, MODEL_ID, SEED
from src.evaluation.scorer import score_output
from src.models.loader import load_model
from src.prompts import PROMPTS
from src.services.inference import build_inputs, generate_once
from src.utils.runtime import (
    clean_memory,
    cuda_sync,
    get_input_device,
    gib,
    hardware_info,
    process_rss_bytes,
)


def warm_up_model(model: Any, tokenizer: Any) -> None:
    """Run one small generation before benchmark timing."""
    warmup_inputs = build_inputs(
        tokenizer=tokenizer,
        prompt="Reply with one word: ready",
        device=get_input_device(model),
    )

    with torch.inference_mode():
        output = model.generate(
            **warmup_inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    cuda_sync()
    del warmup_inputs, output
    gc.collect()


def build_result(
    *,
    mode: str,
    precision: dict[str, Any],
    load_seconds: float,
    model_footprint: int,
    ram_before_load: int,
    ram_after_load: int,
    ram_after_warmup: int,
    vram_before_load: int,
    vram_after_load: int,
    vram_after_warmup: int,
    peak_vram_during_load: int,
    prompt_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the final JSON-serializable result."""
    total_tokens = sum(
        item["generated_tokens"]
        for item in prompt_results
    )
    total_seconds = sum(
        item["elapsed_seconds"]
        for item in prompt_results
    )

    weighted_tps = (
        total_tokens / total_seconds
        if total_seconds
        else 0.0
    )

    valid_tps = [
        item["tokens_per_second"]
        for item in prompt_results
        if item["tokens_per_second"] is not None
    ]
    mean_prompt_tps = sum(valid_tps) / len(valid_tps)

    mean_quality = sum(
        item["automatic_compliance_score"]
        for item in prompt_results
    ) / len(prompt_results)

    peak_inference = max(
        item["peak_vram_total_gib"]
        for item in prompt_results
    )

    return {
        "model_id": MODEL_ID,
        "mode": mode,
        "precision": precision,
        "methodology": {
            "fixed_prompt_count": len(PROMPTS),
            "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": "greedy / deterministic (do_sample=False)",
            "use_kv_cache": True,
            "warmup": True,
            "throughput_scope": (
                "prefill + decoding; tokenization excluded"
            ),
            "seed": SEED,
        },
        "hardware": hardware_info(),
        "memory": {
            "model_footprint_gib": gib(model_footprint),
            "process_ram_before_load_gib": gib(ram_before_load),
            "process_ram_after_load_gib": gib(ram_after_load),
            "process_ram_after_warmup_gib": gib(ram_after_warmup),
            "process_ram_load_delta_gib": gib(
                max(0, ram_after_load - ram_before_load)
            ),
            "vram_before_load_gib": gib(vram_before_load),
            "vram_after_load_gib": gib(vram_after_load),
            "vram_after_warmup_gib": gib(vram_after_warmup),
            "peak_vram_during_load_gib": gib(
                peak_vram_during_load
            ),
            "peak_vram_during_inference_gib": peak_inference,
        },
        "performance": {
            "load_seconds": round(load_seconds, 4),
            "total_generated_tokens": total_tokens,
            "total_generation_seconds": round(total_seconds, 6),
            "weighted_tokens_per_second": round(weighted_tps, 4),
            "mean_prompt_tokens_per_second": round(
                mean_prompt_tps,
                4,
            ),
        },
        "quality": {
            "mean_automatic_compliance_score": round(
                mean_quality,
                2,
            ),
            "warning": (
                "Automatic checks measure format/task compliance only. "
                "Read the five outputs side by side for qualitative judgment."
            ),
        },
        "prompts": prompt_results,
    }


def run_benchmark(mode: str, output_dir: Path) -> Path:
    """Run one model variant and save its JSON result."""
    set_seed(SEED)
    clean_memory()

    output_dir.mkdir(parents=True, exist_ok=True)

    ram_before_load = process_rss_bytes()
    vram_before_load = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    model, tokenizer, precision = load_model(mode)
    cuda_sync()
    load_seconds = time.perf_counter() - load_started

    ram_after_load = process_rss_bytes()
    vram_after_load = torch.cuda.memory_allocated()
    peak_vram_during_load = torch.cuda.max_memory_allocated()
    model_footprint = int(model.get_memory_footprint())

    warm_up_model(model, tokenizer)

    vram_after_warmup = torch.cuda.memory_allocated()
    ram_after_warmup = process_rss_bytes()

    prompt_results: list[dict[str, Any]] = []

    for item in PROMPTS:
        generation = generate_once(
            model=model,
            tokenizer=tokenizer,
            prompt=item["prompt"],
        )
        quality = score_output(
            prompt_id=item["id"],
            text=generation["output"],
        )

        prompt_results.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                **generation,
                **quality,
            }
        )

        print(
            f"[{mode}] {item['id']}: "
            f"{generation['tokens_per_second']} tok/s, "
            "quality proxy="
            f"{quality['automatic_compliance_score']}"
        )

    result = build_result(
        mode=mode,
        precision=precision,
        load_seconds=load_seconds,
        model_footprint=model_footprint,
        ram_before_load=ram_before_load,
        ram_after_load=ram_after_load,
        ram_after_warmup=ram_after_warmup,
        vram_before_load=vram_before_load,
        vram_after_load=vram_after_load,
        vram_after_warmup=vram_after_warmup,
        peak_vram_during_load=peak_vram_during_load,
        prompt_results=prompt_results,
    )

    result_path = output_dir / f"{mode}_results.json"
    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return result_path
