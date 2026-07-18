import csv
import json
from pathlib import Path


RESULTS_DIR = Path("results")
FULL_PATH = RESULTS_DIR / "full_results.json"
QUANT_PATH = RESULTS_DIR / "quantized_results.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct_change(new: float, old: float) -> float:
    return ((new / old) - 1.0) * 100.0


def esc(text: str) -> str:
    return text.replace("|", r"\|").replace("\n", "<br>")


def main() -> None:
    full = load(FULL_PATH)
    quant = load(QUANT_PATH)

    full_mem = full["memory"]["model_footprint_gib"]
    quant_mem = quant["memory"]["model_footprint_gib"]
    full_peak = full["memory"]["peak_vram_during_inference_gib"]
    quant_peak = quant["memory"]["peak_vram_during_inference_gib"]
    full_tps = full["performance"]["weighted_tokens_per_second"]
    quant_tps = quant["performance"]["weighted_tokens_per_second"]
    full_quality = full["quality"]["mean_automatic_compliance_score"]
    quant_quality = quant["quality"]["mean_automatic_compliance_score"]

    memory_saved = (1 - quant_mem / full_mem) * 100
    peak_saved = (1 - quant_peak / full_peak) * 100
    speed_change = pct_change(quant_tps, full_tps)
    quality_change = quant_quality - full_quality

    speed_word = "faster" if speed_change >= 0 else "slower"
    quality_word = (
        "higher" if quality_change > 0
        else "lower" if quality_change < 0
        else "the same"
    )

    rows = [
        {
            "variant": full["precision"]["label"],
            "weight_precision": full["precision"]["weight_precision"],
            "model_footprint_gib": full_mem,
            "peak_vram_gib": full_peak,
            "process_ram_gib": full["memory"]["process_ram_after_warmup_gib"],
            "tokens_per_second": full_tps,
            "auto_quality_score": full_quality,
        },
        {
            "variant": quant["precision"]["label"],
            "weight_precision": quant["precision"]["weight_precision"],
            "model_footprint_gib": quant_mem,
            "peak_vram_gib": quant_peak,
            "process_ram_gib": quant["memory"]["process_ram_after_warmup_gib"],
            "tokens_per_second": quant_tps,
            "auto_quality_score": quant_quality,
        },
    ]

    with (RESULTS_DIR / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Qwen2.5-1.5B: Non-quantized vs 4-bit NF4",
        "",
        "## Experimental setup",
        "",
        f"- Model: `{full['model_id']}`",
        f"- GPU: {full['hardware']['gpu']} ({full['hardware']['gpu_total_memory_gib']} GiB)",
        f"- PyTorch/CUDA: {full['hardware']['pytorch']} / {full['hardware']['cuda_runtime']}",
        "- Same five prompts and deterministic greedy decoding for both variants.",
        "- Throughput includes prefill and decoding but excludes tokenization.",
        "- A warm-up generation is performed before timing.",
        "",
        "## Short trade-off table",
        "",
        "| Variant | Weight precision | Model footprint (GiB) | Peak VRAM (GiB) | Process RAM (GiB) | Tokens/s | Quality proxy /100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['weight_precision']} | "
            f"{row['model_footprint_gib']:.3f} | {row['peak_vram_gib']:.3f} | "
            f"{row['process_ram_gib']:.3f} | {row['tokens_per_second']:.2f} | "
            f"{row['auto_quality_score']:.1f} |"
        )

    lines += [
        "",
        "## Measured trade-off",
        "",
        f"- 4-bit reduced the model footprint by **{memory_saved:.1f}%**.",
        f"- 4-bit reduced peak inference VRAM by **{peak_saved:.1f}%**.",
        f"- 4-bit was **{abs(speed_change):.1f}% {speed_word}** on this GPU and software stack.",
        f"- Its automatic compliance score was **{abs(quality_change):.1f} points {quality_word}**.",
        "",
        "The speed result is hardware-dependent: bitsandbytes must dequantize 4-bit "
        "weights during computation, so memory savings do not guarantee higher tokens/sec.",
        "",
        "## Five-prompt qualitative comparison",
        "",
        "> The automatic score checks correctness and formatting constraints. "
        "Human review should also assess clarity, reasoning, relevance, and hidden factual errors.",
        "",
    ]

    full_prompts = {item["id"]: item for item in full["prompts"]}
    quant_prompts = {item["id"]: item for item in quant["prompts"]}

    for prompt_id, full_item in full_prompts.items():
        quant_item = quant_prompts[prompt_id]
        lines += [
            f"### {prompt_id}",
            "",
            f"**Prompt:** {full_item['prompt']}",
            "",
            f"**Non-quantized output** — {full_item['automatic_compliance_score']}/100, "
            f"{full_item['tokens_per_second']:.2f} tok/s",
            "",
            "```text",
            full_item["output"],
            "```",
            "",
            f"**4-bit NF4 output** — {quant_item['automatic_compliance_score']}/100, "
            f"{quant_item['tokens_per_second']:.2f} tok/s",
            "",
            "```text",
            quant_item["output"],
            "```",
            "",
        ]

    lines += [
        "## Conclusion",
        "",
        "Choose 4-bit NF4 when fitting the model into limited VRAM is the main constraint. "
        "Choose the non-quantized variant when maximum numerical fidelity is more important "
        "and enough memory is available. For a final deployment decision, repeat the test "
        "on the target hardware and add domain-specific prompts.",
        "",
        "## Limitations",
        "",
        "- Five prompts are useful for a small assignment but are not a comprehensive benchmark.",
        "- Greedy decoding improves repeatability but does not represent every production setting.",
        "- The quality proxy is not a full semantic evaluator.",
        "- Results should not be compared across different GPUs as though they were controlled.",
    ]

    report_path = RESULTS_DIR / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {report_path}")
    print(f"Saved: {RESULTS_DIR / 'comparison.csv'}")


if __name__ == "__main__":
    main()
