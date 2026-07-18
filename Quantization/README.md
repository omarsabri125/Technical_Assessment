# Quantization

This folder contains a small PyTorch benchmarking project for comparing two variants of the Qwen 2.5 1.5B instruct model:

- a non-quantized BFLOAT16 variant
- a 4-bit NF4 quantized variant

The goal is to measure the trade-off between memory usage, VRAM footprint, throughput, and output quality on a CUDA GPU.

## Project overview

The project evaluates the same five prompts against both model variants using deterministic greedy decoding. It records:

- model footprint and VRAM usage
- process RAM usage
- generation throughput in tokens per second
- automatic compliance scores for output quality

The benchmarking workflow is implemented in the source folder and the generated results are stored in the results folder.

## Folder structure

- src/main.py - command-line entry point for running the benchmark
- src/config.py - model ID, generation settings, and prompt configuration
- src/prompts.py - the five evaluation prompts
- src/services/ - benchmarking, inference, and model loading logic
- results/ - JSON benchmark outputs and the final report

## Installation

This project requires Python 3.10+ and a CUDA-capable GPU. A Tesla T4 or similar GPU is suitable for running the experiments.

On Windows, you can set up the environment as follows:

```powershell
cd d:\Technical_Assessment\Quantization
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If you are using Google Colab, choose a T4 GPU runtime before running the benchmark.

## Running the benchmark

Run the full-precision benchmark:

```powershell
python -m src.main --mode full --output-dir results
```

Run the 4-bit quantized benchmark:

```powershell
python -m src.main --mode quantized --output-dir results
```

The scripts will save JSON results in the results folder and print the generated metrics to the console.

## Benchmark report

# Qwen2.5-1.5B: Non-quantized vs 4-bit NF4

## Experimental setup

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- GPU: Tesla T4 (14.5632 GiB)
- PyTorch/CUDA: 2.11.0+cu128 / 12.8
- Same five prompts and deterministic greedy decoding for both variants.
- Throughput includes prefill and decoding but excludes tokenization.
- A warm-up generation is performed before timing.

## Short trade-off table

| Variant | Weight precision | Model footprint (GiB) | Peak VRAM (GiB) | Process RAM (GiB) | Tokens/s | Quality proxy /100 |
|---|---:|---:|---:|---:|---:|---:|
| Non-quantized BFLOAT16 | BFLOAT16 | 2.875 | 2.896 | 1.837 | 25.92 | 85.0 |
| 4-bit NF4 / BFLOAT16 compute | 4-bit NF4 | 1.045 | 1.118 | 1.560 | 14.07 | 80.0 |

## Measured trade-off

- 4-bit reduced the model footprint by **63.7%**.
- 4-bit reduced peak inference VRAM by **61.4%**.
- 4-bit was **45.7% slower** on this GPU and software stack.
- Its automatic compliance score was **5.0 points lower**.

The speed result is hardware-dependent: bitsandbytes must dequantize 4-bit weights during computation, so memory savings do not guarantee higher tokens/sec.

## Five-prompt qualitative comparison

> The automatic score checks correctness and formatting constraints. Human review should also assess clarity, reasoning, relevance, and hidden factual errors.

### math_reasoning

**Prompt:** A warehouse has 240 boxes. It ships 35% of them, then receives 18 new boxes. Show the calculation briefly and finish with exactly 'FINAL: <number>'.

**Non-quantized output** — 100.0/100, 24.63 tok/s

````text
The warehouse starts with 240 boxes. Shipping 35% of them means shipping \( 240 \times 0.35 = 84 \) boxes.

After shipping, it has \( 240 - 84 = 156 \) boxes left.

Then, receiving 18 new boxes adds to this number, making it \( 156 + 18 = 174 \).

FINAL: 174
````

**4-bit NF4 output** — 100.0/100, 13.86 tok/s

````text
The number of boxes shipped is calculated as follows:
\[ \text{Number of boxes shipped} = 240 \times 0.35 = 84 \]

After shipping, the remaining boxes in the warehouse are:
\[ \text{Remaining boxes} = 240 - 84 = 156 \]

Then, after receiving 18 new boxes, the final count is:
\[ \text{Final count} = 156 + 18 = 174 \]

FINAL: 174.
````

### json_extraction

**Prompt:** Extract the following text into valid JSON only, with exactly these keys: name, city, product, price_usd, purchase_date.
Text: Nora lives in Cairo. She bought a laptop for 1250 US dollars on 12 March 2025.

**Non-quantized output** — 100.0/100, 27.39 tok/s

````text
```json
{
  "name": "Nora",
  "city": "Cairo",
  "product": "laptop",
  "price_usd": 1250,
  "purchase_date": "12 March 2025"
}
```
````

**4-bit NF4 output** — 100.0/100, 15.02 tok/s

````text
```json
[
    {
        "name": "Nora",
        "city": "Cairo",
        "product": "laptop",
        "price_usd": 1250,
        "purchase_date": "12 March 2025"
    }
]
```
````

### coding

**Prompt:** Write only a Python function named chunk_list(items, size). It must return consecutive chunks as lists, use no imports, and raise ValueError when size <= 0.

**Non-quantized output** — 100.0/100, 27.57 tok/s

````text
```python
def chunk_list(items, size):
    if size <= 0:
        raise ValueError("Size must be greater than 0")
    
    return [items[i:i+size] for i in range(0, len(items), size)]
```
````

**4-bit NF4 output** — 100.0/100, 13.63 tok/s

````text
```python
def chunk_list(items, size):
    if size <= 0:
        raise ValueError("Size must be greater than zero")
    
    return [items[i:i + size] for i in range(0, len(items), size)]

# Example usage:
chunks = chunk_list([1, 2, 3, 4, 5], 2)
print(chunks)  # Output: [[1, 2], [3, 4], [5]]
```
````

### rag_explanation

**Prompt:** Explain retrieval-augmented generation (RAG) to a 12-year-old in at most 80 words. Include one simple analogy and explain both the retrieval step and the answer-generation step.

**Non-quantized output** — 75.0/100, 25.34 tok/s

```text
Retrieval-Augmented Generation (RAG) is like using a magic book of answers to help you write stories or solve puzzles. Imagine you're trying to find a specific page in your storybook, but instead of just looking for it, you also have a magical helper that reads parts of other books nearby. This helps you quickly find what you need without having to search everywhere. In RAG, the "magic book" is a large database of text, and the "helper" uses this to find relevant pieces of information when generating new content. The first part, where it finds these pieces, is called the Retrieval Step. The second part, where it combines them into something new, is Answer-Generation.
```

**4-bit NF4 output** — 75.0/100, 15.22 tok/s

```text
Retrieval-Augmented Generation (RAG) is like using a magical book of answers to help you write stories or solve problems. Imagine you're trying to find a specific fact about dinosaurs, but instead of searching through books yourself, you ask your magical book for help. The book then quickly finds the right dinosaur facts from its vast library and gives them back to you. Then, you use those facts to create or solve something related, just like how you would use the found facts to build a story or figure out an equation. It's like having a super-smart friend who knows all the answers!
```

### quantization_summary

**Prompt:** Summarize the paragraph below in exactly 3 bullet points, each under 18 words. Preserve the ideas of memory, speed, and quality.

Quantization stores model weights with fewer bits. This usually reduces memory use and can make deployment possible on smaller devices. Speed may improve, remain similar, or even decrease depending on hardware and kernel overhead. Very aggressive quantization can slightly reduce output quality, especially on difficult reasoning tasks.

**Non-quantized output** — 50.0/100, 27.24 tok/s

```text
- Quantization uses fewer bits to store model weights.
- Reduces memory usage for efficient deployment.
- May increase/decrease speed based on hardware and kernel overhead.
- Aggressive quantization can slightly degrade output quality on complex tasks.
```

**4-bit NF4 output** — 25.0/100, 12.08 tok/s

```text
- Quantization reduces model size by using fewer bits.
- It improves deployment efficiency on small devices.
- Speed might increase but could also decrease based on hardware and kernel overhead.
- Aggressive quantization can slightly lower output quality for complex tasks.
```

## Production deployment guidance

GPTQ, AWQ, and bitsandbytes all reduce model memory requirements, but they differ significantly in how quantization is performed and how suitable they are for a production-serving workflow.

GPTQ is an offline post-training, weight-only quantization method. It requires a representative calibration dataset and processes the model layer by layer to reduce the reconstruction error introduced by low-bit weights. This produces a pre-quantized checkpoint that can be stored and loaded repeatedly by an inference server. GPTQ is therefore suitable when the model is deployed on a dedicated GPU server, the quantization step can be performed in advance, and the serving engine provides optimized kernels for the selected GPTQ format. However, the final quality and performance depend on the calibration dataset, bit width, group size, model architecture, serving runtime, and target GPU.

AWQ is also an offline weight-only quantization method that uses a calibration dataset. Unlike GPTQ, AWQ analyzes activation statistics to identify the weight channels that are most important to model output. It then applies scaling transformations that reduce quantization error for these salient channels. AWQ can be a strong deployment option when the serving stack has efficient AWQ kernels and when preserving important activation-sensitive channels is valuable. It does not require model retraining or backpropagation, but its accuracy and runtime performance should still be evaluated on representative application data.

Bitsandbytes provides a simpler on-the-fly quantization path. A model can be loaded directly in 8-bit or 4-bit NF4/FP4 without first creating a calibration dataset or running a separate quantization process. This makes bitsandbytes especially useful for experimentation, memory-constrained notebooks, rapid prototyping, and QLoRA-based fine-tuning. Its main benefit is ease of use and reduced memory consumption; however, a reduction in memory does not guarantee improved inference throughput because the runtime may need to dequantize weights during computation and performance depends on the available kernels and hardware.

Therefore, the production decision should not be based only on the quantization method’s name. GPTQ or AWQ may be preferable when a reusable pre-quantized checkpoint and an optimized CUDA-serving path are required. Bitsandbytes may be preferable when simplicity, rapid loading, or quantized fine-tuning is the priority. The final choice should be based on controlled measurements of VRAM usage, throughput, latency, model-loading time, and task-specific quality on the actual deployment hardware.

## Conclusion

Choose 4-bit NF4 when fitting the model into limited VRAM is the main constraint. Choose the non-quantized variant when maximum numerical fidelity is more important and enough memory is available. For a final deployment decision, repeat the test on the target hardware and add domain-specific prompts.

## Limitations

- Five prompts are useful for a small assignment but are not a comprehensive benchmark.
- Greedy decoding improves repeatability but does not represent every production setting.
- The quality proxy is not a full semantic evaluator.
- Results should not be compared across different GPUs as though they were controlled.
