SYSTEM_PROMPT = (
    "You are a careful assistant. Follow the requested format exactly. "
    "Be concise and do not add unrelated information."
)

PROMPTS = [
    {
        "id": "math_reasoning",
        "prompt": (
            "A warehouse has 240 boxes. It ships 35% of them, then receives "
            "18 new boxes. Show the calculation briefly and finish with exactly "
            "'FINAL: <number>'."
        ),
    },
    {
        "id": "json_extraction",
        "prompt": (
            "Extract the following text into valid JSON only, with exactly these "
            "keys: name, city, product, price_usd, purchase_date.\n"
            "Text: Nora lives in Cairo. She bought a laptop for 1250 US dollars "
            "on 12 March 2025."
        ),
    },
    {
        "id": "coding",
        "prompt": (
            "Write only a Python function named chunk_list(items, size). It must "
            "return consecutive chunks as lists, use no imports, and raise "
            "ValueError when size <= 0."
        ),
    },
    {
        "id": "rag_explanation",
        "prompt": (
            "Explain retrieval-augmented generation (RAG) to a 12-year-old in "
            "at most 80 words. Include one simple analogy and explain both the "
            "retrieval step and the answer-generation step."
        ),
    },
    {
        "id": "quantization_summary",
        "prompt": (
            "Summarize the paragraph below in exactly 3 bullet points, each under "
            "18 words. Preserve the ideas of memory, speed, and quality.\n\n"
            "Quantization stores model weights with fewer bits. This usually "
            "reduces memory use and can make deployment possible on smaller "
            "devices. Speed may improve, remain similar, or even decrease "
            "depending on hardware and kernel overhead. Very aggressive "
            "quantization can slightly reduce output quality, especially on "
            "difficult reasoning tasks."
        ),
    },
]
