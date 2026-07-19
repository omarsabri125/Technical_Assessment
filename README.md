# Technical Assessment Repository

This workspace contains four focused AI/ML projects, each organized in its own folder. The sections below summarize each folder and link to the detailed README inside that folder.

## Contents

- [LiveKit_Agents](LiveKit_Agents/README.md)
- [Model_Deployment](Model_Deployment/README.md)
- [Quantization](Quantization/README.md)
- [RAG_Langchain](RAG_Langchain/README.md)

## Folder summaries

### LiveKit_Agents
A voice-enabled conversational agent built with LiveKit and LLMs. This project connects speech-to-text, a language model, and text-to-speech providers to run a real-time voice assistant that can call a live web-search tool for time-sensitive answers.

Key features:
- Real-time voice session handling using LiveKit
- STT + LLM + TTS integration
- Live web search support via a tool pipeline
- Research-oriented assistant behavior and interruption handling

Learn more: [LiveKit_Agents README](LiveKit_Agents/README.md)

### Model_Deployment
A FastAPI deployment scaffold for streaming model responses through OpenRouter. This folder includes the web service, API route definitions, Docker deployment support, and load testing with Locust.

Key features:
- FastAPI app for serving chat/model responses
- Dockerfile for containerized deployment
- Load testing via `locustfile.py`
- OpenRouter integration for LLM inference

Learn more: [Model_Deployment README](Model_Deployment/README.md)

### Quantization
A benchmarking project comparing Qwen 2.5 1.5B model variants in full precision and 4-bit quantized modes. The experiments measure memory, VRAM, throughput, and output quality trade-offs.

Key features:
- Full-precision vs 4-bit NF4 benchmark
- CUDA GPU performance evaluation
- Results reporting and model footprint analysis
- Reproducible experiment scripts and JSON output

Learn more: [Quantization README](Quantization/README.md)

### RAG_Langchain
A retrieval-augmented generation (RAG) application for financial documents. This project ingests PDFs, converts them to Markdown, builds a hybrid retrieval index, and answers user queries grounded in the indexed content.

Key features:
- PDF ingestion and Markdown conversion
- Image-aware chunking and hybrid retrieval
- Chroma vector store and optional reranking
- CLI for indexing and question answering

Learn more: [RAG_Langchain README](RAG_Langchain/README.md)

## How to use this repository

Each folder is a separate project with its own dependencies and setup instructions. Open the project folder you want to run and follow the README inside that folder.

- `LiveKit_Agents/README.md` for voice agent setup
- `Model_Deployment/README.md` for FastAPI deployment and Docker usage
- `Quantization/README.md` for benchmark execution
- `RAG_Langchain/README.md` for document indexing and RAG queries
