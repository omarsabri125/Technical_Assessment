# LiveKit Agents

## Overview

This project is a voice-enabled AI agent built with LiveKit Agents. It connects to a LiveKit server, listens to user speech, uses an LLM to understand the request, and can call a live web-search tool when the user asks for recent or changing information.

The agent is designed for real-time voice conversations and supports:

- speech-to-text (STT)
- large language model (LLM) reasoning
- text-to-speech (TTS)
- live web search via Tavily
- tool-based retrieval for up-to-date answers

In short, this project lets you run a conversational assistant that can answer questions from both the model and the live web.

## What this project does

The main flow is:

1. A client connects to a LiveKit room.
2. The agent session starts with STT, LLM, and TTS providers.
3. The agent receives the user request and can decide to call the web search tool.
4. Search results are returned to the model, which generates a spoken answer.

The current agent is a research-focused assistant named Lina. It is instructed to use the web-search tool for recent news, prices, technology updates, people or company information, and other time-sensitive facts.

## Project structure

- [src/main.py](src/main.py): entry point that starts the LiveKit agent server and registers the session entrypoint
- [src/agents/research_agent.py](src/agents/research_agent.py): the voice agent and its web-search tool definition
- [src/helpers/settings.py](src/helpers/settings.py): environment-based configuration using Pydantic settings
- [src/llms/](src/llms/): factories for LLM, STT, and TTS providers
- [src/tools/web_search_tool.py](src/tools/web_search_tool.py): Tavily-based live web search service

## Requirements

- Python 3.10+
- A LiveKit server or LiveKit Cloud project
- An LLM provider API key
- A Tavily API key for web search
- Optional STT/TTS provider credentials depending on your selected providers

## Installation

1. Open a terminal in the project folder:

   ```bash
   cd LiveKit_Agents
   ```

2. Create and activate a virtual environment:

   On Windows:

   ```bash
   py -3.11 -m venv .venv
   .\.venv\Scripts\activate
   ```

   On macOS/Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Copy the environment template:

   On Windows:

   ```bash
   copy .env.example .env
   ```

   On macOS/Linux:

   ```bash
   cp .env.example .env
   ```

## Configuration

Update your [.env](.env) file with the required values.

### Required variables

- LIVEKIT_URL
- LIVEKIT_API_KEY
- LIVEKIT_API_SECRET
- LLM_PROVIDER
- TAVILY_API_KEY

### Example providers

The project supports:

- LLM providers: OpenRouter or Groq
- STT providers: Deepgram or AssemblyAI
- TTS providers: Cartesia or ElevenLabs

Example configuration values are already provided in [.env.example](.env.example).

## Running the agent

Start the agent server from the project root:

```bash
python -m src.main console
```

This launches the LiveKit agent application and waits for incoming voice sessions.

## Usage

### 1. Start the LiveKit server

You need a working LiveKit environment before running the agent. This can be:

- a local LiveKit server
- a LiveKit Cloud project

Make sure the LiveKit URL, API key, and API secret are set in your environment file.

### 2. Configure the LLM and voice stack

Choose your preferred providers in [.env](.env):

- set LLM_PROVIDER to OPENROUTER or GROQ
- set STT_PROVIDER to DEEPGRAM or ASSEMBLYAI
- set TTS_PROVIDER to CARTESIA or ELEVENLABS

### 3. Connect a client

Once the server is running, connect a LiveKit client or voice app to the room and start speaking. The agent will respond using the configured LLM and TTS provider.

### 4. Ask questions

The agent can answer questions such as:

- What is the latest news about this topic?
- What is the current price of this product?
- What happened recently in this company or market?

When the question is time-sensitive, the agent uses the web-search tool to gather current information.

## How the agent works

- The agent is defined in [src/agents/research_agent.py](src/agents/research_agent.py).
- The web-search tool calls Tavily and returns structured results.
- The agent instructions tell it to use the tool for current information and avoid guessing when no reliable source is found.
- The session factory wires together STT, LLM, and TTS providers so the voice pipeline works end to end.

## Notes

- The project is optimized for voice-first research tasks and real-time question answering.
- Web search is only used when the prompt requires current or changing information.
- If the web search service fails, the agent is instructed to say that live search is unavailable rather than inventing an answer.
- If you run into startup issues, verify that your API keys and LiveKit credentials are correct in [.env](.env).

## Troubleshooting

- If the agent does not start, check that LiveKit credentials are valid.
- If the LLM does not respond, verify the selected provider and API key.
- If web search fails, confirm that the Tavily API key is correctly configured.
- If speech output is missing or poor, verify the STT/TTS provider settings in [.env](.env).

## Barge-In Handling and Safe Tool Extension

The voice agent is designed to support natural barge-in, allowing the user to interrupt the assistant while it is speaking. Interruption handling is enabled in the LiveKit session configuration. When the system detects that the user has started speaking, the current TTS output is stopped so the agent can listen to the new request.

```python
interruption={
    "enabled": True,
    "resume_false_interruption": True,
    "false_interruption_timeout": 1.0,
}
```

The `resume_false_interruption` option protects the conversation from accidental interruptions caused by coughing, background noise, or audio echo. If no valid speech transcript is produced within the configured timeout, the agent treats the interruption as false and resumes its previous response.

The session also uses acoustic echo cancellation warm-up:

```python
aec_warmup_duration=2.0
```

This reduces the chance that the agent hears its own TTS output through the microphone and incorrectly treats it as user speech. For generated greetings and responses, `allow_interruptions=True` is used so the user can interrupt the assistant naturally.

Preemptive generation can also be enabled to reduce response latency. It allows the LLM to begin preparing a response when the system predicts that the user is close to finishing their turn. If the user continues speaking, the incomplete response is discarded and regenerated using the complete input.

To add a second tool safely, I would keep the tool logic separate from the agent. For example, a weather tool would use a dedicated `WeatherService`, while the LiveKit tool method would only validate input, call the service, and return a structured result.

```python
@function_tool()
async def get_current_weather(
    self,
    context: RunContext,
    city: str,
) -> str:
    """Get the current weather for a city.

    Args:
        city: A valid city name, such as Cairo or London.
    """
```

The typed parameter and docstring provide a clear schema that helps the LLM call the tool correctly. The input should be normalized and validated before calling the external API. Empty values, excessively long inputs, and unsupported values should be rejected.

Provider-specific failures such as timeouts, authentication errors, rate limits, or invalid API responses should be caught inside the service layer and converted into a project-specific exception, such as `WeatherServiceError`. The agent then converts that exception into LiveKit’s `ToolError`.

```python
except WeatherServiceError as error:
    raise ToolError(
        "The weather service is temporarily unavailable. "
        "Tell the user that the request could not be completed. "
        "Do not guess or invent weather information."
    ) from error
```

This separation prevents provider-specific errors from leaking into the agent, produces a safe message for the LLM, and ensures that the assistant never fabricates results when a tool call fails. Tool calls, successful results, and failures should also be logged without exposing API keys or other sensitive data.
