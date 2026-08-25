import json
import logging
import sys
from pathlib import Path
from typing import Generator

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingestion"))
from config import cfg


log = logging.getLogger("rag_api")


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Thin wrapper around the Ollama /api/generate endpoint.

    Instantiate once and reuse across requests — no persistent
    connection is held so there is no state to manage.
    """

    def __init__(self):
        self.base_url    = cfg.OLLAMA_BASE_URL
        self.model       = cfg.OLLAMA_MODEL
        self.max_tokens  = cfg.OLLAMA_MAX_TOKENS
        self.temperature = cfg.OLLAMA_TEMPERATURE
        self.timeout     = cfg.OLLAMA_TIMEOUT
        self._generate_url = f"{self.base_url}/api/generate"
        self._tags_url     = f"{self.base_url}/api/tags"

    def is_available(self) -> bool:
        """
        Return True if the Ollama server is reachable.
        Used by the health check and UI startup.
        """
        try:
            resp = requests.get(self._tags_url, timeout=5)
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def list_models(self) -> list[str]:
        """
        Return names of all models currently pulled in Ollama.
        Used by the Streamlit sidebar model selector.
        """
        try:
            resp   = requests.get(self._tags_url, timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
        except Exception:
            return []

    def generate(self, prompt: str, model: str | None = None) -> str:
        """
        Send a prompt to Ollama and return the complete response as a string.

        Args:
            prompt : the full prompt including system message and context
            model  : override the default model from .env for this call

        Returns:
            The LLM's response as a plain string.

        Raises:
            ConnectionError : if Ollama is not running
            RuntimeError    : if the request fails
        """
        payload = {
            "model"  : model or self.model,
            "prompt" : prompt,
            "stream" : False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            resp = requests.post(
                self._generate_url,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: ollama serve"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout}s. "
                "Try a smaller model or increase OLLAMA_TIMEOUT in .env"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    def stream(self, prompt: str, model: str | None = None) -> Generator[str, None, None]:
        """
        Send a prompt to Ollama and yield tokens as they are generated.

        Used by the Streamlit UI to display the answer word-by-word
        rather than waiting for the full response.

        Args:
            prompt : the full prompt
            model  : override the default model for this call

        Yields:
            Individual token strings as they arrive from Ollama.

        Raises:
            ConnectionError : if Ollama is not running
        """
        payload = {
            "model"  : model or self.model,
            "prompt" : prompt,
            "stream" : True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            with requests.post(
                self._generate_url,
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data  = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: ollama serve"
            ) from exc


# ---------------------------------------------------------------------------
# Smoke test:  python llm/ollama_client.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = OllamaClient()

    print(f"Ollama available : {client.is_available()}")
    print(f"Models pulled    : {client.list_models()}")

    if not client.is_available():
        print("\nOllama is not running. Start it with: ollama serve")
        sys.exit(1)

    print(f"\nTesting generate with model: {client.model}")
    print("─" * 50)

    response = client.generate("In one sentence, what is a RAG chatbot?")
    print(f"Response: {response}")

    print("\nTesting streaming ...")
    print("─" * 50)
    for token in client.stream("List 3 benefits of RAG in one sentence each."):
        print(token, end="", flush=True)
    print()