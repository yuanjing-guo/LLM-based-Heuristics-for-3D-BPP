# heuristics/llm_based/llm_interface_ollama.py
import json
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def call_llm_ollama(api_url: str, model: str, prompt: str, timeout: int = 300) -> Optional[str]:
    """
    Call a local Ollama server.
    Expected api_url example: http://localhost:11434/api/generate
    Returns: the 'response' field (plain text).
    """
    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")
    req = Request(api_url, data=data, headers=headers, method="POST")
    print("[LLM][Ollama] request sent")

    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            print("[LLM][Ollama] raw response:", raw[:200])
            obj = json.loads(raw)
            return obj.get("response", None)

    except HTTPError as e:
        print(f"[LLM][Ollama] HTTPError: {e.code} {e.reason}")
        return None

    except URLError as e:
        print(f"[LLM][Ollama] URLError: {e.reason}")
        return None

    except json.JSONDecodeError as e:
        print(f"[LLM][Ollama] JSON decode error: {e}")
        return None