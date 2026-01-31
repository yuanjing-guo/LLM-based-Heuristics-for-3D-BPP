import json
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def _is_ollama_generate(api_url: str) -> bool:
    u = (api_url or "").strip().lower()
    return ("localhost:11434" in u) or u.endswith("/api/generate") or (":11434/" in u)


def call_llm(api_url: str, api_key: str, model: str, prompt: str, timeout: int = 120) -> Optional[str]:
    """
    Supports 2 backends:
      1) Ollama: POST /api/generate  {model, prompt, stream:false}
         returns: {"response": "...", ...}
      2) OpenAI/DeepSeek chat.completions style:
         POST ... {model, messages:[{role:"user",content:prompt}]}
         returns: {"choices":[{"message":{"content":"..."}}]}
    """

    if _is_ollama_generate(api_url):
        headers = {"Content-Type": "application/json"}
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

    # -------------------------
    # DeepSeek / OpenAI style
    # -------------------------
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(api_url, data=data, headers=headers, method="POST")
    print("[LLM][ChatCompletions] request sent")

    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            print("[LLM][ChatCompletions] raw response:", raw[:200])
            obj = json.loads(raw)
            return obj["choices"][0]["message"]["content"]
    except HTTPError as e:
        print(f"[LLM][ChatCompletions] HTTPError: {e.code} {e.reason}")
        return None
    except URLError as e:
        print(f"[LLM][ChatCompletions] URLError: {e.reason}")
        return None
    except json.JSONDecodeError as e:
        print(f"[LLM][ChatCompletions] JSON decode error: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"[LLM][ChatCompletions] Response format error: {e}")
        return None
