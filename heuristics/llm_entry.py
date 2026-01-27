# heuristics/llm_entry.py
import os
import time
import importlib.util
import numpy as np

from heuristics.base import BaseHeuristic
from heuristics.floor_building import FloorBuilding
from heuristics.llm_based.llm_based_function import generate_heuristic, write_heuristic


class LLMBasedHeuristic(BaseHeuristic):
    name = "llm_based"

    def __init__(self):
        super().__init__()

        # --- API config (use env vars; NEVER hardcode secrets in code) ---
        self.api_url = os.getenv("LLM_API_URL", "https://api.deepseek.com/v1/chat/completions")
        self.api_key = os.getenv("LLM_API_KEY", "")  # must be set in env
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")

        # temporary generated heuristic file
        self.out_path = os.path.join(
            os.path.dirname(__file__), "llm_based", "llm_based_heuristic.py"
        )

        self._last_code = None
        self.current_code = None  # <-- expose for debug save
        self._impl = None
        self._feedback_history = []  # rolling constraints

        self._generate(initial=True)

    def get_current_code(self):
        return self.current_code

    def regenerate(self, feedback: str):
        self._feedback_history.append(feedback)
        self._generate(initial=False, feedback=feedback)

    def __call__(self, obs):
        if self._impl is None:
            return FloorBuilding()(obs)
        return self._impl(obs)

    def _generate(self, initial: bool, feedback: str = None):
        if not self.api_key:
            print("[LLM] Missing LLM_API_KEY (env var). Fallback to FloorBuilding.")
            self._impl = FloorBuilding()
            return

        code = generate_heuristic(
            self.api_url,
            self.api_key,
            self.model,
            self._last_code,
            feedback,
            self._feedback_history,
        )
        if not code:
            print("[LLM] Generation failed, fallback to FloorBuilding.")
            self._impl = FloorBuilding()
            return

        # write to temp file and import it
        write_heuristic(self.out_path, code)
        impl = self._load_generated_class()
        if impl is None:
            print("[LLM] Import failed, fallback to FloorBuilding.")
            self._impl = FloorBuilding()
            return

        # save successful result
        self._last_code = code
        self.current_code = code
        self._impl = impl

    def _load_generated_class(self):
        module_name = f"llm_generated_{int(time.time())}"
        spec = importlib.util.spec_from_file_location(module_name, self.out_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print("[LLM] Import exception:", e)
            return None
        cls = getattr(module, "GeneratedHeuristic", None)
        if cls is None:
            return None
        try:
            return cls()
        except Exception as e:
            print("[LLM] Instantiation exception:", e)
            return None
