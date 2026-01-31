# heuristics/llm_entry.py
import os
import time
import importlib.util
import numpy as np
import json

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
        is_ollama = ("localhost:11434" in self.api_url) or (self.api_url.rstrip("/").endswith("/api/generate"))

        if (not is_ollama) and (not self.api_key):
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

        if not self._smoke_test(impl):
            print("[LLM] Generated heuristic rejected by smoke test. Regenerating with hard feedback.")

            hard_feedback = (
                "The function crashes. Fix it strictly:\n"
                "- llm_policy MUST return exactly 4 integers: (box_slot, rot_id, x, y).\n"
                "- Do NOT unpack shapes. Rotated sizes are length-3 vectors.\n"
                "- Use indexing only: dx=int(v[0]), dy=int(v[1]), dz=int(v[2]).\n"
                "- Never use shape[...] to infer dimensions.\n"
                "- If no feasible action exists, return (0,0,0,0).\n"
                "Output ONLY the llm_policy function code."
            )

            # try one automatic regenerate
            try:
                self._feedback_history.append(hard_feedback)
                code2 = generate_heuristic(
                    self.api_url,
                    self.api_key,
                    self.model,
                    self._last_code,
                    hard_feedback,
                    self._feedback_history,
                )
                if code2:
                    write_heuristic(self.out_path, code2)
                    impl2 = self._load_generated_class()
                    if impl2 and self._smoke_test(impl2):
                        self._last_code = code2
                        self.current_code = code2
                        self._impl = impl2
                        print("[LLM] Regeneration succeeded after feedback.")
                        return
            except Exception as e:
                print("[LLM] Regeneration failed:", e)

            print("[LLM] Final fallback to FloorBuilding.")
            self._impl = FloorBuilding()
            return

        # save successful result
        self._last_code = code
        self.current_code = code
        self._impl = impl

    def _smoke_test(self, impl) -> bool:
        """
        Minimal runtime test to avoid crashing the whole episode due to bad generated code.
        We build a dummy obs with correct shapes from runs/latest/run_context.json.
        """
        ctx_path = "runs/latest/run_context.json"
        try:
            with open(ctx_path, "r", encoding="utf-8") as f:
                ctx = json.load(f)
        except Exception as e:
            print("[LLM] Smoke test skipped (cannot read run_context):", e)
            return True  # don't block if context missing

        X, Y, H = ctx.get("X"), ctx.get("Y"), ctx.get("H")
        n_props = ctx.get("n_properties")
        obs_schema = ctx.get("obs_schema", {})

        # Basic sanity
        if not all(isinstance(v, int) and v > 0 for v in [X, Y, H, n_props]):
            print("[LLM] Smoke test skipped (invalid X/Y/H/n_properties in run_context).")
            return True

        # Determine buffer slots N from schema if possible; else pick a safe small N
        N = 10
        try:
            buf_shape = obs_schema.get("buffer", {}).get("shape", None)
            # buffer shape often like [N*n_properties] or similar
            if isinstance(buf_shape, (list, tuple)) and len(buf_shape) == 1 and isinstance(buf_shape[0], int):
                total = int(buf_shape[0])
                if total > 0 and total % int(n_props) == 0:
                    N = total // int(n_props)
        except Exception:
            pass

        import numpy as np  # allowed here, not inside llm_policy

        dummy_obs = {
            "pallet_obs_density": np.zeros((X, Y, H), dtype=np.float32),
            "buffer": np.zeros((N * n_props,), dtype=np.float32),
        }

        # Fill slot 0 with a simple positive size so slot_is_empty won't always skip
        # We don't know your exact props layout; but many of your helpers treat first 3 as dims-bins.
        # If this guess is wrong, the smoke test will still catch exceptions and we fallback safely.
        if N > 0:
            dummy_obs["buffer"][0:3] = np.array([1, 1, 1], dtype=np.float32)

        try:
            _ = impl(dummy_obs)
            return True
        except Exception as e:
            print("[LLM] Smoke test failed. Generated heuristic would crash:", repr(e))
            return False


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
