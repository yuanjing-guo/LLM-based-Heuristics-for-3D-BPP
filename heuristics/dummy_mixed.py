# heuristics/dummy_mixed.py
import numpy as np
from heuristics.base import BaseHeuristic


class DummyMixed(BaseHeuristic):
    """
    Very simple mixed policy:
      - Try to PLACE with a deterministic scan:
          slot: 0..N-1
          rot: 0..5
          x,y: raster scan
        (but we do NOT check feasibility here; we rely on env termination to see failures)
      - If pallet_count > 0 and after some failures, do REMOVE last box.
    """

    name = "dummy_mixed"

    def __init__(self, max_place_tries_per_step: int = 1, remove_after_steps: int = 20):
        super().__init__()
        self.max_place_tries_per_step = int(max_place_tries_per_step)
        self.remove_after_steps = int(remove_after_steps)
        self._t = 0

        # internal cursor for deterministic scan
        self._slot = 0
        self._rot = 0
        self._x = 0
        self._y = 0

    def reset(self):
        self._t = 0
        self._slot = 0
        self._rot = 0
        self._x = 0
        self._y = 0

    def __call__(self, obs):
        self._t += 1

        # Try to remove occasionally if there are boxes on pallet
        pallet_count = int(obs.get("pallet_count", 0))
        if pallet_count > 0 and (self._t % self.remove_after_steps == 0):
            # remove most recent box (safe index)
            return np.array([1, pallet_count - 1, 0, 0, 0], dtype=np.int32)

        # Otherwise, always attempt a place with a simple scan
        # Need X,Y bounds from obs
        pallet = obs["pallet_obs_density"]
        X = int(pallet.shape[0])
        Y = int(pallet.shape[1])

        N = int(obs["buffer"].size // getattr(self, "n_properties", 4))  # fallback 4
        # If your BaseHeuristic has n_properties, it will be used; else fallback.

        # If front_ids exists, prefer non-empty slots; otherwise just cycle.
        front_ids = obs.get("front_ids", None)

        # One candidate per step (keep it dumb)
        for _ in range(self.max_place_tries_per_step):
            slot = self._slot % max(N, 1)
            rot = self._rot % 6
            x = self._x % max(X, 1)
            y = self._y % max(Y, 1)

            # advance cursor
            self._y += 1
            if self._y >= Y:
                self._y = 0
                self._x += 1
            if self._x >= X:
                self._x = 0
                self._rot += 1
            if self._rot >= 6:
                self._rot = 0
                self._slot += 1

            # If slot is empty in front_ids, skip it
            if front_ids is not None:
                if int(front_ids[slot]) < 0:
                    continue

            return np.array([0, int(slot), int(rot), int(x), int(y)], dtype=np.int32)

        # fallback no-op
        return np.array([0, 0, 0, 0, 0], dtype=np.int32)
