# heuristics/base.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from helpers.task_config import TaskConfig
from heuristics.feasibility import FeasibilityChecker


# 6-axis-aligned rotations = 3! permutations of (x,y,z)
ORDERS = {
    0: np.array([0, 1, 2], dtype=int),
    1: np.array([0, 2, 1], dtype=int),
    2: np.array([1, 0, 2], dtype=int),
    3: np.array([1, 2, 0], dtype=int),
    4: np.array([2, 0, 1], dtype=int),
    5: np.array([2, 1, 0], dtype=int),
}


class BaseHeuristic:
    """
    Base class for all heuristics.

    Reads global geometry/config directly from TaskConfig:
      - N: visible buffer slots
      - X,Y: pallet discrete size (bins)
      - H: max pallet height (bins)
      - bin_size: meters per bin

    Provides:
      - Parsing buffer slots -> (dx,dy,dz) in bins
      - Rotation utilities
      - Action logits encoding helper
      - Optional feasibility checker helper
    """

    name = "base"

    def __init__(self):
        # ---- read from TaskConfig (single source of truth) ----
        self.N: int = int(TaskConfig.buffer_size)
        self.bin_size: float = float(TaskConfig.bin_size)

        pallet_size_xy = np.array(TaskConfig.pallet.size, dtype=np.float32)[:2]  # meters
        self.X: int = int(np.round(pallet_size_xy[0] / self.bin_size))
        self.Y: int = int(np.round(pallet_size_xy[1] / self.bin_size))

        self.H: int = int(TaskConfig.pallet.max_pallet_height)
        self.n_properties: int = int(TaskConfig.box.n_properties)

        # Optional helper: if you already have feasibility logic in heuristics.feasibility
        self.feasibility = FeasibilityChecker(X=self.X, Y=self.Y, H=self.H)

    # ------------------------------------------------------------
    # Optional episode hook
    # ------------------------------------------------------------
    def reset(self):
        """Override if heuristic is stateful."""
        return None

    # ------------------------------------------------------------
    # Buffer parsing utilities
    # ------------------------------------------------------------
    def get_slot_props(self, obs: dict, slot_i: int) -> np.ndarray:
        """
        Return properties for one slot.
        If slot is empty, usually it's all zeros (depends on env.update_obs_buffer()).
        """
        buf = obs["buffer"]
        start = slot_i * self.n_properties
        end = start + self.n_properties
        return np.array(buf[start:end], dtype=np.float32)

    def slot_is_empty(self, obs: dict, slot_i: int, eps: float = 1e-6) -> bool:
        """Heuristic-side check for empty slot (buffer padded with zeros)."""
        props = self.get_slot_props(obs, slot_i)
        return bool(np.all(np.abs(props) < eps))

    def props_to_size_bins(self, props: np.ndarray) -> np.ndarray:
        """
        Convert buffer props -> discrete size (dx,dy,dz) in bins.

        IMPORTANT:
          In your env, you stored:
            props[:3] = box_obj.size * 100
          where box_obj.size is HALF-size in meters.
          So props[:3] is HALF-size in cm.

        Env uses:
          size_bins = (box_obj.size * 2 / bin_size).astype(int)
        which is basically full_size / bin_size.

        We follow the same meaning:
          full_cm = 2 * half_cm
          bins = ceil(full_cm / bin_cm)
        """
        half_cm = np.array(props[:3], dtype=np.float32)          # half-size in cm
        full_cm = 2.0 * half_cm                                 # full-size in cm
        bin_cm = self.bin_size * 100.0                           # bin size in cm
        return np.ceil(full_cm / bin_cm).astype(int)             # (dx,dy,dz) in bins

    def rotate_size_bins(self, size_bins: np.ndarray, rot_id: int) -> np.ndarray:
        """Apply ORDERS[rot_id] permutation to (dx,dy,dz)."""
        rot_id = int(np.clip(rot_id, 0, 5))
        return np.array(size_bins[ORDERS[rot_id]], dtype=int)

    # ------------------------------------------------------------
    # Action encoding utilities (logits vector)
    # ------------------------------------------------------------
    def encode_action_logits(
        self,
        box_slot: int,
        rot_id: int,
        x: int,
        y: int,
        low: float = -10.0,
        high: float = 10.0,
    ) -> np.ndarray:
        """
        Build action logits vector of length (N + 6 + X + Y).
        Argmax decoding in env will pick the indices with the highest logits.
        """
        action_dim = int(self.N + 6 + self.X + self.Y)
        a = np.full(action_dim, low, dtype=np.float32)

        box_slot = int(np.clip(box_slot, 0, self.N - 1))
        rot_id = int(np.clip(rot_id, 0, 5))
        x = int(np.clip(x, 0, self.X - 1))
        y = int(np.clip(y, 0, self.Y - 1))

        a[box_slot] = high
        a[self.N + rot_id] = high
        a[self.N + 6 + x] = high
        a[self.N + 6 + self.X + y] = high
        return a

    # ------------------------------------------------------------
    # Useful checks
    # ------------------------------------------------------------
    def in_bounds_xy(self, x: int, y: int, dx: int, dy: int) -> bool:
        """Strict boundary: footprint [x:x+dx, y:y+dy] must be inside pallet."""
        return (x >= 0) and (y >= 0) and (x + dx <= self.X) and (y + dy <= self.Y)

    # ------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------
    def __call__(self, obs: dict) -> np.ndarray:
        """Subclasses must implement: obs -> action logits vector."""
        raise NotImplementedError
