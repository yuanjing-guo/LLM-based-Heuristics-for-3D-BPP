# heuristics/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

from helpers.task_config import TaskConfig
from heuristics.feasibility import FeasibilityChecker


# 6 axis-aligned rotations = 3! permutations of (x, y, z)
# rotated_size = size[ORDERS[rot_id]]
ORDERS: Dict[int, np.ndarray] = {
    0: np.array([0, 1, 2], dtype=int),
    1: np.array([0, 2, 1], dtype=int),
    2: np.array([1, 0, 2], dtype=int),
    3: np.array([1, 2, 0], dtype=int),
    4: np.array([2, 0, 1], dtype=int),
    5: np.array([2, 1, 0], dtype=int),
}


@dataclass(frozen=True)
class GeometrySpec:
    """Static geometry of the environment, derived from TaskConfig."""
    N: int          # visible buffer slots
    X: int          # pallet bins in x
    Y: int          # pallet bins in y
    H: int          # max pallet height in bins
    bin_size: float # meters per bin
    n_properties: int  # properties per slot (e.g., 4: sx,sy,sz,density)


class BaseHeuristic:
    """
    Base class for heuristics.

    Responsibilities (TOOLING only):
      - Read fixed geometry from TaskConfig (single source of truth)
      - Parse obs["buffer"] slots -> props -> size in bins
      - Rotation utility (rot_id -> permute dx,dy,dz)
      - Encode action logits (N + 6 + X + Y)

    Non-responsibilities:
      - No hard-constraint decisions inside BaseHeuristic
        (boundary / height / support -> FeasibilityChecker)
    """

    name = "base"

    def __init__(self):
        # ---- Read from TaskConfig (match env discretization) ----
        N = int(TaskConfig.buffer_size)
        bin_size = float(TaskConfig.bin_size)

        pallet_size_xy_m = np.array(TaskConfig.pallet.size, dtype=np.float32)[:2]  # meters
        # IMPORTANT: match env's (size/bin).astype(int) behavior (floor/truncate)
        X = int((pallet_size_xy_m[0] / bin_size))
        Y = int((pallet_size_xy_m[1] / bin_size))

        H = int(TaskConfig.pallet.max_pallet_height)
        n_properties = int(TaskConfig.box.n_properties)

        self.geom = GeometrySpec(
            N=N, X=X, Y=Y, H=H, bin_size=bin_size, n_properties=n_properties
        )

        # Feasibility checker: owns *all* hard constraints
        self.feasibility = FeasibilityChecker(X=X, Y=Y, H=H)

    # -------------------------
    # Optional episode hook
    # -------------------------
    def reset(self):
        """Override if heuristic is stateful."""
        return None

    # -------------------------
    # Convenience properties
    # -------------------------
    @property
    def N(self) -> int:
        return self.geom.N

    @property
    def X(self) -> int:
        return self.geom.X

    @property
    def Y(self) -> int:
        return self.geom.Y

    @property
    def H(self) -> int:
        return self.geom.H

    @property
    def bin_size(self) -> float:
        return self.geom.bin_size

    @property
    def n_properties(self) -> int:
        return self.geom.n_properties

    def action_dim(self) -> int:
        """Length of action logits vector expected by env."""
        return int(self.N + 6 + self.X + self.Y)

    # -------------------------
    # Buffer parsing utilities
    # -------------------------
    def get_slot_props(self, obs: dict, slot_i: int) -> np.ndarray:
        """
        Return one slot's properties array (length = n_properties).
        Env pads empty slots with zeros.
        """
        buf = obs["buffer"]
        s = slot_i * self.n_properties
        e = s + self.n_properties
        return np.asarray(buf[s:e], dtype=np.float32)

    def slot_is_empty(self, obs: dict, slot_i: int, eps: float = 1e-6) -> bool:
        """Heuristic-side check for padded empty slot."""
        props = self.get_slot_props(obs, slot_i)
        return bool(np.all(np.abs(props) < eps))

    # -------------------------
    # Size conversion utilities
    # -------------------------
    def props_to_size_bins(self, props: np.ndarray) -> np.ndarray:
        """
        Convert buffer props -> discrete size (dx,dy,dz) in bins.

        Buffer convention (from env._setup_references):
          props[:3] = box_obj.size * 100
          where box_obj.size is HALF-size in meters
          => props[:3] is HALF-size in centimeters.

        Env uses:
          size_bins = (box_obj.size * 2 / bin_size).astype(int)

        Here we do a safe equivalent:
          full_cm = 2 * half_cm
          bins = ceil(full_cm / bin_cm)
        """
        half_cm = np.asarray(props[:3], dtype=np.float32)   # half-size in cm
        full_cm = 2.0 * half_cm                             # full-size in cm
        bin_cm = self.bin_size * 100.0                      # bin size in cm

        # ceil is conservative: avoids underestimating footprint
        return np.ceil(full_cm / bin_cm).astype(int)

    def rotate_size_bins(self, size_bins: np.ndarray, rot_id: int) -> np.ndarray:
        """
        Apply rot_id permutation to (dx,dy,dz).
        Example:
          size = [dx,dy,dz]
          rot_id=1 -> order=[0,2,1] -> [dx,dz,dy]
        """
        rid = int(np.clip(rot_id, 0, 5))
        return np.asarray(size_bins[ORDERS[rid]], dtype=int)

    # -------------------------
    # Action encoding utilities
    # -------------------------
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

        Layout:
          [0 : N)           -> choose buffer slot
          [N : N+6)         -> choose rot_id
          [N+6 : N+6+X)     -> choose x
          [N+6+X : end)     -> choose y
        """
        a = np.full(self.action_dim(), low, dtype=np.float32)

        box_slot = int(np.clip(box_slot, 0, self.N - 1))
        rot_id = int(np.clip(rot_id, 0, 5))
        x = int(np.clip(x, 0, self.X - 1))
        y = int(np.clip(y, 0, self.Y - 1))

        a[box_slot] = high
        a[self.N + rot_id] = high
        a[self.N + 6 + x] = high
        a[self.N + 6 + self.X + y] = high
        return a

    # -------------------------
    # Interface
    # -------------------------
    def __call__(self, obs: dict) -> np.ndarray:
        """Subclasses implement: obs -> action logits."""
        raise NotImplementedError
