# envs/scheduler.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RehandleItem:
    box_id: int
    available_step: int  # cooldown until this step (inclusive) is reached


class CandidateScheduler:
    """
    Three-layer candidate management:
      - front: fixed length N list of box_ids (visible buffer), empty slot = None
      - backlog: deque of box_ids (not yet visible)
      - rehandle: deque of RehandleItem (removed boxes, can return after cooldown)

    Refill rule:
      When a front slot is empty:
        1) take from rehandle if available_step <= step_count
        2) else take from backlog
    """

    def __init__(self, N: int):
        self.N = int(N)
        self.front: List[Optional[int]] = [None for _ in range(self.N)]
        self.backlog: Deque[int] = deque()
        self.rehandle: Deque[RehandleItem] = deque()

    def reset(self, all_box_ids: List[int], rng: np.random.Generator):
        ids = list(all_box_ids)
        rng.shuffle(ids)
        self.front = [None for _ in range(self.N)]
        self.backlog = deque(ids)
        self.rehandle = deque()

        # fill front from backlog
        for i in range(self.N):
            if len(self.backlog) == 0:
                break
            self.front[i] = self.backlog.popleft()

    def take_front(self, slot: int) -> Optional[int]:
        slot = int(slot)
        if slot < 0 or slot >= self.N:
            return None
        box_id = self.front[slot]
        self.front[slot] = None
        return box_id

    def push_rehandle(self, box_id: int, available_step: int):
        self.rehandle.append(RehandleItem(int(box_id), int(available_step)))

    def _pop_available_rehandle(self, step_count: int) -> Optional[int]:
        """
        Find and pop the first available rehandle item (stable order).
        If none available, return None.
        """
        step_count = int(step_count)
        if not self.rehandle:
            return None

        # stable order scan (deque), but without O(n) pop from middle complexity
        # We'll rotate until we find one, or give up after full cycle.
        n = len(self.rehandle)
        for _ in range(n):
            it = self.rehandle[0]
            self.rehandle.popleft()
            if it.available_step <= step_count:
                return it.box_id
            # not available yet -> put back
            self.rehandle.append(it)
        return None

    def refill(self, step_count: int):
        """Refill empty slots in front using rehandle (available) then backlog."""
        for i in range(self.N):
            if self.front[i] is not None:
                continue

            b = self._pop_available_rehandle(step_count)
            if b is not None:
                self.front[i] = int(b)
                continue

            if len(self.backlog) > 0:
                self.front[i] = int(self.backlog.popleft())

    def front_ids(self) -> np.ndarray:
        arr = np.full((self.N,), -1, dtype=np.int32)
        for i, bid in enumerate(self.front):
            if bid is None:
                continue
            arr[i] = int(bid)
        return arr

    def counts(self) -> Dict[str, int]:
        return {
            "front_nonempty": int(sum([1 for x in self.front if x is not None])),
            "backlog": int(len(self.backlog)),
            "rehandle": int(len(self.rehandle)),
        }

    def encode_front(
        self,
        id_to_properties: Dict[int, np.ndarray],
        id_to_softness: Dict[int, float],
        id_to_friction_mu: Dict[int, float],
        n_properties: int,
        n_physics_properties: int,
        expose_physics_obs: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          buffer:         float32 shape (N*n_properties,)
          buffer_physics: float32 shape (N*n_physics_properties,)
          front_ids:      int32   shape (N,)
        """
        N = self.N
        n_properties = int(n_properties)
        n_physics_properties = int(n_physics_properties)

        buf = np.zeros((N * n_properties,), dtype=np.float32)
        buf_phy = np.zeros((N * n_physics_properties,), dtype=np.float32)
        ids = self.front_ids()

        for i in range(N):
            box_id = self.front[i]
            if box_id is None:
                continue

            props = id_to_properties[int(box_id)]
            base = i * n_properties
            buf[base : base + n_properties] = props.astype(np.float32)

            if expose_physics_obs:
                s = float(id_to_softness.get(int(box_id), 0.0))
                mu = float(id_to_friction_mu.get(int(box_id), 0.0))
                pbase = i * n_physics_properties
                if n_physics_properties >= 1:
                    buf_phy[pbase + 0] = s
                if n_physics_properties >= 2:
                    buf_phy[pbase + 1] = mu

        return buf, buf_phy, ids
