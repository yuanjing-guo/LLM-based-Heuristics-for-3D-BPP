# heuristics/floor_building.py
import numpy as np

from heuristics.base import BaseHeuristic


class FloorBuildingBuffer(BaseHeuristic):
    """
    Buffer-aware floor-building heuristic (layer-by-layer packing), with:
      - buffer-aware box selection (scan all non-empty slots)
      - rotation enabled (rot_id 0..5)
      - strict boundary/height/support handled ONLY by FeasibilityChecker

    Objective (unchanged):
      - minimize z (fill lowest layer first)
      - tie-break: prefer corner (x+y small)

    NOTE:
      This version chooses the globally best (box_slot, rot_id, x, y) by the same score.
    """

    name = "floor_building_buffer"

    def __init__(self):
        super().__init__()

    def __call__(self, obs: dict) -> np.ndarray:
        pallet = obs["pallet_obs_density"]

        # --------------------------------------------------
        # 0) Scan all candidate box slots (buffer-aware)
        # --------------------------------------------------
        candidate_slots = [i for i in range(self.N) if not self.slot_is_empty(obs, i)]
        if not candidate_slots:
            # No visible box in buffer -> deterministic default
            return self.encode_action_logits(box_slot=0, rot_id=0, x=0, y=0)

        global_best_score = None
        global_best = None  # (box_slot, rot_id, x, y)

        # --------------------------------------------------
        # 1) For each slot, search over rotations and xy placements
        # --------------------------------------------------
        for box_slot in candidate_slots:
            props = self.get_slot_props(obs, box_slot)
            size_bins = self.props_to_size_bins(props)  # (dx,dy,dz) unrotated

            for rot_id in range(6):
                dx, dy, dz = self.rotate_size_bins(size_bins, rot_id)

                # quick prune: if box itself taller than pallet
                if dz > self.H:
                    continue

                for x in range(self.X):
                    for y in range(self.Y):
                        # bounds check (avoid invalid slicing)
                        if not self.feasibility.is_within_pallet(x, y, dx, dy):
                            continue

                        # compute z from current occupancy
                        place_area = pallet[x:x + dx, y:y + dy, :]
                        non_zero_mask = np.any(place_area > 0, axis=(0, 1))
                        z = int(np.max(np.nonzero(non_zero_mask)) + 1) if np.any(non_zero_mask) else 0

                        # feasibility: boundary + height + support
                        if not self.feasibility.is_feasible(
                            pallet_obs=pallet,
                            x=x, y=y,
                            dx=dx, dy=dy, dz=dz,
                            z=z,
                        ):
                            continue

                        # scoring: lowest z, then corner preference
                        score = (z, x + y)

                        if global_best_score is None or score < global_best_score:
                            global_best_score = score
                            global_best = (box_slot, rot_id, x, y)

        # --------------------------------------------------
        # 2) No feasible placement found for any slot
        # --------------------------------------------------
        if global_best is None:
            # deterministic fallback (may fail in env)
            # choose the first non-empty slot if possible
            box_slot = candidate_slots[0]
            return self.encode_action_logits(box_slot=box_slot, rot_id=0, x=0, y=0)

        box_slot, rot_id, x, y = global_best

        # --------------------------------------------------
        # 3) Encode action logits
        # --------------------------------------------------
        return self.encode_action_logits(
            box_slot=box_slot,
            rot_id=rot_id,
            x=x,
            y=y,
        )
