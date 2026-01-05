# heuristics/floor_building.py
import numpy as np

from heuristics.base import BaseHeuristic


class FloorBuilding(BaseHeuristic):
    """
    Floor-building heuristic (layer-by-layer packing), now with:
      - rotation enabled (rot_id 0..5)
      - strict boundary/height/support handled ONLY by FeasibilityChecker
      - box choice locked to buffer slot 0 (for now)

    Objective:
      - minimize z (fill lowest layer first)
      - tie-break: prefer corner (x+y small)
    """

    name = "floor_building"

    def __init__(self):
        super().__init__()

    def __call__(self, obs: dict) -> np.ndarray:
        pallet = obs["pallet_obs_density"]

        # --------------------------------------------------
        # 0) Pick box slot (locked to first slot)
        # --------------------------------------------------
        box_slot = 0
        if self.slot_is_empty(obs, box_slot):
            # Nothing visible / empty slot: return a deterministic default
            return self.encode_action_logits(box_slot=0, rot_id=0, x=0, y=0)

        props = self.get_slot_props(obs, box_slot)
        size_bins = self.props_to_size_bins(props)  # (dx,dy,dz) in bins (unrotated)

        best_score = None
        best_choice = None  # (rot_id, x, y)

        # --------------------------------------------------
        # 1) Search over rotations and xy placements
        # --------------------------------------------------
        for rot_id in range(6):
            dx, dy, dz = self.rotate_size_bins(size_bins, rot_id)

            # quick prune: if box itself taller than pallet
            if dz > self.H:
                continue

            # scan all x,y (bounds check is done inside feasibility)
            for x in range(self.X):
                for y in range(self.Y):

                    # ---- IMPORTANT: avoid invalid slicing before computing z ----
                    # We are NOT doing bounds logic here; we call feasibility's is_within_pallet.
                    if not self.feasibility.is_within_pallet(x, y, dx, dy):
                        continue

                    # ---- compute z from occupancy (same idea as env) ----
                    # now slicing is safe because within_pallet passed
                    place_area = pallet[x:x + dx, y:y + dy, :]
                    non_zero_mask = np.any(place_area > 0, axis=(0, 1))
                    z = int(np.max(np.nonzero(non_zero_mask)) + 1) if np.any(non_zero_mask) else 0

                    # ---- final feasibility: boundary + height + support ----
                    if not self.feasibility.is_feasible(
                        pallet_obs=pallet,
                        x=x, y=y,
                        dx=dx, dy=dy, dz=dz,
                        z=z,
                    ):
                        continue

                    # ---- scoring: lowest z, then corner preference ----
                    score = (z, x + y)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_choice = (rot_id, x, y)

        # --------------------------------------------------
        # 2) No feasible placement found
        # --------------------------------------------------
        if best_choice is None:
            # deterministic fallback (may fail in env)
            return self.encode_action_logits(box_slot=box_slot, rot_id=0, x=0, y=0)

        rot_id, x, y = best_choice

        # --------------------------------------------------
        # 3) Encode action logits
        # --------------------------------------------------
        return self.encode_action_logits(
            box_slot=box_slot,
            rot_id=rot_id,
            x=x,
            y=y,
        )
