# heuristics/floor_building.py
import numpy as np

from heuristics.base import BaseHeuristic


class FloorBuilding(BaseHeuristic):
    """
    Floor-building heuristic (layer-by-layer packing).

    Same as before, except:
      - box selection is buffered over all visible slots (Kb),
        exactly like FirstFit.
    """

    name = "floor_building"

    def __init__(self):
        super().__init__()

    def __call__(self, obs: dict) -> np.ndarray:
        pallet = obs["pallet_obs_density"]
        Kb = self.N  # visible buffer (same as FirstFit)

        best_score = None
        best_choice = None  # (box_slot, rot_id, x, y)

        # --------------------------------------------------
        # 0) Enumerate visible slots (BUFFER = Kb)
        # --------------------------------------------------
        for box_slot in range(Kb):
            if self.slot_is_empty(obs, box_slot):
                continue

            props = self.get_slot_props(obs, box_slot)
            size_bins = self.props_to_size_bins(props)

            # --------------------------------------------------
            # 1) Original Floor-Building search (UNCHANGED)
            # --------------------------------------------------
            for rot_id in range(6):
                dx, dy, dz = self.rotate_size_bins(size_bins, rot_id)

                if dz > self.H:
                    continue

                for x in range(self.X):
                    for y in range(self.Y):

                        if not self.feasibility.is_within_pallet(x, y, dx, dy):
                            continue

                        place_area = pallet[x:x + dx, y:y + dy, :]
                        non_zero_mask = np.any(place_area > 0, axis=(0, 1))
                        z = int(np.max(np.nonzero(non_zero_mask)) + 1) if np.any(non_zero_mask) else 0

                        if not self.feasibility.is_feasible(
                            pallet_obs=pallet,
                            x=x, y=y,
                            dx=dx, dy=dy, dz=dz,
                            z=z,
                        ):
                            continue

                        # ---- SAME scoring as before ----
                        score = (z, x + y)

                        if best_score is None or score < best_score:
                            best_score = score
                            best_choice = (box_slot, rot_id, x, y)

        # --------------------------------------------------
        # 2) No feasible placement
        # --------------------------------------------------
        if best_choice is None:
            return self.encode_action_logits(0, 0, 0, 0)

        box_slot, rot_id, x, y = best_choice

        # --------------------------------------------------
        # 3) Encode action
        # --------------------------------------------------
        return self.encode_action_logits(
            box_slot=box_slot,
            rot_id=rot_id,
            x=x,
            y=y,
        )
