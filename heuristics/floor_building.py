import numpy as np

from heuristics.base import BaseHeuristic
from env import encode_choice_to_action_logits


class FloorBuilding(BaseHeuristic):
    """
    Floor Building heuristic (layer-by-layer packing).

    Strategy:
        - Prefer lower z (fill floor first)
        - Tie-break by smaller x + y (corner preference)

    Feasibility:
        - Fully delegated to FeasibilityChecker (hard constraints)
    """

    name = "floor_building"

    def __init__(
        self,
        N_visible_boxes,
        pallet_size_discrete,
        max_pallet_height,
        bin_size,
    ):
        super().__init__(
            N_visible_boxes=N_visible_boxes,
            pallet_size_discrete=pallet_size_discrete,
            max_pallet_height=max_pallet_height,
            bin_size=bin_size,
        )

    def __call__(self, obs):
        pallet = obs["pallet_obs_density"]
        buffer = obs["buffer"]

        # --------------------------------------------------
        # 1. Select box: largest volume first (deterministic)
        # --------------------------------------------------
        best_box = None
        best_volume = -1

        for i in range(self.N):
            size_cm = buffer[i * 4 : i * 4 + 3]
            if np.any(size_cm <= 0):
                continue

            volume = np.prod(size_cm)
            if volume > best_volume:
                best_volume = volume
                best_box = i

        # No valid box (should not happen normally)
        if best_box is None:
            best_box = 0

        box_i = best_box
        rot_i = 0  # orientation fixed (consistent with env)

        # Box size in bins
        size_cm = buffer[box_i * 4 : box_i * 4 + 3]
        dx, dy, dz = self.box_size_to_bins(size_cm)

        # --------------------------------------------------
        # 2. Search feasible placements (floor building)
        # --------------------------------------------------
        best_score = None
        best_xy = None

        for x in range(self.X):
            for y in range(self.Y):

                # compute required z from pallet occupancy
                x2 = x + dx
                y2 = y + dy

                place_area = pallet[x:x2, y:y2, :]
                if place_area.size == 0:
                    continue

                non_zero = np.any(place_area > 0, axis=(0, 1))
                z = int(np.max(np.nonzero(non_zero)) + 1) if np.any(non_zero) else 0

                # ---------- HARD CONSTRAINT CHECK ----------
                if not self.feasibility.is_feasible(
                    pallet_obs=pallet,
                    x=x, y=y,
                    dx=dx, dy=dy, dz=dz,
                    z=z,
                ):
                    continue

                # ---------- STRATEGY SCORE ----------
                # floor building: lowest z, then closest to corner
                score = (z, x + y)

                if best_score is None or score < best_score:
                    best_score = score
                    best_xy = (x, y)

        # --------------------------------------------------
        # 3. No feasible placement → let env terminate
        # --------------------------------------------------
        if best_xy is None:
            # fallback: env will detect failure (term=2)
            return encode_choice_to_action_logits(
                N=self.N,
                X=self.X,
                Y=self.Y,
                box_i=box_i,
                rot_i=rot_i,
                x_i=0,
                y_i=0,
            )

        x_i, y_i = best_xy

        # --------------------------------------------------
        # 4. Encode final action
        # --------------------------------------------------
        action = encode_choice_to_action_logits(
            N=self.N,
            X=self.X,
            Y=self.Y,
            box_i=box_i,
            rot_i=rot_i,
            x_i=x_i,
            y_i=y_i,
        )

        return action
