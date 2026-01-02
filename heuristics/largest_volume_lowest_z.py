# heuristics/handcrafted.py

import numpy as np
from heuristics.base import BaseHeuristic
from env import encode_choice_to_action_logits


class LargestVolumeLowestZ(BaseHeuristic):
    """
    Handcrafted baseline:
    - Select largest-volume box
    - Place at lowest feasible z
    - Tie-break by minimal (x + y)
    """

    name = "largest_volume_lowest_z"

    def __init__(self, N_visible_boxes, pallet_size_discrete, max_pallet_height):
        # Let BaseHeuristic handle geometry & limits
        super().__init__(
            N_visible_boxes=N_visible_boxes,
            pallet_size_discrete=pallet_size_discrete,
            max_pallet_height=max_pallet_height,
        )

    def __call__(self, obs):
        pallet = obs["pallet_obs_density"]
        buffer = obs["buffer"]

        # ----------------------------------
        # 1. Select box: largest volume first
        # ----------------------------------
        box_sizes = []
        for i in range(self.N):
            size = buffer[i * 4 : i * 4 + 3]  # (dx, dy, dz) in cm
            if np.all(size > 0):
                vol = np.prod(size)
            else:
                vol = -1
            box_sizes.append(vol)

        box_i = int(np.argmax(box_sizes))
        if box_sizes[box_i] <= 0:
            box_i = 0  # fallback

        # ----------------------------------
        # 2. Orientation (fixed)
        # ----------------------------------
        rot_i = 0

        box_size_discrete = buffer[box_i * 4 : box_i * 4 + 3].astype(int)
        dx, dy, dz = box_size_discrete

        # ----------------------------------
        # 3. Find best (x, y)
        # ----------------------------------
        best_score = None
        best_xy = (0, 0)

        for x in range(self.X):
            for y in range(self.Y):
                x2 = min(x + dx, self.X)
                y2 = min(y + dy, self.Y)

                place_area = pallet[x:x2, y:y2, :]
                non_zero_mask = np.any(place_area > 0, axis=(0, 1))
                z = np.max(np.nonzero(non_zero_mask)) + 1 if np.any(non_zero_mask) else 0

                if z + dz > self.H:
                    continue

                score = (z, x + y)
                if best_score is None or score < best_score:
                    best_score = score
                    best_xy = (x, y)

        x_i, y_i = best_xy

        # ----------------------------------
        # 4. Encode into action logits
        # ----------------------------------
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
