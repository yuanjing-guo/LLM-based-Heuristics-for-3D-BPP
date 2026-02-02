# heuristics/best_fit.py
import numpy as np
from heuristics.base import BaseHeuristic


class BestFit(BaseHeuristic):
    """
    Best-Fit 3D packing heuristic (top-loading).

    - Single container
    - Items placed from +z direction
    - Candidate positions derived from heightmap
    - Placement rule: select the best feasible position
    """

    name = "best_fit"

    def __init__(self):
        super().__init__()

    # --------------------------------------------------
    # Candidate position generation (same as First-Fit)
    # --------------------------------------------------
    def extract_candidates(self, pallet: np.ndarray):
        X, Y, H = pallet.shape
        candidates = []

        occ = pallet > 0
        has = occ.any(axis=2)
        occ_rev = occ[..., ::-1]
        first_from_top = occ_rev.argmax(axis=2)
        hmap = np.where(has, H - first_from_top, 0)

        for x in range(X):
            for y in range(Y):
                z = int(hmap[x, y])

                dx = 0
                while x + dx < X and hmap[x + dx, y] == z:
                    dx += 1

                dy = 0
                while y + dy < Y and hmap[x, y + dy] == z:
                    dy += 1

                if dx > 0 and dy > 0:
                    candidates.append((z, x, y, dx, dy))

        # fixed spatial priority (same as First-Fit)
        candidates.sort(key=lambda c: (c[0], c[1], c[2]))
        return candidates

    # --------------------------------------------------
    # Best-Fit placement
    # --------------------------------------------------
    def __call__(self, obs: dict) -> np.ndarray:
        pallet = obs["pallet_obs_density"]
        Kb = self.N  # visible items

        candidates = self.extract_candidates(pallet)

        best_choice = None
        best_score = None  # larger is better (tuple comparison)

        for (z, x, y, dx_max, dy_max) in candidates:
            for slot in range(Kb):
                if self.slot_is_empty(obs, slot):
                    continue

                props = self.get_slot_props(obs, slot)
                size_bins = self.props_to_size_bins(props)

                for rot_id in range(6):
                    dx, dy, dz = self.rotate_size_bins(size_bins, rot_id)

                    if dx > dx_max or dy > dy_max:
                        continue

                    if not self.feasibility.is_feasible(
                        pallet_obs=pallet,
                        x=x, y=y,
                        dx=dx, dy=dy, dz=dz,
                        z=z,
                    ):
                        continue

                    # ----------------------------------
                    # Best-Fit scoring (local optimality)
                    # ----------------------------------
                    footprint_fill = (dx * dy) / (dx_max * dy_max)
                    height_penalty = z

                    score = (footprint_fill, -height_penalty)

                    if best_score is None or score > best_score:
                        best_score = score
                        best_choice = (slot, rot_id, x, y)

        # --------------------------------------------------
        # Commit best placement
        # --------------------------------------------------
        if best_choice is None:
            return self.encode_action_logits(0, 0, 0, 0)

        slot, rot_id, x, y = best_choice

        return self.encode_action_logits(
            box_slot=slot,
            rot_id=rot_id,
            x=x,
            y=y,
        )
