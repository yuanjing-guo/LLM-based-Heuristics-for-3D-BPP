# heuristics/best_fit.py
import numpy as np
from heuristics.base import BaseHeuristic


class BestFit(BaseHeuristic):
    """
    Best-Fit 3D packing heuristic (top-loading).

    Identical candidate space to First-Fit (heightmap-based flat rectangles).
    Only differs in placement selection via a physics-aware Best-Fit score.
    """

    name = "best_fit"

    def __init__(self):
        super().__init__()

        # scoring weights (tuned for palletization + physics)
        self.w_footprint = 1.0     # absolute footprint usage
        self.w_platform = 0.3      # prefer larger platforms
        self.w_height = -0.5       # penalize higher placement
        self.w_shape = -0.05       # mild regularization

    # --------------------------------------------------
    # Candidate position generation (IDENTICAL to First-Fit)
    # --------------------------------------------------
    def extract_candidates(self, pallet: np.ndarray):
        """
        Generate candidate placement positions from heightmap.

        Each candidate:
            (z, x, y, dx_max, dy_max)
        """
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

        # same deterministic order as First-Fit
        candidates.sort(key=lambda c: (c[0], c[1], c[2]))
        return candidates

    # --------------------------------------------------
    # Best-Fit placement (ONLY difference vs First-Fit)
    # --------------------------------------------------
    def __call__(self, obs: dict) -> np.ndarray:
        pallet = obs["pallet_obs_density"]
        Kb = self.N  # visible items

        candidates = self.extract_candidates(pallet)

        best_score = None
        best_action = None

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

                    # --------------------------------
                    # Physics-aware Best-Fit score
                    # --------------------------------
                    footprint = dx * dy                  # absolute area used
                    platform = dx_max * dy_max           # size of supporting surface
                    height = z                           # placement height
                    shape = abs(dx - dy)                 # mild regularization

                    score = (
                        self.w_footprint * footprint +
                        self.w_platform * platform +
                        self.w_height * height +
                        self.w_shape * shape
                    )

                    if best_score is None or score > best_score:
                        best_score = score
                        best_action = self.encode_action_logits(
                            box_slot=slot,
                            rot_id=rot_id,
                            x=x,
                            y=y,
                        )

        # --------------------------------------------------
        # Failure fallback
        # --------------------------------------------------
        if best_action is not None:
            return best_action

        return self.encode_action_logits(0, 0, 0, 0)
