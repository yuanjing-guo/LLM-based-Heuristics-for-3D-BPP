# heuristics/corner_point_online.py
import numpy as np
from heuristics.base import BaseHeuristic


class CornerPoint(BaseHeuristic):
    """
    Online Corner Point heuristic with First-Fit placement.

    - Single container
    - Top-loading (z direction)
    - Corner points derived from heightmap
    - Candidate = (x, y, z)
    - Placement rule: FIRST feasible placement is accepted
    """

    name = "corner_point"

    def __init__(self):
        super().__init__()

    # --------------------------------------------------
    # Corner point extraction from heightmap (UNCHANGED)
    # --------------------------------------------------
    def extract_corner_points(self, pallet: np.ndarray):
        """
        Extract corner points from heightmap.

        Each corner point is:
            (x, y, z)
        where z = heightmap[x, y]
        """
        X, Y, H = pallet.shape
        cps = []

        occ = pallet > 0
        has = occ.any(axis=2)
        occ_rev = occ[..., ::-1]
        first_from_top = occ_rev.argmax(axis=2)
        hmap = np.where(has, H - first_from_top, 0)

        for x in range(X):
            for y in range(Y):
                z = int(hmap[x, y])
                cps.append((x, y, z))

        # top-loading priority: lowest z first, then x, then y
        cps.sort(key=lambda p: (p[2], p[0], p[1]))
        return cps

    # --------------------------------------------------
    # Main heuristic: Corner Point + First-Fit
    # --------------------------------------------------
    def __call__(self, obs: dict) -> np.ndarray:
        pallet = obs["pallet_obs_density"]
        Kb = self.N  # visible items (IPS)

        corner_points = self.extract_corner_points(pallet)

        # --------------------------------------------------
        # First-Fit placement
        # --------------------------------------------------
        for (x, y, z) in corner_points:
            for slot in range(Kb):
                if self.slot_is_empty(obs, slot):
                    continue

                props = self.get_slot_props(obs, slot)
                size_bins = self.props_to_size_bins(props)

                for rot_id in range(6):
                    dx, dy, dz = self.rotate_size_bins(size_bins, rot_id)

                    # boundary check
                    if x + dx > pallet.shape[0] or y + dy > pallet.shape[1]:
                        continue

                    # feasibility check (overlap + support)
                    if not self.feasibility.is_feasible(
                        pallet_obs=pallet,
                        x=x, y=y,
                        dx=dx, dy=dy, dz=dz,
                        z=z,
                    ):
                        continue

                    # ------------------------------------------
                    # First-Fit: accept immediately
                    # ------------------------------------------
                    return self.encode_action_logits(
                        box_slot=slot,
                        rot_id=rot_id,
                        x=x,
                        y=y,
                    )

        # --------------------------------------------------
        # Failure fallback
        # --------------------------------------------------
        return self.encode_action_logits(0, 0, 0, 0)
