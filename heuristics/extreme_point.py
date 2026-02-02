# heuristics/ep_online.py
import numpy as np
from heuristics.base import BaseHeuristic


class ExtremePoint(BaseHeuristic):
    """
    Online Extreme Point (EP) heuristic for 3D bin packing.

    - Single container (pallet)
    - Top-loading (items placed from +z direction)
    - EPs extracted implicitly from heightmap corners
    - Candidate = (x, y, z) points only (no EMS)
    """

    name = "extreme_point"

    def __init__(self):
        super().__init__()

    # --------------------------------------------------
    # Extreme Point extraction from heightmap
    # --------------------------------------------------
    def extract_eps(self, pallet: np.ndarray):
        """
        Extract Extreme Points from heightmap.

        EP = (x, y, z)
        where z = heightmap[x, y]

        Only consider EPs on the top surface.
        """
        X, Y, H = pallet.shape
        eps = []

        occ = pallet > 0
        has = occ.any(axis=2)
        occ_rev = occ[..., ::-1]
        first_from_top = occ_rev.argmax(axis=2)
        hmap = np.where(has, H - first_from_top, 0)

        for x in range(X):
            for y in range(Y):
                z = int(hmap[x, y])
                eps.append((x, y, z))

        # EP priority: lowest z first, then x, then y
        eps.sort(key=lambda p: (p[2], p[0], p[1]))
        return eps

    # --------------------------------------------------
    # Main heuristic
    # --------------------------------------------------
    def __call__(self, obs: dict) -> np.ndarray:
        pallet = obs["pallet_obs_density"]
        Kb = self.N  # visible items (IPS)

        eps = self.extract_eps(pallet)

        best_choice = None  # (slot, rot_id, x, y)
        best_score = None

        # --------------------------------------------------
        # Placement selection
        # --------------------------------------------------
        for (x, y, z) in eps:
            for slot in range(Kb):
                if self.slot_is_empty(obs, slot):
                    continue

                props = self.get_slot_props(obs, slot)
                size_bins = self.props_to_size_bins(props)

                for rot_id in range(6):
                    dx, dy, dz = self.rotate_size_bins(size_bins, rot_id)

                    # container boundary check
                    if x + dx > pallet.shape[0] or y + dy > pallet.shape[1]:
                        continue

                    # feasibility (overlap + gravity consistency)
                    if not self.feasibility.is_feasible(
                        pallet_obs=pallet,
                        x=x, y=y,
                        dx=dx, dy=dy, dz=dz,
                        z=z,
                    ):
                        continue

                    # ------------------------------
                    # EP placement score (Best-Fit)
                    # ------------------------------
                    footprint_area = dx * dy
                    height_penalty = z

                    score = (footprint_area, -height_penalty)

                    if best_score is None or score > best_score:
                        best_score = score
                        best_choice = (slot, rot_id, x, y)

            # EP-style early stop (FFD-like)
            if best_choice is not None:
                break

        # --------------------------------------------------
        # Failure fallback
        # --------------------------------------------------
        if best_choice is None:
            # deterministic explicit failure
            return self.encode_action_logits(0, 0, 0, 0)

        slot, rot_id, x, y = best_choice

        return self.encode_action_logits(
            box_slot=slot,
            rot_id=rot_id,
            x=x,
            y=y,
        )
