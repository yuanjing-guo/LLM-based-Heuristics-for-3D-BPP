# heuristics/ems_online.py
import numpy as np
from heuristics.base import BaseHeuristic


class EMSOnline(BaseHeuristic):
    """
    Online EMS-style 3D packing heuristic.

    - Single container (pallet) version
    - EMS derived from heightmap (no explicit EMS splitting tree)
    - Order: deepest-bottom-left-first  -> (z, x, y)
    - Box choice: iterate visible IPS (buffer slots)
    """

    name = "empty_maximal_space"

    def __init__(self):
        super().__init__()

    # --------------------------------------------------
    # EMS extraction from heightmap
    # --------------------------------------------------
    def extract_ems(self, pallet: np.ndarray):
        """
        Extract EMS candidates from heightmap.

        Each EMS is represented as:
            (z, x, y, dx_max, dy_max)

        z  : placement height
        dx : max contiguous free extent in +x
        dy : max contiguous free extent in +y
        """
        X, Y, H = pallet.shape
        ems_list = []

        # heightmap h(x,y)
        occ = pallet > 0
        has = occ.any(axis=2)
        occ_rev = occ[..., ::-1]
        first_from_top = occ_rev.argmax(axis=2)
        hmap = np.where(has, H - first_from_top, 0)

        for x in range(X):
            for y in range(Y):
                z = int(hmap[x, y])

                # grow dx
                dx = 0
                while x + dx < X and hmap[x + dx, y] == z:
                    dx += 1

                # grow dy
                dy = 0
                while y + dy < Y and hmap[x, y + dy] == z:
                    dy += 1

                if dx > 0 and dy > 0:
                    ems_list.append((z, x, y, dx, dy))

        # deepest-bottom-left-first
        ems_list.sort(key=lambda e: (e[0], e[1], e[2]))
        return ems_list

    # --------------------------------------------------
    # Main heuristic
    # --------------------------------------------------
    def __call__(self, obs: dict) -> np.ndarray:
        pallet = obs["pallet_obs_density"]

        # visible items (IPS)
        Kb = self.N

        # EMS list
        ems_list = self.extract_ems(pallet)

        best_choice = None  # (slot, rot_id, x, y)
        best_score = None   # (z, ems_index)

        # --------------------------------------------------
        # Phase 1: try to pack into opened container
        # --------------------------------------------------
        for ems_idx, (z, ex, ey, dx_max, dy_max) in enumerate(ems_list):
            for slot in range(Kb):
                if self.slot_is_empty(obs, slot):
                    continue

                props = self.get_slot_props(obs, slot)
                size_bins = self.props_to_size_bins(props)

                for rot_id in range(6):
                    dx, dy, dz = self.rotate_size_bins(size_bins, rot_id)

                    # footprint must fit EMS rectangle
                    if dx > dx_max or dy > dy_max:
                        continue

                    # full feasibility check
                    if not self.feasibility.is_feasible(
                        pallet_obs=pallet,
                        x=ex, y=ey,
                        dx=dx, dy=dy, dz=dz,
                        z=z,
                    ):
                        continue

                    score = (z, ems_idx)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_choice = (slot, rot_id, ex, ey)

            # stop early like P[0] semantics
            if best_choice is not None:
                break

        # --------------------------------------------------
        # Failure fallback
        # --------------------------------------------------
        if best_choice is None:
            # deterministic but explicit failure
            return self.encode_action_logits(0, 0, 0, 0)

        slot, rot_id, x, y = best_choice

        return self.encode_action_logits(
            box_slot=slot,
            rot_id=rot_id,
            x=x,
            y=y,
        )
