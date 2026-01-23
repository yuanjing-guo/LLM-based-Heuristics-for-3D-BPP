# heuristics/extreme_point_physics.py
import numpy as np
from heuristics.base import BaseHeuristic


class ExtremePointPhysicsAware(BaseHeuristic):
    """
    Physics-aware Extreme Point heuristic.

    Key modification (vs vanilla EP):
      - EP provides ONLY (x, y)
      - z is ALWAYS recomputed from heightmap envelope
        (same semantics as env.get_target_position_strict)

    Placement score:
      (z, x + y, -support_ratio)
    """

    name = "extreme_point_physics"

    def __init__(self):
        super().__init__()

    # --------------------------------------------------
    # EP extraction (XY only, no z!)
    # --------------------------------------------------
    def extract_extreme_points_xy(self, pallet: np.ndarray):
        """
        Conservative EP extraction:
        return a set of (x, y) that are bottom-left corners
        of flat regions in the heightmap.

        This is essentially:
          EP_xy = {(x,y) | h(x,y) is a local minimum
                             in (-x, -y) directions}
        """
        X, Y, H = pallet.shape
        occ = pallet > 0
        has = occ.any(axis=2)

        occ_rev = occ[..., ::-1]
        first_from_top = occ_rev.argmax(axis=2)
        hmap = np.where(has, H - first_from_top, 0)

        EP_xy = []

        for x in range(X):
            for y in range(Y):
                h = hmap[x, y]

                # block if same-height neighbor exists to the left
                if x > 0 and hmap[x - 1, y] == h:
                    continue
                # block if same-height neighbor exists to the back
                if y > 0 and hmap[x, y - 1] == h:
                    continue

                EP_xy.append((x, y))

        # bottom-left-first ordering
        EP_xy.sort(key=lambda p: (p[0] + p[1], p[0], p[1]))
        return EP_xy

    # --------------------------------------------------
    # Heightmap-based z computation (IDENTICAL to env)
    # --------------------------------------------------
    def compute_z_from_heightmap(self, pallet, x, y, dx, dy):
        """
        Compute placement z as:
          z = highest occupied bin under the footprint + 1
        """
        place_area = pallet[x:x + dx, y:y + dy, :]
        non_zero_mask = np.any(place_area > 0, axis=(0, 1))
        if np.any(non_zero_mask):
            return int(np.max(np.nonzero(non_zero_mask)) + 1)
        return 0

    # --------------------------------------------------
    # Support ratio
    # --------------------------------------------------
    def compute_support_ratio(self, pallet, x, y, dx, dy, z):
        if z == 0:
            return 1.0
        support_area = pallet[x:x + dx, y:y + dy, z - 1]
        if support_area.size == 0:
            return 0.0
        return float(np.mean(support_area > 0))

    # --------------------------------------------------
    # Main heuristic
    # --------------------------------------------------
    def __call__(self, obs: dict) -> np.ndarray:
        pallet = obs["pallet_obs_density"]

        # --------------------------------------------------
        # 0) Select item (online IPS)
        # --------------------------------------------------
        box_slot = 0
        if self.slot_is_empty(obs, box_slot):
            return self.encode_action_logits(0, 0, 0, 0)

        props = self.get_slot_props(obs, box_slot)
        size_bins = self.props_to_size_bins(props)

        # --------------------------------------------------
        # 1) Extract EP in XY only
        # --------------------------------------------------
        EP_xy = self.extract_extreme_points_xy(pallet)

        best_score = None
        best_choice = None  # (rot_id, x, y)

        # --------------------------------------------------
        # 2) EP_xy × rotations
        # --------------------------------------------------
        for (x, y) in EP_xy:
            for rot_id in range(6):
                dx, dy, dz = self.rotate_size_bins(size_bins, rot_id)

                # quick prune
                if dz > self.H:
                    continue
                if not self.feasibility.is_within_pallet(x, y, dx, dy):
                    continue

                # *** CRITICAL FIX ***
                # z is recomputed from heightmap envelope
                z = self.compute_z_from_heightmap(pallet, x, y, dx, dy)

                # full feasibility (boundary + height + support)
                if not self.feasibility.is_feasible(
                    pallet_obs=pallet,
                    x=x, y=y,
                    dx=dx, dy=dy, dz=dz,
                    z=z,
                ):
                    continue

                support = self.compute_support_ratio(
                    pallet, x, y, dx, dy, z
                )

                score = (
                    z,          # primary: lowest layer
                    x + y,      # secondary: corner compactness
                    -support,   # tertiary: stability
                )

                if best_score is None or score < best_score:
                    best_score = score
                    best_choice = (rot_id, x, y)

            # P[0] semantics: first EP_xy with feasible placement wins
            if best_choice is not None:
                break

        # --------------------------------------------------
        # 3) Failure fallback
        # --------------------------------------------------
        if best_choice is None:
            return self.encode_action_logits(box_slot, 0, 0, 0)

        rot_id, x, y = best_choice

        return self.encode_action_logits(
            box_slot=box_slot,
            rot_id=rot_id,
            x=x,
            y=y,
        )
