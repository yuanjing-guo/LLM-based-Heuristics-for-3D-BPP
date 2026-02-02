# heuristics/feasibility.py
from typing import Dict, List, Optional, Tuple
import numpy as np


class FeasibilityChecker:
    """
    Centralized hard-constraint checker for placements.

    Conventions:
      - pallet_obs shape: (X, Y, H)
      - x,y are footprint start indices (lower-left/start corner in discrete grid)
      - dx,dy,dz are box sizes in bins after rotation
      - z is computed placement height index (bin level)
      - Placement occupies [x:x+dx, y:y+dy, z:z+dz)
    """

    def __init__(self, X: int, Y: int, H: int, min_support_ratio: float = 0.8):
        self.X = int(X)
        self.Y = int(Y)
        self.H = int(H)
        self.min_support_ratio = float(min_support_ratio)

    def is_within_pallet(self, x: int, y: int, dx: int, dy: int) -> bool:
        """
        HARD constraint:
        Box footprint must be fully inside pallet.
        """
        x = int(x); y = int(y); dx = int(dx); dy = int(dy)

        if dx <= 0 or dy <= 0:
            return False

        return (
            (x >= 0) and (y >= 0) and
            (x + dx <= self.X) and
            (y + dy <= self.Y)
        )

    def height_ok(self, z: int, dz: int) -> bool:
        """
        HARD constraint:
        Box height must not exceed pallet height.
        """
        z = int(z); dz = int(dz)
        if dz <= 0:
            return False
        return (z >= 0) and (z + dz <= self.H)

    def supported(self, pallet_obs: np.ndarray, x: int, y: int, dx: int, dy: int, z: int) -> bool:
        """
        Support/stability proxy.

        - If z == 0: on the ground -> always supported.
        - Else: check how much of the footprint is supported by the layer below (z-1).

        min_support_ratio is a hard threshold here:
          support_ratio >= min_support_ratio  => OK
        """
        z = int(z)
        if z == 0:
            return True

        if z - 1 < 0:
            return False

        support_area = pallet_obs[x:x + dx, y:y + dy, z - 1]
        if support_area.size == 0:
            return False

        support_ratio = float(np.mean(support_area > 0))
        return support_ratio >= self.min_support_ratio

    def is_feasible(self, pallet_obs: np.ndarray, x: int, y: int, dx: int, dy: int, dz: int, z: int) -> bool:
        """
        Final feasibility = intersection of all HARD constraints + support proxy.
        """
        return (
            self.is_within_pallet(x, y, dx, dy)
            and self.height_ok(z, dz)
            and self.supported(pallet_obs, x, y, dx, dy, z)
        )


class RemovalFeasibilityChecker:
    """
    Centralized checker for REMOVE legality / masks.

    IMPORTANT:
      - This checks GEOMETRIC removability ("top-removable") only.
      - Policy decisions (which one to remove, cooldown strategy, etc.) live in heuristic.
      - But we provide helpers to filter out repeated removals via optional cooldown/history.
    """

    def __init__(self, H: int):
        self.H = int(H)

    @staticmethod
    def _as_fp(fp) -> Tuple[int, int, int, int, int, int]:
        # fp = (x,y,z,dx,dy,dz)
        if fp is None:
            raise ValueError("footprint is None")
        if len(fp) != 6:
            raise ValueError(f"footprint must have len=6, got {len(fp)}: {fp}")
        x, y, z, dx, dy, dz = fp
        return int(x), int(y), int(z), int(dx), int(dy), int(dz)

    def is_top_removable(self, pallet_obs_density: np.ndarray, fp) -> bool:
        """
        True if there is NO occupied voxel above the footprint area.

        This assumes:
          - pallet_obs_density is the truth state.
          - fp corresponds to a committed box on the pallet.
        """
        x, y, z, dx, dy, dz = self._as_fp(fp)

        if dx <= 0 or dy <= 0 or dz <= 0:
            return False

        # Clamp to pallet bounds defensively (never crash on bad fp)
        X, Y, H = pallet_obs_density.shape
        x0 = max(0, min(x, X))
        y0 = max(0, min(y, Y))
        x1 = max(0, min(x + dx, X))
        y1 = max(0, min(y + dy, Y))
        top = int(z + dz)

        # if footprint slice is empty => treat as not removable (bad fp)
        if x1 <= x0 or y1 <= y0:
            return False

        # if it reaches or exceeds top boundary => removable (no space above to block)
        if top >= H:
            return True

        above = pallet_obs_density[x0:x1, y0:y1, top:H]
        return (not np.any(above > 0))

    def compute_removable_mask(
        self,
        pallet_obs_density: np.ndarray,
        footprints: List[Tuple[int, int, int, int, int, int]],
    ) -> np.ndarray:
        """
        Return int8 mask of shape (len(footprints),), where 1 means removable.
        """
        n = int(len(footprints))
        mask = np.zeros((n,), dtype=np.int8)
        for i in range(n):
            try:
                mask[i] = 1 if self.is_top_removable(pallet_obs_density, footprints[i]) else 0
            except Exception:
                mask[i] = 0
        return mask

    def filter_valid_remove_indices(
        self,
        pallet_ids: List[int],
        removable_mask: np.ndarray,
        removed_history: Optional[Dict[int, int]] = None,
        cooldown: int = 0,
        t: int = 0,
    ) -> List[int]:
        """
        Filter indices that are removable AND pass optional cooldown/history rule.

        Params:
          - pallet_ids: aligned with footprints (same indexing as env REMOVE action)
          - removable_mask: int8 array (len==len(pallet_ids))
          - removed_history: dict box_id -> last_removed_step (or any timestamp)
          - cooldown: min steps that must pass before allowing same box_id to be removed again
          - t: current step counter on heuristic side

        Returns:
          list of pallet indices (ints), in natural order (0..n-1). You can reverse it in policy.
        """
        n = int(len(pallet_ids))
        if removable_mask is None or int(removable_mask.size) != n:
            # mismatch => be conservative: no valid removes
            return []

        out: List[int] = []
        for i in range(n):
            if int(removable_mask[i]) != 1:
                continue
            bid = int(pallet_ids[i])

            if removed_history is not None and cooldown > 0:
                last_t = int(removed_history.get(bid, -10**9))
                if int(t) - last_t < int(cooldown):
                    continue

            out.append(int(i))
        return out


class MixedRulebook:
    """
    Optional convenience wrapper:
      - .place : FeasibilityChecker
      - .remove: RemovalFeasibilityChecker

    Useful so env/heuristics both call the SAME code path.
    """

    def __init__(self, X: int, Y: int, H: int, min_support_ratio: float = 0.8):
        self.place = FeasibilityChecker(X=X, Y=Y, H=H, min_support_ratio=min_support_ratio)
        self.remove = RemovalFeasibilityChecker(H=H)
