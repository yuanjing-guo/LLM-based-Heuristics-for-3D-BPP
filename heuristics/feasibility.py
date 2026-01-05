# heuristics/feasibility.py
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

    def __init__(self, X: int, Y: int, H: int, min_support_ratio: float = 0.7):
        self.X = int(X)
        self.Y = int(Y)
        self.H = int(H)
        self.min_support_ratio = float(min_support_ratio)

    def is_within_pallet(self, x: int, y: int, dx: int, dy: int) -> bool:
        """
        HARD constraint:
        Box footprint must be fully inside pallet.

        Correct condition (important!):
          x + dx <= X  and  y + dy <= Y

        Because slicing in Python uses half-open interval [x, x+dx),
        the largest valid x is X-dx (inclusive).
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

        # must have something below, otherwise invalid
        if z - 1 < 0:
            return False

        # NOTE: caller must ensure within pallet, otherwise slicing can be wrong
        support_area = pallet_obs[x:x + dx, y:y + dy, z - 1]

        if support_area.size == 0:
            return False

        support_ratio = float(np.mean(support_area > 0))
        return support_ratio >= self.min_support_ratio

    def is_feasible(self, pallet_obs: np.ndarray, x: int, y: int, dx: int, dy: int, dz: int, z: int) -> bool:
        """
        Final feasibility = intersection of all HARD constraints
        + support proxy.

        This function is the *single source of truth* for:
          - boundary constraints
          - height constraint
          - support constraint
        """
        return (
            self.is_within_pallet(x, y, dx, dy)
            and self.height_ok(z, dz)
            and self.supported(pallet_obs, x, y, dx, dy, z)
        )
