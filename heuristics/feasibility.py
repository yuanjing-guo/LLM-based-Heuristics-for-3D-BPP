import numpy as np

class FeasibilityChecker:
    def __init__(self, X, Y, H, min_support_ratio=0.95):
        self.X = X
        self.Y = Y
        self.H = H
        self.min_support_ratio = min_support_ratio

    def is_within_pallet(self, x, y, dx, dy):
        """
        HARD constraint:
        Box footprint must be fully inside pallet.
        """
        return (x >= 0 and y >= 0
                and x + dx < self.X
                and y + dy < self.Y)

    def height_ok(self, z, dz):
        """
        HARD constraint:
        Box height must not exceed pallet height.
        """
        return z >= 0 and (z + dz) <= self.H

    def supported(self, pallet_obs, x, y, dx, dy, z):
        """
        HARD + SOFT support check.
        """
        # Ground is always stable
        if z == 0:
            return True

        support_area = pallet_obs[x:x+dx, y:y+dy, z-1]

        # 1️⃣ HARD: no cell is allowed to be unsupported
        if not np.all(support_area > 0):
            return False

        # 2️⃣ SOFT: continuous stability proxy
        support_ratio = np.mean(support_area > 0)
        if support_ratio < self.min_support_ratio:
            return False

        return True

    def is_feasible(self, pallet_obs, x, y, dx, dy, dz, z):
        """
        Final feasibility = intersection of all HARD constraints
        + minimal stability proxy.
        """
        return (
            self.is_within_pallet(x, y, dx, dy)
            and self.height_ok(z, dz)
            and self.supported(pallet_obs, x, y, dx, dy, z)
        )
