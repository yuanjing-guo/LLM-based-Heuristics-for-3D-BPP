import numpy as np
from heuristics.feasibility import FeasibilityChecker

class BaseHeuristic:
    """
    Base class for all heuristics.

    Responsibilities:
    - Store global geometry info (pallet size, height, bin size)
    - Provide unit-consistent utilities (e.g. box size -> bin size)
    - Define a common interface (__call__)
    """

    name = "base"

    def __init__(
        self,
        N_visible_boxes,
        pallet_size_discrete,
        max_pallet_height,
        bin_size,
    ):
        """
        Args:
            N_visible_boxes: number of visible boxes in buffer
            pallet_size_discrete: (X, Y) in bins
            max_pallet_height: H in bins
            bin_size: size of one bin in meters (e.g. 0.01)
        """
        self.N = int(N_visible_boxes)
        self.X = int(pallet_size_discrete[0])
        self.Y = int(pallet_size_discrete[1])
        self.H = int(max_pallet_height)
        self.bin_size = float(bin_size)
        self.feasibility = FeasibilityChecker(X=self.X, 
                                              Y=self.Y, 
                                              H=self.H,)
    def reset(self):
        """
        Optional: called at the beginning of an episode.
        Override if the heuristic is stateful.
        """
        pass

    # ------------------------------------------------------------
    # Unit conversion utilities
    # ------------------------------------------------------------

    def box_size_to_bins(self, box_size_cm):
        """
        Convert box size from centimeters to discrete bins.

        Args:
            box_size_cm: array-like (dx_cm, dy_cm, dz_cm)

        Returns:
            (dx, dy, dz) in bins (int)
        """
        box_size_cm = np.array(box_size_cm, dtype=np.float32)

        # bin_size is in meters -> convert to cm
        bin_size_cm = self.bin_size * 100.0

        return np.ceil(box_size_cm / bin_size_cm).astype(int)

    # ------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------

    def __call__(self, obs):
        """
        Given observation, return action logits.

        Must be implemented by subclasses.
        """
        raise NotImplementedError
