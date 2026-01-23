# heuristics/floor_building_buffer_rule_physics.py
import numpy as np
from typing import Tuple

from heuristics.base import BaseHeuristic


class FloorBuildingBufferRulePhysics(BaseHeuristic):
    """
    Buffer-aware floor-building heuristic (layer-by-layer packing),
    RULE + PHYSICS-AWARE (balanced mode), Python 3.8 compatible.

    - scans all non-empty buffer slots
    - tries all 6 rotations
    - scans all x,y
    - feasibility (hard constraints) is still ONLY by FeasibilityChecker
    - scoring is dominated by a physics risk proxy:
        * support_ratio (area support below footprint)
        * support_softness (mean softness under footprint on supporting layer)
        * height penalty (z/H), weighted by density (heavier -> more penalty)
        * box's own softness penalty (heavier+softer -> more risky)

    Space-util bias:
      - prefer left-bottom corner (x+y small)
      - NO "towards center" term
    """

    name = "floor_building_buffer_rule_physics"

    # -------------------------
    # Tunable weights (balanced)
    # -------------------------
    W_PHYS = 10000.0
    W_Z = 500.0
    W_XY = 1.0

    W_HEIGHT = 800.0
    W_SUPPORT_LACK = 25.0
    W_SUPPORT_SOFT = 15.0
    W_BOX_SOFT = 1.0

    EPS = 1e-6

    def __init__(self):
        super().__init__()

    # -------------------------
    # Helpers
    # -------------------------
    def _get_slot_density(self, props: np.ndarray) -> float:
        # props layout: [half_x_cm, half_y_cm, half_z_cm, density_scaled]
        return float(props[3])

    def _get_slot_physics(self, obs: dict, slot_i: int) -> Tuple[float, float]:
        """
        Read (softness, mu) from obs["buffer_physics"].
        If field missing -> return zeros (backward compatible).
        buffer_physics layout per slot: [softness, friction_mu]
        """
        if "buffer_physics" not in obs:
            return 0.0, 0.0

        bufp = np.asarray(obs["buffer_physics"], dtype=np.float32)
        base = slot_i * 2
        if base + 1 >= bufp.shape[0]:
            return 0.0, 0.0

        softness = float(bufp[base + 0])
        mu = float(bufp[base + 1])
        return softness, mu

    def _support_stats(
        self,
        obs: dict,
        pallet: np.ndarray,
        x: int, y: int,
        dx: int, dy: int,
        z: int,
    ) -> Tuple[float, float]:
        """
        Returns:
          support_ratio: fraction of footprint cells that have something directly below
          support_soft : mean softness under supported cells (if available)
        """
        if z <= 0:
            return 1.0, 0.0

        occ = (pallet[x:x + dx, y:y + dy, z - 1] > 0)
        support_ratio = float(occ.mean())

        if "pallet_obs_softness" not in obs:
            return support_ratio, 0.0

        soft_layer = np.asarray(obs["pallet_obs_softness"][x:x + dx, y:y + dy, z - 1], dtype=np.float32)

        # consider softness only where there is actual support, else average footprint
        if np.any(occ):
            support_soft = float(np.mean(soft_layer[occ]))
        else:
            support_soft = float(np.mean(soft_layer))

        return support_ratio, support_soft

    def _phys_cost(
        self,
        density: float,
        box_softness: float,
        z: int,
        support_ratio: float,
        support_soft: float,
    ) -> float:
        """
        density is density/1000 (from env buffer), typical ~0.5 (light) to ~5.0 (heavy)
        """
        z_norm = float(z) / float(max(self.H, 1))

        cost_height = density * z_norm
        cost_support_lack = density * (1.0 - support_ratio)
        cost_support_soft = density * support_soft
        cost_box_soft = density * box_softness

        phys = (
            self.W_HEIGHT * cost_height +
            self.W_SUPPORT_LACK * cost_support_lack +
            self.W_SUPPORT_SOFT * cost_support_soft +
            self.W_BOX_SOFT * cost_box_soft
        )
        return float(phys)

    # -------------------------
    # Main
    # -------------------------
    def __call__(self, obs: dict) -> np.ndarray:
        pallet = np.asarray(obs["pallet_obs_density"], dtype=np.float32)

        # candidate slots
        candidate_slots = [i for i in range(self.N) if not self.slot_is_empty(obs, i)]
        if not candidate_slots:
            return self.encode_action_logits(box_slot=0, rot_id=0, x=0, y=0)

        best_score = None
        best_choice = None  # (box_slot, rot_id, x, y)

        for box_slot in candidate_slots:
            props = self.get_slot_props(obs, box_slot)
            size_bins = self.props_to_size_bins(props)
            density = self._get_slot_density(props)
            box_softness, _mu = self._get_slot_physics(obs, box_slot)

            for rot_id in range(6):
                dx, dy, dz = self.rotate_size_bins(size_bins, rot_id)

                if dz > self.H:
                    continue

                for x in range(self.X):
                    for y in range(self.Y):
                        if not self.feasibility.is_within_pallet(x, y, dx, dy):
                            continue

                        # compute z from occupancy
                        place_area = pallet[x:x + dx, y:y + dy, :]
                        non_zero_mask = np.any(place_area > 0, axis=(0, 1))
                        z = int(np.max(np.nonzero(non_zero_mask)) + 1) if np.any(non_zero_mask) else 0

                        if not self.feasibility.is_feasible(
                            pallet_obs=pallet,
                            x=x, y=y,
                            dx=dx, dy=dy, dz=dz,
                            z=z,
                        ):
                            continue

                        support_ratio, support_soft = self._support_stats(obs, pallet, x, y, dx, dy, z)
                        phys = self._phys_cost(density, box_softness, z, support_ratio, support_soft)

                        score = (
                            self.W_PHYS * phys +
                            self.W_Z * float(z) +
                            self.W_XY * float(x + y)
                        )

                        if best_score is None or score < best_score:
                            best_score = score
                            best_choice = (box_slot, rot_id, x, y)

        if best_choice is None:
            s = candidate_slots[0]
            return self.encode_action_logits(box_slot=s, rot_id=0, x=0, y=0)

        box_slot, rot_id, x, y = best_choice
        return self.encode_action_logits(box_slot=box_slot, rot_id=rot_id, x=x, y=y)
