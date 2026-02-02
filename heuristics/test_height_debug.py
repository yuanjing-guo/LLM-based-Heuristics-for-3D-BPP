import numpy as np
from heuristics.base import BaseHeuristic


class TestHeightDebug(BaseHeuristic):
    """
    Height-debug heuristic for MixedBoxPlanningEnv (expects int32[5] action):
      action = [op, slot, rot_id, x, y]
      op=0 PLACE, op=1 REMOVE
    We do PLACE only, and choose candidate with MAX (z+dz) but <= H (stress height).
    """

    name = "test_height_debug"

    def __init__(self):
        super().__init__()
        self.eps = 1e-6
        self.print_topk = 5
        self._t = 0

    @staticmethod
    def _act_place(slot: int, rot: int, x: int, y: int) -> np.ndarray:
        return np.array([0, int(slot), int(rot), int(x), int(y)], dtype=np.int32)

    def _compute_z_like_env(self, pallet: np.ndarray, x: int, y: int, dx: int, dy: int) -> int:
        place_area = pallet[x:x + dx, y:y + dy, :]
        non_zero = np.any(place_area > 0, axis=(0, 1))
        return int(np.max(np.nonzero(non_zero)) + 1) if np.any(non_zero) else 0

    def _voxel_empty(self, pallet: np.ndarray, x: int, y: int, z: int, dx: int, dy: int, dz: int) -> bool:
        blk = pallet[x:x+dx, y:y+dy, z:z+dz]
        if blk.size == 0:
            return False
        return not bool(np.any(blk > float(self.eps)))

    def __call__(self, obs: dict) -> np.ndarray:
        self._t += 1
        pallet = obs["pallet_obs_density"]
        n_slots = int(obs["buffer"].size // self.n_properties)

        cand = []  # (score, meta)

        for slot in range(n_slots):
            if self.slot_is_empty(obs, slot):
                continue

            props = self.get_slot_props(obs, slot)
            size_bins = self.props_to_size_bins(props)
            if getattr(size_bins, "size", 0) == 0:
                continue
            if np.any(np.array(size_bins, dtype=int) <= 0):
                continue

            for rot_id in range(6):
                dx, dy, dz = self.rotate_size_bins(size_bins, rot_id)
                dx, dy, dz = int(dx), int(dy), int(dz)
                if dx <= 0 or dy <= 0 or dz <= 0:
                    continue
                if dx > self.X or dy > self.Y or dz > self.H:
                    continue

                for x in range(self.X - dx + 1):
                    for y in range(self.Y - dy + 1):
                        if not self.feasibility.is_within_pallet(x, y, dx, dy):
                            continue

                        z = self._compute_z_like_env(pallet, x, y, dx, dy)

                        # HARD HEIGHT RULE
                        if z + dz > self.H:
                            continue

                        if not self.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                            continue

                        if not self._voxel_empty(pallet, x, y, z, dx, dy, dz):
                            continue

                        top = int(z + dz)

                        # MAX top to stress height; tiebreak to keep deterministic
                        score = (top, z, -(x+y), -rot_id, -slot)
                        meta = (slot, rot_id, x, y, z, dx, dy, dz, top)
                        cand.append((score, meta))

        if not cand:
            print(f"[TestHeightDebug] t={self._t} NO CANDIDATE -> PLACE(0,0,0,0)")
            return self._act_place(0, 0, 0, 0)

        cand.sort(key=lambda it: it[0], reverse=True)

        print(f"\n[TestHeightDebug] t={self._t} candidates={len(cand)} (top {self.print_topk})")
        for i in range(min(self.print_topk, len(cand))):
            _, m = cand[i]
            slot, rot, x, y, z, dx, dy, dz, top = m
            print(f"  rank#{i}: slot={slot} rot={rot} xy=({x},{y}) size=({dx},{dy},{dz}) z={z} top={top} H={self.H}")

        _, best = cand[0]
        slot, rot, x, y, z, dx, dy, dz, top = best
        print(f"[TestHeightDebug] CHOSEN: slot={slot} rot={rot} xy=({x},{y}) dx,dy,dz=({dx},{dy},{dz}) z={z} top={top} H={self.H}")

        # Return RAW action for MixedBoxPlanning
        return self._act_place(slot, rot, x, y)
