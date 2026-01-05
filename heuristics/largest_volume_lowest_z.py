# heuristics/handcrafted.py

import numpy as np
from heuristics.base import BaseHeuristic
from env import encode_choice_to_action_logits


class LargestVolumeLowestZ(BaseHeuristic):
    """
    Handcrafted baseline (compatible with current env.py):
    - Select largest-volume box (by discrete bin volume after decoding buffer)
    - Place at lowest feasible z (computed from pallet heightmap within the footprint)
    - Tie-break by minimal (x + y)
    - Uses STRICT bounds (x+dx<=X, y+dy<=Y) to avoid env RuntimeError
    """

    name = "largest_volume_lowest_z"

    def __init__(self):
        # New style: BaseHeuristic reads TaskConfig internally
        # and sets geometry like self.N, self.X, self.Y, self.H, self.bin_size, self.n_properties, ...
        super().__init__()

        # Backward-compat fallback if your BaseHeuristic still expects args
        # (won't trigger if BaseHeuristic already sets these)
        if not hasattr(self, "N"):
            raise RuntimeError(
                "BaseHeuristic does not define N/X/Y/H. "
                "Please update BaseHeuristic to read TaskConfig so heuristics can be constructed with no args."
            )

    def _decode_size_bins_from_buffer(self, obs, box_slot: int) -> np.ndarray:
        """
        Your env buffer layout:
          props = [half_size_x*100, half_size_y*100, half_size_z*100, density/1000]
        So:
          half_size_m = v / 100
          full_size_m = half_size_m * 2
          dxyz_bins = int(full_size_m / bin_size)
        """
        buffer = obs["buffer"]
        n_props = int(getattr(self, "n_properties", 4))
        bin_size = float(getattr(self, "bin_size", 1.0))

        base = box_slot * n_props
        half_sizes_m = (buffer[base : base + 3] / 100.0).astype(np.float32)
        full_sizes_m = half_sizes_m * 2.0
        dxyz = (full_sizes_m / bin_size).astype(int)  # same as env.decode_box_discrete_size_from_buffer

        return dxyz  # (dx,dy,dz) in bins

    def __call__(self, obs):
        pallet = obs["pallet_obs_density"]  # (X,Y,H)
        buffer = obs["buffer"]

        # -------------------------------
        # 1) Select box: largest volume
        # -------------------------------
        volumes = np.full(self.N, -1, dtype=np.int64)
        sizes_bins = [None] * self.N

        # determine which slots are non-empty (size entries > 0)
        for i in range(self.N):
            # quick validity check: the first 3 props must be >0 (they are half_size*100)
            base = i * int(getattr(self, "n_properties", 4))
            size_raw = buffer[base : base + 3]
            if not np.all(size_raw > 0):
                continue

            dxyz = self._decode_size_bins_from_buffer(obs, i)
            if np.any(dxyz <= 0):
                continue

            dx, dy, dz = int(dxyz[0]), int(dxyz[1]), int(dxyz[2])
            # If box footprint itself bigger than pallet, it's never placeable
            if dx > self.X or dy > self.Y or dz > self.H:
                continue

            volumes[i] = dx * dy * dz
            sizes_bins[i] = dxyz

        # If no valid box, fallback to 0
        box_i = int(np.argmax(volumes))
        if volumes[box_i] <= 0 or sizes_bins[box_i] is None:
            box_i = 0
            dxyz = self._decode_size_bins_from_buffer(obs, box_i)
        else:
            dxyz = sizes_bins[box_i]

        # ----------------------------------
        # 2) Orientation (fixed rot=0)
        # ----------------------------------
        rot_i = 0
        dx, dy, dz = int(dxyz[0]), int(dxyz[1]), int(dxyz[2])

        # If chosen one still impossible, try other boxes in descending volume
        if dx > self.X or dy > self.Y or dz > self.H:
            # try find any feasible box
            order_idx = np.argsort(-volumes)  # descending
            found = False
            for j in order_idx:
                if volumes[j] <= 0 or sizes_bins[j] is None:
                    continue
                cand = sizes_bins[j]
                cdx, cdy, cdz = int(cand[0]), int(cand[1]), int(cand[2])
                if cdx <= self.X and cdy <= self.Y and cdz <= self.H:
                    box_i = int(j)
                    dx, dy, dz = cdx, cdy, cdz
                    found = True
                    break
            if not found:
                # hard fallback
                box_i, rot_i, x_i, y_i = 0, 0, 0, 0
                return encode_choice_to_action_logits(
                    N=self.N, X=self.X, Y=self.Y,
                    box_i=box_i, rot_i=rot_i, x_i=x_i, y_i=y_i
                )

        # -------------------------------
        # 3) Find best (x, y) under STRICT
        # -------------------------------
        best_score = None
        best_xy = None

        # STRICT: x in [0, X-dx], y in [0, Y-dy]
        max_x = self.X - dx
        max_y = self.Y - dy
        if max_x < 0 or max_y < 0:
            # can't place anywhere
            x_i, y_i = 0, 0
        else:
            for x in range(max_x + 1):
                x2 = x + dx
                for y in range(max_y + 1):
                    y2 = y + dy

                    place_area = pallet[x:x2, y:y2, :]
                    non_zero_mask = np.any(place_area > 0, axis=(0, 1))
                    z = int(np.max(np.nonzero(non_zero_mask)) + 1) if np.any(non_zero_mask) else 0

                    if z + dz > self.H:
                        continue

                    # same tie-break: minimal z, then minimal x+y
                    score = (z, x + y)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_xy = (x, y)

            # if nothing feasible, just put (0,0) (env may later terminate by instability, but won't out-of-bounds)
            if best_xy is None:
                x_i, y_i = 0, 0
            else:
                x_i, y_i = best_xy

        # -------------------------------
        # 4) Encode into action logits
        # -------------------------------
        action = encode_choice_to_action_logits(
            N=self.N,
            X=self.X,
            Y=self.Y,
            box_i=int(box_i),
            rot_i=int(rot_i),
            x_i=int(x_i),
            y_i=int(y_i),
        )
        return action
