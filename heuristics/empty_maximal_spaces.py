# heuristics/ems_online.py
import numpy as np
from heuristics.base import BaseHeuristic


class EmptyMaximalSpace(BaseHeuristic):
    """
    Online EMS-style 3D packing heuristic (true EMS, top-loading).

    - Single container (pallet)
    - Top-loading (items placed from +z direction)
    - Maintain a list of 3D Empty Maximal Spaces (EMS)
    - After each placement: remove intersecting EMS and split them
    - EMS order: lowest-z-first, then x, then y
    """

    name = "empty_maximal_space"

    def __init__(self):
        super().__init__()
        self.ems_list = None
        self._last_container_shape = None

        # scoring weights (tune if needed)
        self.w_fill = 1.0
        self.w_low = 0.10      # prefer lower z
        self.w_flat = 0.05     # prefer larger footprint usage (top-loading friendly)

    # --------------------------------------------------
    # EMS init / reset
    # --------------------------------------------------
    def _reset_ems(self, pallet: np.ndarray):
        X, Y, H = pallet.shape
        self._last_container_shape = (X, Y, H)
        # one big empty space
        self.ems_list = [(0, 0, 0, X, Y, H)]  # (x, y, z, dx, dy, dz)

    def _maybe_reset(self, pallet: np.ndarray):
        # reset if first call / container shape changed / pallet empty (new episode)
        X, Y, H = pallet.shape
        if self.ems_list is None or self._last_container_shape != (X, Y, H):
            self._reset_ems(pallet)
            return

        if not (pallet > 0).any():
            self._reset_ems(pallet)

    # --------------------------------------------------
    # Geometry helpers
    # --------------------------------------------------
    @staticmethod
    def _intersect_3d(a, b):
        """Check strict overlap between two axis-aligned 3D boxes."""
        ax, ay, az, adx, ady, adz = a
        bx, by, bz, bdx, bdy, bdz = b

        return (
            ax < bx + bdx and ax + adx > bx and
            ay < by + bdy and ay + ady > by and
            az < bz + bdz and az + adz > bz
        )

    @staticmethod
    def _split_ems_by_box(ems, box):
        """
        Split an EMS by removing the overlap region with 'box',
        returning new EMS candidates (up to 6).

        ems: (ex, ey, ez, edx, edy, edz)
        box: (bx, by, bz, bdx, bdy, bdz)
        """
        ex, ey, ez, edx, edy, edz = ems
        bx, by, bz, bdx, bdy, bdz = box

        # overlap (clipped)
        ox1 = max(ex, bx)
        oy1 = max(ey, by)
        oz1 = max(ez, bz)
        ox2 = min(ex + edx, bx + bdx)
        oy2 = min(ey + edy, by + bdy)
        oz2 = min(ez + edz, bz + bdz)

        # if no overlap, keep EMS as-is
        if ox1 >= ox2 or oy1 >= oy2 or oz1 >= oz2:
            return [ems]

        out = []

        # left
        if ox1 > ex:
            out.append((ex, ey, ez, ox1 - ex, edy, edz))
        # right
        if ox2 < ex + edx:
            out.append((ox2, ey, ez, (ex + edx) - ox2, edy, edz))
        # back
        if oy1 > ey:
            out.append((ex, ey, ez, edx, oy1 - ey, edz))
        # front
        if oy2 < ey + edy:
            out.append((ex, oy2, ez, edx, (ey + edy) - oy2, edz))
        # below
        if oz1 > ez:
            out.append((ex, ey, ez, edx, edy, oz1 - ez))
        # above
        if oz2 < ez + edz:
            out.append((ex, ey, oz2, edx, edy, (ez + edz) - oz2))

        # filter invalid
        out = [e for e in out if e[3] > 0 and e[4] > 0 and e[5] > 0]
        return out

    @staticmethod
    def _prune_contained(ems_list):
        """
        Remove EMS that are fully contained in another EMS.
        O(n^2) but usually manageable.
        """
        pruned = []
        for i, a in enumerate(ems_list):
            ax, ay, az, adx, ady, adz = a
            contained = False
            for j, b in enumerate(ems_list):
                if i == j:
                    continue
                bx, by, bz, bdx, bdy, bdz = b
                if (
                    ax >= bx and ay >= by and az >= bz and
                    ax + adx <= bx + bdx and
                    ay + ady <= by + bdy and
                    az + adz <= bz + bdz
                ):
                    contained = True
                    break
            if not contained:
                pruned.append(a)

        # also de-dup (exact tuples)
        pruned = list(dict.fromkeys(pruned))
        return pruned

    def _update_ems_after_place(self, box):
        """
        Update EMS list after placing 'box' by splitting intersecting EMS.
        """
        new_list = []
        for ems in self.ems_list:
            if self._intersect_3d(ems, box):
                new_list.extend(self._split_ems_by_box(ems, box))
            else:
                new_list.append(ems)

        # prune containment to keep EMS maximal-ish
        new_list = self._prune_contained(new_list)

        # stable priority: lowest z, then x, then y
        new_list.sort(key=lambda e: (e[2], e[0], e[1]))
        self.ems_list = new_list

    # --------------------------------------------------
    # Main heuristic
    # --------------------------------------------------
    def __call__(self, obs: dict) -> np.ndarray:
        pallet = obs["pallet_obs_density"]
        Kb = self.N  # visible items (IPS)

        # init/reset EMS if needed
        self._maybe_reset(pallet)

        best_choice = None   # (slot, rot_id, x, y, z, dx, dy, dz)
        best_score = None

        # --------------------------------------------------
        # Search all EMS and all boxes/rotations
        # --------------------------------------------------
        for (ex, ey, ez, edx, edy, edz) in self.ems_list:
            # top-loading: only consider placements at the EMS floor height ez
            z = ez

            for slot in range(Kb):
                if self.slot_is_empty(obs, slot):
                    continue

                props = self.get_slot_props(obs, slot)
                size_bins = self.props_to_size_bins(props)

                for rot_id in range(6):
                    dx, dy, dz = self.rotate_size_bins(size_bins, rot_id)

                    if dx > edx or dy > edy or dz > edz:
                        continue

                    if not self.feasibility.is_feasible(
                        pallet_obs=pallet,
                        x=ex, y=ey,
                        dx=dx, dy=dy, dz=dz,
                        z=z,
                    ):
                        continue

                    # --------------------------------
                    # True EMS scoring (bigger is better)
                    # --------------------------------
                    # volume fill inside EMS
                    fill3d = (dx * dy * dz) / (edx * edy * edz)

                    # prefer low placement height
                    low = -z

                    # top-loading-friendly: prefer using footprint (reduces fragmentation on surface)
                    fill2d = (dx * dy) / (edx * edy)

                    score = (
                        self.w_fill * fill3d +
                        self.w_low * low +
                        self.w_flat * fill2d
                    )

                    if best_score is None or score > best_score:
                        best_score = score
                        best_choice = (slot, rot_id, ex, ey, z, dx, dy, dz)

        # --------------------------------------------------
        # Failure fallback
        # --------------------------------------------------
        if best_choice is None:
            return self.encode_action_logits(0, 0, 0, 0)

        slot, rot_id, x, y, z, dx, dy, dz = best_choice

        # update EMS assuming env will execute this placement
        placed_box = (x, y, z, dx, dy, dz)
        self._update_ems_after_place(placed_box)

        return self.encode_action_logits(
            box_slot=slot,
            rot_id=rot_id,
            x=x,
            y=y,
        )
