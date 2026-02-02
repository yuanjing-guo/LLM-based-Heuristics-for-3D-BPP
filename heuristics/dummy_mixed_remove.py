import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import deque, Counter
import heapq

from heuristics.base import BaseHeuristic


class DummyMixedRemove(BaseHeuristic):
    """
    HARD RULE:
      - HARD boxes must NOT be placed on SOFT support.
      - SOFT boxes are allowed.

    Behavior (kept from your version):
      - If best HARD placement sits on SOFT support: try targeted REMOVE of supporting SOFT box.
      - pending is NOT cleared by SOFT placement.
      - pending期间，SOFT不允许放回刚刚remove的区域（防止振荡）。
      - Circuit breaker remains.

    IMPORTANT FIX (your latest bug):
      - ADD strict voxel non-overlap check EVERYWHERE (hard/soft/fallback).
        We do NOT trust feasibility.is_feasible() to enforce non-overlap.

    FINAL FALLBACK (your request):
      - fallback 不再“拆家”，而是 FORCE PLACE
      - BUT: fallback 也必须满足基本 feasibility + height + NON-OVERLAP
      - If truly nothing feasible exists, we do conservative PLACE(0,0,0,0) as absolute last resort.
    """

    name = "wasted_space_physics_mixed"

    # ---------------- actions ----------------
    @staticmethod
    def act_place(slot: int, rot_id: int, x: int, y: int) -> np.ndarray:
        return np.array([0, int(slot), int(rot_id), int(x), int(y)], dtype=np.int32)

    @staticmethod
    def act_remove(pallet_index: int) -> np.ndarray:
        return np.array([1, int(pallet_index), 0, 0, 0], dtype=np.int32)

    def __init__(self):
        super().__init__()
        self._t = 0

        # classify soft/hard by buffer_physics softness
        self.softness_thr = 0.5  # >thr => soft

        # replacement constraint after remove
        self._pending_fp: Optional[Tuple[int, int, int, int, int, int]] = None  # (x,y,z,dx,dy,dz)
        self._pending_vol: int = 0
        self._pending_active: bool = False
        self._pending_fail_steps: int = 0
        self._pending_fail_limit: int = 8

        # remove throttles
        self.remove_cooldown_steps = 3
        self.min_steps_between_removes = 1
        self._last_remove_step = -10**9
        self._removed_history: Dict[int, int] = {}  # box_id -> last removed step

        # removal ban list for circuit breaker (idx -> ban_until_t)
        self._ban_remove_until: Dict[int, int] = {}

        # HARD search pause for circuit breaker
        self._hard_pause_until: int = -1

        # search budget
        self.max_xy_samples = 0  # 0 => full scan, >0 => sampled grid per dim

        # scoring weights
        self.w_z = 0.30
        self.w_xy = 0.01
        self.w_vol = 1.0

        # IMPORTANT: non-overlap eps (soft mode may leave tiny residues; 1e-6 works if empty truly 0)
        self.eps = 1e-6

        # keep top-k candidates
        self.topk_hard = 25
        self.topk_soft = 25

        # debug + oscillation breaker config
        self._dbg_init()
        self._osc_init()

    # ---------------- debug config ----------------
    def _dbg_init(self):
        self.debug = True
        self.debug_every = 1
        self.debug_topk_hist = 12

        self._dbg_last_events = deque(maxlen=20)
        self._dbg_last_actions = deque(maxlen=30)  # (t, op, idx/slot, x, y, z)

    def _dbg(self, msg: str):
        if getattr(self, "debug", False):
            print(msg)

    def _dbg_event(self, msg: str):
        if not getattr(self, "debug", False):
            return
        self._dbg_last_events.append((int(self._t), str(msg)))
        print(f"[HeurDbg][EVENT] {msg}")

    # ---------------- oscillation breaker config ----------------
    def _osc_init(self):
        self.osc_window = 12
        self.osc_repeat_threshold = 3
        self.osc_ban_steps = 25
        self.osc_hard_pause_steps = 18
        self._osc_recent_removes = deque(maxlen=80)

    def _osc_record_remove(self, idx: int):
        self._osc_recent_removes.append((int(self._t), int(idx)))

    def _osc_check_and_trip(self, idx: int) -> bool:
        now = int(self._t)
        idx = int(idx)

        c = 0
        for (tt, ii) in self._osc_recent_removes:
            if (now - int(tt)) <= int(self.osc_window) and int(ii) == idx:
                c += 1

        if c >= int(self.osc_repeat_threshold):
            ban_until = now + int(self.osc_ban_steps)
            self._ban_remove_until[idx] = max(int(self._ban_remove_until.get(idx, -1)), ban_until)

            # clear pending immediately
            self._pending_active = False
            self._pending_fp = None
            self._pending_vol = 0
            self._pending_fail_steps = 0

            # pause HARD search
            self._hard_pause_until = max(int(self._hard_pause_until), now + int(self.osc_hard_pause_steps))

            self._dbg_event(
                f"OSC BREAKER TRIPPED: remove idx={idx} repeated {c}x within {self.osc_window} steps "
                f"-> BAN until t={ban_until}, clear pending, HARD pause until t={self._hard_pause_until}"
            )
            return True
        return False

    # ---------------- basic helpers ----------------
    def _n_slots(self, obs: dict) -> int:
        return int(obs["buffer"].size // self.n_properties)

    def _slot_softness_mu(self, obs: dict, slot: int) -> Tuple[float, float]:
        buf_phy = obs.get("buffer_physics", None)
        if buf_phy is None:
            return 0.0, 0.0
        base = int(slot) * 2
        if base + 1 >= buf_phy.size:
            return 0.0, 0.0
        return float(buf_phy[base + 0]), float(buf_phy[base + 1])

    def _is_soft_slot(self, obs: dict, slot: int) -> bool:
        s, _ = self._slot_softness_mu(obs, slot)
        return float(s) > float(self.softness_thr)

    def _is_hard_slot(self, obs: dict, slot: int) -> bool:
        return not self._is_soft_slot(obs, slot)

    def _compute_z_like_env(self, pallet: np.ndarray, x: int, y: int, dx: int, dy: int) -> int:
        place_area = pallet[x:x + dx, y:y + dy, :]
        non_zero = np.any(place_area > 0, axis=(0, 1))
        if np.any(non_zero):
            return int(np.max(np.nonzero(non_zero)) + 1)
        return 0

    def _support_softness_mean(self, obs: dict, x: int, y: int, z: int, dx: int, dy: int) -> float:
        if z <= 0:
            return 0.0
        soft = obs.get("pallet_obs_softness", None)
        if soft is None:
            return 0.0
        vals = soft[x:x + dx, y:y + dy, int(z - 1)]
        return float(np.mean(vals)) if vals.size > 0 else 0.0

    def _current_top_height(self, obs: dict) -> int:
        fps = obs.get("pallet_footprints", []) or []
        top = 0
        for (x, y, z, dx, dy, dz) in fps:
            top = max(top, int(z) + int(dz))
        return int(top)

    # -------------- XY iterator --------------
    def _iter_xy(self, dx: int, dy: int):
        max_x = int(self.X - dx)
        max_y = int(self.Y - dy)
        if max_x < 0 or max_y < 0:
            return
        if self.max_xy_samples and self.max_xy_samples > 0:
            k = int(self.max_xy_samples)
            xs = np.linspace(0, max_x, num=min(k, max_x + 1), dtype=int)
            ys = np.linspace(0, max_y, num=min(k, max_y + 1), dtype=int)
            for x in xs:
                for y in ys:
                    yield int(x), int(y)
        else:
            for x in range(max_x + 1):
                for y in range(max_y + 1):
                    yield int(x), int(y)

    # -------------- after-remove constraint --------------
    def _passes_pending_constraint(self, dx: int, dy: int, dz: int) -> bool:
        if not self._pending_active or (self._pending_fp is None):
            return True
        _, _, _, rdx, rdy, rdz = self._pending_fp
        dx = int(dx); dy = int(dy); dz = int(dz)
        rdx = int(rdx); rdy = int(rdy); rdz = int(rdz)
        vol = int(dx * dy * dz)
        if vol > int(self._pending_vol):
            return False
        if (dx > rdx) or (dy > rdy) or (dz > rdz):
            return False
        return True

    def _overlaps_pending(self, x, y, z, dx, dy, dz) -> bool:
        if (not self._pending_active) or (self._pending_fp is None):
            return False
        px, py, pz, pdx, pdy, pdz = [int(v) for v in self._pending_fp]
        x = int(x); y = int(y); z = int(z)
        dx = int(dx); dy = int(dy); dz = int(dz)

        if (x + dx <= px) or (px + pdx <= x):
            return False
        if (y + dy <= py) or (py + pdy <= y):
            return False
        if (z + dz <= pz) or (pz + pdz <= z):
            return False
        return True

    # -------------- pallet soft box detection (for remove decisions) --------------
    def _is_soft_fp(self, obs: dict, fp: Tuple[int, int, int, int, int, int]) -> bool:
        soft = obs.get("pallet_obs_softness", None)
        if soft is None:
            return False
        x, y, z, dx, dy, dz = [int(v) for v in fp]
        blk = soft[x:x + dx, y:y + dy, z:z + dz]
        if blk.size == 0:
            return False
        return float(np.mean(blk)) > float(self.softness_thr)

    def _is_banned_remove_idx(self, idx: int) -> bool:
        until = int(self._ban_remove_until.get(int(idx), -1))
        return int(self._t) < until

    def _valid_removal_indices(self, obs: dict) -> List[int]:
        fps = obs.get("pallet_footprints", [])
        ids = obs.get("pallet_ids", [])
        mask = obs.get("removable_mask", None)
        if (not fps) or (mask is None) or (not ids):
            return []
        mask = np.array(mask, dtype=np.int8).reshape(-1)
        n = min(len(fps), len(ids), int(mask.size))
        out = []
        for i in range(n):
            if int(mask[i]) <= 0:
                continue
            if self._is_banned_remove_idx(i):
                continue
            bid = int(ids[i])
            last_t = int(self._removed_history.get(bid, -10**9))
            if (self._t - last_t) < int(self.remove_cooldown_steps):
                continue
            out.append(int(i))
        return out

    def _choose_soft_support_remove(self, obs: dict, x: int, y: int, z: int, dx: int, dy: int) -> Optional[int]:
        fps = obs.get("pallet_footprints", []) or []
        ids = obs.get("pallet_ids", []) or []
        mask = obs.get("removable_mask", None)
        if mask is None:
            return None
        mask = np.array(mask, dtype=np.int8).reshape(-1)
        n = min(len(fps), len(ids), int(mask.size))
        if n <= 0:
            return None

        best = None
        for i in range(n):
            if int(mask[i]) <= 0:
                continue
            if self._is_banned_remove_idx(i):
                continue

            fp = fps[i]
            fx, fy, fz, fdx, fdy, fdz = [int(v) for v in fp]
            top = int(fz + fdz)

            if top != int(z):
                continue
            if (fx + fdx <= x) or (x + dx <= fx) or (fy + fdy <= y) or (y + dy <= fy):
                continue
            if not self._is_soft_fp(obs, fp):
                continue

            bid = int(ids[i])
            last_t = int(self._removed_history.get(bid, -10**9))
            if (self._t - last_t) < int(self.remove_cooldown_steps):
                continue

            ox = min(fx + fdx, x + dx) - max(fx, x)
            oy = min(fy + fdy, y + dy) - max(fy, y)
            overlap = int(max(0, ox) * max(0, oy))

            key = (-overlap, -top, int(i))
            if best is None or key < best[0]:
                best = (key, int(i))

        return None if best is None else int(best[1])

    # ---------------- debug printing helpers ----------------
    def _slot_density(self, obs: dict, slot: int) -> float:
        base = int(slot) * int(self.n_properties)
        if base + 3 >= obs["buffer"].size:
            return 0.0
        return float(obs["buffer"][base + 3])

    def _dbg_counts(self, obs: dict):
        n = self._n_slots(obs)
        nonempty = 0
        hard = 0
        soft = 0
        for s in range(n):
            if self.slot_is_empty(obs, s):
                continue
            nonempty += 1
            if self._is_soft_slot(obs, s):
                soft += 1
            else:
                hard += 1
        return nonempty, hard, soft

    def _dbg_buffer_hist(self, obs: dict) -> Dict[str, int]:
        n = self._n_slots(obs)
        c = Counter()
        for s in range(n):
            if self.slot_is_empty(obs, s):
                continue
            dens = self._slot_density(obs, s)
            softness, _ = self._slot_softness_mu(obs, s)
            dens_q = float(np.round(dens, 3))
            soft_q = float(np.round(softness, 3))
            c[f"d={dens_q},soft={soft_q}"] += 1
        out = {}
        for k, v in c.most_common(int(self.debug_topk_hist)):
            out[k] = int(v)
        return out

    def _dbg_pending_str(self) -> str:
        if not self._pending_active or (self._pending_fp is None):
            return "OFF"
        _, _, _, rdx, rdy, rdz = self._pending_fp
        return f"ON(vol<={self._pending_vol}, size<={rdx}x{rdy}x{rdz}, fail={self._pending_fail_steps}/{self._pending_fail_limit})"

    def _dbg_pallet_stats(self, obs: dict):
        fps = obs.get("pallet_footprints", []) or []
        mask = obs.get("removable_mask", None)
        removable = 0
        if mask is not None:
            m = np.array(mask, dtype=np.int8).reshape(-1)
            removable = int(np.sum(m > 0))
        topH = self._current_top_height(obs)
        return len(fps), topH, removable

    def _dbg_plan_str(self, obs: dict, meta, need_remove_idx=None, tag="HARD") -> str:
        if meta is None:
            return f"[HeurDbg] best_{tag}: None"
        slot, rot, x, y, z, dx, dy, dz = meta
        support_soft = self._support_softness_mean(obs, x, y, z, dx, dy)
        base = f"[HeurDbg] best_{tag}: slot={slot} rot={rot} xy=({x},{y}) z={z} size={dx}x{dy}x{dz} support_soft={support_soft:.3f}"
        if need_remove_idx is not None:
            base += f" -> NEED_REMOVE idx={int(need_remove_idx)}"
        return base

    def _dbg_step_header(self, obs: dict):
        nonempty, hard, soft = self._dbg_counts(obs)
        fps_n, topH, removable = self._dbg_pallet_stats(obs)
        ban_cnt = sum(1 for _, until in self._ban_remove_until.items() if int(self._t) < int(until))
        hard_paused = int(self._t) < int(self._hard_pause_until)
        self._dbg(
            f"[HeurDbg] t={int(self._t)}  buffer(nonempty={nonempty}, hard={hard}, soft={soft})  "
            f"pallet(fps={fps_n}, topH={topH}, removable={removable})  pending={self._dbg_pending_str()}  "
            f"ban_active={ban_cnt}  hard_paused={hard_paused}"
        )
        self._dbg(f"[HeurDbg] buffer_hist: {self._dbg_buffer_hist(obs)}")

    def _dbg_record_action(self, obs: dict, act: np.ndarray):
        if not getattr(self, "debug", False):
            return
        a = np.array(act, dtype=np.int32).reshape(-1)
        if a.size < 5:
            return
        op = int(a[0])
        if op == 1:
            idx = int(a[1])
            self._dbg_last_actions.append((int(self._t), "REMOVE", idx, None, None, None))
        else:
            slot, rot, x, y = int(a[1]), int(a[2]), int(a[3]), int(a[4])
            self._dbg_last_actions.append((int(self._t), "PLACE", slot, x, y, None))

    def _print_action_physics(self, obs: dict, action: np.ndarray):
        try:
            a = np.array(action, dtype=np.int32).reshape(-1)
            if a.size < 5:
                return
            op = int(a[0])

            if op == 0:
                slot = int(a[1]); rot_id = int(a[2]); x = int(a[3]); y = int(a[4])
                dens = self._slot_density(obs, slot)
                softness, mu = self._slot_softness_mu(obs, slot)
                hardness = 1.0 - float(softness)
                print(
                    f"[HeurPhysics][PLACE] slot={slot} rot={rot_id} xy=({x},{y}) "
                    f"density={dens:.4f} softness={softness:.4f} hardness={hardness:.4f} mu={mu:.4f}"
                )

            elif op == 1:
                idx = int(a[1])
                fps = obs.get("pallet_footprints", []) or []
                ids = obs.get("pallet_ids", []) or []

                if 0 <= idx < len(fps):
                    x, y, z, dx, dy, dz = [int(v) for v in fps[idx]]
                    dens_grid = obs.get("pallet_obs_density", None)
                    soft_grid = obs.get("pallet_obs_softness", None)

                    mean_dens = 0.0
                    mean_soft = 0.0
                    if dens_grid is not None:
                        blk = dens_grid[x:x+dx, y:y+dy, z:z+dz]
                        if blk.size > 0:
                            mean_dens = float(np.mean(blk))
                    if soft_grid is not None:
                        blk2 = soft_grid[x:x+dx, y:y+dy, z:z+dz]
                        if blk2.size > 0:
                            mean_soft = float(np.mean(blk2))

                    bid = int(ids[idx]) if idx < len(ids) else -1
                    hardness = 1.0 - float(mean_soft)

                    print(
                        f"[HeurPhysics][REMOVE] idx={idx} box_id={bid} fp=(x={x},y={y},z={z},dx={dx},dy={dy},dz={dz}) "
                        f"mean_density={mean_dens:.4f} mean_softness={mean_soft:.4f} mean_hardness={hardness:.4f}"
                    )
                else:
                    print(f"[HeurPhysics][REMOVE] idx={idx} (no footprint info)")

        except Exception as e:
            print(f"[HeurPhysics][WARN] failed to print physics info: {e}")

    # ---------------- strict non-overlap check ----------------
    def _voxel_empty(self, pallet: np.ndarray, x: int, y: int, z: int, dx: int, dy: int, dz: int) -> bool:
        """
        True iff the volume block is strictly empty in pallet_obs_density.
        We treat ANY value > eps as occupied to prevent overlaps.
        """
        x = int(x); y = int(y); z = int(z)
        dx = int(dx); dy = int(dy); dz = int(dz)
        blk = pallet[x:x+dx, y:y+dy, z:z+dz]
        if blk.size == 0:
            return False
        return not bool(np.any(blk > float(self.eps)))

    # ---------------- Top-K helper ----------------
    def _push_topk(self, heap: list, item, k: int):
        sc, payload = item
        sc_neg = tuple(-int(v) for v in sc)

        if not hasattr(self, "_heap_counter"):
            self._heap_counter = 0
        self._heap_counter += 1
        tie = int(self._heap_counter)

        entry = (sc_neg, tie, sc, payload)
        if len(heap) < int(k):
            heapq.heappush(heap, entry)
        else:
            if sc_neg > heap[0][0]:
                heapq.heapreplace(heap, entry)

    def _topk_hard_places(self, obs: dict) -> List[Tuple[Tuple, Optional[int], Tuple]]:
        pallet = obs["pallet_obs_density"]
        n_slots = self._n_slots(obs)

        heap = []

        for slot in range(n_slots):
            if self.slot_is_empty(obs, slot):
                continue
            if not self._is_hard_slot(obs, slot):
                continue

            props = self.get_slot_props(obs, slot)
            size_bins = self.props_to_size_bins(props)
            if getattr(size_bins, "size", 0) == 0:
                continue

            for rot_id in range(6):
                dx, dy, dz = self.rotate_size_bins(size_bins, rot_id)
                dx, dy, dz = int(dx), int(dy), int(dz)
                if dx <= 0 or dy <= 0 or dz <= 0 or dz > self.H:
                    continue

                if not self._passes_pending_constraint(dx, dy, dz):
                    continue

                for x, y in self._iter_xy(dx, dy):
                    if not self.feasibility.is_within_pallet(x, y, dx, dy):
                        continue

                    z = self._compute_z_like_env(pallet, x, y, dx, dy)
                    if z + dz > self.H:
                        continue

                    # base feasibility
                    if not self.feasibility.is_feasible(
                        pallet_obs=pallet, x=x, y=y, dx=dx, dy=dy, dz=dz, z=z
                    ):
                        continue

                    # STRICT: voxel non-overlap
                    if not self._voxel_empty(pallet, x, y, z, dx, dy, dz):
                        continue

                    support_soft = self._support_softness_mean(obs, x, y, z, dx, dy)

                    need_remove = None
                    if (z > 0) and (support_soft > self.softness_thr):
                        need_remove = self._choose_soft_support_remove(obs, x, y, z, dx, dy)
                        if need_remove is None:
                            continue

                    vol = int(dx * dy * dz)

                    pending_dist = 0
                    if self._pending_active and (self._pending_fp is not None):
                        px, py, _, _, _, _ = self._pending_fp
                        pending_dist = abs(int(x) - int(px)) + abs(int(y) - int(py))

                    sc = (
                        int(-vol),
                        int(self.w_z * 1000 * z),
                        int(self.w_xy * 1000 * (x + y)),
                        int(pending_dist),
                        int(rot_id),
                    )
                    meta = (slot, rot_id, x, y, z, dx, dy, dz)
                    self._push_topk(heap, (sc, (need_remove, meta)), self.topk_hard)

        out = [(sc, payload[0], payload[1]) for (_, _, sc, payload) in heap]
        out.sort(key=lambda t: t[0])
        return out

    def _topk_soft_places(self, obs: dict) -> List[Tuple[Tuple, Tuple]]:
        pallet = obs["pallet_obs_density"]
        n_slots = self._n_slots(obs)

        heap = []

        for slot in range(n_slots):
            if self.slot_is_empty(obs, slot):
                continue
            if not self._is_soft_slot(obs, slot):
                continue

            props = self.get_slot_props(obs, slot)
            size_bins = self.props_to_size_bins(props)
            if getattr(size_bins, "size", 0) == 0:
                continue

            for rot_id in range(6):
                dx, dy, dz = self.rotate_size_bins(size_bins, rot_id)
                dx, dy, dz = int(dx), int(dy), int(dz)
                if dx <= 0 or dy <= 0 or dz <= 0 or dz > self.H:
                    continue

                if not self._passes_pending_constraint(dx, dy, dz):
                    continue

                for x, y in self._iter_xy(dx, dy):
                    if not self.feasibility.is_within_pallet(x, y, dx, dy):
                        continue

                    z = self._compute_z_like_env(pallet, x, y, dx, dy)
                    if z + dz > self.H:
                        continue

                    if not self.feasibility.is_feasible(
                        pallet_obs=pallet, x=x, y=y, dx=dx, dy=dy, dz=dz, z=z
                    ):
                        continue

                    # STRICT: voxel non-overlap
                    if not self._voxel_empty(pallet, x, y, z, dx, dy, dz):
                        continue

                    # don't place back into pending region
                    if self._overlaps_pending(x, y, z, dx, dy, dz):
                        continue

                    # OPTIONAL: don't put soft on soft support
                    support_soft = self._support_softness_mean(obs, x, y, z, dx, dy)
                    if (z > 0) and (support_soft > self.softness_thr):
                        continue

                    vol = int(dx * dy * dz)
                    sc = (
                        int(-vol),
                        int(self.w_z * 1000 * z),
                        int(self.w_xy * 1000 * (x + y)),
                        int(rot_id),
                    )
                    meta = (slot, rot_id, x, y, z, dx, dy, dz)
                    self._push_topk(heap, (sc, meta), self.topk_soft)

        out = [(sc, payload) for (_, _, sc, payload) in heap]
        out.sort(key=lambda t: t[0])
        return out

    # ---------------- FORCE PLACE (feasible + height-safe + non-overlap) ----------------
    def _force_place_any_feasible(self, obs: dict) -> Optional[np.ndarray]:
        """
        Find ANY feasible placement (including hard-on-soft) that:
          - within pallet
          - z computed by env-like
          - z+dz <= H
          - feasibility.is_feasible == True
          - strict voxel non-overlap == True

        Preference:
          - try HARD slots first (so you keep progressing on hard inventory)
          - then SOFT slots
          - scan rot_id 0..5
          - scan XY in your iterator order
        """
        pallet = obs["pallet_obs_density"]
        n_slots = self._n_slots(obs)

        # two-pass: hard first, then soft
        for pass_id in [0, 1]:
            for slot in range(n_slots):
                if self.slot_is_empty(obs, slot):
                    continue

                is_soft = self._is_soft_slot(obs, slot)
                if pass_id == 0 and is_soft:
                    continue
                if pass_id == 1 and (not is_soft):
                    continue

                props = self.get_slot_props(obs, slot)
                size_bins = self.props_to_size_bins(props)
                if getattr(size_bins, "size", 0) == 0:
                    continue

                for rot_id in range(6):
                    dx, dy, dz = self.rotate_size_bins(size_bins, rot_id)
                    dx, dy, dz = int(dx), int(dy), int(dz)
                    if dx <= 0 or dy <= 0 or dz <= 0 or dz > self.H:
                        continue

                    for x, y in self._iter_xy(dx, dy):
                        if not self.feasibility.is_within_pallet(x, y, dx, dy):
                            continue

                        z = self._compute_z_like_env(pallet, x, y, dx, dy)
                        if z + dz > self.H:
                            continue

                        if not self.feasibility.is_feasible(
                            pallet_obs=pallet, x=x, y=y, dx=dx, dy=dy, dz=dz, z=z
                        ):
                            continue

                        # STRICT non-overlap
                        if not self._voxel_empty(pallet, x, y, z, dx, dy, dz):
                            continue

                        self._dbg_event(
                            f"FORCE PLACE (feasible+height+no-overlap) slot={slot} rot={rot_id} "
                            f"xy=({x},{y}) z={z} size={dx}x{dy}x{dz} pass={'HARD' if pass_id==0 else 'SOFT'}"
                        )
                        return self.act_place(slot, rot_id, x, y)

        return None

    # ---------------- main policy ----------------
    def __call__(self, obs: dict) -> np.ndarray:
        self._t += 1

        fps = obs.get("pallet_footprints", []) or []
        ids = obs.get("pallet_ids", []) or []

        # debug header
        if getattr(self, "debug", False) and (int(self._t) % int(self.debug_every) == 0):
            self._dbg_step_header(obs)

        # pending housekeeping
        if self._pending_active:
            self._pending_fail_steps += 1
            if self._pending_fail_steps >= int(self._pending_fail_limit):
                self._dbg_event("pending auto-cleared by fail_limit")
                self._pending_active = False
                self._pending_fp = None
                self._pending_vol = 0
                self._pending_fail_steps = 0

        hard_paused = int(self._t) < int(self._hard_pause_until)

        # =========================
        # 1) HARD top-k attempt
        # =========================
        hard_list = self._topk_hard_places(obs)
        if hard_paused and hard_list:
            hard_list = [(sc, need_rm, meta) for (sc, need_rm, meta) in hard_list if need_rm is None]

        if getattr(self, "debug", False) and (int(self._t) % int(self.debug_every) == 0):
            if hard_list:
                _, need_rm, meta = hard_list[0]
                self._dbg(self._dbg_plan_str(obs, meta, need_remove_idx=need_rm, tag="HARD"))
            else:
                self._dbg("[HeurDbg] best_HARD: None")

        for sc, need_remove_idx, meta in hard_list:
            slot, rot_id, x, y, z, dx, dy, dz = meta

            # need REMOVE first
            if need_remove_idx is not None:
                ridx = int(need_remove_idx)
                if self._is_banned_remove_idx(ridx):
                    continue
                if (self._t - self._last_remove_step) < int(self.min_steps_between_removes):
                    continue

                # set pending from removed footprint
                if ridx < len(fps):
                    rx, ry, rz, rdx, rdy, rdz = [int(v) for v in fps[ridx]]
                    self._pending_fp = (rx, ry, rz, rdx, rdy, rdz)
                    self._pending_vol = int(rdx * rdy * rdz)
                    self._pending_active = True
                    self._pending_fail_steps = 0

                if ridx < len(ids):
                    bid = int(ids[ridx])
                    self._removed_history[bid] = int(self._t)

                self._last_remove_step = int(self._t)
                act = self.act_remove(ridx)

                self._osc_record_remove(ridx)
                self._osc_check_and_trip(ridx)

                self._dbg_record_action(obs, act)
                self._print_action_physics(obs, act)
                return act

            # direct HARD place
            act = self.act_place(slot, rot_id, x, y)

            # HARD place clears pending
            if self._pending_active:
                self._dbg_event("pending cleared by successful HARD place")
                self._pending_active = False
                self._pending_fp = None
                self._pending_vol = 0
                self._pending_fail_steps = 0

            self._dbg_record_action(obs, act)
            self._print_action_physics(obs, act)
            return act

        # =========================
        # 2) SOFT top-k attempt
        # =========================
        soft_list = self._topk_soft_places(obs)

        if getattr(self, "debug", False) and (int(self._t) % int(self.debug_every) == 0):
            if soft_list:
                _, meta = soft_list[0]
                self._dbg(self._dbg_plan_str(obs, meta, tag="SOFT"))
            else:
                self._dbg("[HeurDbg] best_SOFT: None")

        for sc, meta in soft_list:
            slot, rot_id, x, y, z, dx, dy, dz = meta
            act = self.act_place(slot, rot_id, x, y)
            # IMPORTANT: DO NOT clear pending on SOFT place
            self._dbg_record_action(obs, act)
            self._print_action_physics(obs, act)
            return act

        # =========================
        # 3) FINAL FALLBACK: FORCE PLACE (NO REMOVE)
        # =========================
        self._dbg_event("NO HARD/SOFT place -> FORCE PLACE (still feasible+height+no-overlap; may allow hard-on-soft)")
        act = self._force_place_any_feasible(obs)
        if act is not None:
            self._dbg_record_action(obs, act)
            self._print_action_physics(obs, act)
            return act

        # =========================
        # 4) absolute last resort (should be rare)
        # =========================
        self._dbg_event("ABSOLUTE LAST RESORT: no feasible placement exists -> PLACE(0,0,0,0)")
        act = self.act_place(0, 0, 0, 0)
        self._dbg_record_action(obs, act)
        self._print_action_physics(obs, act)
        return act
