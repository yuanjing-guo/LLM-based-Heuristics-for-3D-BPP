import numpy as np
from heuristics.base import BaseHeuristic


def llm_policy(heur, obs):
    n_slots = int(obs["buffer"].size // heur.n_properties)
    pallet = obs["pallet_obs_density"]

    X, Y, H = pallet.shape[0], pallet.shape[1], pallet.shape[2]

    # ------------------------------------------------------------
    # 1) Heightmap (top-loading): h(x,y) = top surface height
    # ------------------------------------------------------------
    occ = pallet > 0
    has = occ.any(axis=2)
    occ_rev = occ[..., ::-1]
    first_from_top = occ_rev.argmax(axis=2)
    heightmap = np.where(has, H - first_from_top, 0).astype(int)

    # "Current layer" = lowest existing surface anywhere (includes 0)
    layer_z = int(heightmap.min()) if heightmap.size > 0 else 0
    global_max_h = int(heightmap.max()) if heightmap.size > 0 else 0

    # ------------------------------------------------------------
    # 2) Stage 1: pick top-K items by "best possible volume"
    #    (do NOT lock a single rotation here)
    # ------------------------------------------------------------
    K = 3
    rot_candidates_per_item = 3  # evaluate top-R rotations later for this item

    items = []
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue

        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins is None or getattr(size_bins, "size", 0) == 0:
            continue
        if np.any(size_bins <= 0):
            continue

        # estimate "difficulty" / priority by the best possible volume over rotations
        best_vol = 0
        best_fp = 0
        for rot_id in range(6):
            dx, dy, dz = heur.rotate_size_bins(size_bins, rot_id)
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
            if dx > X or dy > Y or dz > H:
                continue
            vol = int(dx * dy * dz)
            fp = int(dx * dy)
            if vol > best_vol or (vol == best_vol and fp > best_fp):
                best_vol, best_fp = vol, fp

        if best_vol > 0:
            items.append((slot_i, size_bins, best_vol, best_fp))

    if not items:
        return (0, 0, 0, 0)

    items.sort(key=lambda t: (t[2], t[3]), reverse=True)
    top_items = items[:K]

    # ------------------------------------------------------------
    # 3) Candidate position generation (pruned):
    #    For each item+rotation, only scan a limited set of low-z cells.
    #    We score placements using:
    #      - layer fill bonus (fills the current lowest layer)
    #      - support fraction (prefer fully supported footprints)
    #      - adjacency / wall contact (compact packing)
    #      - local flatness penalty (avoid creating rough top surface)
    #      - height penalty and above-layer penalty (avoid pillars)
    # ------------------------------------------------------------
    best_action = (0, 0, 0, 0)
    best_score = -1e18

    # parameters (tune-friendly but already sensible)
    MAX_POS_SCAN = 120      # cap how many (x,y) sites we try per rotation
    TOP_M_BY_Z = 8          # among feasible, score only a few lowest-z
    W_HEIGHT = 0.08         # penalize absolute z
    W_ABOVE_LAYER = 0.35    # penalize placing above current layer
    W_SUPPORT = 0.85        # penalize lack of full support
    W_FLATNESS = 0.10       # penalize roughness in footprint after placement
    W_ADJ = 0.06            # reward adjacency to existing boxes
    W_WALL = 0.04           # reward touching walls
    W_LAYERFILL = 0.40      # reward covering cells at current lowest layer
    W_TOPSPREAD = 0.03      # mild reward for not concentrating in one spot

    # precompute an occupancy map per (x,y) at the current surface height
    # "surface occupied" at cell = heightmap[x,y] > 0
    surface_occ = (heightmap > 0).astype(np.uint8)

    def adjacency_score(x, y, z, dx, dy):
        """
        Reward contact with existing mass at same height "z" around footprint perimeter.
        Also reward wall contact to encourage compactness.
        """
        # neighbor checks on the *footprint perimeter* at (x,y,z)
        # We'll approximate by looking at heightmap equality to z in adjacent cells.
        score = 0.0

        # Wall contact
        if x == 0:
            score += 1.0
        if y == 0:
            score += 1.0
        if x + dx == X:
            score += 1.0
        if y + dy == Y:
            score += 1.0
        wall_bonus = score

        # Adjacency (left/right/up/down) for cells along perimeter:
        adj = 0.0
        # left side neighbors
        if x > 0:
            neighbor = heightmap[x - 1, y:y + dy]
            adj += float(np.sum(neighbor >= z))  # supports "touching mass"
        # right
        if x + dx < X:
            neighbor = heightmap[x + dx, y:y + dy]
            adj += float(np.sum(neighbor >= z))
        # bottom
        if y > 0:
            neighbor = heightmap[x:x + dx, y - 1]
            adj += float(np.sum(neighbor >= z))
        # top
        if y + dy < Y:
            neighbor = heightmap[x:x + dx, y + dy]
            adj += float(np.sum(neighbor >= z))

        # normalize by perimeter length (avoid bias to big boxes)
        perim = 2 * dx + 2 * dy
        adj_norm = adj / max(perim, 1)

        return W_WALL * wall_bonus + W_ADJ * adj_norm

    # For diversity / spread: penalize stacking always on the same highest column region
    # We'll use mild penalty on local height concentration.
    def top_spread_bonus(x, y, dx, dy):
        local = heightmap[x:x + dx, y:y + dy]
        # encourage placing on lower-than-avg zones in that footprint region
        return -W_TOPSPREAD * float(local.mean() / max(H, 1))

    for slot_i, size_bins, _, _ in top_items:
        # pick a few promising rotations for this item (by volume then footprint)
        rot_list = []
        for rot_id in range(6):
            dx, dy, dz = heur.rotate_size_bins(size_bins, rot_id)
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
            if dx > X or dy > Y or dz > H:
                continue
            vol = int(dx * dy * dz)
            fp = int(dx * dy)
            rot_list.append((vol, fp, rot_id, dx, dy, dz))
        if not rot_list:
            continue
        rot_list.sort(key=lambda t: (t[0], t[1]), reverse=True)
        rot_list = rot_list[:rot_candidates_per_item]

        for _, _, rot_id, dx, dy, dz in rot_list:
            # quick scan order: low layer first. We generate (x,y) in a pattern that
            # tends to find compact solutions early.
            candidates = []

            # Generate scan sites biased toward corners/walls (compactness)
            scan_sites = []
            # simple pattern: corners -> edges -> interior
            for x0 in (0, max(0, X - dx)):
                for y0 in (0, max(0, Y - dy)):
                    scan_sites.append((x0, y0))
            # edge sweeps
            for x0 in range(0, X - dx + 1):
                scan_sites.append((x0, 0))
                scan_sites.append((x0, max(0, Y - dy)))
            for y0 in range(0, Y - dy + 1):
                scan_sites.append((0, y0))
                scan_sites.append((max(0, X - dx), y0))

            # add a few interior points (diagonal)
            step = max(1, min(X, Y) // 5)
            for t in range(0, min(X - dx + 1, Y - dy + 1), step):
                scan_sites.append((t, t))

            # remove duplicates while preserving order
            seen = set()
            uniq_sites = []
            for s in scan_sites:
                if s not in seen:
                    seen.add(s)
                    uniq_sites.append(s)

            # if still too few sites, pad with raster (but cap total)
            if len(uniq_sites) < MAX_POS_SCAN:
                for x0 in range(0, X - dx + 1):
                    for y0 in range(0, Y - dy + 1):
                        if (x0, y0) in seen:
                            continue
                        uniq_sites.append((x0, y0))
                        if len(uniq_sites) >= MAX_POS_SCAN:
                            break
                    if len(uniq_sites) >= MAX_POS_SCAN:
                        break
            else:
                uniq_sites = uniq_sites[:MAX_POS_SCAN]

            for (x, y) in uniq_sites:
                # top-loading height is the max surface under footprint
                z = int(np.max(heightmap[x:x + dx, y:y + dy]))
                if z + dz > H:
                    continue

                # feasibility check (overlap + support constraints)
                if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                    continue

                candidates.append((z, x, y))

            if not candidates:
                continue

            candidates.sort(key=lambda t: t[0])  # lowest z first
            for (z, x, y) in candidates[:min(TOP_M_BY_Z, len(candidates))]:
                # --- layer fill bonus: reward covering the current lowest layer cells ---
                footprint_heights = heightmap[x:x + dx, y:y + dy]
                layer_fill_fraction = float(np.mean(footprint_heights == layer_z))
                layer_fill_bonus = W_LAYERFILL * layer_fill_fraction

                # --- support: prefer fully supported footprint ---
                support_fraction = 1.0
                if z > 0:
                    support_fraction = float(np.mean(footprint_heights == z))
                support_penalty = W_SUPPORT * (1.0 - support_fraction) * (z / max(H, 1))

                # --- height penalties ---
                height_penalty = W_HEIGHT * (z / max(H, 1))
                above_layer_penalty = 0.0
                if z > layer_z:
                    above_layer_penalty = W_ABOVE_LAYER * ((z - layer_z) / max(H, 1))

                # --- flatness penalty: avoid creating rough top in the footprint ---
                # after placing, those cells become max(old, z+dz)
                new_local = np.maximum(footprint_heights, z + dz)
                local_rough = float(new_local.max() - new_local.min()) / max(H, 1)
                flatness_penalty = W_FLATNESS * local_rough

                # --- compactness reward ---
                compact_reward = adjacency_score(x, y, z, dx, dy) + top_spread_bonus(x, y, dx, dy)

                # --- objective: maximize compactness & layer fill, minimize height & instability ---
                # (No global fill_ratio constant; we use meaningful local signals.)
                score = (
                    compact_reward
                    + layer_fill_bonus
                    - height_penalty
                    - above_layer_penalty
                    - support_penalty
                    - flatness_penalty
                )

                if score > best_score:
                    best_score = score
                    best_action = (slot_i, rot_id, x, y)

    if best_score <= -1e17:
        return (0, 0, 0, 0)
    return best_action


class TestFailedOne(BaseHeuristic):
    name = "failed_one_fixed"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
