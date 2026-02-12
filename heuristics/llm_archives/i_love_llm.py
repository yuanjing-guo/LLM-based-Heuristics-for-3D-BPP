import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Initialize persistent state
    if not hasattr(llm_policy, '_recent'):
        llm_policy._recent = []
    if not hasattr(llm_policy, '_rm_cd'):
        llm_policy._rm_cd = {}
    if not hasattr(llm_policy, '_pending_base'):
        llm_policy._pending_base = None
    if not hasattr(llm_policy, '_place_fail_count'):
        llm_policy._place_fail_count = 0
    
    # Update cooldowns
    for k in list(llm_policy._rm_cd.keys()):
        llm_policy._rm_cd[k] -= 1
        if llm_policy._rm_cd[k] <= 0:
            del llm_policy._rm_cd[k]
    
    # Update pending TTL
    if llm_policy._pending_base is not None:
        fx, fy, fdx, fdy, ttl = llm_policy._pending_base
        ttl -= 1
        if ttl <= 0:
            llm_policy._pending_base = None
        else:
            llm_policy._pending_base = (fx, fy, fdx, fdy, ttl)
    
    # Physics detection
    physics_informative = ('buffer_physics' in obs) and ('pallet_obs_softness' in obs) and \
                          (np.any(obs['buffer_physics'] != 0) or np.any(obs['pallet_obs_softness'] != 0))
    
    # Helper to get softness
    def get_softness(slot):
        if not physics_informative or 2*slot + 0 >= len(obs['buffer_physics']):
            return 0.0
        return float(obs['buffer_physics'][2*slot + 0])
    
    # Helper to check if box is HARD
    def is_hard(slot):
        return get_softness(slot) <= 0.5
    
    # Helper to check if footprint is bottom-soft
    def is_bottom_soft_footprint(i):
        if not physics_informative:
            return False
        fx, fy, fz, fdx, fdy, fdz = obs['pallet_footprints'][i]
        if fz != 0:
            return False
        region = obs['pallet_obs_softness'][fx:fx+fdx, fy:fy+fdy, 0]
        return np.mean(region) > 0.5
    
    # Get pallet array
    pallet = obs['pallet_obs_density']
    N = len(obs['front_ids'])
    
    # Anti-oscillation check
    def is_recent(candidate):
        return candidate in llm_policy._recent[-4:] if llm_policy._recent else False
    
    # Update recent actions
    def record_action(action):
        llm_policy._recent.append(action)
        if len(llm_policy._recent) > 8:
            llm_policy._recent.pop(0)
    
    # --- PHASE A: BASE-HARDIFY (replacement) ---
    if physics_informative:
        # Check for bottom-soft footprints
        bottom_soft_indices = [i for i in range(obs['pallet_count']) 
                               if obs['removable_mask'][i] > 0 and is_bottom_soft_footprint(i)]
        
        if bottom_soft_indices:
            # A1: Remove bottom-soft directly
            for i in bottom_soft_indices:
                if i not in llm_policy._rm_cd and not is_recent((1, i, 0, 0, 0)):
                    fx, fy, fz, fdx, fdy, fdz = obs['pallet_footprints'][i]
                    llm_policy._pending_base = (fx, fy, fdx, fdy, 10)
                    llm_policy._rm_cd[i] = 3
                    action = (1, i, 0, 0, 0)
                    record_action(action)
                    llm_policy._place_fail_count = 0
                    return action
        
        # A2: Find removable blocker above bottom-soft
        bottom_soft_all = [i for i in range(obs['pallet_count']) if is_bottom_soft_footprint(i)]
        for bs_idx in bottom_soft_all:
            fx, fy, fz, fdx, fdy, fdz = obs['pallet_footprints'][bs_idx]
            # Find removable boxes overlapping XY
            candidates = []
            for i in range(obs['pallet_count']):
                if obs['removable_mask'][i] <= 0 or i in llm_policy._rm_cd:
                    continue
                if is_recent((1, i, 0, 0, 0)):
                    continue
                ifx, ify, ifz, ifdx, ifdy, ifdz = obs['pallet_footprints'][i]
                if ifz == 0:
                    continue  # Skip bottom boxes (already handled)
                # Check XY overlap
                x_overlap = max(0, min(fx + fdx, ifx + ifdx) - max(fx, ifx))
                y_overlap = max(0, min(fy + fdy, ify + ifdy) - max(fy, ify))
                if x_overlap > 0 and y_overlap > 0:
                    top_height = ifz + ifdz
                    overlap_area = x_overlap * y_overlap
                    candidates.append((i, top_height, overlap_area))
            
            if candidates:
                # Prefer higher top height, then larger overlap
                candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
                i = candidates[0][0]
                llm_policy._rm_cd[i] = 3
                action = (1, i, 0, 0, 0)
                record_action(action)
                llm_policy._place_fail_count = 0
                return action
    
    # --- PHASE B: FILL PENDING HOLE WITH HARD ---
    if llm_policy._pending_base is not None:
        fx, fy, fdx, fdy, _ = llm_policy._pending_base
        # Try to place HARD box at z==0 overlapping pending region
        for slot in range(N):
            if obs['front_ids'][slot] == 0:
                continue
            if not is_hard(slot):
                continue
            props = heur.get_slot_props(obs, slot)
            size_bins = heur.props_to_size_bins(props)
            for rot_id in range(6):
                dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
                # Check overlap with pending region
                x_overlap = max(0, min(fx + fdx, fx + dx) - max(fx, fx))
                y_overlap = max(0, min(fy + fdy, fy + dy) - max(fy, fy))
                if x_overlap == 0 or y_overlap == 0:
                    continue
                
                # Try placement at (fx, fy) first
                for x in range(max(0, fx - dx + 1), min(heur.X - dx + 1, fx + fdx)):
                    for y in range(max(0, fy - dy + 1), min(heur.Y - dy + 1, fy + fdy)):
                        # Compute z
                        place_area = pallet[x:x+dx, y:y+dy, :]
                        non_zero_mask = np.any(place_area > 0, axis=(0,1))
                        if np.any(non_zero_mask):
                            z = int(np.max(np.nonzero(non_zero_mask)[0]) + 1)
                        else:
                            z = 0
                        if z != 0:
                            continue
                        if z + dz > heur.H:
                            continue
                        # Non-overlap check
                        blk = pallet[x:x+dx, y:y+dy, z:z+dz]
                        if blk.size == 0 or np.any(blk > 1e-6):
                            continue
                        # Feasibility
                        if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                            continue
                        if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                            continue
                        # Anti-oscillation
                        candidate = (0, slot, rot_id, x, y)
                        if is_recent(candidate):
                            continue
                        action = candidate
                        record_action(action)
                        llm_policy._place_fail_count = 0
                        return action
    
    # --- PHASE C: NORMAL PLACEMENT ---
    # Collect legal placements
    legal_placements = []
    
    for slot in range(N):
        if obs['front_ids'][slot] == 0:
            continue
        hard = is_hard(slot) if physics_informative else True
        props = heur.get_slot_props(obs, slot)
        size_bins = heur.props_to_size_bins(props)
        
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    # Compute z
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero_mask = np.any(place_area > 0, axis=(0,1))
                    if np.any(non_zero_mask):
                        z = int(np.max(np.nonzero(non_zero_mask)[0]) + 1)
                    else:
                        z = 0
                    if z + dz > heur.H:
                        continue
                    # Non-overlap check
                    blk = pallet[x:x+dx, y:y+dy, z:z+dz]
                    if blk.size == 0 or np.any(blk > 1e-6):
                        continue
                    # Feasibility
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        continue
                    # Anti-oscillation
                    candidate = (0, slot, rot_id, x, y)
                    if is_recent(candidate):
                        continue
                    legal_placements.append((z, hard, candidate))
    
    if legal_placements:
        # Sort: z==0 HARD first, then z==0 SOFT, then others
        def priority_key(item):
            z, hard, _ = item
            if z == 0 and hard:
                return 0
            if z == 0 and not hard:
                return 1
            return 2 + z  # Higher z gets lower priority
        
        legal_placements.sort(key=priority_key)
        _, _, action = legal_placements[0]
        record_action(action)
        llm_policy._place_fail_count = 0
        return action
    
    # --- REMOVE FALLBACK ---
    llm_policy._place_fail_count += 1
    if llm_policy._place_fail_count >= 2:
        # Find removable box not on cooldown
        for i in range(obs['pallet_count']):
            if obs['removable_mask'][i] > 0 and i not in llm_policy._rm_cd:
                if not is_recent((1, i, 0, 0, 0)):
                    llm_policy._rm_cd[i] = 3
                    action = (1, i, 0, 0, 0)
                    record_action(action)
                    return action
    
    # --- LAST RESORT ---
    action = (0, 0, 0, 0, 0)
    record_action(action)
    return action

class ArchivedHeuristic(BaseHeuristic):
    name = "i_love_llm"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        out = llm_policy(self, obs)
        if isinstance(out, (list, tuple, np.ndarray)):
            out = list(out)
        else:
            out = [out]

        while len(out) < 5:
            out.append(0)
        out = out[:5]

        return np.array([int(v) for v in out], dtype=np.int32)
