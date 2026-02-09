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
    if not hasattr(llm_policy, '_fail_count'):
        llm_policy._fail_count = 0
    
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
    
    # Helper: get softness of slot
    def slot_softness(slot):
        if not physics_informative or 2*slot + 0 >= len(obs['buffer_physics']):
            return 0.0
        return float(obs['buffer_physics'][2*slot + 0])
    
    # Helper: is box HARD?
    def is_hard(slot):
        return slot_softness(slot) <= 0.5
    
    # Helper: check if candidate is in recent history
    def in_recent(candidate):
        return candidate in llm_policy._recent[-4:] if llm_policy._recent else False
    
    # Helper: find bottom-soft footprints
    def find_bottom_soft_footprints():
        soft_footprints = []
        pallet = obs['pallet_obs_density']
        softness_grid = obs.get('pallet_obs_softness', np.zeros_like(pallet))
        for i in range(obs['pallet_count']):
            x, y, z, dx, dy, dz = obs['pallet_footprints'][i]
            if z == 0 and obs['removable_mask'][i] > 0:
                # Check if region is soft
                region = softness_grid[x:x+dx, y:y+dy, 0]
                if region.size > 0 and np.mean(region) > 0.5:
                    soft_footprints.append((i, x, y, dx, dy))
        return soft_footprints
    
    # Helper: find blockers above a bottom footprint
    def find_blockers_above(bx, by, bdx, bdy):
        blockers = []
        for i in range(obs['pallet_count']):
            if obs['removable_mask'][i] <= 0:
                continue
            x, y, z, dx, dy, dz = obs['pallet_footprints'][i]
            if z > 0:  # Must be above bottom
                # Check XY overlap
                x_overlap = max(0, min(bx + bdx, x + dx) - max(bx, x))
                y_overlap = max(0, min(by + bdy, y + dy) - max(by, y))
                if x_overlap > 0 and y_overlap > 0:
                    blockers.append((i, z + dz, x_overlap * y_overlap, x, y, dx, dy, z, dz))
        return blockers
    
    # --- PHASE A: BASE-HARDIFY ---
    if physics_informative:
        # Check for HARD boxes in buffer
        hard_slots = []
        for slot in range(len(obs['front_ids'])):
            props = heur.get_slot_props(obs, slot)
            if props[0] > 0 and is_hard(slot):
                hard_slots.append(slot)
        
        if hard_slots:
            bottom_soft = find_bottom_soft_footprints()
            if bottom_soft:
                # A1: Remove bottom-soft directly if removable
                for i, x, y, dx, dy in bottom_soft:
                    if i not in llm_policy._rm_cd and not in_recent((1, i, 0, 0, 0)):
                        llm_policy._recent.append((1, i, 0, 0, 0))
                        if len(llm_policy._recent) > 8:
                            llm_policy._recent.pop(0)
                        llm_policy._rm_cd[i] = 3
                        llm_policy._pending_base = (x, y, dx, dy, 10)
                        llm_policy._fail_count = 0
                        return (1, i, 0, 0, 0)
                
                # A2: Remove blocker above bottom-soft
                for i, x, y, dx, dy in bottom_soft:
                    blockers = find_blockers_above(x, y, dx, dy)
                    if blockers:
                        # Sort by height then overlap
                        blockers.sort(key=lambda b: (-b[1], -b[2]))
                        for bi, _, _, _, _, _, _, _, _ in blockers:
                            if bi not in llm_policy._rm_cd and not in_recent((1, bi, 0, 0, 0)):
                                llm_policy._recent.append((1, bi, 0, 0, 0))
                                if len(llm_policy._recent) > 8:
                                    llm_policy._recent.pop(0)
                                llm_policy._rm_cd[bi] = 3
                                llm_policy._fail_count = 0
                                return (1, bi, 0, 0, 0)
    
    # --- PHASE B: FILL PENDING HOLE ---
    if llm_policy._pending_base is not None:
        fx, fy, fdx, fdy, _ = llm_policy._pending_base
        pallet = obs['pallet_obs_density']
        
        # Try to place HARD box at z==0 overlapping pending region
        for slot in range(len(obs['front_ids'])):
            props = heur.get_slot_props(obs, slot)
            if props[0] == 0 or not is_hard(slot):
                continue
            
            for rot_id in range(4):
                size_bins = heur.props_to_size_bins(props)
                dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
                
                # Must fit in pallet
                if dx > heur.X or dy > heur.Y or dz > heur.H:
                    continue
                
                # Search placement positions
                for x in range(heur.X - dx + 1):
                    for y in range(heur.Y - dy + 1):
                        # Check overlap with pending region
                        x_overlap = max(0, min(fx + fdx, x + dx) - max(fx, x))
                        y_overlap = max(0, min(fy + fdy, y + dy) - max(fy, y))
                        if x_overlap == 0 or y_overlap == 0:
                            continue
                        
                        # Compute z using exact height rule
                        place_area = pallet[x:x+dx, y:y+dy, :]
                        non_zero_mask = np.any(place_area > 0, axis=(0,1))
                        if np.any(non_zero_mask):
                            z = int(np.max(np.nonzero(non_zero_mask)[0]) + 1)
                        else:
                            z = 0
                        
                        if z != 0:  # Must be bottom placement
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
                        
                        candidate = (0, slot, rot_id, x, y)
                        if in_recent(candidate):
                            continue
                        
                        llm_policy._recent.append(candidate)
                        if len(llm_policy._recent) > 8:
                            llm_policy._recent.pop(0)
                        llm_policy._pending_base = None
                        llm_policy._fail_count = 0
                        return candidate
    
    # --- PHASE C: NORMAL PLACEMENT ---
    pallet = obs['pallet_obs_density']
    candidates = []
    
    # Try HARD boxes first for z==0
    for slot in range(len(obs['front_ids'])):
        props = heur.get_slot_props(obs, slot)
        if props[0] == 0:
            continue
        
        hard_box = physics_informative and is_hard(slot)
        
        for rot_id in range(4):
            size_bins = heur.props_to_size_bins(props)
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            
            if dx > heur.X or dy > heur.Y or dz > heur.H:
                continue
            
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
                    
                    # Non-overlap
                    blk = pallet[x:x+dx, y:y+dy, z:z+dz]
                    if blk.size == 0 or np.any(blk > 1e-6):
                        continue
                    
                    # Feasibility
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        continue
                    
                    candidate = (0, slot, rot_id, x, y)
                    if in_recent(candidate):
                        continue
                    
                    # Prioritize: HARD on bottom > SOFT on bottom > anything above
                    priority = 0
                    if z == 0:
                        if hard_box:
                            priority = 3
                        else:
                            priority = 2
                    else:
                        priority = 1
                    
                    candidates.append((priority, candidate))
    
    if candidates:
        # Sort by priority (descending)
        candidates.sort(key=lambda c: -c[0])
        best = candidates[0][1]
        llm_policy._recent.append(best)
        if len(llm_policy._recent) > 8:
            llm_policy._recent.pop(0)
        llm_policy._fail_count = 0
        return best
    
    # --- REMOVE FALLBACK ---
    llm_policy._fail_count += 1
    if llm_policy._fail_count >= 2:
        # Find removable pallet
        for i in range(obs['pallet_count']):
            if obs['removable_mask'][i] > 0 and i not in llm_policy._rm_cd and not in_recent((1, i, 0, 0, 0)):
                llm_policy._recent.append((1, i, 0, 0, 0))
                if len(llm_policy._recent) > 8:
                    llm_policy._recent.pop(0)
                llm_policy._rm_cd[i] = 3
                llm_policy._fail_count = 0
                return (1, i, 0, 0, 0)
    
    # Last resort
    llm_policy._fail_count = 0
    return (0, 0, 0, 0, 0)

class GeneratedHeuristic(BaseHeuristic):
    name = "llm_generated"

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
