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
    
    # Update recent actions (keep last 8)
    if len(llm_policy._recent) > 8:
        llm_policy._recent.pop(0)
    
    # Helper: check if candidate is in recent history
    def in_recent(cand):
        return cand in llm_policy._recent[-4:] if len(llm_policy._recent) >= 4 else False
    
    # Physics detection
    physics_informative = ('buffer_physics' in obs) and ('pallet_obs_softness' in obs) and \
                          (np.any(obs['buffer_physics'] != 0) or np.any(obs['pallet_obs_softness'] != 0))
    
    # Get pallet array
    pallet = obs['pallet_obs_density']
    
    # Helper: get softness for slot
    def get_softness(slot):
        if not physics_informative or 2*slot + 0 >= len(obs['buffer_physics']):
            return 0.0
        return float(obs['buffer_physics'][2*slot + 0])
    
    # Helper: check if box is HARD
    def is_hard(slot):
        return not physics_informative or get_softness(slot) <= 0.5
    
    # Helper: find bottom-soft footprints
    def find_bottom_soft_footprints():
        soft_footprints = []
        for i in range(obs['pallet_count']):
            fx, fy, fz, fdx, fdy, fdz = obs['pallet_footprints'][i]
            if fz == 0 and fdx > 0 and fdy > 0:
                # Check if region has mean softness > 0.5
                region = obs['pallet_obs_softness'][fx:fx+fdx, fy:fy+fdy, 0]
                if region.size > 0 and np.mean(region) > 0.5:
                    soft_footprints.append((i, fx, fy, fdx, fdy))
        return soft_footprints
    
    # Helper: find removable blocker above a given XY region
    def find_blocker_above(x, y, dx, dy):
        candidates = []
        for i in range(obs['pallet_count']):
            if obs['removable_mask'][i] <= 0 or i in llm_policy._rm_cd:
                continue
            fx, fy, fz, fdx, fdy, fdz = obs['pallet_footprints'][i]
            if fz > 0:  # Must be above bottom
                # Check XY overlap
                x_overlap = max(0, min(x+dx, fx+fdx) - max(x, fx))
                y_overlap = max(0, min(y+dy, fy+fdy) - max(y, fy))
                if x_overlap > 0 and y_overlap > 0:
                    top_height = fz + fdz
                    overlap_area = x_overlap * y_overlap
                    candidates.append((i, top_height, overlap_area, fx, fy, fdx, fdy))
        if not candidates:
            return None
        # Sort by top_height desc, then overlap_area desc
        candidates.sort(key=lambda c: (c[1], c[2]), reverse=True)
        return candidates[0]
    
    # PHASE A: BASE-HARDIFY
    if physics_informative:
        # Check for HARD boxes in buffer
        hard_slots = []
        for slot in range(len(obs['front_ids'])):
            props = heur.get_slot_props(obs, slot)
            if props[0] > 0 and is_hard(slot):  # non-empty and HARD
                hard_slots.append(slot)
        
        if hard_slots:
            soft_footprints = find_bottom_soft_footprints()
            if soft_footprints:
                # A1: Try to remove bottom-soft footprint directly
                for i, fx, fy, fdx, fdy in soft_footprints:
                    if obs['removable_mask'][i] > 0 and i not in llm_policy._rm_cd:
                        llm_policy._pending_base = (fx, fy, fdx, fdy, 10)
                        llm_policy._rm_cd[i] = 3
                        llm_policy._recent.append((1, i, 0, 0, 0))
                        return (1, i, 0, 0, 0)
                
                # A2: Remove blocker above a bottom-soft footprint
                for i, fx, fy, fdx, fdy in soft_footprints:
                    blocker = find_blocker_above(fx, fy, fdx, fdy)
                    if blocker:
                        idx, _, _, _, _, _, _ = blocker
                        llm_policy._rm_cd[idx] = 3
                        llm_policy._recent.append((1, idx, 0, 0, 0))
                        return (1, idx, 0, 0, 0)
    
    # PHASE B: FILL PENDING HOLE
    if llm_policy._pending_base is not None:
        fx, fy, fdx, fdy, ttl = llm_policy._pending_base
        if ttl <= 0:
            llm_policy._pending_base = None
        else:
            # Try to place HARD box at z==0 overlapping pending region
            for slot in range(len(obs['front_ids'])):
                props = heur.get_slot_props(obs, slot)
                if props[0] <= 0 or not is_hard(slot):
                    continue
                
                size_bins = heur.props_to_size_bins(props)
                for rot_id in range(6):
                    dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
                    if dx <= 0 or dy <= 0 or dz <= 0:
                        continue
                    
                    # Search positions overlapping pending region
                    for x in range(max(0, fx - dx + 1), min(heur.X - dx + 1, fx + fdx)):
                        for y in range(max(0, fy - dy + 1), min(heur.Y - dy + 1, fy + fdy)):
                            # Check overlap
                            x_overlap = max(0, min(x+dx, fx+fdx) - max(x, fx))
                            y_overlap = max(0, min(y+dy, fy+fdy) - max(y, fy))
                            if x_overlap == 0 or y_overlap == 0:
                                continue
                            
                            # Compute z
                            place_area = pallet[x:x+dx, y:y+dy, :]
                            non_zero_mask = np.any(place_area > 0, axis=(0,1))
                            if np.any(non_zero_mask):
                                z = int(np.max(np.nonzero(non_zero_mask)[0]) + 1)
                            else:
                                z = 0
                            
                            if z != 0 or z + dz > heur.H:
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
                            cand = (0, slot, rot_id, x, y)
                            if in_recent(cand):
                                continue
                            
                            llm_policy._pending_base = None
                            llm_policy._place_fail_count = 0
                            llm_policy._recent.append(cand)
                            return cand
            
            # Decrement TTL
            llm_policy._pending_base = (fx, fy, fdx, fdy, ttl - 1)
    
    # PHASE C: NORMAL PLACEMENT
    # First try HARD boxes for z==0
    hard_candidates = []
    soft_candidates = []
    
    for slot in range(len(obs['front_ids'])):
        props = heur.get_slot_props(obs, slot)
        if props[0] <= 0:
            continue
        
        hard_box = is_hard(slot)
        size_bins = heur.props_to_size_bins(props)
        
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            if dx <= 0 or dy <= 0 or dz <= 0:
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
                    
                    # Non-overlap check
                    blk = pallet[x:x+dx, y:y+dy, z:z+dz]
                    if blk.size == 0 or np.any(blk > 1e-6):
                        continue
                    
                    # Feasibility
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        continue
                    
                    cand = (0, slot, rot_id, x, y)
                    if in_recent(cand):
                        continue
                    
                    if z == 0:
                        if hard_box:
                            hard_candidates.append(cand)
                        else:
                            soft_candidates.append(cand)
                    else:
                        # For z>0, accept any
                        llm_policy._place_fail_count = 0
                        llm_policy._recent.append(cand)
                        return cand
    
    # Try HARD z==0 candidates
    if hard_candidates:
        llm_policy._place_fail_count = 0
        cand = hard_candidates[0]
        llm_policy._recent.append(cand)
        return cand
    
    # Try SOFT z==0 candidates
    if soft_candidates:
        llm_policy._place_fail_count = 0
        cand = soft_candidates[0]
        llm_policy._recent.append(cand)
        return cand
    
    # No placement found
    llm_policy._place_fail_count += 1
    
    # REMOVE as last resort after 2 failed placement attempts
    if llm_policy._place_fail_count >= 2:
        for i in range(obs['pallet_count']):
            if obs['removable_mask'][i] > 0 and i not in llm_policy._rm_cd:
                cand = (1, i, 0, 0, 0)
                if in_recent(cand):
                    continue
                llm_policy._rm_cd[i] = 3
                llm_policy._place_fail_count = 0
                llm_policy._recent.append(cand)
                return cand
    
    # Absolute fallback
    return (0, 0, 0, 0, 0)

class ArchivedHeuristic(BaseHeuristic):
    name = "success"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
