import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Initialize recent actions storage
    if not hasattr(llm_policy, "_recent"):
        llm_policy._recent = []
    
    # Constants
    eps = 1e-6
    X, Y, H = heur.X, heur.Y, heur.H
    
    # Observation data
    pallet = obs['pallet_obs_density']
    pallet_softness = obs['pallet_obs_softness']
    buffer = obs['buffer']
    buffer_physics = obs['buffer_physics']
    pallet_count = obs['pallet_count']
    removable_mask = obs['removable_mask']
    
    # Calculate number of slots
    n_slots = int(buffer.size // heur.n_properties)
    
    # PASS-1: Safe place (avoid hard-on-soft) with FLOOR BUILDING strategy
    best_action = None
    best_score = -float('inf')
    
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
            
        # Get box properties
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        # Get softness for this slot
        softness = 0.0
        if 2 * slot_i + 0 < len(buffer_physics):
            softness = float(buffer_physics[2 * slot_i + 0])
        box_is_hard = (softness <= 0.5)
        
        # Try all rotations
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            
            # FLOOR BUILDING STRATEGY: Prioritize building complete floor first
            # We'll try ALL positions at ground level (z=0) first before considering elevated positions
            
            # First pass: ONLY ground positions (z=0)
            for x in range(X):
                for y in range(Y):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Check if this position is at ground level
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero_mask = np.any(place_area > 0, axis=(0, 1))
                    if np.any(non_zero_mask):  # Not ground level, skip in first pass
                        continue
                    
                    z = 0  # Ground level
                    
                    # Check height overflow
                    if z + dz > H:
                        continue
                    
                    # Check non-overlap
                    if z + dz <= H:
                        blk = pallet[x:x+dx, y:y+dy, z:z+dz]
                        if blk.size > 0 and np.any(blk > eps):
                            continue
                    
                    # Check feasibility
                    if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        continue
                    
                    # At ground level (z=0), there's no support, so no hard-on-soft issue
                    # This automatically passes the PASS-1 rule
                    
                    # Anti-repeat check
                    action = (0, slot_i, rot_id, x, y)
                    if action in llm_policy._recent:
                        continue
                    
                    # Score ground positions: prioritize corners and near other boxes
                    # Find all occupied positions for proximity calculation
                    occupied_positions = []
                    for ox in range(X):
                        for oy in range(Y):
                            if np.any(pallet[ox, oy, :] > eps):
                                occupied_positions.append((ox, oy))
                    
                    # Corner preference: start from corner and near other boxes
                    corner_dist = min(x, X - x - dx) + min(y, Y - y - dy)
                    box_proximity = 0
                    if occupied_positions:
                        min_dist = float('inf')
                        for ox, oy in occupied_positions:
                            dist = abs(x - ox) + abs(y - oy)
                            if dist < min_dist:
                                min_dist = dist
                        box_proximity = min_dist
                    
                    # Ground positions get highest priority
                    ground_bonus = 10000
                    # Larger boxes preferred
                    volume_bonus = (dx * dy * dz) * 10
                    # Corner and proximity scoring
                    corner_penalty = corner_dist * 5
                    proximity_bonus = -box_proximity * 2  # Negative because we want SMALL distance
                    
                    score = ground_bonus + volume_bonus - corner_penalty + proximity_bonus
                    
                    if score > best_score:
                        best_score = score
                        best_action = action
    
    # If we found a ground position, use it immediately
    if best_action is not None:
        llm_policy._recent.append(best_action)
        if len(llm_policy._recent) > 8:
            llm_policy._recent.pop(0)
        return best_action
    
    # Second pass: elevated positions (only if no ground position found)
    # Still following PASS-1 rules (avoid hard-on-soft)
    best_action = None
    best_score = -float('inf')
    
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
            
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        # Get softness for this slot
        softness = 0.0
        if 2 * slot_i + 0 < len(buffer_physics):
            softness = float(buffer_physics[2 * slot_i + 0])
        box_is_hard = (softness <= 0.5)
        
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            
            for x in range(X):
                for y in range(Y):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Compute height
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero_mask = np.any(place_area > 0, axis=(0, 1))
                    if np.any(non_zero_mask):
                        z = int(np.max(np.nonzero(non_zero_mask)) + 1)
                    else:
                        z = 0
                    
                    # Check height overflow
                    if z + dz > H:
                        continue
                    
                    # Check non-overlap
                    if z + dz <= H:
                        blk = pallet[x:x+dx, y:y+dy, z:z+dz]
                        if blk.size > 0 and np.any(blk > eps):
                            continue
                    
                    # Check feasibility
                    if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        continue
                    
                    # Hard-on-soft check (only if z > 0)
                    if box_is_hard and z > 0:
                        support = pallet_softness[x:x+dx, y:y+dy, z-1]
                        if support.size > 0 and np.max(support) > 0.5:
                            continue  # Reject in PASS-1
                    
                    # Anti-repeat check
                    action = (0, slot_i, rot_id, x, y)
                    if action in llm_policy._recent:
                        continue
                    
                    # Score elevated positions
                    # Find all occupied positions for proximity calculation
                    occupied_positions = []
                    for ox in range(X):
                        for oy in range(Y):
                            if np.any(pallet[ox, oy, :] > eps):
                                occupied_positions.append((ox, oy))
                    
                    corner_dist = min(x, X - x - dx) + min(y, Y - y - dy)
                    box_proximity = 0
                    if occupied_positions:
                        min_dist = float('inf')
                        for ox, oy in occupied_positions:
                            dist = abs(x - ox) + abs(y - oy)
                            if dist < min_dist:
                                min_dist = dist
                        box_proximity = min_dist
                    
                    # Elevated positions: prefer lower heights to build floor first
                    # Lower z gets higher score (we want to fill lower layers first)
                    height_score = (H - z) * 100  # Inverted: lower z = higher score
                    volume_bonus = (dx * dy * dz) * 10
                    corner_penalty = corner_dist * 5
                    proximity_bonus = -box_proximity * 2
                    
                    score = height_score + volume_bonus - corner_penalty + proximity_bonus
                    
                    if score > best_score:
                        best_score = score
                        best_action = action
    
    if best_action is not None:
        llm_policy._recent.append(best_action)
        if len(llm_policy._recent) > 8:
            llm_policy._recent.pop(0)
        return best_action
    
    # PASS-2: Remove action if available
    if pallet_count > 0 and len(removable_mask) > 0:
        # Find removable indices
        removable_indices = [idx for idx in range(pallet_count) if removable_mask[idx] > 0]
        
        if removable_indices:
            # Simple deterministic rule: choose the LAST removable index
            # This mimics top-removal behavior
            best_remove_idx = removable_indices[-1]
            
            # Anti-repeat check
            action = (1, best_remove_idx, 0, 0, 0)
            if action in llm_policy._recent:
                # Try other indices from end to beginning
                for idx in reversed(removable_indices[:-1]):
                    action = (1, idx, 0, 0, 0)
                    if action not in llm_policy._recent:
                        best_remove_idx = idx
                        break
                else:
                    # All removable indices are in recent actions
                    best_remove_idx = None
            
            if best_remove_idx is not None:
                action = (1, best_remove_idx, 0, 0, 0)
                llm_policy._recent.append(action)
                if len(llm_policy._recent) > 8:
                    llm_policy._recent.pop(0)
                return action
    
    # PASS-3: Relaxed place (allow hard-on-soft) with FLOOR BUILDING strategy
    best_action = None
    best_score = -float('inf')
    
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
            
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            
            # FLOOR BUILDING: Try ground positions first
            for x in range(X):
                for y in range(Y):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Check if this position is at ground level
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero_mask = np.any(place_area > 0, axis=(0, 1))
                    if np.any(non_zero_mask):  # Not ground level, skip in first pass
                        continue
                    
                    z = 0  # Ground level
                    
                    if z + dz > H:
                        continue
                    
                    if z + dz <= H:
                        blk = pallet[x:x+dx, y:y+dy, z:z+dz]
                        if blk.size > 0 and np.any(blk > eps):
                            continue
                    
                    if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        continue
                    
                    action = (0, slot_i, rot_id, x, y)
                    if action in llm_policy._recent:
                        continue
                    
                    # Score ground positions highly
                    occupied_positions = []
                    for ox in range(X):
                        for oy in range(Y):
                            if np.any(pallet[ox, oy, :] > eps):
                                occupied_positions.append((ox, oy))
                    
                    corner_dist = min(x, X - x - dx) + min(y, Y - y - dy)
                    box_proximity = 0
                    if occupied_positions:
                        min_dist = float('inf')
                        for ox, oy in occupied_positions:
                            dist = abs(x - ox) + abs(y - oy)
                            if dist < min_dist:
                                min_dist = dist
                        box_proximity = min_dist
                    
                    ground_bonus = 10000
                    volume_bonus = (dx * dy * dz) * 10
                    corner_penalty = corner_dist * 5
                    proximity_bonus = -box_proximity * 2
                    
                    score = ground_bonus + volume_bonus - corner_penalty + proximity_bonus
                    
                    if score > best_score:
                        best_score = score
                        best_action = action
            
            # If no ground position, try all positions
            if best_action is None:
                for x in range(X):
                    for y in range(Y):
                        if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                            continue
                        
                        place_area = pallet[x:x+dx, y:y+dy, :]
                        non_zero_mask = np.any(place_area > 0, axis=(0, 1))
                        if np.any(non_zero_mask):
                            z = int(np.max(np.nonzero(non_zero_mask)) + 1)
                        else:
                            z = 0
                        
                        if z + dz > H:
                            continue
                        
                        if z + dz <= H:
                            blk = pallet[x:x+dx, y:y+dy, z:z+dz]
                            if blk.size > 0 and np.any(blk > eps):
                                continue
                        
                        if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                            continue
                        
                        action = (0, slot_i, rot_id, x, y)
                        if action in llm_policy._recent:
                            continue
                        
                        occupied_positions = []
                        for ox in range(X):
                            for oy in range(Y):
                                if np.any(pallet[ox, oy, :] > eps):
                                    occupied_positions.append((ox, oy))
                        
                        corner_dist = min(x, X - x - dx) + min(y, Y - y - dy)
                        box_proximity = 0
                        if occupied_positions:
                            min_dist = float('inf')
                            for ox, oy in occupied_positions:
                                dist = abs(x - ox) + abs(y - oy)
                                if dist < min_dist:
                                    min_dist = dist
                            box_proximity = min_dist
                        
                        # Prefer lower heights to continue floor building
                        height_score = (H - z) * 100  # Inverted: lower z = higher score
                        volume_bonus = (dx * dy * dz) * 10
                        corner_penalty = corner_dist * 5
                        proximity_bonus = -box_proximity * 2
                        
                        score = height_score + volume_bonus - corner_penalty + proximity_bonus
                        
                        if score > best_score:
                            best_score = score
                            best_action = action
    
    if best_action is not None:
        llm_policy._recent.append(best_action)
        if len(llm_policy._recent) > 8:
            llm_policy._recent.pop(0)
        return best_action
    
    # Last resort: return (0,0,0,0,0)
    action = (0, 0, 0, 0, 0)
    llm_policy._recent.append(action)
    if len(llm_policy._recent) > 8:
        llm_policy._recent.pop(0)
    return action

class ArchivedHeuristic(BaseHeuristic):
    name = "firsttest_of_llm_unpacking"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
