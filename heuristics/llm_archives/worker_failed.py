'''
util dosn't improve while still repeating.

If utilization does not improve for 10 steps, do not place boxes at the same or similar positions again and temporarily disable unpacking or force placements in a different area. At the same time, keep stable while maximal the util
'''

import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Initialize recent actions storage
    if not hasattr(llm_policy, '_recent'):
        llm_policy._recent = []
    
    # Initialize utilization tracking
    if not hasattr(llm_policy, '_util_history'):
        llm_policy._util_history = []
    if not hasattr(llm_policy, '_bad_positions'):
        llm_policy._bad_positions = set()
    if not hasattr(llm_policy, '_unpack_disabled'):
        llm_policy._unpack_disabled = False
    
    # Get observation data
    pallet = obs['pallet_obs_density']
    buffer = obs['buffer']
    pallet_softness = obs['pallet_obs_softness']
    buffer_physics = obs['buffer_physics']
    front_ids = obs['front_ids']
    pallet_count = obs['pallet_count']
    pallet_footprints = obs['pallet_footprints']
    pallet_ids = obs['pallet_ids']
    removable_mask = obs['removable_mask']
    
    X, Y, H = pallet.shape
    eps = 1e-6
    
    # Calculate current utilization
    current_util = np.sum(pallet > eps) / (X * Y * H)
    llm_policy._util_history.append(current_util)
    if len(llm_policy._util_history) > 10:
        llm_policy._util_history.pop(0)
    
    # Check if utilization hasn't improved for 10 steps
    if len(llm_policy._util_history) >= 10:
        util_improved = False
        for i in range(1, 10):
            if llm_policy._util_history[-i] > llm_policy._util_history[-i-1]:
                util_improved = True
                break
        
        if not util_improved:
            # Mark recent placement positions as bad
            for action in llm_policy._recent:
                if action[0] == 0:  # PLACE action
                    x, y = action[3], action[4]
                    # Mark this position and neighboring positions as bad
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            bad_x = max(0, min(X-1, x + dx))
                            bad_y = max(0, min(Y-1, y + dy))
                            llm_policy._bad_positions.add((bad_x, bad_y))
            
            # Temporarily disable unpacking
            llm_policy._unpack_disabled = True
        else:
            # Clear bad positions if utilization improved
            llm_policy._bad_positions.clear()
            llm_policy._unpack_disabled = False
    
    # Calculate number of slots
    n_slots = int(buffer.size // heur.n_properties)
    
    # PASS-1: Safe place (avoid hard on soft support)
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
            
        # Get box properties
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        # Get box softness
        if 2 * slot_i + 0 < len(buffer_physics):
            softness = float(buffer_physics[2 * slot_i + 0])
            box_is_soft = (softness > 0.5)
        else:
            box_is_soft = False
        
        # Try all rotations
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            
            # Try all positions
            for x in range(X):
                for y in range(Y):
                    # Skip bad positions if utilization stuck
                    if (x, y) in llm_policy._bad_positions:
                        continue
                    
                    # Check if within pallet bounds
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Calculate placement height
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
                    if z + dz <= H and dx > 0 and dy > 0 and dz > 0:
                        blk = pallet[x:x+dx, y:y+dy, z:z+dz]
                        if blk.size > 0 and np.any(blk > eps):
                            continue
                    
                    # Check feasibility
                    if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        continue
                    
                    # Check hard-on-soft rule for PASS-1
                    if not box_is_soft and z > 0:
                        support = pallet_softness[x:x+dx, y:y+dy, z-1]
                        if support.size > 0 and np.max(support) > 0.5:
                            # Hard box on soft support - reject in PASS-1
                            continue
                    
                    # Create candidate action
                    candidate = (0, slot_i, rot_id, x, y)
                    
                    # Check anti-repeat rule
                    if candidate in llm_policy._recent:
                        continue
                    
                    # Update recent actions
                    llm_policy._recent.append(candidate)
                    if len(llm_policy._recent) > 8:
                        llm_policy._recent.pop(0)
                    
                    return (int(candidate[0]), int(candidate[1]), int(candidate[2]), 
                            int(candidate[3]), int(candidate[4]))
    
    # PASS-2: Remove action if no safe place found and removable boxes exist
    # Skip removal if unpacking is temporarily disabled
    if not llm_policy._unpack_disabled and pallet_count > 0 and len(removable_mask) > 0:
        # Find removable indices
        removable_indices = [i for i in range(pallet_count) if removable_mask[i] > 0]
        
        if removable_indices:
            # Prefer last removable index (highest index)
            for idx in reversed(removable_indices):
                candidate = (1, idx, 0, 0, 0)
                
                # Check anti-repeat rule
                if candidate in llm_policy._recent:
                    continue
                
                # Update recent actions
                llm_policy._recent.append(candidate)
                if len(llm_policy._recent) > 8:
                    llm_policy._recent.pop(0)
                
                return (int(candidate[0]), int(candidate[1]), int(candidate[2]), 
                        int(candidate[3]), int(candidate[4]))
    
    # PASS-3: Relax rule - allow hard on soft when no other option
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
            
        # Get box properties
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        # Try all rotations
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            
            # Try all positions, avoiding bad positions if possible
            # First try positions not in bad_positions
            positions_to_try = []
            for x in range(X):
                for y in range(Y):
                    if (x, y) not in llm_policy._bad_positions:
                        positions_to_try.append((x, y))
            
            # If all positions are bad, try them anyway
            if not positions_to_try:
                for x in range(X):
                    for y in range(Y):
                        positions_to_try.append((x, y))
            
            for x, y in positions_to_try:
                # Check if within pallet bounds
                if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                    continue
                
                # Calculate placement height
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
                if z + dz <= H and dx > 0 and dy > 0 and dz > 0:
                    blk = pallet[x:x+dx, y:y+dy, z:z+dz]
                    if blk.size > 0 and np.any(blk > eps):
                        continue
                
                # Check feasibility
                if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                    continue
                
                # Create candidate action (hard-on-soft allowed here)
                candidate = (0, slot_i, rot_id, x, y)
                
                # Check anti-repeat rule
                if candidate in llm_policy._recent:
                    continue
                
                # Update recent actions
                llm_policy._recent.append(candidate)
                if len(llm_policy._recent) > 8:
                    llm_policy._recent.pop(0)
                
                return (int(candidate[0]), int(candidate[1]), int(candidate[2]), 
                        int(candidate[3]), int(candidate[4]))
    
    # Last resort: return zero action
    candidate = (0, 0, 0, 0, 0)
    llm_policy._recent.append(candidate)
    if len(llm_policy._recent) > 8:
        llm_policy._recent.pop(0)
    
    return (int(candidate[0]), int(candidate[1]), int(candidate[2]), 
            int(candidate[3]), int(candidate[4]))

class ArchivedHeuristic(BaseHeuristic):
    name = "worker_failed"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
