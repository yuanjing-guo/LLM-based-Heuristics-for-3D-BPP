'''
still repeats, util decreases.

the agent try to keep stable, so it repeats the same but safe action.

If utilization does not improve for 10 steps, do not place boxes at the same or similar positions again and temporarily disable unpacking or force placements in a different area. and keep stable

'''

import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Initialize recent actions storage
    if not hasattr(llm_policy, "_recent"):
        llm_policy._recent = []
    if not hasattr(llm_policy, "_utilization_history"):
        llm_policy._utilization_history = []
    if not hasattr(llm_policy, "_no_improve_steps"):
        llm_policy._no_improve_steps = 0
    if not hasattr(llm_policy, "_bad_positions"):
        llm_policy._bad_positions = set()
    if not hasattr(llm_policy, "_unpacking_disabled"):
        llm_policy._unpacking_disabled = False
    
    # Helper to check if action repeats last action
    def is_repeating(action_tuple):
        if len(llm_policy._recent) == 0:
            return False
        return action_tuple == llm_policy._recent[-1]
    
    # Helper to check if position is similar to bad positions
    def is_bad_position(x, y, dx, dy):
        for (bx, by, bdx, bdy) in llm_policy._bad_positions:
            # Check if positions overlap or are adjacent (within 2 bins)
            x_overlap = not (x + dx <= bx or bx + bdx <= x)
            y_overlap = not (y + dy <= by or by + bdy <= y)
            if x_overlap and y_overlap:
                return True
            # Check if centers are close (within 3 bins)
            cx1, cy1 = x + dx//2, y + dy//2
            cx2, cy2 = bx + bdx//2, by + bdy//2
            if abs(cx1 - cx2) <= 3 and abs(cy1 - cy2) <= 3:
                return True
        return False
    
    # Get basic info
    pallet = obs['pallet_obs_density']
    X, Y, H = pallet.shape
    eps = 1e-6
    
    # Track utilization (approximate as filled volume)
    filled_volume = np.sum(pallet > eps)
    total_volume = X * Y * H
    current_util = filled_volume / total_volume if total_volume > 0 else 0
    
    # Update utilization history
    llm_policy._utilization_history.append(current_util)
    if len(llm_policy._utilization_history) > 10:
        llm_policy._utilization_history.pop(0)
    
    # Check if utilization improved in last 10 steps
    if len(llm_policy._utilization_history) >= 10:
        if llm_policy._utilization_history[-1] <= max(llm_policy._utilization_history[:-1]):
            llm_policy._no_improve_steps += 1
        else:
            llm_policy._no_improve_steps = 0
    
    # If no improvement for 10+ steps, activate avoidance measures
    if llm_policy._no_improve_steps >= 10:
        llm_policy._unpacking_disabled = True
    else:
        llm_policy._unpacking_disabled = False
        # Clear bad positions when utilization improves
        llm_policy._bad_positions.clear()
    
    # Slot iteration
    n_slots = int(obs['buffer'].size // heur.n_properties)
    
    # PASS-1: Safe place (hard boxes not on soft support)
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
            
        # Get box properties
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        # Get softness info for this box
        if 'buffer_physics' in obs and 2*slot_i + 0 < len(obs['buffer_physics']):
            softness = float(obs['buffer_physics'][2*slot_i + 0])
            box_is_soft = (softness > 0.5)
        else:
            box_is_soft = False
        box_is_hard = not box_is_soft
        
        # Try all rotations
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            
            # Try all positions
            for x in range(X):
                for y in range(Y):
                    # Check within pallet
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Skip bad positions if utilization stuck
                    if llm_policy._no_improve_steps >= 10 and is_bad_position(x, y, dx, dy):
                        continue
                    
                    # Compute height
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero_mask = np.any(place_area > 0, axis=(0,1))
                    if np.any(non_zero_mask):
                        z = int(np.max(np.nonzero(non_zero_mask)) + 1)
                    else:
                        z = 0
                    
                    # Check height
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
                    
                    # Check hard-on-soft rule for PASS-1
                    if box_is_hard and z > 0:
                        if 'pallet_obs_softness' in obs:
                            support = obs['pallet_obs_softness'][x:x+dx, y:y+dy, z-1]
                            if np.max(support) > 0.5:  # soft support
                                continue  # REJECT in PASS-1
                    
                    # Check anti-repeat
                    action = (0, slot_i, rot_id, x, y)
                    if is_repeating(action):
                        continue
                    
                    # Update recent actions and return
                    llm_policy._recent.append(action)
                    if len(llm_policy._recent) > 8:
                        llm_policy._recent.pop(0)
                    
                    # Record this position if utilization stuck
                    if llm_policy._no_improve_steps >= 10:
                        llm_policy._bad_positions.add((x, y, dx, dy))
                    
                    return (int(action[0]), int(action[1]), int(action[2]), 
                            int(action[3]), int(action[4]))
    
    # PASS-2: Remove if possible (but skip if unpacking disabled)
    if not llm_policy._unpacking_disabled:
        pallet_count = obs['pallet_count']
        if pallet_count > 0 and 'removable_mask' in obs:
            removable_indices = [i for i in range(pallet_count) 
                               if obs['removable_mask'][i] > 0]
            
            # Prefer LAST removable index (highest index)
            for idx in reversed(removable_indices):
                action = (1, idx, 0, 0, 0)
                if is_repeating(action):
                    continue
                    
                llm_policy._recent.append(action)
                if len(llm_policy._recent) > 8:
                    llm_policy._recent.pop(0)
                return (int(action[0]), int(action[1]), int(action[2]), 
                        int(action[3]), int(action[4]))
    
    # PASS-3: Relax rule (allow hard-on-soft)
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
            
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            
            for x in range(X):
                for y in range(Y):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Skip bad positions if utilization stuck
                    if llm_policy._no_improve_steps >= 10 and is_bad_position(x, y, dx, dy):
                        continue
                    
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero_mask = np.any(place_area > 0, axis=(0,1))
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
                    if is_repeating(action):
                        continue
                    
                    llm_policy._recent.append(action)
                    if len(llm_policy._recent) > 8:
                        llm_policy._recent.pop(0)
                    
                    # Record this position if utilization stuck
                    if llm_policy._no_improve_steps >= 10:
                        llm_policy._bad_positions.add((x, y, dx, dy))
                    
                    return (int(action[0]), int(action[1]), int(action[2]), 
                            int(action[3]), int(action[4]))
    
    # Last resort
    action = (0, 0, 0, 0, 0)
    llm_policy._recent.append(action)
    if len(llm_policy._recent) > 8:
        llm_policy._recent.pop(0)
    return (int(action[0]), int(action[1]), int(action[2]), 
            int(action[3]), int(action[4]))

class ArchivedHeuristic(BaseHeuristic):
    name = "worker_failed_2"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
