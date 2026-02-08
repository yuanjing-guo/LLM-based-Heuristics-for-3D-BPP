'''
break repeating, but still don't improve util

If utilization does not improve for 10 steps, do not place boxes at the same or similar positions again and temporarily disable unpacking or force placements in a different area. And maximal the util
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
    if not hasattr(llm_policy, '_no_improve_steps'):
        llm_policy._no_improve_steps = 0
    if not hasattr(llm_policy, '_avoid_positions'):
        llm_policy._avoid_positions = set()
    if not hasattr(llm_policy, '_disable_unpack'):
        llm_policy._disable_unpack = False
    
    # Get constants
    X, Y, H = heur.X, heur.Y, heur.H
    eps = 1e-6
    
    # Get pallet data
    pallet = obs['pallet_obs_density']
    pallet_soft = obs['pallet_obs_softness']
    
    # Calculate current utilization
    current_util = np.sum(pallet > eps)
    if llm_policy._util_history:
        last_util = llm_policy._util_history[-1]
        if current_util <= last_util:
            llm_policy._no_improve_steps += 1
        else:
            llm_policy._no_improve_steps = 0
            # Clear avoidance when utilization improves
            llm_policy._avoid_positions.clear()
            llm_policy._disable_unpack = False
    llm_policy._util_history.append(current_util)
    if len(llm_policy._util_history) > 10:
        llm_policy._util_history.pop(0)
    
    # Check if utilization hasn't improved for 10 steps
    if llm_policy._no_improve_steps >= 10:
        llm_policy._disable_unpack = True
        # Add recent placement positions to avoid set
        for action in llm_policy._recent:
            if action[0] == 0:  # PLACE action
                x, y = action[3], action[4]
                # Add this position and neighboring positions
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < X and 0 <= ny < Y:
                            llm_policy._avoid_positions.add((nx, ny))
    
    # Calculate number of slots
    n_slots = int(obs['buffer'].size // heur.n_properties)
    
    # PASS-1: Try to find safe PLACE action
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
            
        # Get box properties
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        # Get softness for this slot
        if 2*slot_i + 1 < obs['buffer_physics'].size:
            softness = float(obs['buffer_physics'][2*slot_i + 0])
            box_is_soft = (softness > 0.5)
        else:
            box_is_soft = False
        
        # Try all rotations
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            
            # Try all positions
            for x in range(X):
                for y in range(Y):
                    # Skip avoided positions if utilization stuck
                    if llm_policy._no_improve_steps >= 10 and (x, y) in llm_policy._avoid_positions:
                        continue
                    
                    # Check within pallet bounds
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Compute height
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero_mask = np.any(place_area > 0, axis=(0,1))
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
                    
                    # Check hard-on-soft rule for PASS-1
                    if not box_is_soft and z > 0:
                        support = pallet_soft[x:x+dx, y:y+dy, z-1]
                        if support.size > 0 and np.max(support) > 0.5:
                            # Hard box on soft support - reject in PASS-1
                            continue
                    
                    # Create action
                    action = (0, slot_i, rot_id, x, y)
                    
                    # Check anti-repeat
                    if action in llm_policy._recent:
                        continue
                    
                    # Update recent actions
                    llm_policy._recent.append(action)
                    if len(llm_policy._recent) > 8:
                        llm_policy._recent.pop(0)
                    
                    return (int(action[0]), int(action[1]), int(action[2]), 
                            int(action[3]), int(action[4]))
    
    # PASS-2: Try REMOVE action if available and not disabled
    if not llm_policy._disable_unpack:
        pallet_count = obs['pallet_count']
        if pallet_count > 0 and 'removable_mask' in obs:
            removable_mask = obs['removable_mask']
            # Find removable indices (prefer last/highest index)
            removable_indices = []
            for idx in range(pallet_count):
                if removable_mask[idx] > 0:
                    removable_indices.append(idx)
            
            if removable_indices:
                # Choose last removable index
                pallet_index = removable_indices[-1]
                action = (1, pallet_index, 0, 0, 0)
                
                # Check anti-repeat
                if action not in llm_policy._recent:
                    # Update recent actions
                    llm_policy._recent.append(action)
                    if len(llm_policy._recent) > 8:
                        llm_policy._recent.pop(0)
                    
                    return (int(action[0]), int(action[1]), int(action[2]), 
                            int(action[3]), int(action[4]))
    
    # PASS-3: Relax rule - allow hard-on-soft placement
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
            
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            
            for x in range(X):
                for y in range(Y):
                    # Skip avoided positions if utilization stuck
                    if llm_policy._no_improve_steps >= 10 and (x, y) in llm_policy._avoid_positions:
                        continue
                    
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
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
                    
                    if action in llm_policy._recent:
                        continue
                    
                    llm_policy._recent.append(action)
                    if len(llm_policy._recent) > 8:
                        llm_policy._recent.pop(0)
                    
                    return (int(action[0]), int(action[1]), int(action[2]), 
                            int(action[3]), int(action[4]))
    
    # Last resort: return default action
    action = (0, 0, 0, 0, 0)
    llm_policy._recent.append(action)
    if len(llm_policy._recent) > 8:
        llm_policy._recent.pop(0)
    
    return (int(action[0]), int(action[1]), int(action[2]), 
            int(action[3]), int(action[4]))

class ArchivedHeuristic(BaseHeuristic):
    name = "worker_failed_1"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
