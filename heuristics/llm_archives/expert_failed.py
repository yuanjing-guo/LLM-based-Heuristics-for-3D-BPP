'''
util=0.5395 status=height_oob boxes=21
When breaking repetition, do not just slightly change positions; move to a different region or layer of the pallet instead of reusing the same local area.
'''


import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Initialize recent actions storage
    if not hasattr(llm_policy, "_recent"):
        llm_policy._recent = []
    
    # Get observation data
    pallet = obs['pallet_obs_density']
    pallet_soft = obs['pallet_obs_softness']
    buffer = obs['buffer']
    buffer_physics = obs['buffer_physics']
    pallet_count = obs['pallet_count']
    removable_mask = obs['removable_mask']
    
    X, Y, H = pallet.shape
    eps = 1e-6
    n_slots = int(buffer.size // heur.n_properties)
    
    # Helper to check if action repeats recent pattern
    def is_repetitive(action, recent_actions):
        if not recent_actions:
            return False
        
        # For PLACE actions, check if we're placing in same local area
        if action[0] == 0:  # PLACE
            slot_i, rot_id, x, y = action[1], action[2], action[3], action[4]
            
            # Get box size for this placement
            if not heur.slot_is_empty(obs, slot_i):
                props = heur.get_slot_props(obs, slot_i)
                size_bins = heur.props_to_size_bins(props)
                dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
                
                # Compute placement region bounds
                x_end = x + dx
                y_end = y + dy
                
                # Check against recent PLACE actions
                for recent in recent_actions:
                    if recent[0] == 0:  # PLACE
                        r_slot, r_rot, r_x, r_y = recent[1], recent[2], recent[3], recent[4]
                        if not heur.slot_is_empty(obs, r_slot):
                            r_props = heur.get_slot_props(obs, r_slot)
                            r_size = heur.props_to_size_bins(r_props)
                            r_dx, r_dy, r_dz = [int(v) for v in heur.rotate_size_bins(r_size, r_rot)]
                            r_x_end = r_x + r_dx
                            r_y_end = r_y + r_dy
                            
                            # Check if regions overlap significantly (same local area)
                            x_overlap = max(0, min(x_end, r_x_end) - max(x, r_x))
                            y_overlap = max(0, min(y_end, r_y_end) - max(y, r_y))
                            
                            # If overlap area is more than 50% of either box's area, it's repetitive
                            if x_overlap > 0 and y_overlap > 0:
                                area1 = dx * dy
                                area2 = r_dx * r_dy
                                overlap_area = x_overlap * y_overlap
                                if overlap_area > 0.5 * min(area1, area2):
                                    return True
        
        # For REMOVE actions, check if same index
        elif action[0] == 1:  # REMOVE
            for recent in recent_actions:
                if recent[0] == 1 and recent[1] == action[1]:
                    return True
        
        return False
    
    # PASS-1: Find safe place (hard boxes not on soft support)
    candidates = []
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        # Get box properties
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        # Get softness for this slot
        softness = 0.0
        if 2*slot_i + 0 < len(buffer_physics):
            softness = float(buffer_physics[2*slot_i + 0])
        box_is_soft = (softness > 0.5)
        
        # Try all rotations
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            
            # Try all positions
            for x in range(X):
                for y in range(Y):
                    # Check if within pallet bounds
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Compute height z
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
                            # Hard box on soft support - reject for PASS-1
                            continue
                    
                    # Candidate found
                    action = (0, slot_i, rot_id, x, y)
                    # Check anti-repeat (both exact match and region pattern)
                    if (len(llm_policy._recent) == 0 or action != llm_policy._recent[-1]) and \
                       not is_repetitive(action, llm_policy._recent):
                        candidates.append(action)
    
    # Return first safe candidate
    if candidates:
        action = candidates[0]
        llm_policy._recent.append(action)
        if len(llm_policy._recent) > 8:
            llm_policy._recent.pop(0)
        return (int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4]))
    
    # PASS-2: Remove if possible
    if pallet_count > 0 and len(removable_mask) > 0:
        # Find removable indices
        removable_indices = [idx for idx in range(pallet_count) if removable_mask[idx] > 0]
        if removable_indices:
            # Prefer last removable index (highest index)
            pallet_index = removable_indices[-1]
            action = (1, pallet_index, 0, 0, 0)
            # Check anti-repeat
            if (len(llm_policy._recent) == 0 or action != llm_policy._recent[-1]) and \
               not is_repetitive(action, llm_policy._recent):
                llm_policy._recent.append(action)
                if len(llm_policy._recent) > 8:
                    llm_policy._recent.pop(0)
                return (int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4]))
    
    # PASS-3: Relax rule - allow hard-on-soft
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
                    # Check anti-repeat (both exact match and region pattern)
                    if (len(llm_policy._recent) == 0 or action != llm_policy._recent[-1]) and \
                       not is_repetitive(action, llm_policy._recent):
                        llm_policy._recent.append(action)
                        if len(llm_policy._recent) > 8:
                            llm_policy._recent.pop(0)
                        return (int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4]))
    
    # Last resort: return zero action
    action = (0, 0, 0, 0, 0)
    llm_policy._recent.append(action)
    if len(llm_policy._recent) > 8:
        llm_policy._recent.pop(0)
    return (int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4]))

class ArchivedHeuristic(BaseHeuristic):
    name = "expert_failed"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
