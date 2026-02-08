'''
still repeat, dosn't improve

I try to put higher weight on util than stable but still repeat.

If utilization does not improve for 10 steps, prioritize breaking repetition over stability or maximal utilization: forbid similar placements even if they seem more stable or optimal, and only restore stability preferences after progress resumes.


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
    if not hasattr(llm_policy, '_stuck_counter'):
        llm_policy._stuck_counter = 0
    
    # Constants
    eps = 1e-6
    X, Y, H = heur.X, heur.Y, heur.H
    
    # Get observation data
    pallet = obs['pallet_obs_density']
    pallet_softness = obs['pallet_obs_softness']
    buffer = obs['buffer']
    buffer_physics = obs['buffer_physics']
    pallet_count = obs['pallet_count']
    removable_mask = obs['removable_mask']
    
    # Calculate current utilization
    current_util = np.sum(pallet > eps)
    llm_policy._util_history.append(current_util)
    if len(llm_policy._util_history) > 1:
        if llm_policy._util_history[-1] <= llm_policy._util_history[-2]:
            llm_policy._stuck_counter += 1
        else:
            llm_policy._stuck_counter = 0
    # Keep history manageable
    if len(llm_policy._util_history) > 20:
        llm_policy._util_history.pop(0)
    
    # Calculate number of slots
    n_slots = int(buffer.size // heur.n_properties)
    
    # PASS-1: Safe place (avoid hard-on-soft)
    candidates_pass1 = []
    for slot_i in range(n_slots):
        # Skip empty slots
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
        box_is_hard = not box_is_soft
        
        # Try all rotations
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            
            # Try all positions
            for x in range(X):
                for y in range(Y):
                    # Check within pallet bounds
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Compute placement height
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
                    
                    # Check hard-on-soft rule (only if z > 0)
                    if box_is_hard and z > 0:
                        support = pallet_softness[x:x+dx, y:y+dy, z-1]
                        if support.size > 0 and np.max(support) > 0.5:
                            # Reject in PASS-1
                            continue
                    
                    action = (0, slot_i, rot_id, x, y)
                    candidates_pass1.append(action)
    
    # Filter candidates by anti-repeat and stuck condition
    filtered_candidates_pass1 = []
    for action in candidates_pass1:
        # Always check basic anti-repeat
        if action in llm_policy._recent:
            continue
        
        # If stuck for 10+ steps, apply stricter similarity check
        if llm_policy._stuck_counter >= 10:
            # Forbid placements with similar coordinates (within 2 bins)
            slot_i, rot_id, x, y = action[1], action[2], action[3], action[4]
            too_similar = False
            for past_action in llm_policy._recent:
                if past_action[0] == 0:  # Only compare PLACE actions
                    past_slot, past_rot, past_x, past_y = past_action[1], past_action[2], past_action[3], past_action[4]
                    # Consider similar if same slot and nearby position
                    if slot_i == past_slot and abs(x - past_x) <= 2 and abs(y - past_y) <= 2:
                        too_similar = True
                        break
            if too_similar:
                continue
        
        filtered_candidates_pass1.append(action)
    
    # Return first filtered candidate from PASS-1
    if filtered_candidates_pass1:
        action = filtered_candidates_pass1[0]
        llm_policy._recent.append(action)
        if len(llm_policy._recent) > 8:
            llm_policy._recent.pop(0)
        return action
    
    # PASS-2: Remove action if available
    if pallet_count > 0 and len(removable_mask) > 0:
        # Find removable indices
        removable_indices = [idx for idx in range(pallet_count) if removable_mask[idx] > 0]
        
        if removable_indices:
            # Prefer last removable index (highest index)
            for idx in reversed(removable_indices):
                action = (1, idx, 0, 0, 0)
                # Check anti-repeat
                if action in llm_policy._recent:
                    continue
                
                # Update recent actions
                llm_policy._recent.append(action)
                if len(llm_policy._recent) > 8:
                    llm_policy._recent.pop(0)
                
                return (1, idx, 0, 0, 0)
    
    # PASS-3: Relaxed place (ignore hard-on-soft rule)
    candidates_pass3 = []
    for slot_i in range(n_slots):
        # Skip empty slots
        if heur.slot_is_empty(obs, slot_i):
            continue
            
        # Get box properties
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        # Try all rotations
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            
            # Try all positions
            for x in range(X):
                for y in range(Y):
                    # Check within pallet bounds
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Compute placement height
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
                    
                    action = (0, slot_i, rot_id, x, y)
                    candidates_pass3.append(action)
    
    # Filter candidates by anti-repeat and stuck condition
    filtered_candidates_pass3 = []
    for action in candidates_pass3:
        # Always check basic anti-repeat
        if action in llm_policy._recent:
            continue
        
        # If stuck for 10+ steps, apply stricter similarity check
        if llm_policy._stuck_counter >= 10:
            # Forbid placements with similar coordinates (within 2 bins)
            slot_i, rot_id, x, y = action[1], action[2], action[3], action[4]
            too_similar = False
            for past_action in llm_policy._recent:
                if past_action[0] == 0:  # Only compare PLACE actions
                    past_slot, past_rot, past_x, past_y = past_action[1], past_action[2], past_action[3], past_action[4]
                    # Consider similar if same slot and nearby position
                    if slot_i == past_slot and abs(x - past_x) <= 2 and abs(y - past_y) <= 2:
                        too_similar = True
                        break
            if too_similar:
                continue
        
        filtered_candidates_pass3.append(action)
    
    # Return first filtered candidate from PASS-3
    if filtered_candidates_pass3:
        action = filtered_candidates_pass3[0]
        llm_policy._recent.append(action)
        if len(llm_policy._recent) > 8:
            llm_policy._recent.pop(0)
        return action
    
    # Last resort: return zero action
    action = (0, 0, 0, 0, 0)
    llm_policy._recent.append(action)
    if len(llm_policy._recent) > 8:
        llm_policy._recent.pop(0)
    return (0, 0, 0, 0, 0)

class ArchivedHeuristic(BaseHeuristic):
    name = "worker_failed_3"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
