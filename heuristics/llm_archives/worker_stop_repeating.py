'''
util doesn't improve.

before training, the agent repeat every 5 steps.
If utilization does not improve for 10 steps, do not place boxes at the same or similar positions again and temporarily disable unpacking or force placements in a different area.
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
    
    # Constants
    eps = 1e-6
    X, Y, H = heur.X, heur.Y, heur.H
    n_slots = int(obs['buffer'].size // heur.n_properties)
    pallet = obs['pallet_obs_density']
    
    # Helper to check if action repeats recent ones
    def is_recent_action(action_tuple):
        return action_tuple in llm_policy._recent
    
    # Helper to check if position is similar to bad positions
    def is_bad_position(x, y, dx, dy):
        if not llm_policy._bad_positions:
            return False
        # Check if any corner of the placement area is near a bad position
        for bad_x, bad_y in llm_policy._bad_positions:
            if abs(x - bad_x) <= 2 and abs(y - bad_y) <= 2:
                return True
            if abs(x + dx - 1 - bad_x) <= 2 and abs(y + dy - 1 - bad_y) <= 2:
                return True
        return False
    
    # Track current utilization
    current_util = np.sum(pallet > eps)
    llm_policy._util_history.append(current_util)
    if len(llm_policy._util_history) > 10:
        llm_policy._util_history.pop(0)
    
    # Check if utilization hasn't improved for 10 steps
    if len(llm_policy._util_history) >= 10:
        if llm_policy._util_history[0] >= llm_policy._util_history[-1]:
            # Utilization hasn't improved - mark current occupied positions as bad
            occupied_positions = np.where(pallet[:, :, :] > eps)
            if len(occupied_positions[0]) > 0:
                for i in range(len(occupied_positions[0])):
                    x, y = occupied_positions[0][i], occupied_positions[1][i]
                    llm_policy._bad_positions.add((int(x), int(y)))
            # Disable unpacking
            llm_policy._unpack_disabled = True
        else:
            # Utilization improved - reset
            llm_policy._bad_positions.clear()
            llm_policy._unpack_disabled = False
    
    # PASS-1: Safe place (avoid hard-on-soft, avoid bad positions)
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        # Get box properties
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        # Get softness
        softness = 0.0
        if slot_i * 2 + 1 < len(obs['buffer_physics']):
            softness = float(obs['buffer_physics'][2 * slot_i + 0])
        box_is_hard = (softness <= 0.5)
        
        # Try all rotations
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            
            # Try all positions
            for x in range(X):
                for y in range(Y):
                    # Check within pallet
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Skip bad positions if utilization hasn't improved
                    if is_bad_position(x, y, dx, dy):
                        continue
                    
                    # Compute z height
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero_mask = np.any(place_area > 0, axis=(0, 1))
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
                    
                    # Hard-on-soft check
                    if box_is_hard and z > 0:
                        support = obs['pallet_obs_softness'][x:x+dx, y:y+dy, z-1]
                        if np.max(support) > 0.5:  # soft support
                            continue  # Reject in PASS-1
                    
                    # Check repeat
                    action = (0, slot_i, rot_id, x, y)
                    if is_recent_action(action):
                        continue
                    
                    # Update recent actions and return
                    llm_policy._recent.append(action)
                    if len(llm_policy._recent) > 8:
                        llm_policy._recent.pop(0)
                    return (int(0), int(slot_i), int(rot_id), int(x), int(y))
    
    # PASS-2: Remove action if available and not disabled
    pallet_count = obs['pallet_count']
    if not llm_policy._unpack_disabled and pallet_count > 0 and 'removable_mask' in obs:
        removable_mask = obs['removable_mask']
        # Find removable indices (prefer last/highest index)
        removable_indices = []
        for idx in range(pallet_count):
            if removable_mask[idx] > 0:
                removable_indices.append(idx)
        
        if removable_indices:
            # Choose highest index (last)
            idx_to_remove = removable_indices[-1]
            action = (1, idx_to_remove, 0, 0, 0)
            
            if not is_recent_action(action):
                llm_policy._recent.append(action)
                if len(llm_policy._recent) > 8:
                    llm_policy._recent.pop(0)
                return (int(1), int(idx_to_remove), int(0), int(0), int(0))
    
    # PASS-3: Relaxed place (allow hard-on-soft, avoid bad positions)
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
                    
                    # Skip bad positions if utilization hasn't improved
                    if is_bad_position(x, y, dx, dy):
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
                    if is_recent_action(action):
                        continue
                    
                    llm_policy._recent.append(action)
                    if len(llm_policy._recent) > 8:
                        llm_policy._recent.pop(0)
                    return (int(0), int(slot_i), int(rot_id), int(x), int(y))
    
    # Last resort: default action (only if not a bad position)
    if not is_bad_position(0, 0, 1, 1):
        default_action = (0, 0, 0, 0, 0)
        if not is_recent_action(default_action):
            llm_policy._recent.append(default_action)
            if len(llm_policy._recent) > 8:
                llm_policy._recent.pop(0)
            return (int(0), int(0), int(0), int(0), int(0))
    
    # If default position is bad, try to find any non-bad position
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
                    
                    # Skip bad positions
                    if is_bad_position(x, y, dx, dy):
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
                    llm_policy._recent.append(action)
                    if len(llm_policy._recent) > 8:
                        llm_policy._recent.pop(0)
                    return (int(0), int(slot_i), int(rot_id), int(x), int(y))
    
    # Absolute last resort: default action even if bad position
    default_action = (0, 0, 0, 0, 0)
    llm_policy._recent.append(default_action)
    if len(llm_policy._recent) > 8:
        llm_policy._recent.pop(0)
    return (int(0), int(0), int(0), int(0), int(0))

class ArchivedHeuristic(BaseHeuristic):
    name = "worker_stop_repeating"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
