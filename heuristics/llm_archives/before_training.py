import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Initialize recent actions storage
    if not hasattr(llm_policy, '_recent'):
        llm_policy._recent = []
    
    # Constants
    eps = 1e-6
    X, Y, H = heur.X, heur.Y, heur.H
    n_slots = int(obs['buffer'].size // heur.n_properties)
    pallet = obs['pallet_obs_density']
    pallet_soft = obs['pallet_obs_softness']
    
    # PASS-1: Safe place (hard boxes not on soft support)
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        # Get box properties
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        # Get softness for this slot
        softness = 0.0
        if obs['buffer_physics'].size > 2 * slot_i + 1:
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
                    
                    # Hard-on-soft check (only if z > 0)
                    if box_is_hard and z > 0:
                        support = pallet_soft[x:x+dx, y:y+dy, z-1]
                        if support.size > 0 and np.max(support) > 0.5:
                            continue  # Reject in PASS-1
                    
                    # Anti-repeat check
                    action = (0, slot_i, rot_id, x, y)
                    if action in llm_policy._recent:
                        continue
                    
                    # Update recent actions
                    llm_policy._recent.append(action)
                    if len(llm_policy._recent) > 8:
                        llm_policy._recent.pop(0)
                    
                    return (0, slot_i, rot_id, x, y)
    
    # PASS-2: Remove action if available
    pallet_count = obs['pallet_count']
    if pallet_count > 0 and 'removable_mask' in obs:
        removable_mask = obs['removable_mask']
        # Find removable indices (prefer last/highest index)
        removable_indices = [i for i in range(pallet_count) if removable_mask[i] > 0]
        
        for idx in reversed(removable_indices):  # Try from highest to lowest
            action = (1, idx, 0, 0, 0)
            if action in llm_policy._recent:
                continue
            
            # Update recent actions
            llm_policy._recent.append(action)
            if len(llm_policy._recent) > 8:
                llm_policy._recent.pop(0)
            
            return (1, idx, 0, 0, 0)
    
    # PASS-3: Relaxed placement (allow hard-on-soft)
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
                    if action in llm_policy._recent:
                        continue
                    
                    # Update recent actions
                    llm_policy._recent.append(action)
                    if len(llm_policy._recent) > 8:
                        llm_policy._recent.pop(0)
                    
                    return (0, slot_i, rot_id, x, y)
    
    # Last resort: return zero action
    action = (0, 0, 0, 0, 0)
    llm_policy._recent.append(action)
    if len(llm_policy._recent) > 8:
        llm_policy._recent.pop(0)
    return (0, 0, 0, 0, 0)

class ArchivedHeuristic(BaseHeuristic):
    name = "before_training"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
