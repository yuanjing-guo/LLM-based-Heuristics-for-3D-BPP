import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Initialize recent actions storage
    if not hasattr(llm_policy, '_recent'):
        llm_policy._recent = []
    
    # Get observation data
    pallet = obs['pallet_obs_density']
    buffer = obs['buffer']
    pallet_softness = obs['pallet_obs_softness']
    buffer_physics = obs['buffer_physics']
    pallet_count = obs['pallet_count']
    removable_mask = obs['removable_mask']
    
    # Constants
    X, Y, H = pallet.shape
    eps = 1e-6
    n_slots = int(buffer.size // heur.n_properties)
    
    # Helper to check if action repeats recent ones
    def is_recent_action(action_tuple):
        return action_tuple in llm_policy._recent
    
    # PASS-1: Safe place (avoid hard-on-soft)
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        # Get box properties
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        # Get softness from buffer_physics
        softness = 0.0
        if 2 * slot_i + 0 < len(buffer_physics):
            softness = float(buffer_physics[2 * slot_i + 0])
        box_is_soft = (softness > 0.5)
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
                    
                    # Check hard-on-soft rule for PASS-1
                    if box_is_hard and z > 0:
                        support = pallet_softness[x:x+dx, y:y+dy, z-1]
                        support_is_soft = (np.max(support) > 0.5)
                        if support_is_soft:
                            continue  # Reject hard-on-soft in PASS-1
                    
                    # Check for repeat
                    action_tuple = (0, slot_i, rot_id, x, y)
                    if is_recent_action(action_tuple):
                        continue
                    
                    # Update recent actions
                    llm_policy._recent.append(action_tuple)
                    if len(llm_policy._recent) > 8:
                        llm_policy._recent.pop(0)
                    
                    return (0, slot_i, rot_id, x, y)
    
    # PASS-2: Remove if possible
    if pallet_count > 0 and len(removable_mask) > 0:
        # Find removable indices
        removable_indices = [idx for idx in range(pallet_count) if removable_mask[idx] > 0]
        
        if removable_indices:
            # Prefer LAST removable index (highest index)
            for idx in reversed(removable_indices):
                action_tuple = (1, idx, 0, 0, 0)
                if is_recent_action(action_tuple):
                    continue
                
                # Update recent actions
                llm_policy._recent.append(action_tuple)
                if len(llm_policy._recent) > 8:
                    llm_policy._recent.pop(0)
                
                return (1, idx, 0, 0, 0)
    
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
                    
                    action_tuple = (0, slot_i, rot_id, x, y)
                    if is_recent_action(action_tuple):
                        continue
                    
                    # Update recent actions
                    llm_policy._recent.append(action_tuple)
                    if len(llm_policy._recent) > 8:
                        llm_policy._recent.pop(0)
                    
                    return (0, slot_i, rot_id, x, y)
    
    # Last resort: default action
    action_tuple = (0, 0, 0, 0, 0)
    llm_policy._recent.append(action_tuple)
    if len(llm_policy._recent) > 8:
        llm_policy._recent.pop(0)
    return (0, 0, 0, 0, 0)

class GeneratedHeuristic(BaseHeuristic):
    name = "llm_generated"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        out = llm_policy(self, obs)

        # robust unpack
        if isinstance(out, (list, tuple, np.ndarray)):
            out = list(out)
        else:
            out = [out]

        # pad / truncate to 5
        while len(out) < 5:
            out.append(0)
        if len(out) > 5:
            out = out[:5]

        op, a1, a2, a3, a4 = [int(x) for x in out]

        # IMPORTANT: Mixed env consumes raw int32[5] directly (no logits)
        return np.array([op, a1, a2, a3, a4], dtype=np.int32)
