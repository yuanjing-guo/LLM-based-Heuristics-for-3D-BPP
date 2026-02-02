import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Initialize recent actions storage
    if not hasattr(llm_policy, '_recent'):
        llm_policy._recent = []
    
    # Helper to check if action is recent
    def is_recent(action_tuple):
        return any(np.array_equal(np.array(action_tuple), np.array(prev)) for prev in llm_policy._recent)
    
    # Update recent actions (keep last 8)
    if len(llm_policy._recent) >= 8:
        llm_policy._recent.pop(0)
    
    pallet = obs['pallet_obs_density']
    X, Y, H = pallet.shape
    eps = 1e-6
    
    # PASS-1: Safe place (avoid hard-on-soft) with large box preference
    n_slots = int(obs['buffer'].size // heur.n_properties)
    
    # First, collect all safe candidates with their volume
    safe_candidates = []
    
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        # Get box properties
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        # Get softness
        if 2*slot_i + 0 < len(obs['buffer_physics']):
            softness = float(obs['buffer_physics'][2*slot_i + 0])
            box_is_soft = (softness > 0.5)
        else:
            box_is_soft = False
        box_is_hard = not box_is_soft
        
        # Try rotations
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            volume = dx * dy * dz
            
            # Try positions
            for x in range(X):
                for y in range(Y):
                    # Check within pallet
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Compute z height
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
                    
                    # Hard-on-soft check (only if z>0)
                    if box_is_hard and z > 0:
                        support = obs['pallet_obs_softness'][x:x+dx, y:y+dy, z-1]
                        if support.size > 0 and np.max(support) > 0.5:
                            continue  # Reject hard-on-soft in PASS-1
                    
                    action = (0, slot_i, rot_id, x, y)
                    if not is_recent(action):
                        safe_candidates.append((volume, action))
    
    # Sort safe candidates by volume (largest first) and return first non-recent
    if safe_candidates:
        safe_candidates.sort(key=lambda x: x[0], reverse=True)  # Sort by volume descending
        for volume, action in safe_candidates:
            if not is_recent(action):
                llm_policy._recent.append(action)
                return (int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4]))
    
    # PASS-2: Remove small boxes to make room for large ones
    pallet_count = obs['pallet_count']
    if pallet_count > 0 and 'removable_mask' in obs:
        removable_indices = [i for i in range(pallet_count) if obs['removable_mask'][i] > 0]
        if removable_indices:
            # Find smallest removable box to replace
            smallest_volume = float('inf')
            smallest_idx = -1
            
            for idx in removable_indices:
                # Get box footprint
                fp = obs['pallet_footprints'][idx]
                x, y, z, dx, dy, dz = fp
                volume = dx * dy * dz
                
                if volume < smallest_volume:
                    smallest_volume = volume
                    smallest_idx = idx
            
            if smallest_idx >= 0:
                action = (1, smallest_idx, 0, 0, 0)
                if not is_recent(action):
                    llm_policy._recent.append(action)
                    return (int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4]))
    
    # PASS-3: Relax rule (allow hard-on-soft) with large box preference
    relaxed_candidates = []
    
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            volume = dx * dy * dz
            
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
                    if not is_recent(action):
                        relaxed_candidates.append((volume, action))
    
    # Sort relaxed candidates by volume (largest first)
    if relaxed_candidates:
        relaxed_candidates.sort(key=lambda x: x[0], reverse=True)  # Sort by volume descending
        for volume, action in relaxed_candidates:
            if not is_recent(action):
                llm_policy._recent.append(action)
                return (int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4]))
    
    # Last resort
    action = (0, 0, 0, 0, 0)
    llm_policy._recent.append(action)
    return (int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4]))

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
