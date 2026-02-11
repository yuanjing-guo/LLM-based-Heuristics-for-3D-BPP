import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    n_slots = int(obs['buffer'].size // heur.n_properties)
    pallet = obs['pallet_obs_density']
    
    # Get softness info if available
    physics_mode_soft = True  # from context
    expose_physics = True  # from context
    use_physics = physics_mode_soft and expose_physics
    
    # Get softness arrays if available
    pallet_softness = None
    buffer_physics = None
    if use_physics:
        if 'pallet_obs_softness' in obs:
            pallet_softness = obs['pallet_obs_softness']
        if 'buffer_physics' in obs:
            buffer_physics = obs['buffer_physics']
    
    # Helper to check if box is hard
    def is_hard_box(slot_idx):
        if buffer_physics is not None and slot_idx < len(buffer_physics):
            return buffer_physics[slot_idx] < 0.5  # Assuming 0=hard, 1=soft
        # Fallback: check if we can determine from properties
        props = heur.get_slot_props(obs, slot_idx)
        # Property 3 might indicate softness (0=hard, 1=soft)
        if len(props) > 3:
            return props[3] < 0.5
        return True  # Default to hard if unknown
    
    # Helper to check if support at (x,y,z-1) is hard
    def has_hard_support(x, y, z, dx, dy):
        if z == 0:
            return True  # Ground support is always hard
        if pallet_softness is None:
            return True  # Can't check without softness info
        
        # Check the entire footprint at layer z-1
        support_layer = pallet_softness[x:x+dx, y:y+dy, z-1]
        # If any cell in the support area is soft (>=0.5), support is not fully hard
        return not np.any(support_layer >= 0.5)
    
    # Helper to check if placement reuses same footprint and increases height
    def reuses_footprint_and_increases_height(x, y, dx, dy, z):
        if z == 0:
            return False
        # Check if there's already a box at (x,y,z) with same footprint
        current_footprint = pallet[x:x+dx, y:y+dy, z]
        return np.any(current_footprint > 0)
    
    # Simple EMS-like space generation: prioritize lower z, then lower x, then lower y
    # We'll generate candidate positions in a "good" order
    candidate_positions = []
    for x in range(heur.X):
        for y in range(heur.Y):
            # Find current height at this (x,y)
            column = pallet[x, y, :]
            non_zero = np.where(column > 0)[0]
            z = int(non_zero[-1] + 1) if non_zero.size > 0 else 0
            candidate_positions.append((z, x, y))
    
    # Sort by z (lowest first), then x, then y
    candidate_positions.sort(key=lambda pos: (pos[0], pos[1], pos[2]))
    
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        # Check if box is hard
        box_is_hard = is_hard_box(slot_i)
        
        for rot_id in range(6):
            rotated = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated[0], rotated[1], rotated[2]
            
            # Rule (1): Only allow horizontal placements (largest face parallel to pallet)
            # This means dz should be the smallest dimension
            sorted_dims = sorted([dx, dy, dz])
            if dz != sorted_dims[0]:  # dz is not the smallest
                continue
            
            # Check if box fits in pallet
            if dx > heur.X or dy > heur.Y or dz > heur.H:
                continue
            
            # Try candidate positions from EMS-like heuristic
            for z_candidate, x_candidate, y_candidate in candidate_positions:
                x, y = x_candidate, y_candidate
                
                # Adjust position to ensure box fits
                if x + dx > heur.X:
                    x = heur.X - dx
                if y + dy > heur.Y:
                    y = heur.Y - dy
                if x < 0 or y < 0:
                    continue
                
                if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                    continue
                
                # Compute actual z at this position
                place_area = pallet[x:x+dx, y:y+dy, :]
                non_zero = np.any(place_area > 0, axis=(0,1))
                if np.any(non_zero):
                    z = int(np.max(np.nonzero(non_zero)) + 1)
                else:
                    z = 0
                
                # Rule (2): When z==0, only hard boxes on bottom
                if z == 0 and not box_is_hard:
                    continue
                
                if z + dz > heur.H:
                    continue
                
                # Rule (3): When z>0, allow placement only on hard support
                if z > 0 and not has_hard_support(x, y, z, dx, dy):
                    continue
                
                # Rule (3): Skip placements that reuse same footprint and increase height
                if reuses_footprint_and_increases_height(x, y, dx, dy, z):
                    continue
                
                if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                    return (slot_i, rot_id, x, y)
    
    # Fallback: original scanning if EMS-like fails
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        box_is_hard = is_hard_box(slot_i)
        
        for rot_id in range(6):
            rotated = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated[0], rotated[1], rotated[2]
            
            # Rule (1)
            sorted_dims = sorted([dx, dy, dz])
            if dz != sorted_dims[0]:
                continue
            
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero = np.any(place_area > 0, axis=(0,1))
                    if np.any(non_zero):
                        z = int(np.max(np.nonzero(non_zero)) + 1)
                    else:
                        z = 0
                    
                    # Rule (2)
                    if z == 0 and not box_is_hard:
                        continue
                    
                    if z + dz > heur.H:
                        continue
                    
                    # Rule (3)
                    if z > 0 and not has_hard_support(x, y, z, dx, dy):
                        continue
                    
                    # Rule (3) footprint reuse check
                    if reuses_footprint_and_increases_height(x, y, dx, dy, z):
                        continue
                    
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        return (slot_i, rot_id, x, y)
    
    return (0, 0, 0, 0)

class ArchivedHeuristic(BaseHeuristic):
    name = "test4"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
