import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Determine number of visible slots
    n_slots = int(obs['buffer'].size // heur.n_properties)
    
    # First pass: try to place hard boxes at z=0
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        # Check if box is hard (property index 3 > 0 means hard)
        if props[3] <= 0:
            continue  # Skip soft boxes in first pass
        
        # Scan rotations
        for rot_id in range(6):
            rotated_size = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated_size[0], rotated_size[1], rotated_size[2]
            
            # Ensure box fits within pallet dimensions
            if dx > heur.X or dy > heur.Y or dz > heur.H:
                continue
            
            # Scan x, y positions
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    # Check within pallet bounds
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # For hard boxes at z=0, we must check if bottom layer is empty
                    pallet = obs['pallet_obs_density']
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    
                    # Check if bottom layer (z=0) is completely empty
                    bottom_slice = place_area[:, :, 0]
                    if np.any(bottom_slice > 0):
                        continue  # Bottom layer not empty, can't place hard box here
                    
                    z = 0  # Hard boxes must be placed at z=0
                    
                    # Check height overflow
                    if z + dz > heur.H:
                        continue
                    
                    # Final feasibility check
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        return (slot_i, rot_id, x, y)
    
    # Second pass: try all boxes (including soft) at any feasible z
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        # Scan rotations
        for rot_id in range(6):
            rotated_size = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated_size[0], rotated_size[1], rotated_size[2]
            
            # Ensure box fits within pallet dimensions
            if dx > heur.X or dy > heur.Y or dz > heur.H:
                continue
            
            # Scan x, y positions
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    # Check within pallet bounds
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Compute z exactly as in env.py
                    pallet = obs['pallet_obs_density']
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero = np.any(place_area > 0, axis=(0, 1))
                    if np.any(non_zero):
                        z = int(np.max(np.nonzero(non_zero)) + 1)
                    else:
                        z = 0
                    
                    # Check height overflow
                    if z + dz > heur.H:
                        continue
                    
                    # Final feasibility check
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        return (slot_i, rot_id, x, y)
    
    # No feasible action found
    return (0, 0, 0, 0)

class ArchivedHeuristic(BaseHeuristic):
    name = "test3"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
