import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    n_slots = int(obs['buffer'].size // heur.n_properties)
    pallet = obs['pallet_obs_density']
    
    # First pass: try to find a non-soft box for bottom layer placement
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        # Check if box is soft (property index 3 > 0.5)
        is_soft = props[3] > 0.5
        
        for rot_id in range(6):
            rotated = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated[0], rotated[1], rotated[2]
            
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
            
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero = np.any(place_area > 0, axis=(0, 1))
                    if np.any(non_zero):
                        z = int(np.max(np.nonzero(non_zero)) + 1)
                    else:
                        z = 0
                    
                    if z + dz > heur.H:
                        continue
                    
                    # Don't put soft box on bottom layer (z == 0)
                    if z == 0 and is_soft:
                        continue
                    
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        return (slot_i, rot_id, x, y)
    
    # Second pass: if no non-soft box found for bottom layer, allow any box
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        for rot_id in range(6):
            rotated = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated[0], rotated[1], rotated[2]
            
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
            
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero = np.any(place_area > 0, axis=(0, 1))
                    if np.any(non_zero):
                        z = int(np.max(np.nonzero(non_zero)) + 1)
                    else:
                        z = 0
                    
                    if z + dz > heur.H:
                        continue
                    
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        return (slot_i, rot_id, x, y)
    
    return (0, 0, 0, 0)

class GeneratedHeuristic(BaseHeuristic):
    name = "llm_generated"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
