import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    n_slots = int(obs['buffer'].size // heur.n_properties)
    pallet = obs['pallet_obs_density']
    
    # First pass: try to place hard boxes at z=0
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        # Check if box is hard (4th property > 0.5)
        if props[3] <= 0.5:
            continue  # Skip soft boxes in first pass
        
        for rot_id in range(6):
            rotated = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated[0], rotated[1], rotated[2]
            
            if dx > heur.X or dy > heur.Y or dz > heur.H:
                continue
            
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Force placement at bottom layer (z=0)
                    z = 0
                    
                    if z + dz > heur.H:
                        continue
                    
                    # Check if area at bottom is empty
                    place_area = pallet[x:x+dx, y:y+dy, 0]
                    if np.any(place_area > 0):
                        continue  # Bottom layer not empty here
                    
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        return (slot_i, rot_id, x, y)
    
    # Second pass: place any box (hard or soft) on top of hard boxes
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
            
            if dx > heur.X or dy > heur.Y or dz > heur.H:
                continue
            
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Find highest z with support
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero = np.any(place_area > 0, axis=(0,1))
                    if np.any(non_zero):
                        z = int(np.max(np.nonzero(non_zero)) + 1)
                    else:
                        z = 0
                    
                    # Must be above bottom layer (z > 0) for this pass
                    if z == 0:
                        continue
                    
                    if z + dz > heur.H:
                        continue
                    
                    # Check support layer (z-1) for soft boxes
                    if 'pallet_obs_softness' in obs:
                        support_layer = obs['pallet_obs_softness'][x:x+dx, y:y+dy, z-1]
                        # If any supporting cell is soft (value > 0.5), disallow placement
                        if np.any(support_layer > 0.5):
                            continue
                    
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        return (slot_i, rot_id, x, y)
    
    # Third pass: fallback to any feasible placement (including soft boxes at z=0 if needed)
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
            
            if dx > heur.X or dy > heur.Y or dz > heur.H:
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
                    
                    if z + dz > heur.H:
                        continue
                    
                    # For z > 0, check support layer if physics obs available
                    if z > 0 and 'pallet_obs_softness' in obs:
                        support_layer = obs['pallet_obs_softness'][x:x+dx, y:y+dy, z-1]
                        if np.any(support_layer > 0.5):
                            continue
                    
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        return (slot_i, rot_id, x, y)
    
    return (0, 0, 0, 0)

class ArchivedHeuristic(BaseHeuristic):
    name = "test2"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
