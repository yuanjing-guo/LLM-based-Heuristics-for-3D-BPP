'''
You MUST use box and pallet rigidity/softness information. Select hard boxes first from the buffer and place them on the bottom layer (z==0) whenever feasible. Do NOT place hard boxes on soft or low-rigidity support regions. Use pallet softness/rigidity to validate all placements with z > 0. Soft boxes may be placed on top of hard boxes; feasibility rules still apply.
'''
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
        
        # Check if box is hard (rigidity property is in props)
        # Assuming rigidity is the 4th property (index 3)
        if props.size >= 4 and props[3] > 0.5:  # Hard box
            # Scan rotations
            for rot_id in range(6):
                rotated = heur.rotate_size_bins(size_bins, rot_id)
                dx, dy, dz = rotated[0], rotated[1], rotated[2]
                
                if dx > heur.X or dy > heur.Y or dz > heur.H:
                    continue
                
                # For hard boxes, only consider z=0 placements
                z = 0
                if z + dz > heur.H:
                    continue
                
                # Scan positions at bottom layer
                for x in range(heur.X - dx + 1):
                    for y in range(heur.Y - dy + 1):
                        if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                            continue
                        
                        # Check if placement area at z=0 is rigid enough
                        # Use pallet_obs_softness if available
                        if 'pallet_obs_softness' in obs:
                            pallet_softness = obs['pallet_obs_softness']
                            support_area = pallet_softness[x:x+dx, y:y+dy, 0]
                            if np.any(support_area < 0.5):  # Soft support region
                                continue
                        
                        pallet = obs['pallet_obs_density']
                        if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                            return (slot_i, rot_id, x, y)
    
    # Second pass: try all boxes (including soft ones) with normal placement
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        # Check if box is hard
        is_hard = props.size >= 4 and props[3] > 0.5
        
        for rot_id in range(6):
            rotated = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated[0], rotated[1], rotated[2]
            
            if dx > heur.X or dy > heur.Y or dz > heur.H:
                continue
            
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    pallet = obs['pallet_obs_density']
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero = np.any(place_area > 0, axis=(0, 1))
                    if np.any(non_zero):
                        z = int(np.max(np.nonzero(non_zero)) + 1)
                    else:
                        z = 0
                    
                    if z + dz > heur.H:
                        continue
                    
                    # For hard boxes placed above z=0, check support rigidity
                    if is_hard and z > 0 and 'pallet_obs_softness' in obs:
                        pallet_softness = obs['pallet_obs_softness']
                        # Check the layer below the box
                        support_layer = pallet_softness[x:x+dx, y:y+dy, z-1]
                        if np.any(support_layer < 0.5):  # Soft support
                            continue
                    
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        return (slot_i, rot_id, x, y)
    
    return (0, 0, 0, 0)

class ArchivedHeuristic(BaseHeuristic):
    name = "test_soft2"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
