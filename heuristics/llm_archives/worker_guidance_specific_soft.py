'''
code length: 5713
reason=unstable util=0.2189 n_boxes=10
worker feedback：
cover the bottom layer with hard boxes, then place soft boxes on them, maximal the util
'''
import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    n_slots = int(obs['buffer'].size // heur.n_properties)
    pallet = obs['pallet_obs_density']
    
    # Check if physics-aware fields are available
    has_physics = 'pallet_obs_softness' in obs and 'buffer_physics' in obs
    
    # Extract softness information if available
    if has_physics:
        pallet_softness = obs['pallet_obs_softness']
    
    # First pass: try to place hard boxes at z=0 (bottom layer)
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        # Check if box is hard (assuming softness is in props[3])
        box_is_hard = has_physics and props[3] <= 0.5 if len(props) > 3 else True
        
        # Only consider hard boxes for bottom layer
        if not box_is_hard:
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
                    
                    # Try to place at z=0 (bottom layer)
                    z = 0
                    
                    if z + dz > heur.H:
                        continue
                    
                    # Check if area is empty at bottom layer
                    place_area = pallet[x:x+dx, y:y+dy, z:z+dz]
                    if np.any(place_area > 0):
                        continue
                    
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        return (slot_i, rot_id, x, y)
    
    # Second pass: try to place soft boxes on top of hard boxes
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        # Check if box is soft (assuming softness is in props[3])
        box_is_soft = has_physics and props[3] > 0.5 if len(props) > 3 else False
        
        # Only consider soft boxes for this pass
        if not box_is_soft:
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
                    z = int(np.max(np.nonzero(non_zero)) + 1) if np.any(non_zero) else 0
                    
                    if z + dz > heur.H:
                        continue
                    
                    # Check if placing on top of hard boxes (z > 0)
                    if z == 0:
                        continue
                    
                    # Check the layer directly below current placement
                    below_z = z - 1
                    if below_z >= 0:
                        # Check if area below has hard boxes
                        area_below = pallet[x:x+dx, y:y+dy, below_z]
                        if np.any(area_below > 0) and has_physics:
                            softness_below = pallet_softness[x:x+dx, y:y+dy, below_z]
                            # Check if any box below is hard (low softness)
                            if np.any(softness_below <= 0.5):
                                if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                                    return (slot_i, rot_id, x, y)
    
    # Third pass: default naive strategy for any remaining boxes
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
                    z = int(np.max(np.nonzero(non_zero)) + 1) if np.any(non_zero) else 0
                    
                    if z + dz > heur.H:
                        continue
                    
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        return (slot_i, rot_id, x, y)
    
    return (0, 0, 0, 0)

class WorkerGuidanceSpecificSoft(BaseHeuristic):
    name = "worker_guidance_specific_soft"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
