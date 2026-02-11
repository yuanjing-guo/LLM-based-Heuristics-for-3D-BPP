'''
gen Your current policy does not use physical rigidity / softness information and therefore treats all boxes as identical. This is incorrect in the soft-physics palletization setting and can easily lead to collapse. From now on, you MUST explicitly use both buffer-level and pallet-level soft/hard information to decide which box to place and where to place it. Required changes: Use buffer physics (box rigidity / softness) When iterating over boxes in the buffer, read each box’s rigidity/softness from the available buffer information (e.g., box properties or buffer physics fields). Classify boxes into hard (high rigidity) and soft (low rigidity). Hard boxes first, bottom layer priority
'''
import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    n_slots = int(obs['buffer'].size // heur.n_properties)
    pallet = obs['pallet_obs_density']
    
    # Check if we can use physics-aware fields
    use_physics = (hasattr(heur, 'physics_mode') and heur.physics_mode == 'soft' and 
                   hasattr(heur, 'expose_physics_obs') and heur.expose_physics_obs and 
                   'buffer_physics' in obs and 'pallet_obs_softness' in obs)
    
    # First pass: collect hard boxes
    hard_boxes = []
    soft_boxes = []
    
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        # Classify box as hard or soft if physics info is available
        is_hard = False
        if use_physics:
            # Assuming buffer_physics contains rigidity info for each slot
            # Each slot might have multiple physics properties
            physics_per_slot = obs['buffer_physics'].size // n_slots
            if physics_per_slot > 0:
                start_idx = slot_i * physics_per_slot
                end_idx = start_idx + physics_per_slot
                rigidity = obs['buffer_physics'][start_idx:end_idx]
                # Simple threshold: if any rigidity value > 0.5, consider it hard
                if np.any(rigidity > 0.5):
                    is_hard = True
        
        if is_hard:
            hard_boxes.append((slot_i, props, size_bins))
        else:
            soft_boxes.append((slot_i, props, size_bins))
    
    # Process hard boxes first, then soft boxes
    box_groups = hard_boxes + soft_boxes
    
    for slot_i, props, size_bins in box_groups:
        for rot_id in range(6):
            rotated = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated[0], rotated[1], rotated[2]
            
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
            
            if not heur.feasibility.is_within_pallet(0, 0, dx, dy):
                continue
            
            # For hard boxes, prioritize bottom layers
            # For soft boxes, we can place higher up
            if use_physics and slot_i in [h[0] for h in hard_boxes]:
                # Hard box: scan x,y normally, but prioritize lower z positions
                # We'll use the standard scanning order but check feasibility
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
                        
                        # Additional check for hard boxes: avoid placing on top of soft material
                        if use_physics:
                            pallet_softness = obs['pallet_obs_softness']
                            # Check if the placement area has high softness at the base
                            if z > 0:
                                base_softness = pallet_softness[x:x+dx, y:y+dy, z-1]
                                if np.any(base_softness > 0.7):  # High softness threshold
                                    continue
                        
                        if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                            return (slot_i, rot_id, x, y)
            else:
                # Soft box or no physics: use standard scanning
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
                        
                        if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                            return (slot_i, rot_id, x, y)
    
    return (0, 0, 0, 0)

class ArchivedHeuristic(BaseHeuristic):
    name = "test_soft"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
