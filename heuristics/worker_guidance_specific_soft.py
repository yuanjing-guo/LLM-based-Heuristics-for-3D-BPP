'''
You MUST use box and pallet rigidity/softness information. Select hard boxes first from the buffer and place them on the bottom layer (z==0) whenever feasible. Do NOT place hard boxes on soft or low-rigidity support regions. Use pallet softness/rigidity to validate all placements with z > 0. Soft boxes may be placed on top of hard boxes; feasibility rules still apply.


'''
import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Get number of visible slots
    n_slots = int(obs['buffer'].size // heur.n_properties)
    
    # Separate hard and soft boxes
    hard_slots = []
    soft_slots = []
    
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
            
        props = heur.get_slot_props(obs, slot_i)
        if props.size < 4:
            continue
            
        # Property 3 is rigidity (0=soft, 1=hard)
        rigidity = props[3]
        if rigidity > 0.5:  # Hard box
            hard_slots.append(slot_i)
        else:  # Soft box
            soft_slots.append(slot_i)
    
    # Process hard boxes first - try to cover bottom layer
    for slot_i in hard_slots:
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
            
        # Only consider rotations where largest face is parallel to pallet (height aligned with +Z)
        # This means dz should be the smallest dimension (box lying flat)
        valid_rotations = []
        for rot_id in range(6):
            rotated_size = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated_size
            
            # Check if this rotation puts the box flat (dz is smallest dimension)
            if dz <= dx and dz <= dy:
                valid_rotations.append((rot_id, rotated_size))
        
        # Try bottom layer (z=0) placements first for hard boxes
        for rot_id, (dx, dy, dz) in valid_rotations:
            # Scan all possible positions on bottom layer
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    z = 0
                    
                    # Check if placing on soft support region (only for physics mode)
                    if 'pallet_obs_softness' in obs and obs.get('pallet_obs_softness') is not None:
                        support_area = obs['pallet_obs_softness'][x:x+dx, y:y+dy, 0]
                        if np.any(support_area > 0.5):  # Soft support
                            continue
                    
                    # Check height overflow
                    if z + dz > heur.H:
                        continue
                    
                    # Check if this position is already occupied at bottom layer
                    pallet = obs['pallet_obs_density']
                    bottom_area = pallet[x:x+dx, y:y+dy, 0]
                    if np.any(bottom_area > 0):
                        continue  # Skip occupied bottom positions
                    
                    # Final feasibility check
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        return (slot_i, rot_id, x, y)
    
    # If no hard boxes can be placed at bottom, try placing them above if supported
    for slot_i in hard_slots:
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
            
        valid_rotations = []
        for rot_id in range(6):
            rotated_size = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated_size
            
            if dz <= dx and dz <= dy:
                valid_rotations.append((rot_id, rotated_size))
        
        for rot_id, (dx, dy, dz) in valid_rotations:
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Compute z from existing density
                    pallet = obs['pallet_obs_density']
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero = np.any(place_area > 0, axis=(0, 1))
                    
                    if np.any(non_zero):
                        z = int(np.max(np.nonzero(non_zero)) + 1)
                    else:
                        z = 0
                    
                    # For hard boxes at z>0, ensure wide, flat support
                    if z > 0:
                        # Check support contact area
                        support_area = pallet[x:x+dx, y:y+dy, z-1]
                        if np.sum(support_area > 0) < dx * dy:  # Not fully supported
                            continue
                        
                        # Reject edge or point contact (require substantial support)
                        if dx == 1 or dy == 1:
                            continue  # Avoid single-column stacking
                    
                    # Check height overflow
                    if z + dz > heur.H:
                        continue
                    
                    # Final feasibility check
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        return (slot_i, rot_id, x, y)
    
    # Process soft boxes - place on top of hard boxes
    for slot_i in soft_slots:
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
            
        # Soft boxes also need to be placed flat
        valid_rotations = []
        for rot_id in range(6):
            rotated_size = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated_size
            
            if dz <= dx and dz <= dy:
                valid_rotations.append((rot_id, rotated_size))
        
        for rot_id, (dx, dy, dz) in valid_rotations:
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Compute z from existing density
                    pallet = obs['pallet_obs_density']
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero = np.any(place_area > 0, axis=(0, 1))
                    
                    if np.any(non_zero):
                        z = int(np.max(np.nonzero(non_zero)) + 1)
                    else:
                        z = 0
                    
                    # For soft boxes, prefer placement on hard boxes (z>0)
                    # But also allow placement at bottom if no hard boxes available
                    if z > 0:
                        # Check if supported by hard boxes
                        support_layer = pallet[x:x+dx, y:y+dy, z-1]
                        if np.sum(support_layer > 0) < dx * dy:  # Not fully supported
                            continue
                        
                        # Reject single-column stacking
                        if dx == 1 and dy == 1:
                            continue
                    
                    # Check height overflow
                    if z + dz > heur.H:
                        continue
                    
                    # Final feasibility check
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        return (slot_i, rot_id, x, y)
    
    # No feasible action found
    return (0, 0, 0, 0)

class WorkerGuidanceSpecificSoft(BaseHeuristic):
    name = "worker_guidance_specific_soft"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
