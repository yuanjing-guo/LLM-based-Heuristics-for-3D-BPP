'''
1-Apply the following feasibility rules while keeping the naive scan-and-return strategy: only consider horizontal placements where the largest face is parallel to the pallet; when z==0, treat placements of soft boxes as infeasible and skip them so that only hard boxes may contact the pallet; only when z>0, evaluate support rigidity by checking the boxes at layer z−1 and skip placements supported by soft boxes; all other logic must follow the existing naive first-feasible strategy.

2-Any rotation where the box is not placed flat (i.e., its largest face is not parallel to the pallet) must be treated as infeasible and skipped before feasibility checks

3-When the computed placement height z equals 0, any placement of a soft box must be treated as infeasible and skipped; only hard boxes are allowed to directly contact the pallet.

4-fill the full bottom layer first!!

5-consider the softness of boxes and place hard under the softboxes.

...


the performance remains low and llm dosn't understand my command at last.
'''





import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Determine number of visible slots
    n_slots = int(obs['buffer'].size // heur.n_properties)
    
    # Get pallet density observation
    pallet = obs['pallet_obs_density']
    
    # Check if physics-aware fields are available
    has_physics = 'pallet_obs_softness' in obs and 'buffer_physics' in obs
    
    # Naive scan-and-return: return first feasible action
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        # Skip invalid size bins
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        # Scan rotations
        for rot_id in range(6):
            rotated_size = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated_size[0], rotated_size[1], rotated_size[2]
            
            # CRITICAL: Only consider horizontal placements where largest face is parallel to pallet
            # This means the box must be placed flat: dz must be the smallest dimension
            # Check if this rotation places the box flat
            sorted_dims = sorted([dx, dy, dz])
            if dz != sorted_dims[0]:  # dz is not the smallest dimension
                continue  # Skip this rotation - box is not placed flat
            
            # Skip if box doesn't fit within pallet dimensions
            if dx > heur.X or dy > heur.Y or dz > heur.H:
                continue
            
            # Scan x,y positions
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    # Check if within pallet bounds
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Compute z exactly as in env.py
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero = np.any(place_area > 0, axis=(0, 1))
                    
                    if np.any(non_zero):
                        z = int(np.max(np.nonzero(non_zero)) + 1)
                    else:
                        z = 0
                    
                    # Check height overflow
                    if z + dz > heur.H:
                        continue
                    
                    # CRITICAL: Softness constraints
                    if has_physics:
                        # Get box softness from physics buffer
                        box_softness = obs['buffer_physics'][0]  # First element is box softness
                        
                        # When z==0: only hard boxes may contact the pallet
                        if z == 0 and box_softness > 0.5:  # Assuming softness > 0.5 means soft box
                            continue  # Skip placement of soft box at z==0
                        
                        # When z>0: evaluate support rigidity
                        if z > 0:
                            softness = obs['pallet_obs_softness']
                            # Check the support layer (z-1) in the placement area
                            support_area = softness[x:x+dx, y:y+dy, z-1]
                            if support_area.size > 0:
                                # Skip if supported by any soft box (softness > 0.5)
                                if np.any(support_area > 0.5):
                                    continue
                    
                    # Final feasibility check
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        return (slot_i, rot_id, x, y)
    
    # No feasible action found
    return (0, 0, 0, 0)

class multi_iteration_soft(BaseHeuristic):
    name = "multi_iteration_soft"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
