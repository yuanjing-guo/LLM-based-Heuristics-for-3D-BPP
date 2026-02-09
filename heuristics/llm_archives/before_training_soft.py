import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Determine number of visible slots
    n_slots = int(obs['buffer'].size // heur.n_properties)
    
    # Main loop over slots
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        # Try all rotations
        for rot_id in range(6):
            rotated_size = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated_size
            
            # Check if rotation makes box too large for pallet
            if dx > heur.X or dy > heur.Y or dz > heur.H:
                continue
            
            # Scan x,y positions
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    # Check within pallet bounds (redundant but safe)
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

class BeforeTrainingSoft(BaseHeuristic):
    name = "before_training_soft"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
