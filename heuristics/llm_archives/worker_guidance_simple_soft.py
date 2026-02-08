'''
gen keep stable while maximal the util
'''
import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Determine number of visible slots
    n_slots = int(obs['buffer'].size // heur.n_properties)
    
    # Get pallet density array
    pallet = obs['pallet_obs_density']
    
    # Track best action for maximal utility (stability prioritized)
    best_action = (0, 0, 0, 0)
    best_score = -1.0
    
    # Scan slots
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
            dx, dy, dz = rotated_size
            
            # Skip if box dimensions exceed pallet bounds
            if dx > heur.X or dy > heur.Y or dz > heur.H:
                continue
            
            # Scan x,y positions
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    # Check within pallet bounds
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
                    
                    # Final feasibility check
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        # Stability score: lower z is more stable
                        stability_score = 1.0 / (z + 1.0)
                        
                        # Utility score: maximize volume utilization
                        volume_util = (dx * dy * dz) / (heur.X * heur.Y * heur.H)
                        
                        # Combined score: stability first, then utility
                        combined_score = stability_score + 0.1 * volume_util
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_action = (slot_i, rot_id, x, y)
    
    return best_action

class ArchivedHeuristic(BaseHeuristic):
    name = "worker_guidance_simple_soft"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
