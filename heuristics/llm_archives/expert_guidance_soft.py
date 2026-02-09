'''
expert feedback: 
When placing items at z > 0, prioritize placements with strong support from below. Estimate support by checking how much of the footprint is supported at the placement height, and prefer placements with larger supported area. This helps maintain stability in soft physics while still maximizing utilization.

Keep volume utilization as the primary objective. When placing items at z > 0, enforce a minimum support requirement to avoid unstable placements, but do not rank placements primarily by support area. Among placements with similar volume contribution, prefer those with stronger support and lower z.
'''
import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Determine number of visible slots
    n_slots = int(obs['buffer'].size // heur.n_properties)
    
    best_action = (0, 0, 0, 0)
    best_volume = -1
    best_support = -1.0
    best_z = heur.H + 1
    
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
            volume = dx * dy * dz
            
            # Scan positions
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
                        # For z > 0, enforce minimum support requirement
                        if z > 0:
                            # Check support from below
                            below_layer = pallet[x:x+dx, y:y+dy, z-1]
                            supported_area = np.sum(below_layer > 0)
                            total_area = dx * dy
                            support_ratio = supported_area / total_area
                            
                            # Minimum support threshold (at least 50% supported)
                            if support_ratio < 0.5:
                                continue
                            
                            # Volume is primary objective
                            if volume > best_volume:
                                best_volume = volume
                                best_support = support_ratio
                                best_z = z
                                best_action = (slot_i, rot_id, x, y)
                            elif volume == best_volume:
                                # Among placements with same volume, prefer stronger support and lower z
                                if support_ratio > best_support or (support_ratio == best_support and z < best_z):
                                    best_support = support_ratio
                                    best_z = z
                                    best_action = (slot_i, rot_id, x, y)
                        else:
                            # Ground placements (z=0) have perfect support
                            # Volume is primary objective
                            if volume > best_volume:
                                best_volume = volume
                                best_support = 1.0
                                best_z = 0
                                best_action = (slot_i, rot_id, x, y)
    
    # Return best action found
    if best_volume > 0:
        return best_action
    
    # No feasible action found
    return (0, 0, 0, 0)

class ExpertGuidanceSoft(BaseHeuristic):
    name = "expert_guidance_soft"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
