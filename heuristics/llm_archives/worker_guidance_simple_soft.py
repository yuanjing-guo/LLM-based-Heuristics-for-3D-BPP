'''
gen consider the softness and keep stable, while maximize the space util.
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
    
    # Initialize best action tracking
    best_action = None
    best_score = -float('inf')
    
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
            
            # Skip if box doesn't fit within pallet dimensions
            if dx > heur.X or dy > heur.Y or dz > heur.H:
                continue
            
            # Scan x,y positions
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    # Check if placement is within pallet bounds
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
                    
                    # Check feasibility
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        # Calculate score for this placement
                        score = 0.0
                        
                        # 1. Maximize space utilization (prefer higher z placements)
                        score += z * 0.1
                        
                        # 2. Consider softness if physics data is available
                        if has_physics:
                            softness = obs['pallet_obs_softness']
                            # Check softness in the placement area
                            if z > 0:
                                # Look at the layer below the placement
                                below_layer = softness[x:x+dx, y:y+dy, z-1]
                                avg_softness = np.mean(below_layer)
                                # Prefer placements on softer surfaces for stability
                                score += avg_softness * 0.05
                        
                        # 3. Prefer placements that fill empty spaces
                        # Check how much empty space this placement covers
                        empty_space = np.sum(pallet[x:x+dx, y:y+dy, z:z+dz] == 0)
                        score += empty_space * 0.01
                        
                        # 4. Prefer placements closer to edges for better stability
                        edge_distance = min(x, heur.X - (x + dx), y, heur.Y - (y + dy))
                        score -= edge_distance * 0.005
                        
                        # Update best action if this score is better
                        if score > best_score:
                            best_score = score
                            best_action = (slot_i, rot_id, x, y)
    
    # Return best action if found, otherwise default
    if best_action is not None:
        return best_action
    
    # No feasible action found
    return (0, 0, 0, 0)

class WorkerGuidanceSimpleSoft(BaseHeuristic):
    name = "worker_guidance_simple_soft"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
