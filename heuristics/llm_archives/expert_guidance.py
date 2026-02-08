'''
code length: 1957
reason=height_oob util=0.7930 n_boxes=29
expert feedback：
Use heightmap-based corner points as candidate positions. Evaluate all feasible placements and select the one that maximizes footprint fill ratio while penalizing higher placement heights. Prioritize completing lower layers before stacking upward.
'''
import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    n_slots = int(obs['buffer'].size // heur.n_properties)
    pallet = obs['pallet_obs_density']
    
    best_action = (0, 0, 0, 0)
    best_score = -float('inf')
    
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        for rot_id in range(6):
            rot_size = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rot_size
            
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
            
            # Generate heightmap-based corner points
            heightmap = np.zeros((heur.X, heur.Y), dtype=int)
            for i in range(heur.X):
                for j in range(heur.Y):
                    column = pallet[i, j, :]
                    occupied = np.where(column > 0)[0]
                    heightmap[i, j] = occupied[-1] + 1 if occupied.size > 0 else 0
            
            # Collect candidate positions from heightmap corners
            candidates = []
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Compute z from heightmap (max height in footprint)
                    footprint_heights = heightmap[x:x+dx, y:y+dy]
                    z = np.max(footprint_heights) if footprint_heights.size > 0 else 0
                    
                    if z + dz > heur.H:
                        continue
                    
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        candidates.append((x, y, z))
            
            # Evaluate all feasible placements
            for x, y, z in candidates:
                # Footprint fill ratio: area / (pallet area)
                footprint_area = dx * dy
                pallet_area = heur.X * heur.Y
                fill_ratio = footprint_area / pallet_area
                
                # Penalty for height (prioritize lower layers)
                height_penalty = z / heur.H
                
                # Combined score
                score = fill_ratio - height_penalty
                
                if score > best_score:
                    best_score = score
                    best_action = (slot_i, rot_id, x, y)
    
    return best_action

class ArchivedHeuristic(BaseHeuristic):
    name = "expert_guidance"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
