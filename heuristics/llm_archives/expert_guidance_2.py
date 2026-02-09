'''
code length: 3026
reason=height_oob util=0.8074 n_boxes=30
expert feedback：
Further improve space utilization by reducing small residual gaps. In addition to footprint fill and height penalty, add a mild penalty for creating thin slivers of free space (narrow gaps) that are smaller than typical box footprints. Prefer placements that leave more ‘regular’ rectangular remaining areas.
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
            rotated = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated[0], rotated[1], rotated[2]
            
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
            
            if not heur.feasibility.is_within_pallet(0, 0, dx, dy):
                continue
            
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero = np.any(place_area > 0, axis=(0, 1))
                    if np.any(non_zero):
                        z = int(np.max(np.nonzero(non_zero)) + 1)
                    else:
                        z = 0
                    
                    if z + dz > heur.H:
                        continue
                    
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        footprint_fill = dx * dy / (heur.X * heur.Y)
                        height_penalty = z / heur.H
                        
                        gap_penalty = 0.0
                        remaining_x_left = x
                        remaining_x_right = heur.X - (x + dx)
                        remaining_y_bottom = y
                        remaining_y_top = heur.Y - (y + dy)
                        
                        min_dim = min(dx, dy)
                        if remaining_x_left > 0 and remaining_x_left < min_dim:
                            gap_penalty += 0.1 * (min_dim - remaining_x_left) / min_dim
                        if remaining_x_right > 0 and remaining_x_right < min_dim:
                            gap_penalty += 0.1 * (min_dim - remaining_x_right) / min_dim
                        if remaining_y_bottom > 0 and remaining_y_bottom < min_dim:
                            gap_penalty += 0.1 * (min_dim - remaining_y_bottom) / min_dim
                        if remaining_y_top > 0 and remaining_y_top < min_dim:
                            gap_penalty += 0.1 * (min_dim - remaining_y_top) / min_dim
                        
                        score = footprint_fill - height_penalty - gap_penalty
                        
                        if score > best_score:
                            best_score = score
                            best_action = (slot_i, rot_id, x, y)
    
    return best_action

class ExpertGuidance2(BaseHeuristic):
    name = "expert_guidance_2"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
