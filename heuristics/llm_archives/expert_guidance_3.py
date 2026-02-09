'''
code length: 3597
reason=height_oob util=0.6758 n_boxes=27
expert feedback：
To further maximize space utilization, emphasize maintaining a smoother and more uniform top surface.Prioritize placements that minimize local height differences and reduce surface roughness, even if this slightly reduces immediate footprint fill, as a flatter surface is expected to enable denser packing in subsequent steps.
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
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero = np.any(place_area > 0, axis=(0, 1))
                    if np.any(non_zero):
                        z = int(np.max(np.nonzero(non_zero)) + 1)
                    else:
                        z = 0
                    
                    if z + dz > heur.H:
                        continue
                    
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        footprint_fill = (dx * dy) / (heur.X * heur.Y)
                        height_penalty = z / heur.H
                        
                        gap_penalty = 0.0
                        remaining_x = heur.X - (x + dx)
                        remaining_y = heur.Y - (y + dy)
                        
                        if remaining_x > 0 and remaining_x < 3:
                            gap_penalty += 0.1 * (3 - remaining_x)
                        if remaining_y > 0 and remaining_y < 3:
                            gap_penalty += 0.1 * (3 - remaining_y)
                        
                        if x > 0 and x < 3:
                            gap_penalty += 0.1 * (3 - x)
                        if y > 0 and y < 3:
                            gap_penalty += 0.1 * (3 - y)
                        
                        surface_roughness_penalty = 0.0
                        if z > 0:
                            neighbor_heights = []
                            for nx in range(max(0, x-1), min(heur.X, x+dx+1)):
                                for ny in range(max(0, y-1), min(heur.Y, y+dy+1)):
                                    if nx < x or nx >= x+dx or ny < y or ny >= y+dy:
                                        column = pallet[nx, ny, :]
                                        if np.any(column > 0):
                                            h = int(np.max(np.nonzero(column > 0)) + 1)
                                            neighbor_heights.append(h)
                            
                            if neighbor_heights:
                                avg_height = np.mean(neighbor_heights)
                                height_diff = abs(z + dz - avg_height)
                                surface_roughness_penalty = height_diff / heur.H * 0.3
                        
                        score = footprint_fill - height_penalty - gap_penalty - surface_roughness_penalty
                        
                        if score > best_score:
                            best_score = score
                            best_action = (slot_i, rot_id, x, y)
    
    return best_action

class ExpertGuidance3(BaseHeuristic):
    name = "expert_guidance_3"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
