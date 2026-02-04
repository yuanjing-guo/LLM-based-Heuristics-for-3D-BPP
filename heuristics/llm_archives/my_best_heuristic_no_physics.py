import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    n_slots = int(obs['buffer'].size // heur.n_properties)
    pallet = obs['pallet_obs_density']
    
    best_action = (0, 0, 0, 0)
    best_score = -1
    
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
                        # Score: prioritize lower height to fill gaps
                        height_score = heur.H - (z + dz)
                        # Penalize leaving small gaps at edges
                        edge_x = min(x, heur.X - (x + dx))
                        edge_y = min(y, heur.Y - (y + dy))
                        edge_penalty = edge_x + edge_y
                        # Prioritize larger boxes for better space utilization
                        volume_score = dx * dy * dz
                        # Combined score
                        score = volume_score * 100 + height_score * 10 - edge_penalty
                        
                        if score > best_score:
                            best_score = score
                            best_action = (slot_i, rot_id, x, y)
    
    return best_action

class ArchivedHeuristic(BaseHeuristic):
    name = "my_best_heuristic_no_physics"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
