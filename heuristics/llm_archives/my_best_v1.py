import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    pallet = obs["pallet_obs_density"]
    best_score = None
    best_action = (0, 0, 0, 0)
    
    # Fixed column position - choose a corner (0,0)
    fixed_x = 0
    fixed_y = 0
    
    for slot_i in range(len(obs["buffer"])):
        if heur.slot_is_empty(obs, slot_i):
            continue
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or not np.all(size_bins > 0):
            continue

        for rot_id in range(6):
            dx, dy, dz = heur.rotate_size_bins(size_bins, rot_id)
            if dz > heur.H:
                continue

            # Only consider the fixed column position
            x = fixed_x
            y = fixed_y
            
            if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                continue

            place_area = pallet[x:x + dx, y:y + dy, :]
            non_zero = np.any(place_area > 0, axis=(0, 1))
            z = int(np.max(np.nonzero(non_zero)) + 1) if np.any(non_zero) else 0

            if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                continue

            # Primary: minimize z (height) - building a single column
            # Secondary: prefer fixed corner position (0,0)
            score = (z, 0)
            if best_score is None or score < best_score:
                best_score = score
                best_action = (slot_i, rot_id, x, y)

    return best_action

class ArchivedHeuristic(BaseHeuristic):
    name = "my_best_v1"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
