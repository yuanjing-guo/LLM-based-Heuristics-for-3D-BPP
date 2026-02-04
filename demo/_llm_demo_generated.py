import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Get slot count
    n_slots = int(obs['buffer'].size // heur.n_properties)
    
    # Get pallet density observation
    pallet = obs['pallet_obs_density']
    
    # Precompute height map for faster z calculation
    height_map = np.zeros((heur.X, heur.Y), dtype=int)
    for x in range(heur.X):
        for y in range(heur.Y):
            column = pallet[x, y, :]
            non_zero_indices = np.where(column > 0)[0]
            if non_zero_indices.size > 0:
                height_map[x, y] = np.max(non_zero_indices) + 1
            else:
                height_map[x, y] = 0
    
    # Track best placement (lowest height, most centered)
    best_score = float('inf')
    best_placement = (0, 0, 0, 0)
    found_any = False
    
    # Scan slots in increasing order
    for slot_i in range(n_slots):
        # Check if slot is empty
        if heur.slot_is_empty(obs, slot_i):
            continue
            
        # Get box properties and convert to size bins
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        # Try all rotations
        for rot_id in range(6):
            # Get rotated dimensions
            rot_size = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rot_size
            
            # Check if dimensions are valid
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
                
            # Try all positions
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    # Check if within pallet bounds
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Calculate z (height to place on) using height map
                    z = 0
                    for i in range(x, x + dx):
                        for j in range(y, y + dy):
                            if height_map[i, j] > z:
                                z = height_map[i, j]
                    
                    # Check height constraint
                    if z + dz > heur.H:
                        continue
                    
                    # Check feasibility
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        # Score: lower height and more centered is better
                        center_x = x + dx / 2 - heur.X / 2
                        center_y = y + dy / 2 - heur.Y / 2
                        score = z * 100 + abs(center_x) + abs(center_y)
                        
                        if not found_any or score < best_score:
                            best_score = score
                            best_placement = (slot_i, rot_id, x, y)
                            found_any = True
    
    if found_any:
        return best_placement
    else:
        return (0, 0, 0, 0)

def _clamp_action_if_needed(heur, obs, action):
    try:
        box_slot, rot_id, x, y = action
        # buffer-first: force box_slot=0
        if getattr(heur, "_caps", {}).get("buffer", "full") == "first":
            box_slot = 0
        return (int(box_slot), int(rot_id), int(x), int(y))
    except Exception:
        return (0, 0, 0, 0)

class GeneratedHeuristic(BaseHeuristic):
    name = "llm_demo_generated"

    def __init__(self):
        super().__init__()
        # 注意：这里为了让 clamp 能拿到 caps，我们在 wrapper 里会把 _caps 注入到 impl 上（见下面说明）
        # 如果没注入，也不会崩，只是不会 clamp。

    def __call__(self, obs):
        action = llm_policy(self, obs)
        action = _clamp_action_if_needed(self, obs, action)
        box_slot, rot_id, x, y = action
        return self.encode_action_logits(box_slot, rot_id, x, y)
