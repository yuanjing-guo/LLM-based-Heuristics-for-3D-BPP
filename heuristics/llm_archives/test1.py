import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Determine number of visible slots
    n_slots = int(obs['buffer'].size // heur.n_properties)
    
    best_action = (0, 0, 0, 0)
    best_score = float('-inf')
    
    # Check if physics-aware fields are available
    physics_available = ('pallet_obs_softness' in obs and 'buffer_physics' in obs)
    
    # Scan slots
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        # Extract box properties if physics available
        box_rigidity = None
        box_weight = None
        if physics_available:
            # Assuming rigidity and weight are in buffer_physics for each slot
            phys_start = slot_i * 2  # Assuming 2 physics properties per box
            if phys_start + 1 < obs['buffer_physics'].size:
                box_rigidity = obs['buffer_physics'][phys_start]
                box_weight = obs['buffer_physics'][phys_start + 1]
        
        # Scan rotations
        for rot_id in range(6):
            rotated_size = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated_size
            
            # Skip if box dimensions exceed pallet bounds
            if dx > heur.X or dy > heur.Y or dz > heur.H:
                continue
            
            # Scan positions
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
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
                    
                    # Check feasibility
                    if heur.feasibility.is_within_pallet(x, y, dx, dy) and \
                       heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        
                        # Calculate placement score based on rigidity/softness rules
                        score = 0.0
                        
                        if physics_available and box_rigidity is not None and box_weight is not None:
                            # Get support rigidity from pallet_obs_softness
                            if z > 0:
                                support_layer = obs['pallet_obs_softness'][x:x+dx, y:y+dy, z-1]
                                avg_support_rigidity = np.mean(support_layer) if support_layer.size > 0 else 0.0
                                
                                # Rule 1: Reward soft boxes on top of hard boxes
                                if box_rigidity < 0.5:  # Soft box
                                    score += avg_support_rigidity * 2.0
                                # Rule 2: Penalize soft boxes on soft support
                                if box_rigidity < 0.5 and avg_support_rigidity < 0.5:
                                    score -= 3.0
                                
                                # Rule 3: Reward heavy boxes on hard support
                                if box_weight > 0.5:  # Heavy box
                                    score += avg_support_rigidity * box_weight
                                # Rule 4: Penalize heavy boxes on low-rigidity support
                                if box_weight > 0.5 and avg_support_rigidity < 0.3:
                                    score -= 4.0 * box_weight
                            else:
                                # On floor - good for heavy/rigid boxes
                                if box_weight > 0.5 or box_rigidity > 0.5:
                                    score += 1.0
                            
                            # Rule 5: Penalize height increase without horizontal footprint increase
                            if z > 0:
                                # Check if this placement increases height significantly without expanding base
                                footprint_below = pallet[x:x+dx, y:y+dy, z-1]
                                current_footprint = np.ones((dx, dy))
                                if np.array_equal(footprint_below > 0, current_footprint):
                                    # Same footprint as below - penalize height increase
                                    score -= 2.0 * (dz / heur.H)
                        
                        # Always prefer lower z for stability
                        score -= z * 0.1
                        
                        # Prefer larger horizontal coverage
                        horizontal_coverage = (dx * dy) / (heur.X * heur.Y)
                        score += horizontal_coverage * 0.5
                        
                        if score > best_score:
                            best_score = score
                            best_action = (slot_i, rot_id, x, y)
    
    # Return best action found, or (0,0,0,0) if none
    if best_score > float('-inf'):
        return best_action
    return (0, 0, 0, 0)

class ArchivedHeuristic(BaseHeuristic):
    name = "test1"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
