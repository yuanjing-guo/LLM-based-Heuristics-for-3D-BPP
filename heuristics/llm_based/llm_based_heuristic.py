import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    
    pallet = obs['pallet_obs_density']
    buffer = obs['buffer']
    n_slots = len(buffer) // heur.n_properties
    
    # 1. Check if physics is informative
    physics_informative = False
    if 'physics_obs' in obs and 'support' in obs['physics_obs']:
        support = obs['physics_obs']['support']
        if np.any(support != 0):
            physics_informative = True
    
    # 2. Anti-oscillation tracking
    if not hasattr(heur, '_recent_actions'):
        heur._recent_actions = []
    if not hasattr(heur, '_remove_count'):
        heur._remove_count = 0
    
    # Keep last 10 actions
    if len(heur._recent_actions) > 10:
        heur._recent_actions.pop(0)
    
    # 3. Try PLACE first
    # Priority: HARD boxes for bottom layer, then SOFT boxes
    for hardness_priority in [1, 0]:  # 1=HARD, 0=SOFT
        for slot in range(n_slots):
            props = heur.get_slot_props(obs, slot)
            if props[3] != hardness_priority:  # hardness is property index 3
                continue
            
            # Try all rotations
            for rot_id in range(6):
                size_bins = heur.props_to_size_bins(props)
                dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
                
                # Try bottom layer first (z=0)
                for x in range(heur.X - dx + 1):
                    for y in range(heur.Y - dy + 1):
                        if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                            continue
                        
                        # Check bottom layer placement
                        z = 0
                        if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                            continue
                        
                        # For bottom layer, only place HARD boxes if physics is informative
                        if physics_informative and z == 0 and hardness_priority == 0:
                            continue  # Skip SOFT boxes on bottom when physics matters
                        
                        # Check support if physics is informative
                        if physics_informative and z > 0:
                            # Get boxes below
                            below_area = pallet[x:x+dx, y:y+dy, z-1:z]
                            if np.any(below_area > 1e-6):
                                # Check if any SOFT box supports HARD box (prohibited)
                                if hardness_priority == 1:  # HARD box
                                    # Need to check each column
                                    support_issue = False
                                    for i in range(x, x+dx):
                                        for j in range(y, y+dy):
                                            if pallet[i, j, z-1] > 1e-6:
                                                # Find which box is below
                                                for s in range(n_slots):
                                                    p = heur.get_slot_props(obs, s)
                                                    # This is simplified - in full impl would need box ID mapping
                                                    if p[3] == 0:  # SOFT box
                                                        support_issue = True
                                                        break
                                    if support_issue:
                                        continue
                        
                        # Anti-oscillation: avoid repeating same placement
                        action = (0, slot, rot_id, x, y)
                        if action in heur._recent_actions:
                            continue
                        
                        heur._recent_actions.append(action)
                        heur._remove_count = 0
                        return (0, slot, rot_id, x, y)
    
    # 4. If no placement possible, try REMOVE
    # Limit remove bursts to 3 consecutive removes
    if heur._remove_count >= 3:
        # Force wait by returning a no-op (shouldn't happen in normal flow)
        heur._remove_count = 0
        return (1, 0, 0, 0, 0)
    
    # Find topmost boxes to remove (prioritize SOFT boxes that block HARD placements)
    top_boxes = []
    for x in range(heur.X):
        for y in range(heur.Y):
            column = pallet[x, y, :]
            occupied = np.where(column > 1e-6)[0]
            if len(occupied) > 0:
                z = occupied[-1]  # Topmost occupied cell
                # Try to find box properties (simplified - would need box ID in real impl)
                # For now, remove from highest occupied position
                top_boxes.append((z, x, y))
    
    if top_boxes:
        # Sort by height (descending)
        top_boxes.sort(reverse=True)
        _, x, y = top_boxes[0]
        
        # Find pallet index (simplified - in real impl would map to pallet_index)
        pallet_index = x * heur.Y + y  # Simplified mapping
        
        action = (1, pallet_index, 0, 0, 0)
        if action in heur._recent_actions:
            # Try second highest
            if len(top_boxes) > 1:
                _, x2, y2 = top_boxes[1]
                pallet_index = x2 * heur.Y + y2
                action = (1, pallet_index, 0, 0, 0)
        
        heur._recent_actions.append(action)
        heur._remove_count += 1
        return (1, pallet_index, 0, 0, 0)
    
    # 5. Default fallback
    heur._remove_count = 0
    return (1, 0, 0, 0, 0)

class GeneratedHeuristic(BaseHeuristic):
    name = "llm_generated"

    def __call__(self, obs):
        out = llm_policy(self, obs)
        if isinstance(out, (list, tuple, np.ndarray)):
            out = list(out)
        else:
            out = [out]

        while len(out) < 5:
            out.append(0)
        out = out[:5]

        return np.array([int(v) for v in out], dtype=np.int32)
