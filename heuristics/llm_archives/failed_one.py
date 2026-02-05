import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    n_slots = int(obs['buffer'].size // heur.n_properties)
    pallet = obs['pallet_obs_density']
    
    # Build heightmap
    heightmap = np.zeros((heur.X, heur.Y), dtype=int)
    for x in range(heur.X):
        for y in range(heur.Y):
            column = pallet[x, y, :]
            occupied = np.where(column > 0)[0]
            heightmap[x, y] = occupied[-1] + 1 if occupied.size > 0 else 0
    
    # Find current minimum and maximum heights
    if heightmap.size > 0:
        non_zero_heights = heightmap[heightmap > 0]
        current_min_height = np.min(non_zero_heights) if non_zero_heights.size > 0 else 0
        current_max_height = np.max(heightmap)
    else:
        current_min_height = 0
        current_max_height = 0
    
    # Stage 1: Select top-K largest items by volume (footprint as tiebreaker)
    K = 3
    candidate_items = []
    
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        best_rot_id = None
        best_dims = None
        best_volume = 0
        best_footprint = 0
        
        for rot_id in range(6):
            rotated = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated[0], rotated[1], rotated[2]
            
            if dx > heur.X or dy > heur.Y or dz > heur.H:
                continue
            
            volume = dx * dy * dz
            footprint = dx * dy
            
            if volume > best_volume or (volume == best_volume and footprint > best_footprint):
                best_volume = volume
                best_footprint = footprint
                best_rot_id = rot_id
                best_dims = (dx, dy, dz)
        
        if best_volume > 0:
            candidate_items.append((slot_i, best_rot_id, best_dims[0], best_dims[1], best_dims[2], best_volume, best_footprint))
    
    if len(candidate_items) == 0:
        return (0, 0, 0, 0)
    
    candidate_items.sort(key=lambda x: (x[5], x[6]), reverse=True)
    top_items = candidate_items[:K]
    
    best_action = (0, 0, 0, 0)
    best_score = -float('inf')
    
    for slot_i, rot_id, dx, dy, dz, volume, footprint in top_items:
        candidates = []
        
        for x in range(heur.X - dx + 1):
            for y in range(heur.Y - dy + 1):
                if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                    continue
                
                place_area = pallet[x:x+dx, y:y+dy, :]
                non_zero = np.any(place_area > 0, axis=(0,1))
                z = int(np.max(np.nonzero(non_zero)) + 1) if np.any(non_zero) else 0
                
                if z + dz > heur.H:
                    continue
                
                if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                    candidates.append((x, y, z))
        
        if len(candidates) == 0:
            continue
        
        candidates.sort(key=lambda pos: pos[2])
        M = min(5, len(candidates))
        top_candidates = candidates[:M]
        
        for x, y, z in top_candidates:
            fill_ratio = (dx * dy) / (heur.X * heur.Y)
            height_penalty = 0.1 * (z / heur.H)
            
            # NEW: Layer-by-layer packing enforcement
            layer_penalty = 0.0
            if z > current_min_height and current_min_height > 0:
                # Strong penalty for placing above current minimum incomplete layer
                # This encourages completing lower layers first
                height_diff = z - current_min_height
                layer_penalty = 0.5 * (height_diff / heur.H)
            
            # Check if this placement helps complete the current layer
            layer_completion_bonus = 0.0
            if z == current_min_height:
                # Calculate how much this placement expands the occupied area at current layer
                occupied_before = np.sum(heightmap == current_min_height)
                
                # Simulate heightmap after placement
                temp_heightmap = heightmap.copy()
                temp_heightmap[x:x+dx, y:y+dy] = np.maximum(temp_heightmap[x:x+dx, y:y+dy], z + dz)
                occupied_after = np.sum(temp_heightmap == current_min_height)
                
                if occupied_before > 0:
                    expansion_ratio = (occupied_after - occupied_before) / occupied_before
                    layer_completion_bonus = 0.3 * expansion_ratio
            
            # Top-layer filling mode
            top_layer_bonus = 0.0
            if z >= current_max_height - 1 and current_max_height > 0:
                layer_slice = pallet[x:x+dx, y:y+dy, z:min(z+dz, heur.H)]
                top_layer_occupied = np.sum(layer_slice > 0)
                total_top_cells = dx * dy * min(dz, heur.H - z)
                if total_top_cells > 0:
                    fill_fraction = top_layer_occupied / total_top_cells
                    top_layer_bonus = 0.5 * fill_fraction
            
            # Support score using heightmap
            support_penalty = 0.0
            if z > 0:
                footprint_heights = heightmap[x:x+dx, y:y+dy]
                fully_supported = np.sum(footprint_heights == z)
                total_cells = dx * dy
                support_fraction = fully_supported / total_cells if total_cells > 0 else 0
                
                if z >= current_max_height - 1 and current_max_height > 0:
                    support_penalty = 0.3 * (1.0 - support_fraction) * (z / heur.H)
                else:
                    support_penalty = 0.7 * (1.0 - support_fraction) * (z / heur.H)
            
            # Combined score with layer-by-layer enforcement
            score = fill_ratio - height_penalty - layer_penalty - support_penalty + layer_completion_bonus + top_layer_bonus
            
            if score > best_score:
                best_score = score
                best_action = (slot_i, rot_id, x, y)
    
    if best_score > -float('inf'):
        return best_action
    
    return (0, 0, 0, 0)

class ArchivedHeuristic(BaseHeuristic):
    name = "failed_one"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
