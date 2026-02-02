import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    n_slots = int(obs['buffer'].size // heur.n_properties)
    pallet = obs['pallet_obs_density']
    
    best_placement = None
    best_score = -1e9
    
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
            
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        for rot_id in range(6):
            dx, dy, dz = heur.rotate_size_bins(size_bins, rot_id)
            
            if dx > heur.X or dy > heur.Y or dz > heur.H:
                continue
                
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero = np.any(place_area > 0, axis=(0, 1))
                    if np.any(non_zero):
                        z = int(np.max(np.nonzero(non_zero)[0]) + 1)
                    else:
                        z = 0
                    
                    if z + dz > heur.H:
                        continue
                    
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        # CRITICAL: Gap elimination strategy
                        
                        # 1. Height minimization (build from bottom up)
                        height_score = -(z + dz) * 1000
                        
                        # 2. Perfect contact with supporting layer (NO GAPS)
                        contact_score = 0
                        if z > 0:
                            support_layer = pallet[x:x+dx, y:y+dy, z-1]
                            contact_area = np.sum(support_layer > 0)
                            if contact_area == dx * dy:  # Perfect full support
                                contact_score = 1000
                            elif contact_area > 0:        # Partial support
                                contact_score = contact_area * 100
                            else:                         # No support (overhang)
                                contact_score = -1000
                        
                        # 3. Gap filling in current layer (CRITICAL FOR NO GAPS)
                        gap_filling_score = 0
                        if z == 0:
                            # Check if box touches left wall or another box
                            left_touch = (x == 0) or np.any(pallet[x-1, y:y+dy, 0] > 0)
                            # Check if box touches right wall or another box
                            right_touch = (x+dx == heur.X) or np.any(pallet[x+dx, y:y+dy, 0] > 0)
                            # Check if box touches front wall or another box
                            front_touch = (y == 0) or np.any(pallet[x:x+dx, y-1, 0] > 0)
                            # Check if box touches back wall or another box
                            back_touch = (y+dy == heur.Y) or np.any(pallet[x:x+dx, y+dy, 0] > 0)
                            
                            # Bonus for touching on both sides (filling gaps)
                            if left_touch and right_touch:
                                gap_filling_score += 500
                            if front_touch and back_touch:
                                gap_filling_score += 500
                            
                            # Check if this placement completely fills a rectangular hole
                            if left_touch and right_touch and front_touch and back_touch:
                                gap_filling_score += 1000
                        
                        # 4. Column building (strong stability, no gaps)
                        column_bonus = 0
                        if z > 0:
                            below = pallet[x:x+dx, y:y+dy, z-1]
                            if np.all(below > 0):
                                column_bonus = 800
                        
                        # 5. Wall proximity for base layer (helps eliminate gaps)
                        wall_score = 0
                        if z == 0:
                            # Strong preference for touching walls to eliminate edge gaps
                            if x == 0:
                                wall_score += 200
                            if x + dx == heur.X:
                                wall_score += 200
                            if y == 0:
                                wall_score += 200
                            if y + dy == heur.Y:
                                wall_score += 200
                        
                        # 6. Penalize creating new gaps
                        gap_creation_penalty = 0
                        if z == 0:
                            # Check what's left and right of this placement
                            left_empty = x > 0 and np.all(pallet[x-1, y:y+dy, 0] == 0)
                            right_empty = x+dx < heur.X and np.all(pallet[x+dx, y:y+dy, 0] == 0)
                            front_empty = y > 0 and np.all(pallet[x:x+dx, y-1, 0] == 0)
                            back_empty = y+dy < heur.Y and np.all(pallet[x:x+dx, y+dy, 0] == 0)
                            
                            # If we create a narrow vertical gap, penalize heavily
                            if left_empty and right_empty:
                                gap_width = heur.X - dx
                                if gap_width > 0:
                                    gap_creation_penalty += 300 * (2 - min(x, heur.X-(x+dx)))
                            
                            if front_empty and back_empty:
                                gap_depth = heur.Y - dy
                                if gap_depth > 0:
                                    gap_creation_penalty += 300 * (2 - min(y, heur.Y-(y+dy)))
                        
                        # 7. Volume utilization
                        volume_util = dx * dy * dz * 2
                        
                        # 8. Prefer placements that align with existing boxes
                        alignment_bonus = 0
                        if z > 0:
                            # Check alignment with box below
                            below_boxes = pallet[x:x+dx, y:y+dy, z-1]
                            if np.any(below_boxes > 0):
                                alignment_bonus = 200
                        
                        # 9. STRATEGY: Fill from corners outward to minimize gaps
                        corner_score = 0
                        if z == 0:
                            # Distance to nearest corner
                            corners = [(0,0), (0,heur.Y-1), (heur.X-1,0), (heur.X-1,heur.Y-1)]
                            min_corner_dist = min(abs(x-cx) + abs(y-cy) for cx, cy in corners)
                            corner_score = -min_corner_dist * 50
                        
                        total_score = (
                            height_score +
                            contact_score +
                            gap_filling_score +
                            column_bonus +
                            wall_score +
                            volume_util +
                            alignment_bonus +
                            corner_score -
                            gap_creation_penalty
                        )
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_placement = (slot_i, rot_id, x, y)
    
    if best_placement is not None:
        return best_placement
    
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
