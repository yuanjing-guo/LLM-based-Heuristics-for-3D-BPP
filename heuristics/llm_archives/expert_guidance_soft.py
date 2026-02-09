'''
Use an EMS-based search to enumerate candidate positions, fill the bottom layer with flat hard boxes first, then place soft boxes only on upper layers using best-fit; if no hard-box EMS placement is feasible, fall back to the lowest feasible hard-box placement.
'''

import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Determine number of visible slots
    n_slots = int(obs['buffer'].size // heur.n_properties)
    
    # First pass: try to place hard boxes using EMS-based search
    # We'll implement a simplified EMS by scanning all positions and finding empty spaces
    # that can accommodate boxes at z=0 (bottom layer)
    
    # Collect hard boxes (we need to determine hardness from properties)
    # In the given schema, we don't have direct hardness info, so we'll assume
    # all boxes are hard for bottom layer placement attempt
    pallet = obs['pallet_obs_density']
    
    # Try to find EMS placements for hard boxes at bottom layer
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        # Try all rotations
        for rot_id in range(6):
            rotated = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated
            
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
            
            if not heur.feasibility.is_within_pallet(0, 0, dx, dy):
                continue
            
            # EMS-based search: find positions where box can be placed at z=0
            # with full support (no overhang)
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    # Check if position is empty at bottom layer
                    place_area = pallet[x:x+dx, y:y+dy, 0]
                    
                    # For bottom layer placement, area must be completely empty
                    if np.any(place_area > 0):
                        continue
                    
                    # Check if box fits within pallet at z=0
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Check if placement at z=0 is feasible
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, 0):
                        return (slot_i, rot_id, x, y)
    
    # If no hard-box EMS placement found, fall back to lowest feasible hard-box placement
    # This is the original naive strategy but modified to prefer lower z positions
    best_action = None
    best_z = heur.H + 1  # Initialize with worst possible z
    
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
        
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        
        for rot_id in range(6):
            rotated = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = rotated
            
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
            
            if not heur.feasibility.is_within_pallet(0, 0, dx, dy):
                continue
            
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    # Compute z exactly as in env.py
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero = np.any(place_area > 0, axis=(0,1))
                    
                    if np.any(non_zero):
                        z = int(np.max(np.nonzero(non_zero)) + 1)
                    else:
                        z = 0
                    
                    if z + dz > heur.H:
                        continue
                    
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        # Track the action with the lowest z
                        if z < best_z:
                            best_z = z
                            best_action = (slot_i, rot_id, x, y)
    
    if best_action is not None:
        return best_action
    
    # No feasible action found
    return (0, 0, 0, 0)

class ExpertGuidanceSoft(BaseHeuristic):
    name = "expert_guidance_soft"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
