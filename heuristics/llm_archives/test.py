import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    n_slots = int(obs['buffer'].size // heur.n_properties)
    
    # First pass: try to place at low heights (z <= H - dz - 2)
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
            
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Try low heights first
                    max_z_low = max(0, heur.H - dz - 2)
                    for z in range(max_z_low + 1):
                        if heur.feasibility.is_feasible(obs['pallet_obs_density'], x, y, dx, dy, dz, z):
                            return (slot_i, rot_id, x, y)
    
    # Second pass: if no low-height placement found, try other positions at any height
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
            
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    
                    # Try all heights but don't get stuck at high z
                    for z in range(heur.H - dz + 1):
                        if heur.feasibility.is_feasible(obs['pallet_obs_density'], x, y, dx, dy, dz, z):
                            return (slot_i, rot_id, x, y)
    
    return (0, 0, 0, 0)

class Test(BaseHeuristic):
    name = "test"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
