import numpy as np

def llm_policy(heur, obs):
    n_slots = int(obs['buffer'].size // heur.n_properties)
    
    best_score = -float('inf')
    best_action = (0, 0, 0, 0)
    
    for slot_i in range(n_slots):
        if heur.slot_is_empty(obs, slot_i):
            continue
            
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        current_hardness = props[3]
        
        for rot_id in range(6):
            dx, dy, dz = heur.rotate_size_bins(size_bins, rot_id)
            
            if dx > heur.X or dy > heur.Y or dz > heur.H:
                continue
                
            if not heur.feasibility.is_within_pallet(0, 0, dx, dy):
                continue
                
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    pallet = obs['pallet_obs_density']
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero = np.any(place_area > 0, axis=(0, 1))
                    
                    if np.any(non_zero):
                        z = int(np.max(np.nonzero(non_zero)) + 1)
                    else:
                        z = 0
                    
                    if z + dz > heur.H:
                        continue
                    
                    # Check "don't put heavy boxes onto soft boxes"
                    if current_hardness > 0.5 and z > 0:
                        support_layer = pallet[x:x+dx, y:y+dy, z-1]
                        soft_mask = support_layer > 0
                        if np.any(soft_mask):
                            supporting_densities = support_layer[soft_mask]
                            if np.any(supporting_densities < 0.5):
                                continue
                    
                    if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        # Scoring: prioritize lower z, then larger volume, then higher hardness
                        volume = dx * dy * dz
                        score = -z * 1000 + volume * 10 + current_hardness
                        
                        if score > best_score:
                            best_score = score
                            best_action = (slot_i, rot_id, x, y)
    
    return best_action
