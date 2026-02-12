import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    
    # Initialize persistent state
    if not hasattr(llm_policy, '_base_lock'):
        llm_policy._base_lock = 0
    if not hasattr(llm_policy, '_pending_base'):
        llm_policy._pending_base = None
    if not hasattr(llm_policy, '_recent'):
        llm_policy._recent = []
    if not hasattr(llm_policy, '_rm_cd'):
        llm_policy._rm_cd = {}
    if not hasattr(llm_policy, '_fail_count'):
        llm_policy._fail_count = 0
    
    # Update state
    llm_policy._base_lock = max(0, llm_policy._base_lock - 1)
    for k in list(llm_policy._rm_cd.keys()):
        llm_policy._rm_cd[k] -= 1
        if llm_policy._rm_cd[k] <= 0:
            del llm_policy._rm_cd[k]
    
    # Helper: check if action is in recent history
    def in_recent(action, lookback=4):
        return any(tuple(action) == tuple(a) for a in llm_policy._recent[-lookback:])
    
    # Helper: get physics info
    physics_informative = ('buffer_physics' in obs) and ('pallet_obs_softness' in obs) and \
                          (np.any(obs['buffer_physics'] != 0) or np.any(obs['pallet_obs_softness'] != 0))
    
    # Helper: classify slot as HARD/SOFT
    def is_hard(slot):
        if not physics_informative:
            return True
        if 2 * slot + 0 >= len(obs['buffer_physics']):
            return True
        softness = float(obs['buffer_physics'][2 * slot + 0])
        return softness <= 0.5
    
    # Helper: get bottom soft footprints
    def get_bottom_soft_footprints():
        soft_footprints = []
        pallet_count = obs['pallet_count']
        for i in range(pallet_count):
            fx, fy, fz, fdx, fdy, fdz = obs['pallet_footprints'][i]
            if fz != 0:
                continue
            # Check if region is soft
            region = obs['pallet_obs_softness'][fx:fx+fdx, fy:fy+fdy, fz:fdz]
            if region.size > 0 and np.mean(region) > 0.5:
                soft_footprints.append((i, fx, fy, fdx, fdy))
        return soft_footprints
    
    # Helper: find legal placements
    def find_placements(require_hard=False, require_z0=False, require_overlap_pending=False):
        placements = []
        N = len(obs['front_ids'])
        pallet = obs['pallet_obs_density']
        
        for slot in range(N):
            if obs['front_ids'][slot] == 0:
                continue
            if require_hard and not is_hard(slot):
                continue
            
            props = heur.get_slot_props(obs, slot)
            size_bins = heur.props_to_size_bins(props)
            
            for rot_id in range(6):
                dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
                
                for x in range(heur.X - dx + 1):
                    for y in range(heur.Y - dy + 1):
                        # Compute z
                        place_area = pallet[x:x+dx, y:y+dy, :]
                        non_zero_mask = np.any(place_area > 0, axis=(0,1))
                        if np.any(non_zero_mask):
                            z = int(np.max(np.nonzero(non_zero_mask)[0]) + 1)
                        else:
                            z = 0
                        
                        if require_z0 and z != 0:
                            continue
                        if z + dz > heur.H:
                            continue
                        
                        # Check overlap with pending region
                        if require_overlap_pending and llm_policy._pending_base:
                            px, py, pdx, pdy, _ = llm_policy._pending_base
                            if not (x < px + pdx and x + dx > px and y < py + pdy and y + dy > py):
                                continue
                        
                        # Non-overlap check
                        blk = pallet[x:x+dx, y:y+dy, z:z+dz]
                        if blk.size == 0:
                            continue
                        if np.any(blk > 1e-6):
                            continue
                        
                        # Feasibility
                        if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                            continue
                        if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                            continue
                        
                        placements.append((slot, rot_id, x, y, z, is_hard(slot)))
        
        return placements
    
    # Helper: find removable blocker above a bottom soft footprint
    def find_blocker_above(bottom_fx, bottom_fy, bottom_fdx, bottom_fdy):
        best_i = -1
        best_score = -1
        pallet_count = obs['pallet_count']
        
        for i in range(pallet_count):
            if obs['removable_mask'][i] <= 0:
                continue
            if i in llm_policy._rm_cd:
                continue
            
            fx, fy, fz, fdx, fdy, fdz = obs['pallet_footprints'][i]
            if fz == 0:
                continue
            
            # Check XY overlap
            overlap_x = max(0, min(bottom_fx + bottom_fdx, fx + fdx) - max(bottom_fx, fx))
            overlap_y = max(0, min(bottom_fy + bottom_fdy, fy + fdy) - max(bottom_fy, fy))
            if overlap_x == 0 or overlap_y == 0:
                continue
            
            score = (fz + fdz) * 100 + overlap_x * overlap_y
            if score > best_score:
                best_score = score
                best_i = i
        
        return best_i
    
    # Main logic
    N = len(obs['front_ids'])
    strict_online = (N == 1)
    
    # Phase A: Base-hardify
    phase_a_allowed = (physics_informative and 
                       llm_policy._base_lock == 0 and 
                       any(obs['front_ids'][slot] != 0 and is_hard(slot) for slot in range(N)))
    
    if phase_a_allowed:
        soft_footprints = get_bottom_soft_footprints()
        if soft_footprints:
            # Try direct removal first
            for i, fx, fy, fdx, fdy in soft_footprints:
                if obs['removable_mask'][i] > 0 and i not in llm_policy._rm_cd:
                    action = (1, i, 0, 0, 0)
                    if not in_recent(action):
                        llm_policy._pending_base = (fx, fy, fdx, fdy, 2 if strict_online else 10)
                        llm_policy._recent.append(action)
                        if len(llm_policy._recent) > 8:
                            llm_policy._recent.pop(0)
                        llm_policy._rm_cd[i] = 3
                        return action
            
            # Try removing a blocker
            for i, fx, fy, fdx, fdy in soft_footprints:
                blocker = find_blocker_above(fx, fy, fdx, fdy)
                if blocker != -1:
                    action = (1, blocker, 0, 0, 0)
                    if not in_recent(action):
                        llm_policy._pending_base = (fx, fy, fdx, fdy, 2 if strict_online else 10)
                        llm_policy._recent.append(action)
                        if len(llm_policy._recent) > 8:
                            llm_policy._recent.pop(0)
                        llm_policy._rm_cd[blocker] = 3
                        return action
    
    # Phase B: Fill pending hole
    if llm_policy._pending_base:
        px, py, pdx, pdy, ttl = llm_policy._pending_base
        ttl -= 1
        if ttl <= 0:
            llm_policy._pending_base = None
        else:
            llm_policy._pending_base = (px, py, pdx, pdy, ttl)
            
            placements = find_placements(
                require_hard=(not strict_online),
                require_z0=True,
                require_overlap_pending=True
            )
            
            # In strict online mode, if no HARD placements, try SOFT
            if strict_online and not placements:
                placements = find_placements(
                    require_hard=False,
                    require_z0=True,
                    require_overlap_pending=True
                )
            
            for slot, rot_id, x, y, z, hard in placements:
                action = (0, slot, rot_id, x, y)
                if not in_recent(action):
                    llm_policy._pending_base = None
                    llm_policy._base_lock = 3 if N > 1 else 1
                    llm_policy._recent.append(action)
                    if len(llm_policy._recent) > 8:
                        llm_policy._recent.pop(0)
                    llm_policy._fail_count = 0
                    return action
    
    # Phase C: Normal placement
    # Try z==0 HARD first
    placements = find_placements(require_hard=True, require_z0=False)
    z0_hard = [p for p in placements if p[4] == 0]
    if z0_hard:
        for slot, rot_id, x, y, z, hard in z0_hard:
            action = (0, slot, rot_id, x, y)
            if not in_recent(action):
                llm_policy._recent.append(action)
                if len(llm_policy._recent) > 8:
                    llm_policy._recent.pop(0)
                llm_policy._fail_count = 0
                return action
    
    # Try any z==0 placement
    placements = find_placements(require_hard=False, require_z0=False)
    z0_any = [p for p in placements if p[4] == 0]
    if z0_any:
        for slot, rot_id, x, y, z, hard in z0_any:
            action = (0, slot, rot_id, x, y)
            if not in_recent(action):
                llm_policy._recent.append(action)
                if len(llm_policy._recent) > 8:
                    llm_policy._recent.pop(0)
                llm_policy._fail_count = 0
                return action
    
    # Try any placement
    if placements:
        for slot, rot_id, x, y, z, hard in placements:
            action = (0, slot, rot_id, x, y)
            if not in_recent(action):
                llm_policy._recent.append(action)
                if len(llm_policy._recent) > 8:
                    llm_policy._recent.pop(0)
                llm_policy._fail_count = 0
                return action
    
    # No placement found
    llm_policy._fail_count += 1
    
    # General REMOVE
    remove_allowed = (strict_online and llm_policy._fail_count >= 1) or \
                     (not strict_online and llm_policy._fail_count >= 2)
    
    if remove_allowed:
        for i in range(obs['pallet_count']):
            if obs['removable_mask'][i] > 0 and i not in llm_policy._rm_cd:
                action = (1, i, 0, 0, 0)
                if not in_recent(action):
                    llm_policy._recent.append(action)
                    if len(llm_policy._recent) > 8:
                        llm_policy._recent.pop(0)
                    llm_policy._rm_cd[i] = 3
                    llm_policy._fail_count = 0
                    return action
    
    # Last resort
    return (0, 0, 0, 0, 0)

class ArchivedHeuristic(BaseHeuristic):
    name = "worked1"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
