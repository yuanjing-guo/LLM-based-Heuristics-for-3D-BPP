import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    # Persistent state initialization
    if not hasattr(llm_policy, '_base_lock'):
        llm_policy._base_lock = 0
    if not hasattr(llm_policy, '_base_stable_steps'):
        llm_policy._base_stable_steps = 0
    if not hasattr(llm_policy, '_pending_base'):
        llm_policy._pending_base = None
    if not hasattr(llm_policy, '_pending_ttl'):
        llm_policy._pending_ttl = 0
    if not hasattr(llm_policy, '_place_fail_count'):
        llm_policy._place_fail_count = 0

    # Helper lambdas
    np_any = lambda arr: any(arr.flat) if hasattr(arr, 'flat') else any(arr)
    np_max = lambda arr: max(arr) if len(arr) > 0 else -1
    np_nonzero = lambda arr: [i for i, v in enumerate(arr) if v]

    # 1. Update persistent states
    llm_policy._base_lock = max(0, llm_policy._base_lock - 1)

    # Compute bottom-soft footprint count
    pallet = obs['pallet_obs_density']
    bottom_soft_count = 0
    if 'pallet_obs_softness' in obs and np_any(obs['pallet_obs_softness'] != 0):
        soft_grid = obs['pallet_obs_softness']
        for x in range(heur.X):
            for y in range(heur.Y):
                if pallet[x, y, 0] > 1e-6 and soft_grid[x, y, 0] > 0.5:
                    bottom_soft_count += 1
    if bottom_soft_count == 0:
        llm_policy._base_stable_steps = min(5, llm_policy._base_stable_steps + 1)
    else:
        llm_policy._base_stable_steps = 0

    # Update pending TTL
    if llm_policy._pending_base is not None:
        llm_policy._pending_ttl -= 1
        if llm_policy._pending_ttl <= 0:
            llm_policy._pending_base = None

    # 2. Determine mode and physics info
    N = len(obs['front_ids'])
    strict_online = (N == 1)
    physics_informative = ('buffer_physics' in obs) and ('pallet_obs_softness' in obs) and \
                          (np_any(obs['buffer_physics'] != 0) or np_any(obs['pallet_obs_softness'] != 0))

    # Helper: get hardness of a slot
    def is_hard_slot(slot):
        if not physics_informative:
            return True  # treat all as HARD when physics not informative
        if slot * 2 + 0 >= len(obs['buffer_physics']):
            return True
        softness = float(obs['buffer_physics'][2 * slot + 0])
        return softness <= 0.5

    # Helper: find removable pallet indices
    removable = [i for i, m in enumerate(obs['removable_mask']) if m == 1]

    # Helper: check if a placement overlaps pending region
    def overlaps_pending(x, y, dx, dy):
        if llm_policy._pending_base is None:
            return False
        px, py, pdx, pdy = llm_policy._pending_base
        return not (x + dx <= px or x >= px + pdx or y + dy <= py or y >= py + pdy)

    # 3. Phase A: Base-Hardify (replacement)
    phase_a_allowed = (
        physics_informative and
        bottom_soft_count > 0 and
        llm_policy._base_lock == 0 and
        llm_policy._base_stable_steps < 5
    )
    # Check existence of at least one HARD valid slot now
    hard_slot_exists = False
    for slot in range(N):
        if obs['front_ids'][slot] != 0 and is_hard_slot(slot):
            hard_slot_exists = True
            break
    phase_a_allowed = phase_a_allowed and hard_slot_exists

    if phase_a_allowed:
        # Find a bottom-soft footprint that is directly removable
        target_remove = None
        for idx, fp in enumerate(obs['pallet_footprints']):
            x, y, z, dx, dy, dz = fp
            if z == 0 and obs['removable_mask'][idx] == 1:
                if 'pallet_obs_softness' in obs:
                    # Check if any cell in this footprint is soft at bottom
                    soft_grid = obs['pallet_obs_softness']
                    for ix in range(x, x + dx):
                        for iy in range(y, y + dy):
                            if soft_grid[ix, iy, 0] > 0.5:
                                target_remove = idx
                                break
                        if target_remove is not None:
                            break
                else:
                    # Without softness grid, assume it's soft
                    target_remove = idx
                if target_remove is not None:
                    break

        if target_remove is not None:
            # A1: Directly removable soft bottom box
            llm_policy._pending_base = (x, y, dx, dy)
            llm_policy._pending_ttl = 2 if strict_online else 10
            llm_policy._place_fail_count = 0
            return (1, target_remove, 0, 0, 0)
        else:
            # A2: Remove a blocker above a soft bottom footprint
            # Find any soft bottom cell
            for x in range(heur.X):
                for y in range(heur.Y):
                    if pallet[x, y, 0] > 1e-6:
                        if 'pallet_obs_softness' in obs and obs['pallet_obs_softness'][x, y, 0] <= 0.5:
                            continue
                        # Check if there's something above it
                        for z in range(1, heur.H):
                            if pallet[x, y, z] > 1e-6:
                                # Find which pallet index occupies (x,y,z)
                                for idx, fp in enumerate(obs['pallet_footprints']):
                                    fx, fy, fz, fdx, fdy, fdz = fp
                                    if fx <= x < fx + fdx and fy <= y < fy + fdy and fz <= z < fz + fdz:
                                        if obs['removable_mask'][idx] == 1:
                                            llm_policy._pending_base = (x, y, 1, 1)
                                            llm_policy._pending_ttl = 2 if strict_online else 10
                                            llm_policy._place_fail_count = 0
                                            return (1, idx, 0, 0, 0)
                                break
            # If no blocker found, fall through to next phase

    # 4. Phase B: Fill pending hole
    if llm_policy._pending_base is not None:
        px, py, pdx, pdy = llm_policy._pending_base
        # Try to place a HARD box at z==0 overlapping pending region
        for slot in range(N):
            if obs['front_ids'][slot] == 0:
                continue
            if not is_hard_slot(slot):
                continue
            props = heur.get_slot_props(obs, slot)
            size_bins = heur.props_to_size_bins(props)
            for rot_id in range(6):
                dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
                # Search positions overlapping pending region
                for x in range(max(0, px - dx + 1), min(heur.X - dx + 1, px + pdx)):
                    for y in range(max(0, py - dy + 1), min(heur.Y - dy + 1, py + pdy)):
                        if not overlaps_pending(x, y, dx, dy):
                            continue
                        # Compute z
                        place_area = pallet[x:x+dx, y:y+dy, :]
                        non_zero_mask = [np_any(place_area[:, :, k] > 1e-6) for k in range(heur.H)]
                        z = np_max(np_nonzero(non_zero_mask)) + 1 if np_any(non_zero_mask) else 0
                        if z + dz > heur.H:
                            continue
                        # Non-overlap check
                        if z < heur.H and dz > 0:
                            blk = pallet[x:x+dx, y:y+dy, z:z+dz]
                            if np_any(blk > 1e-6):
                                continue
                        # Feasibility
                        if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                            continue
                        if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                            continue
                        # Success
                        llm_policy._pending_base = None
                        llm_policy._base_lock = 3 if N > 1 else 1
                        llm_policy._place_fail_count = 0
                        return (0, slot, rot_id, x, y)

    # 5. Phase C: Normal placement
    # First try HARD at z==0
    for slot in range(N):
        if obs['front_ids'][slot] == 0:
            continue
        if not is_hard_slot(slot):
            continue
        props = heur.get_slot_props(obs, slot)
        size_bins = heur.props_to_size_bins(props)
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    # Compute z
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero_mask = [np_any(place_area[:, :, k] > 1e-6) for k in range(heur.H)]
                    z = np_max(np_nonzero(non_zero_mask)) + 1 if np_any(non_zero_mask) else 0
                    if z != 0 or z + dz > heur.H:
                        continue
                    # Non-overlap check
                    if dz > 0:
                        blk = pallet[x:x+dx, y:y+dy, z:z+dz]
                        if np_any(blk > 1e-6):
                            continue
                    # Feasibility
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        continue
                    llm_policy._place_fail_count = 0
                    return (0, slot, rot_id, x, y)

    # In strict online mode, allow SOFT at z==0 if no HARD found
    if strict_online:
        for slot in range(N):
            if obs['front_ids'][slot] == 0:
                continue
            props = heur.get_slot_props(obs, slot)
            size_bins = heur.props_to_size_bins(props)
            for rot_id in range(6):
                dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
                for x in range(heur.X - dx + 1):
                    for y in range(heur.Y - dy + 1):
                        place_area = pallet[x:x+dx, y:y+dy, :]
                        non_zero_mask = [np_any(place_area[:, :, k] > 1e-6) for k in range(heur.H)]
                        z = np_max(np_nonzero(non_zero_mask)) + 1 if np_any(non_zero_mask) else 0
                        if z != 0 or z + dz > heur.H:
                            continue
                        if dz > 0:
                            blk = pallet[x:x+dx, y:y+dy, z:z+dz]
                            if np_any(blk > 1e-6):
                                continue
                        if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                            continue
                        if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                            continue
                        llm_policy._place_fail_count = 0
                        return (0, slot, rot_id, x, y)

    # Try any legal placement (any z)
    for slot in range(N):
        if obs['front_ids'][slot] == 0:
            continue
        props = heur.get_slot_props(obs, slot)
        size_bins = heur.props_to_size_bins(props)
        for rot_id in range(6):
            dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    place_area = pallet[x:x+dx, y:y+dy, :]
                    non_zero_mask = [np_any(place_area[:, :, k] > 1e-6) for k in range(heur.H)]
                    z = np_max(np_nonzero(non_zero_mask)) + 1 if np_any(non_zero_mask) else 0
                    if z + dz > heur.H:
                        continue
                    if dz > 0:
                        blk = pallet[x:x+dx, y:y+dy, z:z+dz]
                        if np_any(blk > 1e-6):
                            continue
                    if not heur.feasibility.is_within_pallet(x, y, dx, dy):
                        continue
                    if not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                        continue
                    llm_policy._place_fail_count = 0
                    return (0, slot, rot_id, x, y)

    # 6. Removal fallback
    llm_policy._place_fail_count += 1
    remove_condition = (strict_online and llm_policy._place_fail_count >= 1) or (not strict_online and llm_policy._place_fail_count >= 2)
    if remove_condition and removable:
        # Remove the highest removable pallet (simple heuristic)
        idx = removable[-1]
        llm_policy._place_fail_count = 0
        return (1, idx, 0, 0, 0)

    # Last resort
    return (0, 0, 0, 0, 0)

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
