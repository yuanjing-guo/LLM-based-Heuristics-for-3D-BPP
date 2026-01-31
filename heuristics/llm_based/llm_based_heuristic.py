import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur, obs):
    n_slots = int(obs['buffer'].size // heur.n_properties)
    for slot_i in range(n_slots):
        props = heur.get_slot_props(obs, slot_i)
        if heur.slot_is_empty(obs, slot_i):
            continue
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0 or np.any(size_bins <= 0):
            continue
        for rot_id in range(6):
            size_rotated = heur.rotate_size_bins(size_bins, rot_id)
            dx, dy, dz = int(size_rotated[0]), int(size_rotated[1]), int(size_rotated[2])
            for x in range(heur.X - dx + 1):
                for y in range(heur.Y - dy + 1):
                    if heur.feasibility.is_within_pallet(x, y, dx, dy):
                        pallet = obs['pallet_obs_density']
                        place_area = pallet[x:x+dx, y:y+dy, :]
                        non_zero = np.any(place_area > 0, axis=(0,1))
                        z = int(np.max(np.nonzero(non_zero)) + 1) if np.any(non_zero) else 0
                        if z + dz > heur.H:
                            continue
                        if heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z):
                            return (slot_i, rot_id, x, y)
    return (0, 0, 0, 0)

class GeneratedHeuristic(BaseHeuristic):
    name = "llm_generated"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
